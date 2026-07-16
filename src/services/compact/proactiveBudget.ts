/**
 * Proactive message budgeting — content-aware pruning of redundant tool outputs.
 *
 * ## The Core Problem
 * OpenClaude sends ALL session messages (up to 500k+ tokens) on every API call.
 * A simple "bye" message takes hours because it re-sends everything since session
 * start — every `Read` result, every `Write`, every `Edit`, from the very first turn.
 *
 * ## The Insight
 * Old file reads are useless because files change between turns. The model does not
 * need every `cat` result — it only needs the LATEST read/write/edit per file.
 * Keeping all user messages and assistant reasoning intact while stripping old
 * redundant tool outputs preserves 100% intelligence while reducing token count
 * from 500k to 5k-10k.
 *
 * ## Strategy
 * 1. First pass: build a map of tool_use_id → {toolName, filePath} from assistant
 *    messages' tool_use blocks (the `input.file_path` parameter).
 * 2. Walk messages backwards (newest → oldest), tracking which file_paths have
 *    been "touched" by MORE RECENT tool results.
 * 3. For tool results (Read/Write/Edit) whose file_path has already been seen
 *    in a newer result → strip the content to a short marker.
 * 4. Keep ALL user messages, assistant reasoning/decisions intact — we only
 *    replace the content of tool_result blocks, never remove messages.
 * 5. If still over budget after content-aware pruning, the pruned messages are
 *    sent as-is rather than dropping messages (which would cause context forgetting).
 * 6. Target: user-configurable via `proactiveBudgetLimit` config. 0 = disabled.
 *    Default: 100_000. Recommended range: 25_000–100_000.
 */

import type { Message } from '../../types/message.js'
import { roughTokenCountEstimationForMessages } from '../tokenEstimation.js'
import { getGlobalConfig } from '../../utils/config.js'
import { getSettings_DEPRECATED } from '../../utils/settings/settings.js'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Default target token budget for the pruned message list.
 * 100K = safe default: strips old redundant reads while preserving
 * full context awareness (content-aware pruning never drops messages).
 * Users can override via the `proactiveBudgetLimit` config setting.
 * Set to 0 to disable (send full context, original behavior).
 */
export const PROACTIVE_BUDGET_TARGET_TOKENS_DEFAULT = 100_000

/**
 * Tools whose outputs contain file contents that become stale between turns.
 * Only Read is included — Write/Edit acknowledgments are tiny and not worth
 * stripping (they cost ~10-20 tokens and the model may reference them).
 */
export const FILE_CONTENT_TOOLS = new Set(['Read'])

/**
 * Tools whose output is dedup-based: only strip when the same identity
 * (command/pattern/url) was re-run. A safety cap prevents pathological
 * accumulation of many unique results.
 */
export const DEDUP_TOOLS = new Set(['Bash', 'Grep', 'Glob', 'WebFetch'])

/**
 * Safety cap: maximum unique tool results to keep per tool type.
 * Prevents context explosion when agent runs many different commands
 * (e.g. 50 unique greps). Dedup handles normal cases; cap handles edge cases.
 */
export const SAFETY_CAP = 20

/**
 * Number of most recent user messages within which tool results are NEVER
 * stripped. The model is most likely to reference results it just produced.
 */
export const RECENCY_GUARD_TURNS = 5

/**
 * Generate a marker for a stripped tool result.
 * Includes a descriptive label so the model can locate the latest result in
 * context or re-run the tool if needed.
 * Kept short to minimize token overhead.
 */
export function strippedMarker(label: string): string {
  return `[Content from earlier ${label} — see latest result in context]`
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProactiveBudgetResult {
  /** The (possibly pruned) message list. */
  messages: Message[]
  /** Whether any pruning was applied. */
  wasPruned: boolean
  /** Estimated token count after pruning. */
  estimatedTokens: number
  /** Number of tool result content blocks that were stripped. */
  strippedCount: number
}

/**
 * Metadata extracted from an assistant message's tool_use block.
 */
interface ToolUseMeta {
  /** Tool name, e.g. "Read", "Write", "Edit", "Bash", etc. */
  name: string
  /** The raw input object passed to the tool. */
  input: Record<string, unknown>
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a map of tool_use_id → {name, input} from assistant messages.
 *
 * Iterates all messages once to collect every tool_use block. This map
 * is the key lookup for linking a tool_result (which only carries
 * tool_use_id) back to the tool name and its input parameters (which
 * contain file_path).
 */
export function buildToolUseMap(messages: Message[]): Map<string, ToolUseMeta> {
  const map = new Map<string, ToolUseMeta>()

  for (const msg of messages) {
    if (msg.type !== 'assistant') continue
    const content = msg.message.content
    if (!Array.isArray(content)) continue

    for (const block of content) {
      if (block.type !== 'tool_use') continue
      const id = (block as any).id
      const name = (block as any).name
      const input = (block as any).input
      if (typeof id === 'string' && typeof name === 'string' && input) {
        map.set(id, {
          name,
          input: typeof input === 'object' && input !== null
            ? (input as Record<string, unknown>)
            : {},
        })
      }
    }
  }

  return map
}

/**
 * Extract the file_path from a tool's input parameters.
 *
 * All file-operation tools (Read, Write, Edit) accept a `file_path`
 * parameter containing the absolute path of the target file. This is
 * the canonical way to identify which file a tool result refers to,
 * rather than trying to parse it out of the natural-language output.
 */
export function getFilePathFromInput(input: Record<string, unknown>): string | undefined {
  const fp = input.file_path
  if (typeof fp === 'string' && fp.length > 0) return fp
  return undefined
}

/**
 * Check whether a tool result's content is large enough to be worth stripping.
 * Small results (e.g. "File not found", brief errors) cost nothing to keep.
 */
export function isLargeContent(content: unknown): boolean {
  if (typeof content === 'string') return content.length > 200
  if (Array.isArray(content)) {
    let total = 0
    for (const block of content) {
      if (typeof block === 'object' && block !== null && 'text' in block) {
        total += (block.text as string).length
      }
    }
    return total > 200
  }
  return false
}

/**
 * Generate a human-readable label for a tool result, used in the stripped
 * marker so the model can identify what was removed.
 */
export function getResultLabel(
  toolName: string,
  input: Record<string, unknown>,
): string {
  switch (toolName) {
    case 'Read':
    case 'Write':
    case 'Edit': {
      const fp = input.file_path
      if (typeof fp === 'string' && fp.length > 0) return `read of ${fp}`
      return `${toolName} result`
    }
    case 'Bash': {
      const cmd = input.command
      if (typeof cmd === 'string') return `command: ${cmd.slice(0, 120)}`
      return 'Bash result'
    }
    case 'Grep': {
      const pattern = input.pattern
      if (typeof pattern === 'string') return `grep: ${pattern.slice(0, 120)}`
      return 'Grep result'
    }
    case 'Glob': {
      const pattern = input.pattern
      if (typeof pattern === 'string') return `glob: ${pattern.slice(0, 120)}`
      return 'Glob result'
    }
    case 'WebFetch': {
      const url = input.url
      if (typeof url === 'string') return `fetch: ${url.slice(0, 120)}`
      return 'WebFetch result'
    }
    default:
      return `${toolName} result`
  }
}

/**
 * Extract a stable identity for dedup-based pruning.
 *
 * Different tools use different parameters to identify what resource they
 * operate on:
 *   Bash   → command
 *   Grep   → pattern[:path]
 *   Glob   → pattern[:path]
 *   WebFetch → url
 *
 * Returns a namespaced identity string or undefined when no identity
 * can be determined.
 */
export function getResourceIdentity(
  toolName: string,
  input: Record<string, unknown>,
): string | undefined {
  switch (toolName) {
    case 'Bash': {
      const cmd = input.command
      if (typeof cmd === 'string' && cmd.length > 0) return `cmd:${cmd}`
      return undefined
    }
    case 'Grep': {
      const pattern = input.pattern
      if (typeof pattern === 'string' && pattern.length > 0) {
        const path = input.path
        return `grep:${pattern}|${typeof path === 'string' ? path : ''}`
      }
      return undefined
    }
    case 'Glob': {
      const pattern = input.pattern
      if (typeof pattern === 'string' && pattern.length > 0) {
        const path = input.path
        return `glob:${pattern}|${typeof path === 'string' ? path : ''}`
      }
      return undefined
    }
    case 'WebFetch': {
      const url = input.url
      if (typeof url === 'string' && url.length > 0) return `url:${url}`
      return undefined
    }
    default:
      return undefined
  }
}

/**
 * Replace a tool_result block's content with a marker.
 * Modifies the block in-place.
 */
export function stripToolResultContent(
  block: Record<string, unknown>,
  label: string,
): void {
  const marker = strippedMarker(label)
  if (typeof block.content === 'string') {
    block.content = marker
  } else if (Array.isArray(block.content)) {
    block.content = [{ type: 'text', text: marker }]
  }
}

// ---------------------------------------------------------------------------
// Phase 1: Content-aware pruning
// ---------------------------------------------------------------------------

/**
 * Walk messages backwards and strip redundant tool outputs.
 *
 * Algorithm:
 * 1. First pass → build tool_use_id → {name, input} map from assistant
 *    messages.
 * 2. Second pass (backwards) → track which file_paths have been seen in
 *    newer messages. When a tool_result's file_path has already been seen
 *    in a newer full read, strip its content.
 *
 * Only the latest full Read per file is preserved. Older reads for the
 * same file are stripped as redundant — the model only needs the most
 * recent file snapshot.
 *
 * This preserves ALL user messages, assistant reasoning, and decisions.
 * Only the content of old redundant tool_result blocks is replaced.
 */
export function pruneRedundantToolOutputs(
  messages: Message[] | readonly Message[],
  toolUseMap: Map<string, ToolUseMeta>,
  recencyGuard = RECENCY_GUARD_TURNS,
): { messages: Message[]; strippedCount: number } {
  const seenFilePaths = new Set<string>()
  const seenIdentities = new Map<string, number>() // identity → count of unique results seen
  const identityFirstSeen = new Map<string, number>() // identity → reversed index (for safety cap)
  let strippedCount = 0
  const strippedToolUseIds = new Set<string>()

  // Walk messages backwards (newest first) so the most recent result per file
  // is seen first and preserved, while older results for the same file are
  // stripped. We avoid cloning until we know a change is needed.
  const reversed = [...messages].reverse()

  // Phase 1: First pass — count user messages from the end for recency guard.
  // We need to know the index of each message in the reversed array to apply
  // the recency guard (skip stripping results within the last N user messages).
  let userMessageCount = 0
  const reversedUserIndex = new Map<number, number>() // reversed index → user message count
  for (let ri = 0; ri < reversed.length; ri++) {
    const msg = reversed[ri]!
    if (msg.type === 'user') {
      reversedUserIndex.set(ri, userMessageCount)
      userMessageCount++
    }
  }

  const result = reversed.map((msg, ri) => {
    // ── Assistant messages: strip tool_use blocks for stripped results ──
    if (msg.type === 'assistant') {
      if (strippedToolUseIds.size > 0) {
        const content = msg.message.content
        if (!Array.isArray(content)) return msg
        let workingContent: typeof content | undefined
        for (let bi = content.length - 1; bi >= 0; bi--) {
          const block = content[bi]!
          if (block.type !== 'tool_use') continue
          const id = block.id as string
          if (!strippedToolUseIds.has(id)) continue
          const meta = toolUseMap.get(id)
          const toolName = meta?.name ?? 'tool'
          if (!workingContent) workingContent = [...content]
          workingContent[bi] = {
            type: 'text' as const,
            text: `[tool_use: ${toolName} — paired result stripped as redundant]`,
          }
        }
        if (workingContent) {
          return {
            ...msg,
            message: { ...msg.message, content: workingContent },
          } as Message
        }
      }
      return msg
    }

    if (msg.type !== 'user') return msg

    // ── Recency guard: don't strip recent results, but still claim ─────
    const userIdx = reversedUserIndex.get(ri) ?? userMessageCount
    const canStrip = recencyGuard <= 0 || userIdx >= recencyGuard

    const content = msg.message.content
    if (!Array.isArray(content)) return msg

    const hasToolResult = content.some(b => b.type === 'tool_result')
    if (!hasToolResult) return msg

    // Lazy clone — only when a mutation is actually needed
    let workingContent: typeof content | undefined
    let changed = false

    for (let bi = content.length - 1; bi >= 0; bi--) {
      const block = content[bi]!
      if (block.type !== 'tool_result') continue

      const toolUseId = block.tool_use_id as string | undefined
      if (!toolUseId) continue

      const meta = toolUseMap.get(toolUseId)
      if (!meta) continue

      // Only process tools in FILE_CONTENT_TOOLS or DEDUP_TOOLS
      const isFileTool = FILE_CONTENT_TOOLS.has(meta.name)
      const isDedupTool = DEDUP_TOOLS.has(meta.name)
      if (!isFileTool && !isDedupTool) continue

      const filePath = getFilePathFromInput(meta.input)
      const identity = isDedupTool ? getResourceIdentity(meta.name, meta.input) : undefined

      // ── Should we strip this result? ─────────────────────────────────

      let shouldStrip = false

      // Check 1: File-path redundancy (Read) — a newer full Read has
      // already covered this file.
      if (isFileTool && filePath && seenFilePaths.has(filePath)) {
        shouldStrip = true
      }

      // Check 2: Dedup-based — strip when the same identity was seen in
      // a newer result. Claiming is handled below alongside Read claiming.
      if (!shouldStrip && isDedupTool && identity) {
        if (seenIdentities.has(identity)) {
          // Duplicate: same command/pattern/url was re-run → strip
          shouldStrip = true
        } else if (identityFirstSeen.size >= SAFETY_CAP) {
          // Safety cap hit: strip even unique results beyond the cap
          shouldStrip = true
        }
      }

      if (shouldStrip && canStrip) {
        // ── Strip this result ──────────────────────────────────────────
        if (isLargeContent(block.content)) {
          if (!workingContent) {
            workingContent = structuredClone(content) as typeof content
          }
          const label = getResultLabel(meta.name, meta.input)
          stripToolResultContent(
            workingContent[bi] as unknown as Record<string, unknown>,
            label,
          )
          changed = true
          strippedCount++
          strippedToolUseIds.add(toolUseId)
        }
      } else if (isLargeContent(block.content)) {
        // ── Claim resources for future redundancy detection ────────────
        // Always claim regardless of recency guard — this ensures that
        // older duplicates get detected even when the most recent result
        // is within the recency window.
        if (isFileTool && meta.name === 'Read' && filePath) {
          // Full Read claims the file_path — older reads for the same file
          // get stripped. (Partial reads with offset/limit skip claiming.)
          if (meta.input.offset === undefined && meta.input.limit === undefined) {
            seenFilePaths.add(filePath)
          }
        }
        if (isDedupTool && identity && !seenIdentities.has(identity)) {
          // Claim identity even when canStrip is false — this ensures
          // older duplicates outside the recency guard get detected.
          const uniqueCount = identityFirstSeen.size
          if (uniqueCount < SAFETY_CAP) {
            seenIdentities.set(identity, 1)
            identityFirstSeen.set(identity, ri)
          }
        }
      }
    }

    if (changed) {
      return { ...msg, message: { ...msg.message, content: workingContent! } } as Message
    }

    return msg
  })

  return { messages: result.reverse(), strippedCount }
}

// ---------------------------------------------------------------------------
// Phase 2: Thinking block pruning
// ---------------------------------------------------------------------------

/**
 * Walk messages backwards and strip old thinking blocks from assistant
 * messages, keeping only the most recent one.
 *
 * Qwen3.6 and other reasoning models produce massive thinking blocks
 * (often 10K+ tokens per turn). Old thinking is redundant — only the
 * latest reasoning chain matters for the current context. Stripping old
 * thinking blocks saves significant tokens while preserving all tool_use
 * and text blocks (the actual decisions and actions).
 *
 * Algorithm:
 * 1. Walk messages backwards (newest → oldest).
 * 2. Track whether we've seen a thinking block.
 * 3. The first (most recent) assistant message with thinking is kept intact.
 * 4. All older assistant messages have their thinking/redacted_thinking
 *    blocks stripped, preserving text and tool_use blocks.
 * 5. If a message has ONLY thinking blocks (no text/tool_use), it gets a
 *    minimal placeholder so the message structure is preserved.
 */
export function pruneOldThinkingBlocks(
  messages: Message[],
): { messages: Message[]; strippedCount: number } {
  let hasSeenThinking = false
  let strippedCount = 0

  const reversed = [...messages].reverse()
  const result = reversed.map(msg => {
    if (msg.type !== 'assistant') return msg

    const content = msg.message.content
    if (!Array.isArray(content)) return msg

    // Check if this message has thinking blocks
    const hasThinking = content.some(
      (block: any) => block.type === 'thinking' || block.type === 'redacted_thinking',
    )
    if (!hasThinking) return msg

    if (!hasSeenThinking) {
      // This is the most recent assistant message with thinking — keep it
      hasSeenThinking = true
      return msg
    }

    // Strip thinking blocks from this older message
    const filtered = content.filter(
      (block: any) => block.type !== 'thinking' && block.type !== 'redacted_thinking',
    )

    if (filtered.length === content.length) return msg // No thinking blocks to strip (shouldn't happen since hasThinking was true)

    strippedCount++

    if (filtered.length === 0) {
      // All blocks were thinking — keep a minimal placeholder so the
      // message structure is preserved and the model doesn't lose a turn.
      return {
        ...msg,
        message: {
          ...msg.message,
          content: [{ type: 'text' as const, text: '[thinking stripped]' }],
        },
      } as Message
    }

    return {
      ...msg,
      message: {
        ...msg.message,
        content: filtered,
      },
    } as Message
  })

  return { messages: result.reverse(), strippedCount }
}

/**
 * Apply proactive message budgeting to strip redundant tool outputs.
 *
 * Call this AFTER `getMessagesAfterCompactBoundary()` but BEFORE the message
 * list is sent to the model. It reduces token count by removing old redundant
 * file-read/write/edit results while preserving all user messages and assistant
 * reasoning.
 *
 * IMPORTANT: This function ONLY does content-aware pruning — it strips old
 * redundant tool outputs but NEVER removes messages. This ensures the model
 * never loses context awareness. If still over budget after pruning, the
 * messages are sent as-is rather than dropping middle messages (which was
 * causing the model to forget where it was).
 *
 * Reads `proactiveBudgetLimit` from user config or settings.json:
 *  - 0 = disabled (send full context, original behavior)
 *  - undefined = uses default (100K)
 *  - positive number = target token budget (e.g. 50000 = try to stay under 50K)
 *  Precedence: /config > settings.json > code default (100K).
 */
export function applyProactiveBudget(
  messages: Message[],
): ProactiveBudgetResult {
  // Read user config. 0 = disabled. Unset defaults to 100K.
  // Check settings.json as fallback when /config hasn't set this value.
  const configValue = getGlobalConfig().proactiveBudgetLimit
  const settingsValue = getSettings_DEPRECATED()?.proactiveBudgetLimit
  const limit = configValue ?? settingsValue ?? PROACTIVE_BUDGET_TARGET_TOKENS_DEFAULT
  if (limit <= 0) {
    return {
      messages,
      wasPruned: false,
      estimatedTokens: roughTokenCountEstimationForMessages(messages),
      strippedCount: 0,
    }
  }

  if (messages.length === 0) {
    return {
      messages: [],
      wasPruned: false,
      estimatedTokens: 0,
      strippedCount: 0,
    }
  }

  const targetTokens = limit
  const initialTokens = roughTokenCountEstimationForMessages(messages)

  // Skip pruning if already under target budget
  if (initialTokens <= targetTokens) {
    return {
      messages,
      wasPruned: false,
      estimatedTokens: initialTokens,
      strippedCount: 0,
    }
  }

  // Phase 1: Content-aware pruning — strip old redundant tool outputs.
  // This preserves ALL user messages and assistant reasoning, only replacing
  // old tool result content with a short marker. The model keeps full context
  // awareness. If still over budget after pruning, we send what we have
  // rather than dropping messages (which caused context forgetting).
  const toolUseMap = buildToolUseMap(messages)
  const { messages: prunedMessages, strippedCount } =
    pruneRedundantToolOutputs(messages, toolUseMap)

  // Phase 2: Strip old thinking blocks from older assistant messages.
  // Reasoning models (Qwen3.6, DeepSeek, etc.) produce massive thinking
  // blocks that are redundant across turns — only the latest reasoning
  // chain matters. This preserves all text and tool_use blocks.
  const { messages: finalMessages, strippedCount: thinkingStripped } =
    pruneOldThinkingBlocks(prunedMessages)

  return {
    messages: finalMessages,
    wasPruned: strippedCount > 0 || thinkingStripped > 0,
    estimatedTokens: roughTokenCountEstimationForMessages(finalMessages),
    strippedCount: strippedCount + thinkingStripped,
  }
}
