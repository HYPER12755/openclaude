import { describe, expect, test } from 'bun:test'

import type { Message } from '../../types/message.js'
import { createAssistantMessage, createUserMessage } from '../../utils/messages.js'
import { saveGlobalConfig } from '../../utils/config.js'

import {
  applyProactiveBudget,
  buildToolUseMap,
  DEDUP_TOOLS,
  getFilePathFromInput,
  getResourceIdentity,
  isLargeContent,
  pruneOldThinkingBlocks,
  pruneRedundantToolOutputs,
  SAFETY_CAP,
  strippedMarker,
  stripToolResultContent,
} from './proactiveBudget.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function assistantWithToolUse(
  toolName: string,
  toolId: string,
  input: Record<string, unknown> = {},
): Message {
  return createAssistantMessage({
    content: [
      {
        type: 'tool_use' as const,
        id: toolId,
        name: toolName,
        input,
      },
    ],
  })
}

function userWithToolResult(toolId: string, content: string): Message {
  return createUserMessage({
    content: [
      {
        type: 'tool_result' as const,
        tool_use_id: toolId,
        content,
      },
    ],
  })
}

function userWithToolResultArray(toolId: string, text: string): Message {
  return createUserMessage({
    content: [
      {
        type: 'tool_result' as const,
        tool_use_id: toolId,
        content: [{ type: 'text' as const, text }],
      },
    ],
  })
}

function userTextMessage(text: string): Message {
  return createUserMessage({ content: text })
}

// ---------------------------------------------------------------------------
// buildToolUseMap
// ---------------------------------------------------------------------------

describe('buildToolUseMap', () => {
  test('extracts tool_use blocks from assistant messages', () => {
    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'content'),
      assistantWithToolUse('Bash', 'tool-2', { command: 'ls' }),
    ]

    const map = buildToolUseMap(messages)
    expect(map.size).toBe(2)
    expect(map.get('tool-1')!.name).toBe('Read')
    expect(map.get('tool-1')!.input).toEqual({ file_path: '/a.ts' })
    expect(map.get('tool-2')!.name).toBe('Bash')
    expect(map.get('tool-2')!.input).toEqual({ command: 'ls' })
  })

  test('skips non-assistant messages', () => {
    const map = buildToolUseMap([userWithToolResult('t1', 'x'), userTextMessage('hi')])
    expect(map.size).toBe(0)
  })

  test('skips content blocks that are not tool_use', () => {
    const map = buildToolUseMap([
      createAssistantMessage({ content: [{ type: 'text' as const, text: 'hello' }] }),
    ])
    expect(map.size).toBe(0)
  })

  test('handles tool_use blocks with non-string id (empty string is still a string)', () => {
    // typeof '' === 'string' is true, so empty-string id IS added
    const msg = createAssistantMessage({
      content: [
        { type: 'tool_use' as const, id: 't1', name: 'Read', input: { file_path: '/a.ts' } },
        { type: 'tool_use' as const, id: '' as any, name: 'Read', input: {} },
      ],
    })

    const map = buildToolUseMap([msg])
    // Both blocks have typeof id === 'string', both are valid
    expect(map.size).toBe(2)
  })
})

// ---------------------------------------------------------------------------
// getFilePathFromInput
// ---------------------------------------------------------------------------

describe('getFilePathFromInput', () => {
  test('extracts file_path from input', () => {
    expect(getFilePathFromInput({ file_path: '/a.ts' })).toBe('/a.ts')
  })

  test('returns undefined when file_path is missing', () => {
    expect(getFilePathFromInput({})).toBeUndefined()
  })

  test('returns undefined when file_path is empty string', () => {
    expect(getFilePathFromInput({ file_path: '' })).toBeUndefined()
  })

  test('returns undefined when file_path is not a string', () => {
    expect(getFilePathFromInput({ file_path: 42 })).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// isLargeContent
// ---------------------------------------------------------------------------

describe('isLargeContent', () => {
  test('returns true for string content > 200 chars', () => {
    expect(isLargeContent('x'.repeat(201))).toBe(true)
  })

  test('returns false for string content <= 200 chars', () => {
    expect(isLargeContent('x'.repeat(200))).toBe(false)
    expect(isLargeContent('small error')).toBe(false)
    expect(isLargeContent('')).toBe(false)
  })

  test('returns true for array content with total text > 200 chars', () => {
    expect(isLargeContent([{ type: 'text' as const, text: 'x'.repeat(201) }])).toBe(true)
  })

  test('returns false for array content with total text <= 200 chars', () => {
    expect(isLargeContent([{ type: 'text' as const, text: 'small' }])).toBe(false)
  })

  test('returns false for non-string non-array content', () => {
    expect(isLargeContent(null)).toBe(false)
    expect(isLargeContent(undefined)).toBe(false)
    expect(isLargeContent(42)).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// stripToolResultContent
// ---------------------------------------------------------------------------

describe('stripToolResultContent', () => {
  test('replaces string content with marker', () => {
    const block = { type: 'tool_result', tool_use_id: 't1', content: 'old content' }
    stripToolResultContent(block, 'read of /a.ts')
    expect(block.content).toBe(strippedMarker('read of /a.ts'))
  })

  test('replaces array content with marker array', () => {
    const block = {
      type: 'tool_result',
      tool_use_id: 't1',
      content: [{ type: 'text', text: 'old content' }],
    }
    stripToolResultContent(block, 'read of /b.ts')
    expect(Array.isArray(block.content)).toBe(true)
    expect((block.content as Array<{ text: string }>)[0]!.text).toBe(strippedMarker('read of /b.ts'))
  })
})

// ---------------------------------------------------------------------------
// pruneRedundantToolOutputs
// ---------------------------------------------------------------------------

describe('pruneRedundantToolOutputs', () => {
  test('strips old Read result when same file is re-read', () => {
    // Turn 1: Read /a.ts, Turn 2: Read /a.ts
    // Old (turn 1) should be stripped, new (turn 2) preserved
    // Pad with 6 filler user messages so the interesting messages are beyond
    // the recency guard (last 5 user messages never stripped).
    const messages: Message[] = [
      userTextMessage('pad 1'),
      userTextMessage('pad 2'),
      userTextMessage('pad 3'),
      userTextMessage('pad 4'),
      userTextMessage('pad 5'),
      userTextMessage('pad 6'),
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(1)
    expect(
      (result[7]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe(strippedMarker('read of /a.ts'))
    expect(
      (result[9]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('y'.repeat(500))
  })

  test('keeps most recent per file across multiple files', () => {
    // Read A, Read B, Read A → first Read A stripped, Read B kept, second Read A kept
    // Pad:6 filler user messages before interesting content
    const messages: Message[] = [
      userTextMessage('p1'), userTextMessage('p2'), userTextMessage('p3'),
      userTextMessage('p4'), userTextMessage('p5'), userTextMessage('p6'),
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/b.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
      assistantWithToolUse('Read', 'tool-3', { file_path: '/a.ts' }),
      userWithToolResult('tool-3', 'z'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(1)
    // First Read of /a.ts stripped
    expect(
      (result[7]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe(strippedMarker('read of /a.ts'))
    // Read of /b.ts preserved
    expect(
      (result[9]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('y'.repeat(500))
    // Second Read of /a.ts preserved
    expect(
      (result[11]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('z'.repeat(500))
  })

  test('preserves different files read once each', () => {
    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/b.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(0)
    expect(
      (result[1]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('x'.repeat(500))
    expect(
      (result[3]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('y'.repeat(500))
  })

  // ── Dedup-based: Bash/Grep/Glob/WebFetch ─────────────────────────────

  test('strips duplicate Bash commands, preserves unique ones', () => {
    // Same command run twice → second supersedes first
    const messages: Message[] = [
      assistantWithToolUse('Bash', 't1', { command: 'git status' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      userTextMessage('hmm'),
      assistantWithToolUse('Bash', 't2', { command: 'git status' }),
      userWithToolResult('t2', 'y'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    expect(strippedCount).toBe(1)
    const blocks = result[1]!.message.content as Array<{ content: string }>
    expect(blocks[0]!.content).toBe(strippedMarker('command: git status'))
  })

  test('preserves all unique Bash commands', () => {
    const messages: Message[] = [
      assistantWithToolUse('Bash', 't1', { command: 'ls' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      assistantWithToolUse('Bash', 't2', { command: 'pwd' }),
      userWithToolResult('t2', 'y'.repeat(500)),
      assistantWithToolUse('Bash', 't3', { command: 'whoami' }),
      userWithToolResult('t3', 'z'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    // All unique → nothing stripped
    expect(strippedCount).toBe(0)
  })

  test('Write does not supersede Read for same file', () => {
    // Write is an acknowledgement, not file content. It should NOT strip an
    // earlier Read (otherwise the model loses the file snapshot).
    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Write', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(0)
    expect(
      (result[1]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('x'.repeat(500))
    expect(
      (result[3]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('y'.repeat(500))
  })

  test('Edit does not supersede Read for same file', () => {
    // Edit is an acknowledgement, not file content. It should NOT strip an
    // earlier Read (otherwise the model loses the file snapshot).
    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Edit', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(0)
    expect(
      (result[1]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('x'.repeat(500))
    expect(
      (result[3]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('y'.repeat(500))
  })

  test('does not strip small content (< 200 chars)', () => {
    // First Read of /a.ts has small content → not stripped even though newer result exists
    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'File not found'),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(0)
    expect(
      (result[1]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe('File not found')
  })

  test('handles array-format tool results', () => {
    const messages: Message[] = [
      userTextMessage('p1'), userTextMessage('p2'), userTextMessage('p3'),
      userTextMessage('p4'), userTextMessage('p5'), userTextMessage('p6'),
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResultArray('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResultArray('tool-2', 'y'.repeat(500)),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(1)
    // Indices: [0-5]padding, [6]asst t1, [7]user t1, [8]asst t2, [9]user t2
    const oldContent = (result[7]!.message.content as Array<{ content: Array<{ text: string }> }>)[0]!.content
    expect(Array.isArray(oldContent)).toBe(true)
    expect((oldContent as Array<{ text: string }>)[0]!.text).toBe(strippedMarker('read of /a.ts'))

    const newContent = (result[9]!.message.content as Array<{ content: Array<{ text: string }> }>)[0]!.content
    expect(Array.isArray(newContent)).toBe(true)
    expect((newContent as Array<{ text: string }>)[0]!.text).toBe('y'.repeat(500))
  })

  test('handles empty messages array', () => {
    const { messages, strippedCount } = pruneRedundantToolOutputs([], new Map())
    expect(messages).toEqual([])
    expect(strippedCount).toBe(0)
  })

  test('handles messages with no tool results', () => {
    const messages: Message[] = [
      userTextMessage('hello'),
      createAssistantMessage({ content: 'world' }),
    ]

    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, new Map())
    expect(strippedCount).toBe(0)
    expect(result).toEqual(messages)
  })

  test('does not mutate original messages when stripping', () => {
    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    const originalContent = (messages[1]!.message.content as Array<{ content: string }>)[0]!.content
    pruneRedundantToolOutputs(messages, buildToolUseMap(messages))

    expect(
      (messages[1]!.message.content as Array<{ content: string }>)[0]!.content,
    ).toBe(originalContent)
  })

  test('strips older same-file block within a single message (regression)', () => {
    // Two tool_results for the same file in one user message.
    // The later block (tool-2) should be preserved, the earlier (tool-1) stripped.
    // Pad with 6 filler user messages so this message is beyond the recency guard.
    const messages: Message[] = [
      userTextMessage('p1'), userTextMessage('p2'), userTextMessage('p3'),
      userTextMessage('p4'), userTextMessage('p5'), userTextMessage('p6'),
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      createUserMessage({
        content: [
          { type: 'tool_result' as const, tool_use_id: 'tool-1', content: 'x'.repeat(500) },
          { type: 'tool_result' as const, tool_use_id: 'tool-2', content: 'y'.repeat(500) },
        ],
      }),
    ]

    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)

    expect(strippedCount).toBe(1)
    const blocks = result[8]!.message.content as Array<{ content: string }>
    // tool-1 (earlier in array) should be stripped, tool-2 (later) preserved
    expect(blocks[0]!.content).toBe(strippedMarker('read of /a.ts'))
    expect(blocks[1]!.content).toBe('y'.repeat(500))
  })

  // ── Dedup-based pruning: Bash/Grep/Glob/WebFetch ─────────────────────

  test('strips duplicate Bash commands, preserves unique ones', () => {
    const messages: Message[] = [
      assistantWithToolUse('Bash', 't1', { command: 'git status' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      userTextMessage('hmm'),
      assistantWithToolUse('Bash', 't2', { command: 'git status' }),
      userWithToolResult('t2', 'y'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    expect(strippedCount).toBe(1)
    const blocks = result[1]!.message.content as Array<{ content: string }>
    expect(blocks[0]!.content).toBe(strippedMarker('command: git status'))
  })

  test('Grep preserves all unique patterns, strips duplicates', () => {
    // All unique → nothing stripped
    const messages: Message[] = [
      assistantWithToolUse('Grep', 't1', { pattern: 'foo' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      userTextMessage('hmm'),
      assistantWithToolUse('Grep', 't2', { pattern: 'bar' }),
      userWithToolResult('t2', 'y'.repeat(500)),
      userTextMessage('ok'),
      assistantWithToolUse('Grep', 't3', { pattern: 'baz' }),
      userWithToolResult('t3', 'z'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    // All unique patterns → nothing stripped
    expect(strippedCount).toBe(0)
  })

  test('Grep strips duplicate pattern', () => {
    const messages: Message[] = [
      assistantWithToolUse('Grep', 't1', { pattern: 'TODO' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      userTextMessage('hmm'),
      assistantWithToolUse('Grep', 't2', { pattern: 'TODO' }),
      userWithToolResult('t2', 'y'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    expect(strippedCount).toBe(1)
    const blocks = result[1]!.message.content as Array<{ content: string }>
    expect(blocks[0]!.content).toBe(strippedMarker('grep: TODO'))
  })

  test('preserves single Grep result (no duplicates)', () => {
    const messages: Message[] = [
      assistantWithToolUse('Grep', 't1', { pattern: 'function' }),
      userWithToolResult('t1', 'x'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    expect(strippedCount).toBe(0)
    const blocks = result[1]!.message.content as Array<{ content: string }>
    expect(blocks[0]!.content).toBe('x'.repeat(500))
  })

  // ── Safety cap: many unique results ──────────────────────────────────

  test('strips unique results beyond safety cap', () => {
    const messages: Message[] = []
    // Create SAFETY_CAP + 5 unique Bash results
    for (let i = 0; i < SAFETY_CAP + 5; i++) {
      messages.push(assistantWithToolUse('Bash', `t${i}`, { command: `unique-cmd-${i}` }))
      messages.push(userWithToolResult(`t${i}`, `${i}`.repeat(500)))
      if (i < SAFETY_CAP + 4) messages.push(userTextMessage(`msg ${i}`))
    }
    const map = buildToolUseMap(messages)
    const { strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    // 5 unique results beyond the cap → stripped
    expect(strippedCount).toBe(5)
  })

  // ── Recency guard ────────────────────────────────────────────────────

  test('preserves results within last N user messages', () => {
    // Two duplicate Bash commands — second supersedes first, but the first
    // is within the recency guard window (last 5 user messages) so it
    // should NOT be stripped. Uses default recency guard (5).
    const messages: Message[] = [
      assistantWithToolUse('Bash', 't1', { command: 'git status' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      assistantWithToolUse('Bash', 't2', { command: 'git status' }),
      userWithToolResult('t2', 'y'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map)
    // Both are within last 5 user messages → nothing stripped
    expect(strippedCount).toBe(0)
    const firstBlocks = result[1]!.message.content as Array<{ content: string }>
    expect(firstBlocks[0]!.content).toBe('x'.repeat(500))
  })

  // ── Write/Edit not stripped (too small) ──────────────────────────────

  test('Write acknowledgment is not stripped', () => {
    const messages: Message[] = [
      assistantWithToolUse('Write', 't1', { file_path: '/a.ts' }),
      userWithToolResult('t1', 'File written successfully'),
    ]
    const map = buildToolUseMap(messages)
    const { strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    // Write is no longer in FILE_CONTENT_TOOLS → not stripped
    expect(strippedCount).toBe(0)
  })

  test('Edit acknowledgment is not stripped', () => {
    const messages: Message[] = [
      assistantWithToolUse('Edit', 't1', { file_path: '/a.ts' }),
      userWithToolResult('t1', 'File updated successfully'),
    ]
    const map = buildToolUseMap(messages)
    const { strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    expect(strippedCount).toBe(0)
  })

  // ── getResourceIdentity ──────────────────────────────────────────────

  test('getResourceIdentity returns command identity for Bash', () => {
    expect(getResourceIdentity('Bash', { command: 'ls -la' })).toBe('cmd:ls -la')
  })

  test('getResourceIdentity returns pattern identity for Grep', () => {
    expect(getResourceIdentity('Grep', { pattern: 'function', path: '/src' }))
      .toBe('grep:function|/src')
  })

  test('getResourceIdentity returns url identity for WebFetch', () => {
    expect(getResourceIdentity('WebFetch', { url: 'https://example.com' }))
      .toBe('url:https://example.com')
  })

  test('getResourceIdentity returns undefined for unknown tool', () => {
    expect(getResourceIdentity('Unknown', {})).toBeUndefined()
  })

  // ── Tool_use block stripping ─────────────────────────────────────────

  test('strips tool_use blocks when paired tool_result is stripped', () => {
    const messages: Message[] = [
      assistantWithToolUse('Read', 't1', { file_path: '/a.ts' }),
      userWithToolResult('t1', 'x'.repeat(500)),
      userTextMessage('thinking...'),
      assistantWithToolUse('Read', 't2', { file_path: '/a.ts' }),
      userWithToolResult('t2', 'y'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    // 1 content strip (tool_result) — tool_use replacement doesn't count
    expect(strippedCount).toBe(1)

    // Check tool_result was stripped
    const userBlocks = result[1]!.message.content as Array<{ content: string }>
    expect(userBlocks[0]!.content).toBe(strippedMarker('read of /a.ts'))

    // Check tool_use was replaced with text marker
    const asstBlocks = result[0]!.message.content as Array<{ type: string; text: string }>
    expect(asstBlocks[0]!.type).toBe('text')
    expect(asstBlocks[0]!.text).toContain('tool_use: Read')
    expect(asstBlocks[0]!.text).toContain('paired result stripped as redundant')
  })

  test('preserves tool_use blocks when tool_result is NOT stripped', () => {
    const messages: Message[] = [
      assistantWithToolUse('Read', 't1', { file_path: '/a.ts' }),
      userWithToolResult('t1', 'x'.repeat(500)),
    ]
    const map = buildToolUseMap(messages)
    const { messages: result, strippedCount } = pruneRedundantToolOutputs(messages, map, 0)
    expect(strippedCount).toBe(0)
    const asstBlocks = result[0]!.message.content as Array<{ type: string }>
    expect(asstBlocks[0]!.type).toBe('tool_use')
  })
})

// ---------------------------------------------------------------------------
// pruneOldThinkingBlocks
// ---------------------------------------------------------------------------

describe('pruneOldThinkingBlocks', () => {
  function assistantWithThinking(reasoning: string): Message {
    return createAssistantMessage({
      content: [
        { type: 'thinking' as const, thinking: reasoning },
        { type: 'text' as const, text: 'OK, here is my response.' },
      ],
    })
  }

  function assistantWithToolUseAndThinking(
    toolName: string,
    toolId: string,
    input: Record<string, unknown> = {},
    thinking: string,
  ): Message {
    return createAssistantMessage({
      content: [
        { type: 'thinking' as const, thinking },
        { type: 'tool_use' as const, id: toolId, name: toolName, input },
      ],
    })
  }

  test('keeps latest thinking, strips older thinking blocks', () => {
    const messages: Message[] = [
      assistantWithThinking('First long thinking'),
      userTextMessage('user msg 1'),
      assistantWithThinking('Second long thinking'),
      userTextMessage('user msg 2'),
      assistantWithThinking('Third long thinking'),
    ]

    const { messages: result, strippedCount } = pruneOldThinkingBlocks(messages)

    expect(strippedCount).toBe(2)
    // Oldest assistant message: thinking stripped
    const firstContent = result[0]!.message.content as Array<{ type: string; text?: string; thinking?: string }>
    expect(firstContent[0]!.type).toBe('text')
    expect(firstContent[0]!.text).toBe('OK, here is my response.')
    // Middle assistant message: thinking stripped
    const midContent = result[2]!.message.content as Array<{ type: string; text?: string; thinking?: string }>
    expect(midContent[0]!.type).toBe('text')
    expect(midContent[0]!.text).toBe('OK, here is my response.')
    // Latest assistant message: thinking kept
    const latestContent = result[4]!.message.content as Array<{ type: string; text?: string; thinking?: string }>
    expect(latestContent[0]!.type).toBe('thinking')
    expect(latestContent[0]!.thinking).toBe('Third long thinking')
  })

  test('preserves tool_use blocks when stripping thinking', () => {
    const messages: Message[] = [
      assistantWithToolUseAndThinking('Read', 'tool-1', { file_path: '/a.ts' }, 'Old thinking'),
      userWithToolResult('tool-1', 'content'),
      assistantWithToolUseAndThinking('Read', 'tool-2', { file_path: '/b.ts' }, 'New thinking'),
    ]

    const { messages: result, strippedCount } = pruneOldThinkingBlocks(messages)

    expect(strippedCount).toBe(1)
    // Oldest assistant: thinking stripped, tool_use preserved
    const firstContent = result[0]!.message.content as Array<{ type: string; name?: string; id?: string }>
    expect(firstContent[0]!.type).toBe('tool_use')
    expect(firstContent[0]!.name).toBe('Read')
    expect(firstContent[0]!.id).toBe('tool-1')
    // Latest assistant: thinking kept
    const latestContent = result[2]!.message.content as Array<{ type: string; thinking?: string }>
    expect(latestContent[0]!.type).toBe('thinking')
    expect(latestContent[0]!.thinking).toBe('New thinking')
  })

  test('no-op when there are no thinking blocks', () => {
    const messages: Message[] = [
      createAssistantMessage({ content: 'Just text, no thinking' }),
      userTextMessage('user msg'),
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
    ]

    const { messages: result, strippedCount } = pruneOldThinkingBlocks(messages)

    expect(strippedCount).toBe(0)
    expect(result).toEqual(messages)
  })

  test('handles single assistant message with thinking', () => {
    const messages: Message[] = [
      assistantWithThinking('Only thinking block'),
    ]

    const { messages: result, strippedCount } = pruneOldThinkingBlocks(messages)

    expect(strippedCount).toBe(0)
    const content = result[0]!.message.content as Array<{ type: string; thinking?: string }>
    expect(content[0]!.type).toBe('thinking')
  })

  test('handles empty messages array', () => {
    const { messages, strippedCount } = pruneOldThinkingBlocks([])
    expect(messages).toEqual([])
    expect(strippedCount).toBe(0)
  })

  test('does not strip non-assistant messages', () => {
    const messages: Message[] = [
      userTextMessage('hello'),
      assistantWithThinking('Some thinking'),
      userWithToolResult('tool-1', 'result'),
    ]

    const { messages: result, strippedCount } = pruneOldThinkingBlocks(messages)

    expect(strippedCount).toBe(0)
    // User messages unchanged
    expect(result[0]!.message.content).toBe('hello')
    expect((result[2]!.message.content as Array<{ content: string }>)[0]!.content).toBe('result')
  })

  test('strips redacted_thinking blocks too', () => {
    const messages: Message[] = [
      createAssistantMessage({
        content: [
          { type: 'redacted_thinking' as const, data: 'redacted-old' },
          { type: 'text' as const, text: 'Old response' },
        ],
      }),
      createAssistantMessage({
        content: [
          { type: 'redacted_thinking' as const, data: 'redacted-new' },
          { type: 'text' as const, text: 'New response' },
        ],
      }),
    ]

    const { messages: result, strippedCount } = pruneOldThinkingBlocks(messages)

    expect(strippedCount).toBe(1)
    const oldContent = result[0]!.message.content as Array<{ type: string; text?: string; data?: string }>
    expect(oldContent[0]!.type).toBe('text')
    expect(oldContent[0]!.text).toBe('Old response')
    const newContent = result[1]!.message.content as Array<{ type: string; data?: string }>
    expect(newContent[0]!.type).toBe('redacted_thinking')
    expect(newContent[0]!.data).toBe('redacted-new')
  })

  test('does not mutate original messages', () => {
    const messages: Message[] = [
      assistantWithThinking('First thinking'),
      assistantWithThinking('Second thinking'),
    ]

    const originalContent = (messages[0]!.message.content as Array<{ thinking: string }>)[0]!.thinking
    pruneOldThinkingBlocks(messages)

    expect(
      (messages[0]!.message.content as Array<{ thinking: string }>)[0]!.thinking,
    ).toBe(originalContent)
  })
})

// ---------------------------------------------------------------------------
// applyProactiveBudget (config-driven integration)
//
// applyProactiveBudget reads proactiveBudgetLimit from user config.
// 0 = disabled. Unset → falls back to PROACTIVE_BUDGET_TARGET_TOKENS_DEFAULT (100K).
// In production, the config system also defaults proactiveBudgetLimit to 100K.
// ---------------------------------------------------------------------------

describe('applyProactiveBudget', () => {
  test('prunes redundant outputs when over the configured limit', () => {
    // Set a small limit so the messages exceed it
    saveGlobalConfig((c) => ({ ...c, proactiveBudgetLimit: 100 }))

    // Need the duplicate Read to be >5 user messages from the end (recency
    // guard = 5). Pad with 10 messages before + 5 after the duplicate.
    const messages: Message[] = [
      userTextMessage('p1'), userTextMessage('p2'), userTextMessage('p3'),
      userTextMessage('p4'), userTextMessage('p5'),
      userTextMessage('p6'), userTextMessage('p7'), userTextMessage('p8'),
      userTextMessage('p9'), userTextMessage('p10'),
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
      userTextMessage('t1'), userTextMessage('t2'), userTextMessage('t3'),
      userTextMessage('t4'), userTextMessage('t5'),
    ]

    try {
      const result = applyProactiveBudget(messages)

      expect(result.wasPruned).toBe(true)
      expect(result.strippedCount).toBe(1)
      expect(result.messages).not.toBe(messages)
    } finally {
      // Clean up config state for subsequent tests
      saveGlobalConfig((c) => {
        delete (c as Record<string, unknown>).proactiveBudgetLimit
        return { ...c }
      })
    }
  })

  test('no-op when config limit is explicitly 0 (disabled)', () => {
    // Set config to 0 to disable proactive budget
    saveGlobalConfig((c) => ({ ...c, proactiveBudgetLimit: 0 }))

    const messages: Message[] = [
      assistantWithToolUse('Read', 'tool-1', { file_path: '/a.ts' }),
      userWithToolResult('tool-1', 'x'.repeat(500)),
      assistantWithToolUse('Read', 'tool-2', { file_path: '/a.ts' }),
      userWithToolResult('tool-2', 'y'.repeat(500)),
    ]

    try {
      const result = applyProactiveBudget(messages)

      expect(result.wasPruned).toBe(false)
      expect(result.strippedCount).toBe(0)
      expect(result.messages).toBe(messages)
    } finally {
      // Clean up config state for subsequent tests
      saveGlobalConfig((c) => {
        delete (c as Record<string, unknown>).proactiveBudgetLimit
        return { ...c }
      })
    }
  })
})
