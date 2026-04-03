/**
 * OpenAI-compatible API adapter for Anthropic SDK interface.
 *
 * Translates Anthropic Messages API calls into OpenAI chat/completions format,
 * enabling claude-code to work with any OpenAI-compatible model server
 * (vLLM, Ollama, LiteLLM, etc.).
 *
 * Activated by setting CLAUDE_CODE_USE_OPENAI=1.
 *
 * Required env vars:
 *   OPENAI_BASE_URL  - e.g. http://10.190.179.61:11999/qwen3_5/v1
 *   OPENAI_API_KEY   - API key (use "dummy" if not required)
 *   OPENAI_MODEL     - model name, e.g. Qwen3.5-27B-FP16
 */
import type Anthropic from '@anthropic-ai/sdk'
import { APIUserAbortError } from '@anthropic-ai/sdk'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | null
  tool_calls?: OpenAIToolCall[]
  tool_call_id?: string
  name?: string
}

interface OpenAIToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters?: Record<string, unknown>
  }
}

interface OpenAIStreamChoice {
  index: number
  delta: {
    role?: string
    content?: string | null
    tool_calls?: Array<{
      index: number
      id?: string
      type?: string
      function?: { name?: string; arguments?: string }
    }>
  }
  finish_reason: string | null
}

interface OpenAIStreamChunk {
  id: string
  object: string
  created: number
  model: string
  choices: OpenAIStreamChoice[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

// ---------------------------------------------------------------------------
// Anthropic → OpenAI message conversion
// ---------------------------------------------------------------------------
function convertSystemPrompt(
  system: unknown,
): OpenAIMessage[] {
  if (!system) return []

  if (typeof system === 'string') {
    return [{ role: 'system', content: system }]
  }

  if (Array.isArray(system)) {
    const text = system
      .filter((b: any) => b.type === 'text')
      .map((b: any) => b.text)
      .join('\n\n')
    return text ? [{ role: 'system', content: text }] : []
  }

  return []
}

function convertMessages(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = []

  for (const msg of messages) {
    if (msg.role === 'user') {
      if (typeof msg.content === 'string') {
        result.push({ role: 'user', content: msg.content })
      } else if (Array.isArray(msg.content)) {
        const parts: string[] = []
        const toolResults: OpenAIMessage[] = []

        for (const block of msg.content) {
          if (block.type === 'text') {
            parts.push(block.text)
          } else if (block.type === 'tool_result') {
            let resultContent = ''
            if (typeof block.content === 'string') {
              resultContent = block.content
            } else if (Array.isArray(block.content)) {
              resultContent = block.content
                .filter((c: any) => c.type === 'text')
                .map((c: any) => c.text)
                .join('\n')
            }
            toolResults.push({
              role: 'tool',
              content: resultContent || (block.is_error ? 'Error' : 'Success'),
              tool_call_id: block.tool_use_id,
            })
          } else if (block.type === 'image') {
            parts.push('[Image content omitted - not supported by this model]')
          }
        }

        // Tool results come first (they need to follow the assistant tool_calls)
        result.push(...toolResults)
        if (parts.length > 0) {
          result.push({ role: 'user', content: parts.join('\n') })
        }
      }
    } else if (msg.role === 'assistant') {
      if (typeof msg.content === 'string') {
        result.push({ role: 'assistant', content: msg.content })
      } else if (Array.isArray(msg.content)) {
        const textParts: string[] = []
        const toolCalls: OpenAIToolCall[] = []

        for (const block of msg.content) {
          if (block.type === 'text') {
            textParts.push(block.text)
          } else if (block.type === 'thinking') {
            // Include thinking as text prefixed with marker
            if (block.thinking) {
              textParts.push(block.thinking)
            }
          } else if (block.type === 'tool_use') {
            toolCalls.push({
              id: block.id,
              type: 'function',
              function: {
                name: block.name,
                arguments:
                  typeof block.input === 'string'
                    ? block.input
                    : JSON.stringify(block.input),
              },
            })
          }
        }

        const assistantMsg: OpenAIMessage = {
          role: 'assistant',
          content: textParts.join('\n') || null,
        }
        if (toolCalls.length > 0) {
          assistantMsg.tool_calls = toolCalls
        }
        result.push(assistantMsg)
      }
    }
  }

  return result
}

function convertTools(tools: any[]): OpenAITool[] | undefined {
  if (!tools || tools.length === 0) return undefined

  return tools
    .filter((t: any) => t.type !== 'computer_20241022')
    .map((tool: any) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description || '',
        parameters: tool.input_schema || { type: 'object', properties: {} },
      },
    }))
}

// ---------------------------------------------------------------------------
// OpenAI SSE stream → Anthropic BetaRawMessageStreamEvent
// ---------------------------------------------------------------------------
function makeMessageId(): string {
  return 'msg_' + Math.random().toString(36).slice(2, 14)
}

function makeToolUseId(): string {
  return 'toolu_' + Math.random().toString(36).slice(2, 14)
}

// ---------------------------------------------------------------------------
// Core adapter: create a fake Anthropic client that routes to OpenAI
// ---------------------------------------------------------------------------
export function createOpenAIAdaptedClient(options: {
  baseURL: string
  model: string
  apiKey: string
  timeout: number
  defaultHeaders?: Record<string, string>
  maxRetries?: number
  fetchOverride?: any
}): Anthropic {
  const {
    baseURL,
    model: defaultModel,
    apiKey,
    timeout,
    defaultHeaders = {},
    fetchOverride,
  } = options

  const chatCompletionsURL = baseURL.endsWith('/')
    ? `${baseURL}chat/completions`
    : `${baseURL}/chat/completions`

  const fetchFn: typeof globalThis.fetch = fetchOverride ?? globalThis.fetch

  // Build the mock Anthropic client with only the methods claude-code uses
  const client = {
    beta: {
      messages: {
        create(
          params: any,
          requestOptions?: any,
        ): any {
          const model = params.model || defaultModel
          const stream = params.stream === true

          const systemMsgs = convertSystemPrompt(params.system)
          const userMsgs = convertMessages(params.messages || [])
          const allMessages = [...systemMsgs, ...userMsgs]
          const openaiTools = convertTools(params.tools)

          const contextLimit = parseInt(
            process.env.OPENAI_CONTEXT_LENGTH || '131072',
            10,
          )
          const reserveForOutput = Math.min(params.max_tokens || 4096, 16384)
          const maxTokens = Math.max(reserveForOutput, 1024)

          const body: Record<string, unknown> = {
            model,
            messages: allMessages,
            max_tokens: maxTokens,
            stream,
            temperature: params.temperature ?? 0.7,
            top_p: 0.8,
            top_k: 20,
          }

          if (params.thinking && params.thinking.type === 'enabled') {
            body.chat_template_kwargs = { enable_thinking: true }
          }

          if (openaiTools && openaiTools.length > 0) {
            body.tools = openaiTools
            if (params.tool_choice) {
              if (params.tool_choice.type === 'auto') {
                body.tool_choice = 'auto'
              } else if (
                params.tool_choice.type === 'tool' &&
                params.tool_choice.name
              ) {
                body.tool_choice = {
                  type: 'function',
                  function: { name: params.tool_choice.name },
                }
              } else if (params.tool_choice.type === 'any') {
                body.tool_choice = 'required'
              }
            }
          }

          if (stream) {
            body.stream_options = { include_usage: true }
          }

          const signal = requestOptions?.signal
          const abortController = new AbortController()
          if (signal) {
            signal.addEventListener('abort', () => abortController.abort())
          }

          const fetchPromise = fetchFn(chatCompletionsURL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${apiKey}`,
              ...defaultHeaders,
            },
            body: JSON.stringify(body),
            signal: abortController.signal,
          })

          if (stream) {
            return createStreamProxy(
              fetchPromise,
              model,
              params.max_tokens || 4096,
              params.tools,
            )
          }

          // Non-streaming: convert response
          return fetchPromise.then(async resp => {
            if (!resp.ok) {
              const errBody = await resp.text()
              throw new Error(
                `OpenAI API error ${resp.status}: ${errBody}`,
              )
            }
            const data = await resp.json()
            return convertNonStreamingResponse(
              data,
              model,
              params.max_tokens || 4096,
            )
          })
        },
      },
    },
  }

  return client as unknown as Anthropic
}

function convertNonStreamingResponse(
  data: any,
  model: string,
  maxTokens: number,
): any {
  const choice = data.choices?.[0]
  if (!choice) {
    return {
      id: makeMessageId(),
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: 'end_turn',
      usage: { input_tokens: 0, output_tokens: 0 },
    }
  }

  const content: any[] = []
  if (choice.message?.content) {
    content.push({ type: 'text', text: choice.message.content })
  }
  if (choice.message?.tool_calls) {
    for (const tc of choice.message.tool_calls) {
      let input: any = {}
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {}
      content.push({
        type: 'tool_use',
        id: tc.id || makeToolUseId(),
        name: tc.function.name,
        input,
      })
    }
  }

  const stopReason =
    choice.finish_reason === 'tool_calls'
      ? 'tool_use'
      : choice.finish_reason === 'length'
        ? 'max_tokens'
        : 'end_turn'

  return {
    id: makeMessageId(),
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: stopReason,
    usage: {
      input_tokens: data.usage?.prompt_tokens || 0,
      output_tokens: data.usage?.completion_tokens || 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  }
}

// ---------------------------------------------------------------------------
// Streaming adapter
// ---------------------------------------------------------------------------
function createStreamProxy(
  fetchPromise: Promise<Response>,
  model: string,
  maxTokens: number,
  tools: any[],
): any {
  const messageId = makeMessageId()

  let inputTokens = 0
  let outputTokens = 0
  let contentBlockIndex = 0
  const toolCallBuffers: Map<
    number,
    { id: string; name: string; args: string }
  > = new Map()

  async function* generateEvents(
    response: Response,
  ): AsyncGenerator<any> {
    if (!response.ok) {
      const errBody = await response.text()
      throw new Error(`OpenAI API error ${response.status}: ${errBody}`)
    }

    // Emit message_start — must include a complete BetaMessage shape
    yield {
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        content: [],
        model,
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    }

    // Start first content block (text)
    yield {
      type: 'content_block_start',
      index: contentBlockIndex,
      content_block: { type: 'text', text: '' },
    }

    let pendingToolCalls: Map<
      number,
      { id: string; name: string; args: string }
    > = new Map()
    let streamFinished = false

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    function resolveStopReason(reason: string | null): string {
      if (reason === 'tool_calls') return 'tool_use'
      if (reason === 'length') return 'max_tokens'
      return 'end_turn'
    }

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') continue
          if (!data) continue

          let chunk: OpenAIStreamChunk
          try {
            chunk = JSON.parse(data)
          } catch {
            continue
          }

          if (chunk.usage) {
            inputTokens = chunk.usage.prompt_tokens || 0
            outputTokens = chunk.usage.completion_tokens || 0
          }

          for (const choice of chunk.choices || []) {
            const delta = choice.delta

            // Text content
            if (delta.content) {
              yield {
                type: 'content_block_delta',
                index: contentBlockIndex,
                delta: { type: 'text_delta', text: delta.content },
              }
            }

            // Tool calls
            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                const idx = tc.index
                if (!pendingToolCalls.has(idx)) {
                  pendingToolCalls.set(idx, {
                    id: tc.id || makeToolUseId(),
                    name: tc.function?.name || '',
                    args: '',
                  })
                }
                const buf = pendingToolCalls.get(idx)!
                if (tc.id) buf.id = tc.id
                if (tc.function?.name) buf.name = tc.function.name
                if (tc.function?.arguments)
                  buf.args += tc.function.arguments
              }
            }

            // Stop — the model finished generating
            if (choice.finish_reason) {
              streamFinished = true

              // Close the current text block
              yield {
                type: 'content_block_stop',
                index: contentBlockIndex,
              }
              contentBlockIndex++

              // Emit accumulated tool_use blocks
              for (const [, tcBuf] of pendingToolCalls) {
                yield {
                  type: 'content_block_start',
                  index: contentBlockIndex,
                  content_block: {
                    type: 'tool_use',
                    id: tcBuf.id,
                    name: tcBuf.name,
                    input: '',
                  },
                }
                yield {
                  type: 'content_block_delta',
                  index: contentBlockIndex,
                  delta: {
                    type: 'input_json_delta',
                    partial_json: tcBuf.args,
                  },
                }
                yield {
                  type: 'content_block_stop',
                  index: contentBlockIndex,
                }
                contentBlockIndex++
              }

              yield {
                type: 'message_delta',
                delta: { stop_reason: resolveStopReason(choice.finish_reason) },
                usage: { output_tokens: outputTokens },
              }

              yield { type: 'message_stop' }
            }
          }
        }
      }

      // Safety net: if the stream ended without a finish_reason, close gracefully
      if (!streamFinished) {
        yield {
          type: 'content_block_stop',
          index: contentBlockIndex,
        }
        contentBlockIndex++

        for (const [, tcBuf] of pendingToolCalls) {
          yield {
            type: 'content_block_start',
            index: contentBlockIndex,
            content_block: {
              type: 'tool_use',
              id: tcBuf.id,
              name: tcBuf.name,
              input: '',
            },
          }
          yield {
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: tcBuf.args,
            },
          }
          yield {
            type: 'content_block_stop',
            index: contentBlockIndex,
          }
          contentBlockIndex++
        }

        yield {
          type: 'message_delta',
          delta: { stop_reason: 'end_turn' },
          usage: { output_tokens: outputTokens },
        }

        yield { type: 'message_stop' }
      }
    } finally {
      reader.releaseLock()
    }
  }

  // The Anthropic SDK's Stream has a `controller` property — claude.ts:1854
  // uses `'controller' in e.value` to distinguish streams from error messages.
  const abortController = new AbortController()

  const streamObj = {
    controller: abortController,
    _response: null as Response | null,
    _requestId: null as string | null,
    messages: [] as any[],
    receivedMessages: [] as any[],
    finalMessage: null as any,

    async *[Symbol.asyncIterator]() {
      const response = await fetchPromise
      streamObj._response = response
      streamObj._requestId =
        response.headers.get('x-request-id') || makeMessageId()
      yield* generateEvents(response)
    },

    withResponse() {
      return fetchPromise.then(async response => {
        streamObj._response = response
        const requestId =
          response.headers.get('x-request-id') || makeMessageId()
        return {
          data: streamObj,
          response,
          request_id: requestId,
        }
      })
    },

    abort() {
      abortController.abort()
    },
  }

  return streamObj
}
