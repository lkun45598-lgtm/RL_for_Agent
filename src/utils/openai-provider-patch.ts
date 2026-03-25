import {
  OpenAIProvider,
  type ContentBlock,
  type Message,
  type ModelResponse,
  type ModelStreamChunk,
} from '@shareai-lab/kode-sdk'

type OpenAIApiMode = 'chat' | 'responses'

type ToolSchema = {
  name?: string
  description?: string
  input_schema?: Record<string, any>
}

type CompletionOptionsLike = {
  tools?: ToolSchema[]
  maxTokens?: number
  temperature?: number
  system?: string
}

function getMessageBlocks(message: Message): ContentBlock[] {
  if (message.metadata?.transport === 'omit') {
    return message.content
  }
  return message.metadata?.content_blocks ?? message.content
}

function safeJsonStringify(value: unknown): string {
  try {
    const json = JSON.stringify(value ?? {})
    return json === undefined ? '{}' : json
  } catch {
    return '{}'
  }
}

function safeJsonParse(value: unknown): any {
  if (typeof value !== 'string') {
    return value ?? {}
  }

  try {
    return JSON.parse(value)
  } catch {
    return { raw: value }
  }
}

function formatToolOutput(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  return safeJsonStringify(content)
}

function extractReasoningText(item: any): string {
  if (!item || typeof item !== 'object') {
    return ''
  }

  if (typeof item.text === 'string' && item.text.trim()) {
    return item.text
  }

  const summary = Array.isArray(item.summary) ? item.summary : []
  const pieces = summary
    .map((entry: any) => {
      if (!entry || typeof entry !== 'object') return ''
      if (typeof entry.text === 'string') return entry.text
      if (typeof entry.summary_text === 'string') return entry.summary_text
      return ''
    })
    .filter(Boolean)

  return pieces.join('\n')
}

function buildResponsesTools(tools: ToolSchema[] | undefined): any[] | undefined {
  if (!tools?.length) {
    return undefined
  }

  return tools.map(tool => ({
    type: 'function',
    name: tool.name ?? 'tool',
    description: tool.description ?? '',
    parameters: tool.input_schema ?? { type: 'object', properties: {}, additionalProperties: true },
  }))
}

function buildResponsesInput(
  messages: Message[],
  reasoningTransport: 'omit' | 'text' | 'provider' = 'text',
): any[] {
  const input: any[] = []
  const seenToolCallIds = new Set<string>()

  for (const msg of messages) {
    if (msg.role === 'system') {
      continue
    }

    const blocks = getMessageBlocks(msg)
    const parts: any[] = []
    const textType = msg.role === 'assistant' ? 'output_text' : 'input_text'

    const flushMessage = () => {
      if (parts.length === 0) {
        return
      }
      input.push({
        role: msg.role,
        content: [...parts],
      })
      parts.length = 0
    }

    for (const block of blocks) {
      if (block.type === 'text') {
        parts.push({ type: textType, text: block.text })
        continue
      }

      if (block.type === 'reasoning') {
        const responseItem = block.meta?.response_item
        if (msg.role === 'assistant' && responseItem && typeof responseItem === 'object') {
          flushMessage()
          input.push(responseItem)
          continue
        }

        if (reasoningTransport === 'text' && block.reasoning) {
          parts.push({ type: textType, text: `<think>${block.reasoning}</think>` })
        }
        continue
      }

      if (block.type === 'tool_use') {
        flushMessage()
        if (block.id) {
          seenToolCallIds.add(block.id)
        }
        input.push({
          type: 'function_call',
          call_id: block.id,
          name: block.name,
          arguments: safeJsonStringify(block.input ?? {}),
        })
        continue
      }

      if (block.type === 'tool_result') {
        if (!block.tool_use_id || !seenToolCallIds.has(block.tool_use_id)) {
          continue
        }
        flushMessage()
        input.push({
          type: 'function_call_output',
          call_id: block.tool_use_id,
          output: formatToolOutput(block.content),
        })
        continue
      }

      if (msg.role !== 'user') {
        continue
      }

      if (block.type === 'image') {
        if (block.url) {
          parts.push({ type: 'input_image', image_url: block.url })
        } else if (block.base64 && block.mime_type) {
          parts.push({ type: 'input_image', image_url: `data:${block.mime_type};base64,${block.base64}` })
        } else {
          parts.push({ type: 'input_text', text: '[image unsupported] Please provide a URL or base64 image.' })
        }
        continue
      }

      if (block.type === 'audio') {
        parts.push({ type: 'input_text', text: '[audio unsupported] Please provide a transcript.' })
        continue
      }

      if (block.type === 'file') {
        if (block.file_id) {
          parts.push({ type: 'input_file', file_id: block.file_id })
        } else if (block.url) {
          parts.push({ type: 'input_file', file_url: block.url })
        } else if (block.base64 && block.mime_type) {
          parts.push({
            type: 'input_file',
            filename: block.filename || 'file.pdf',
            file_data: `data:${block.mime_type};base64,${block.base64}`,
          })
        } else {
          parts.push({ type: 'input_text', text: '[file unsupported] Please provide a file id, URL, or base64 payload.' })
        }
      }
    }

    flushMessage()
  }

  return input
}

function extractBlocksFromResponses(data: any): ContentBlock[] {
  const contentBlocks: ContentBlock[] = []
  const outputs = Array.isArray(data?.output) ? data.output : []

  for (const item of outputs) {
    if (!item || typeof item !== 'object') {
      continue
    }

    if (item.type === 'message') {
      const parts = Array.isArray(item.content) ? item.content : []
      for (const part of parts) {
        if (!part || typeof part !== 'object') {
          continue
        }

        if (part.type === 'output_text' && typeof part.text === 'string') {
          contentBlocks.push({ type: 'text', text: part.text })
        } else if (part.type === 'text' && typeof part.text === 'string') {
          contentBlocks.push({ type: 'text', text: part.text })
        } else if (part.type === 'refusal' && typeof part.refusal === 'string') {
          contentBlocks.push({ type: 'text', text: part.refusal })
        }
      }
      continue
    }

    if (item.type === 'reasoning') {
      const reasoning = extractReasoningText(item)
      if (reasoning) {
        contentBlocks.push({
          type: 'reasoning',
          reasoning,
          meta: { response_item: item },
        })
      }
      continue
    }

    if (item.type === 'function_call') {
      contentBlocks.push({
        type: 'tool_use',
        id: String(item.call_id || item.id || `toolcall-${contentBlocks.length}`),
        name: String(item.name || 'tool'),
        input: safeJsonParse(item.arguments),
        meta: { response_item: item },
      })
    }
  }

  if (contentBlocks.length === 0 && typeof data?.output_text === 'string' && data.output_text) {
    contentBlocks.push({ type: 'text', text: data.output_text })
  }

  return contentBlocks
}

async function completeWithForcedResponses(
  provider: any,
  messages: Message[],
  opts?: CompletionOptionsLike,
): Promise<ModelResponse> {
  const reasoningTransport = (provider.reasoningTransport ?? 'text') as 'omit' | 'text' | 'provider'
  const input = buildResponsesInput(messages, reasoningTransport)
  const body: Record<string, any> = {
    model: provider.model,
    input,
  }

  if (opts?.system) {
    body.instructions = opts.system
  }
  if (opts?.temperature !== undefined) {
    body.temperature = opts.temperature
  }
  if (opts?.maxTokens !== undefined) {
    body.max_output_tokens = opts.maxTokens
  }

  const tools = buildResponsesTools(opts?.tools as ToolSchema[] | undefined)
  if (tools?.length) {
    body.tools = tools
  }

  provider.applyReasoningDefaults?.(body)

  const response = await fetch(`${provider.baseUrl}/responses`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${provider.apiKey}`,
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`OpenAI API error: ${response.status} ${error}`)
  }

  const data: any = await response.json()
  return {
    role: 'assistant',
    content: extractBlocksFromResponses(data),
    usage: data?.usage
      ? {
          input_tokens: data.usage.input_tokens ?? 0,
          output_tokens: data.usage.output_tokens ?? 0,
        }
      : undefined,
    stop_reason: data?.status,
  }
}

async function* streamWithForcedResponses(
  provider: any,
  messages: Message[],
  opts?: CompletionOptionsLike,
): AsyncIterable<ModelStreamChunk> {
  const response = await completeWithForcedResponses(provider, messages, opts)
  let index = 0

  for (const block of response.content) {
    if (block.type === 'text') {
      yield { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }
      if (block.text) {
        yield { type: 'content_block_delta', index, delta: { type: 'text_delta', text: block.text } }
      }
      yield { type: 'content_block_stop', index }
      index += 1
      continue
    }

    if (block.type === 'reasoning') {
      yield {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'reasoning',
          reasoning: '',
          ...(block.meta ? { meta: block.meta } : {}),
        },
      }
      if (block.reasoning) {
        yield { type: 'content_block_delta', index, delta: { type: 'reasoning_delta', text: block.reasoning } }
      }
      yield { type: 'content_block_stop', index }
      index += 1
      continue
    }

    if (block.type === 'tool_use') {
      yield {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: block.id,
          name: block.name,
          input: block.input ?? {},
          ...(block.meta ? { meta: block.meta } : {}),
        },
      }
      yield { type: 'content_block_stop', index }
      index += 1
    }
  }

  if (response.usage) {
    yield {
      type: 'message_delta',
      usage: {
        input_tokens: response.usage.input_tokens ?? 0,
        output_tokens: response.usage.output_tokens ?? 0,
      },
    }
  }

  yield { type: 'message_stop' }
}

export function createOpenAIProvider(
  apiKey: string,
  model: string,
  baseUrl: string,
  apiMode: OpenAIApiMode,
): OpenAIProvider {
  const provider = new OpenAIProvider(
    apiKey,
    model,
    baseUrl,
    undefined,
    {
      api: apiMode,
      reasoningTransport: 'text',
    },
  )

  if (apiMode !== 'responses') {
    return provider
  }

  const patched = provider as any
  patched.resolveOpenAIApi = () => 'responses'
  patched.buildOpenAIResponsesInput = (messages, reasoningTransport = 'text') => (
    buildResponsesInput(messages, reasoningTransport as 'omit' | 'text' | 'provider')
  )
  patched.completeWithResponses = (messages, opts) => completeWithForcedResponses(patched, messages, opts)
  provider.complete = (messages, opts) => completeWithForcedResponses(patched, messages, opts)
  provider.stream = (messages, opts) => streamWithForcedResponses(patched, messages, opts)

  return provider
}
