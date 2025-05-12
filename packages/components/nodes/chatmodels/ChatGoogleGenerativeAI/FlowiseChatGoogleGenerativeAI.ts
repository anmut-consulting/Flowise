import { BaseMessage, AIMessage, AIMessageChunk, isBaseMessage, ChatMessage, MessageContentComplex } from '@langchain/core/messages'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { BaseChatModel, type BaseChatModelParams } from '@langchain/core/language_models/chat_models'
import { ChatGeneration, ChatGenerationChunk, ChatResult } from '@langchain/core/outputs'
import { ToolCallChunk } from '@langchain/core/messages/tool'
import { NewTokenIndices } from '@langchain/core/callbacks/base'

// Import from Google Generative AI
import {
    EnhancedGenerateContentResponse,
    Content,
    Part,
    Tool,
    GenerativeModel,
    GoogleGenerativeAI as GenerativeAI
} from '@google/generative-ai'
import type {
    FunctionCallPart,
    FunctionResponsePart,
    SafetySetting,
    UsageMetadata,
    FunctionDeclarationsTool as GoogleGenerativeAIFunctionDeclarationsTool,
    GenerateContentRequest
} from '@google/generative-ai'
import { ICommonObject, IMultiModalOption, IVisionChatModal } from '../../../src'
import { StructuredToolInterface } from '@langchain/core/tools'
import { isStructuredTool } from '@langchain/core/utils/function_calling'
import { zodToJsonSchema } from 'zod-to-json-schema'
import { BaseLanguageModelCallOptions } from '@langchain/core/language_models/base'
import type FlowiseGoogleAICacheManager from '../../cache/GoogleGenerativeAIContextCache/FlowiseGoogleAICacheManager'

const DEFAULT_IMAGE_MAX_TOKEN = 8192
const DEFAULT_IMAGE_MODEL = 'gemini-1.5-flash-latest'

interface TokenUsage {
    completionTokens?: number
    promptTokens?: number
    totalTokens?: number
}

interface GoogleGenerativeAIChatCallOptions extends BaseLanguageModelCallOptions {
    tools?: StructuredToolInterface[] | GoogleGenerativeAIFunctionDeclarationsTool[]
    /**
     * Whether or not to include usage data, like token counts
     * in the streamed response chunks.
     * @default true
     */
    streamUsage?: boolean
}

export interface GoogleGenerativeAIChatInput extends BaseChatModelParams, Pick<GoogleGenerativeAIChatCallOptions, 'streamUsage'> {
    modelName?: string
    model?: string
    temperature?: number
    maxOutputTokens?: number
    topP?: number
    topK?: number
    stopSequences?: string[]
    safetySettings?: SafetySetting[]
    apiKey?: string
    apiVersion?: string
    baseUrl?: string
    streaming?: boolean
    useSearchGrounding?: boolean
}

class LangchainChatGoogleGenerativeAI
    extends BaseChatModel<GoogleGenerativeAIChatCallOptions, AIMessageChunk>
    implements GoogleGenerativeAIChatInput
{
    modelName = 'gemini-pro'

    temperature?: number

    maxOutputTokens?: number

    topP?: number

    topK?: number

    stopSequences: string[] = []

    safetySettings?: SafetySetting[]

    apiKey?: string

    baseUrl?: string

    streaming = false

    streamUsage = true
    
    // Whether to use Google Search Grounding
    useSearchGrounding = false

    private client: GenerativeModel

    private contextCache?: FlowiseGoogleAICacheManager

    get _isMultimodalModel() {
        return this.modelName.includes('vision') || this.modelName.startsWith('gemini-1.5')
    }

    constructor(fields?: GoogleGenerativeAIChatInput) {
        super(fields ?? {})

        this.modelName = fields?.model?.replace(/^models\//, '') ?? fields?.modelName?.replace(/^models\//, '') ?? 'gemini-pro'

        this.maxOutputTokens = fields?.maxOutputTokens ?? this.maxOutputTokens

        if (this.maxOutputTokens && this.maxOutputTokens < 0) {
            throw new Error('`maxOutputTokens` must be a positive integer')
        }

        this.temperature = fields?.temperature ?? this.temperature
        if (this.temperature && (this.temperature < 0 || this.temperature > 1)) {
            throw new Error('`temperature` must be in the range of [0.0,1.0]')
        }

        this.topP = fields?.topP ?? this.topP
        if (this.topP && this.topP < 0) {
            throw new Error('`topP` must be a positive integer')
        }

        if (this.topP && this.topP > 1) {
            throw new Error('`topP` must be below 1.')
        }

        this.topK = fields?.topK ?? this.topK
        if (this.topK && this.topK < 0) {
            throw new Error('`topK` must be a positive integer')
        }

        // Set search grounding if provided, otherwise use default (false)
        this.useSearchGrounding = fields?.useSearchGrounding ?? false

        this.stopSequences = fields?.stopSequences ?? this.stopSequences

        this.apiKey = fields?.apiKey ?? process.env['GOOGLE_API_KEY']
        if (!this.apiKey) {
            throw new Error(
                'Please set an API key for Google GenerativeAI ' +
                    'in the environment variable GOOGLE_API_KEY ' +
                    'or in the `apiKey` field of the ' +
                    'ChatGoogleGenerativeAI constructor'
            )
        }

        this.safetySettings = fields?.safetySettings ?? this.safetySettings
        if (this.safetySettings && this.safetySettings.length > 0) {
            const safetySettingsSet = new Set(this.safetySettings.map((s) => s.category))
            if (safetySettingsSet.size !== this.safetySettings.length) {
                throw new Error('The categories in `safetySettings` array must be unique')
            }
        }

        this.streaming = fields?.streaming ?? this.streaming

        this.streamUsage = fields?.streamUsage ?? this.streamUsage

        this.getClient()
    }

    async getClient(prompt?: Content[], tools?: Tool[]) {
        // Prepare tools based on configuration and input
        let modelTools = tools ? [...tools] : [];
        
        // Add Google Search tool if enabled
        if (this.useSearchGrounding) {
            // Create search grounding tool using the simple google_search field
            // as suggested by the Google API error message
            modelTools.push({
                google_search: {}
            } as Tool);
        }
        
        // Initialize the Google API client with all configuration
        this.client = new GenerativeAI(this.apiKey ?? '').getGenerativeModel({
            model: this.modelName,
            tools: modelTools.length > 0 ? modelTools : undefined,
            safetySettings: this.safetySettings,
            generationConfig: {
                candidateCount: 1,
                stopSequences: this.stopSequences,
                maxOutputTokens: this.maxOutputTokens,
                temperature: this.temperature,
                topP: this.topP,
                topK: this.topK
            }
        });
        
        // Handle context cache for the Google API client
        if (this.contextCache) {
            const cachedContent = await this.contextCache.lookup({
                contents: prompt ? [{ ...prompt[0], parts: prompt[0].parts.slice(0, 1) }] : [],
                model: this.modelName,
                tools
            });
            // Apply cached content to the client if applicable
            if (cachedContent) {
                this.client.cachedContent = cachedContent;
            }
        }
    }

    _combineLLMOutput() {
        return []
    }

    _llmType() {
        return 'googlegenerativeai'
    }

    override bindTools(tools: (StructuredToolInterface | Record<string, unknown>)[], kwargs?: Partial<ICommonObject>) {
        // Process the tools and create a new client with them
        let toolsArray = [];
        
        // Process tools for compatibility with the Google API
        if (Array.isArray(tools)) {
            // Convert structured tools to Gemini tools format
            if (!tools.some((t: any) => !('lc_namespace' in t))) {
                toolsArray.push(...convertToGeminiTools(tools as StructuredToolInterface[]));
            } else {
                // Tools are already in the correct format
                toolsArray.push(...tools as any[]);
            }
        }
        
        // Initialize the client with the processed tools
        this.getClient(undefined, toolsArray as Tool[]);
        
        // Return this for method chaining
        return this;
    }

    invocationParams(options?: this['ParsedCallOptions']): Omit<GenerateContentRequest, 'contents'> {
        const tools = options?.tools as GoogleGenerativeAIFunctionDeclarationsTool[] | StructuredToolInterface[] | undefined
        let toolsArray = [];
        
        // Add standard tool conversions if needed
        if (Array.isArray(tools) && !tools.some((t: any) => !('lc_namespace' in t))) {
            toolsArray.push(...convertToGeminiTools(tools as StructuredToolInterface[]));
        } else if (tools) {
            toolsArray.push(...tools);
        }
        
        // Add Google Search tool if enabled
        if (this.useSearchGrounding) {
            toolsArray.push({
                google_search: {}
            } as any);
        }
        
        return {
            tools: toolsArray.length > 0 ? toolsArray as any : undefined
        }
    }
    
    /**
     * This method recreates the model with any tools that need to be added
     * @param options The parsed call options with any tools to add
     * @returns The model instance with the appropriate tools
     */
    _prepareForCall(options?: this['ParsedCallOptions']) {
        // Process tools from the options
        const tools = options?.tools as StructuredToolInterface[] | undefined;
        
        // If we have tools or search grounding is enabled, create a new client
        if ((tools && tools.length > 0) || this.useSearchGrounding) {
            let toolsArray: any[] = [];
            
            // Add tools from the options if they exist
            if (tools && tools.length > 0) {
                if (!tools.some((t: any) => !('lc_namespace' in t))) {
                    // Convert structured tools to Gemini tools
                    toolsArray.push(...convertToGeminiTools(tools));
                } else {
                    // Tools are already in the correct format
                    toolsArray.push(...tools as any[]);
                }
            }
            
            // Initialize a new client with the tools and search grounding if needed
            this.getClient(undefined, toolsArray as Tool[]);
        }
        
        // Return this for method chaining
        return this;
    }
    
    /**
     * Handles the actual call to the client with proper formatting
     * @param messages The messages to send
     * @param options Call options
     * @returns The response from the client
     */
    async _handleCall(messages: BaseMessage[], options?: this['ParsedCallOptions']) {
        try {
            // Convert the messages to Google API format
            const contents: Content[] = [];
            
            for (const message of messages) {
                if (isBaseMessage(message)) {
                    const role = message._getType() === 'human' ? 'user' : 
                                message._getType() === 'ai' ? 'model' : 
                                message._getType() === 'system' ? 'model' : 'user';
                    
                    // Convert text content to Google API format
                    contents.push({
                        role,
                        parts: [{ text: message.content as string }]
                    });
                }
            }
            
            // Create the request
            const request: GenerateContentRequest = { contents };
            
            // Add tools including Google Search if enabled
            if (this.useSearchGrounding) {
                if (!request.tools) {
                    request.tools = [];
                }
                
                // Add Google Search tool - use google_search as per the API error message
                request.tools.push({
                    google_search: {}
                } as Tool);
            }
            
            // Use the Google API client
            const result = await this.client.generateContent(request);
            
            // Convert response to the expected format
            const text = result.response.text();
            return {
                generations: [{
                    text,
                    message: new AIMessage(text)
                }]
            };
        } catch (error) {
            console.error('Error in _handleCall:', error);
            throw error;
        }
    }
    
    /**
     * Process function response format for Google API
     */
    convertFunctionResponse(prompts: Content[]) {
        for (let i = 0; i < prompts.length; i += 1) {
            if (prompts[i].role === 'function') {
                if (prompts[i - 1].role === 'model') {
                    const toolName = prompts[i - 1].parts[0].functionCall?.name ?? ''
                    prompts[i].parts = [
                        {
                            functionResponse: {
                                name: toolName,
                                response: {
                                    name: toolName,
                                    content: prompts[i].parts[0].text
                                }
                            }
                        }
                    ]
                }
            }
        }
    }

    /**
     * Set the context cache manager
     */
    setContextCache(contextCache: FlowiseGoogleAICacheManager): void {
        this.contextCache = contextCache
    }

    /**
     * Count tokens in the prompt
     */
    async getNumTokens(prompt: BaseMessage[]) {
        const contents = convertBaseMessagesToContent(prompt, this._isMultimodalModel)
        const { totalTokens } = await this.client.countTokens({ contents })
        return totalTokens
    }

    /**
     * Non-streaming implementation
     */
    async _generateNonStreaming(
        prompt: Content[],
        options: this['ParsedCallOptions'],
        _runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
        const tools = options?.tools ?? []

        // Process any function responses to the correct format
        this.convertFunctionResponse(prompt)

        // Make sure we have a client with the right tools and configuration
        await this.getClient(prompt, tools as Tool[])

        // Generate content with the Google API directly
        const res = await this.caller.callWithOptions({ signal: options?.signal }, async () => {
            try {
                // Create the request, including search grounding if enabled
                const request: GenerateContentRequest = { contents: prompt }
                
                // Add tools if needed
                if (tools.length > 0 || this.useSearchGrounding) {
                    const invocationParams = this.invocationParams(options)
                    Object.assign(request, invocationParams)
                }
                
                const output = await this.client.generateContent(request)
                return output
            } catch (e: any) {
                // Enhance error information
                if (e.message?.includes('400 Bad Request')) {
                    e.status = 400
                }
                throw e
            }
        })

        // Convert response to ChatResult format
        const generationResult = mapGenerateContentResultToChatResult(res.response)
        
        // Handle token tracking
        await _runManager?.handleLLMNewToken(
            generationResult.generations?.length ? generationResult.generations[0].text : ''
        )
        
        return generationResult
    }

    /**
     * Override the _generate method to use our implementation
     */
    async _generate(messages: BaseMessage[], options: this['ParsedCallOptions'], runManager?: CallbackManagerForLLMRun): Promise<ChatResult> {
        // Process messages into Google API format
        let prompt = convertBaseMessagesToContent(messages, this._isMultimodalModel)
        prompt = checkIfEmptyContentAndSameRole(prompt)

        // Handle streaming
        if (this.streaming) {
            const tokenUsage: TokenUsage = {}
            const stream = this._streamResponseChunks(messages, options, runManager)
            const finalChunks: Record<number, ChatGenerationChunk> = {}

            for await (const chunk of stream) {
                const index = (chunk.generationInfo as NewTokenIndices)?.completion ?? 0
                if (finalChunks[index] === undefined) {
                    finalChunks[index] = chunk
                } else {
                    finalChunks[index] = finalChunks[index].concat(chunk)
                }
            }
            const generations = Object.entries(finalChunks)
                .sort(([aKey], [bKey]) => parseInt(aKey, 10) - parseInt(bKey, 10))
                .map(([_, value]) => value)

            return { generations, llmOutput: { estimatedTokenUsage: tokenUsage } }
        }
        
        // Handle non-streaming case
        return this._generateNonStreaming(prompt, options, runManager)
    }

    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        let prompt = convertBaseMessagesToContent(messages, this._isMultimodalModel)
        prompt = checkIfEmptyContentAndSameRole(prompt)

        const parameters = this.invocationParams(options)
        const request = {
            ...parameters,
            contents: prompt
        }

        const tools = options.tools ?? []
        if (tools.length > 0) {
            await this.getClient(prompt, tools as Tool[])
        } else {
            await this.getClient(prompt)
        }

        const stream = await this.caller.callWithOptions({ signal: options?.signal }, async () => {
            const { stream } = await this.client.generateContentStream(request)
            return stream
        })

        let usageMetadata: UsageMetadata | ICommonObject | undefined
        let index = 0
        for await (const response of stream) {
            if ('usageMetadata' in response && this.streamUsage !== false && options.streamUsage !== false) {
                const genAIUsageMetadata = response.usageMetadata as {
                    promptTokenCount: number
                    candidatesTokenCount: number
                    totalTokenCount: number
                }
                if (!usageMetadata) {
                    usageMetadata = {
                        input_tokens: genAIUsageMetadata.promptTokenCount,
                        output_tokens: genAIUsageMetadata.candidatesTokenCount,
                        total_tokens: genAIUsageMetadata.totalTokenCount
                    }
                } else {
                    // Under the hood, LangChain combines the prompt tokens. Google returns the updated
                    // total each time, so we need to find the difference between the tokens.
                    const outputTokenDiff = genAIUsageMetadata.candidatesTokenCount - (usageMetadata as ICommonObject).output_tokens
                    usageMetadata = {
                        input_tokens: 0,
                        output_tokens: outputTokenDiff,
                        total_tokens: outputTokenDiff
                    }
                }
            }

            const chunk = convertResponseContentToChatGenerationChunk(response, {
                usageMetadata: usageMetadata as UsageMetadata,
                index
            })
            index += 1
            if (!chunk) {
                continue
            }

            yield chunk
            await runManager?.handleLLMNewToken(chunk.text ?? '')
        }
    }
}

export class ChatGoogleGenerativeAI extends LangchainChatGoogleGenerativeAI implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken?: number
    multiModalOption: IMultiModalOption
    id: string

    constructor(id: string, fields?: GoogleGenerativeAIChatInput) {
        super(fields)
        this.id = id
        this.configuredModel = fields?.modelName ?? ''
        this.configuredMaxToken = fields?.maxOutputTokens
    }

    revertToOriginalModel(): void {
        this.modelName = this.configuredModel
        this.maxOutputTokens = this.configuredMaxToken
    }

    setMultiModalOption(multiModalOption: IMultiModalOption): void {
        this.multiModalOption = multiModalOption
    }

    setVisionModel(): void {
        if (this.modelName === 'gemini-1.0-pro-latest') {
            this.modelName = DEFAULT_IMAGE_MODEL
            this.maxOutputTokens = this.configuredMaxToken ? this.configuredMaxToken : DEFAULT_IMAGE_MAX_TOKEN
        }
    }
}

function messageContentMedia(content: MessageContentComplex): Part {
    if ('mimeType' in content && 'data' in content) {
        return {
            inlineData: {
                mimeType: content.mimeType,
                data: content.data
            }
        }
    }

    throw new Error('Invalid media content')
}

function getMessageAuthor(message: BaseMessage) {
    const type = message._getType()
    if (ChatMessage.isInstance(message)) {
        return message.role
    }
    return message.name ?? type
}

function convertAuthorToRole(author: string) {
    switch (author.toLowerCase()) {
        case 'ai':
        case 'assistant':
        case 'model':
            return 'model'
        case 'function':
        case 'tool':
            return 'function'
        case 'system':
        case 'human':
        default:
            return 'user'
    }
}

function convertMessageContentToParts(message: BaseMessage, isMultimodalModel: boolean): Part[] {
    if (typeof message.content === 'string' && message.content !== '') {
        return [{ text: message.content }]
    }

    let functionCalls: FunctionCallPart[] = []
    let functionResponses: FunctionResponsePart[] = []
    let messageParts: Part[] = []

    if ('tool_calls' in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
        functionCalls = message.tool_calls.map((tc) => ({
            functionCall: {
                name: tc.name,
                args: tc.args
            }
        }))
    } else if (message._getType() === 'tool' && message.name && message.content) {
        functionResponses = [
            {
                functionResponse: {
                    name: message.name,
                    response: message.content
                }
            }
        ]
    } else if (Array.isArray(message.content)) {
        messageParts = message.content.map((c) => {
            if (c.type === 'text') {
                return {
                    text: c.text
                }
            }

            if (c.type === 'image_url') {
                if (!isMultimodalModel) {
                    throw new Error(`This model does not support images`)
                }
                let source
                if (typeof c.image_url === 'string') {
                    source = c.image_url
                } else if (typeof c.image_url === 'object' && 'url' in c.image_url) {
                    source = c.image_url.url
                } else {
                    throw new Error('Please provide image as base64 encoded data URL')
                }
                const [dm, data] = source.split(',')
                if (!dm.startsWith('data:')) {
                    throw new Error('Please provide image as base64 encoded data URL')
                }

                const [mimeType, encoding] = dm.replace(/^data:/, '').split(';')
                if (encoding !== 'base64') {
                    throw new Error('Please provide image as base64 encoded data URL')
                }

                return {
                    inlineData: {
                        data,
                        mimeType
                    }
                }
            } else if (c.type === 'media') {
                return messageContentMedia(c)
            } else if (c.type === 'tool_use') {
                return {
                    functionCall: {
                        name: c.name,
                        args: c.input
                    }
                }
            }
            throw new Error(`Unknown content type ${(c as { type: string }).type}`)
        })
    }

    return [...messageParts, ...functionCalls, ...functionResponses]
}

/*
 * This is a dedicated logic for Multi Agent Supervisor to handle the case where the content is empty, and the role is the same
 */

function checkIfEmptyContentAndSameRole(contents: Content[]) {
    let prevRole = ''
    const validContents: Content[] = []

    for (const content of contents) {
        // Skip only if completely empty
        if (!content.parts || !content.parts.length) {
            continue
        }

        // Ensure role is always either 'user' or 'model'
        content.role = content.role === 'model' ? 'model' : 'user'

        // Handle consecutive messages
        if (content.role === prevRole && validContents.length > 0) {
            // Merge with previous content if same role
            validContents[validContents.length - 1].parts.push(...content.parts)
            continue
        }

        validContents.push(content)
        prevRole = content.role
    }

    return validContents
}

function convertBaseMessagesToContent(messages: BaseMessage[], isMultimodalModel: boolean) {
    return messages.reduce<{
        content: Content[]
        mergeWithPreviousContent: boolean
    }>(
        (acc, message, index) => {
            if (!isBaseMessage(message)) {
                throw new Error('Unsupported message input')
            }
            const author = getMessageAuthor(message)
            if (author === 'system' && index !== 0) {
                throw new Error('System message should be the first one')
            }
            const role = convertAuthorToRole(author)

            const prevContent = acc.content[acc.content.length]
            if (!acc.mergeWithPreviousContent && prevContent && prevContent.role === role) {
                throw new Error('Google Generative AI requires alternate messages between authors')
            }

            const parts = convertMessageContentToParts(message, isMultimodalModel)

            if (acc.mergeWithPreviousContent) {
                const prevContent = acc.content[acc.content.length - 1]
                if (!prevContent) {
                    throw new Error('There was a problem parsing your system message. Please try a prompt without one.')
                }
                prevContent.parts.push(...parts)

                return {
                    mergeWithPreviousContent: false,
                    content: acc.content
                }
            }
            let actualRole = role
            if (actualRole === 'function' || actualRole === 'tool') {
                // GenerativeAI API will throw an error if the role is not "user" or "model."
                actualRole = 'user'
            }
            const content: Content = {
                role: actualRole,
                parts
            }
            return {
                mergeWithPreviousContent: author === 'system',
                content: [...acc.content, content]
            }
        },
        { content: [], mergeWithPreviousContent: false }
    ).content
}

function mapGenerateContentResultToChatResult(
    response: EnhancedGenerateContentResponse,
    extra?: {
        usageMetadata: UsageMetadata | undefined
    }
): ChatResult {
    // if rejected or error, return empty generations with reason in filters
    if (!response.candidates || response.candidates.length === 0 || !response.candidates[0]) {
        return {
            generations: [],
            llmOutput: {
                filters: response.promptFeedback
            }
        }
    }

    const functionCalls = response.functionCalls()
    const [candidate] = response.candidates
    const { content, ...generationInfo } = candidate
    const text = content?.parts[0]?.text ?? ''

    const generation: ChatGeneration = {
        text,
        message: new AIMessage({
            content: text,
            tool_calls: functionCalls,
            additional_kwargs: {
                ...generationInfo
            },
            usage_metadata: extra?.usageMetadata as any
        }),
        generationInfo
    }

    return {
        generations: [generation]
    }
}

function convertResponseContentToChatGenerationChunk(
    response: EnhancedGenerateContentResponse,
    extra: {
        usageMetadata?: UsageMetadata | undefined
        index: number
    }
): ChatGenerationChunk | null {
    if (!response || !response.candidates || response.candidates.length === 0) {
        return null
    }
    const functionCalls = response.functionCalls()
    const [candidate] = response.candidates
    const { content, ...generationInfo } = candidate
    const text = content?.parts?.[0]?.text ?? ''

    const toolCallChunks: ToolCallChunk[] = []
    if (functionCalls) {
        toolCallChunks.push(
            ...functionCalls.map((fc) => ({
                ...fc,
                args: JSON.stringify(fc.args),
                index: extra.index
            }))
        )
    }
    return new ChatGenerationChunk({
        text,
        message: new AIMessageChunk({
            content: text,
            name: !content ? undefined : content.role,
            tool_call_chunks: toolCallChunks,
            // Each chunk can have unique "generationInfo", and merging strategy is unclear,
            // so leave blank for now.
            additional_kwargs: {},
            usage_metadata: extra.usageMetadata as any
        }),
        generationInfo
    })
}

function zodToGeminiParameters(zodObj: any) {
    // Gemini doesn't accept either the $schema or additionalProperties
    // attributes, so we need to explicitly remove them.
    const jsonSchema: any = zodToJsonSchema(zodObj)
    // eslint-disable-next-line unused-imports/no-unused-vars
    const { $schema, additionalProperties, ...rest } = jsonSchema

    // Ensure all properties have type specified
    if (rest.properties) {
        Object.keys(rest.properties).forEach((key) => {
            const prop = rest.properties[key]

            // Handle enum types
            if (prop.enum?.length) {
                rest.properties[key] = {
                    type: 'string',
                    format: 'enum',
                    enum: prop.enum
                }
            }
            // Handle missing type
            else if (!prop.type && !prop.oneOf && !prop.anyOf && !prop.allOf) {
                // Infer type from other properties
                if (prop.minimum !== undefined || prop.maximum !== undefined) {
                    prop.type = 'number'
                } else if (prop.format === 'date-time') {
                    prop.type = 'string'
                } else if (prop.items) {
                    prop.type = 'array'
                } else if (prop.properties) {
                    prop.type = 'object'
                } else {
                    // Default to string if type can't be inferred
                    prop.type = 'string'
                }
            }
        })
    }

    return rest
}

function convertToGeminiTools(structuredTools: (StructuredToolInterface | Record<string, unknown>)[]) {
    return [
        {
            functionDeclarations: structuredTools.map((structuredTool) => {
                if (isStructuredTool(structuredTool)) {
                    const jsonSchema = zodToGeminiParameters(structuredTool.schema)
                    return {
                        name: structuredTool.name,
                        description: structuredTool.description,
                        parameters: jsonSchema
                    }
                }
                return structuredTool
            })
        }
    ]
}
