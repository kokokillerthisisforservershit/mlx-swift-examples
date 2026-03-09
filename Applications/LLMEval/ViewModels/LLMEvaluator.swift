// Original work Copyright © Apple Inc.
// Modifications and additional code
// Copyright © 2026 Godless Architecture
// Inspired by MLX LLM examples from Apple.
// Implementation rewritten by Godless Architecture.

import Hub
import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftUI

@Observable
@MainActor
class LLMEvaluator {

    var running = false
    var includeWeatherTool = false
    var enableThinking = false
    var maxTokens = 2048

    var prompt = ""
    var output = ""
    var modelInfo = ""
    
    // Core chat history for context
    var chatHistory: [Chat.Message] = []

    var downloadProgress: Double?
    var totalSize: String?

    var tokensPerSecond: Double = 0.0
    var timeToFirstToken: Double = 0.0
    var promptLength: Int = 0
    var totalTokens: Int = 0
    var totalTime: Double = 0.0

    private var ttftTimer: Timer?
    private var generationStartTime: TimeInterval = 0
    private var firstTokenTime: TimeInterval = 0

    var modelConfiguration = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    )

    var generateParameters: GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, temperature: 0.6)
    }

    var generationTask: Task<Void, Error>?
    private let toolExecutor = ToolExecutor()

    enum LoadState {
        case idle
        case loading
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    var isLoading: Bool {
        if case .loading = loadState { return true }
        return false
    }

    private var modelName: String {
        modelConfiguration.name.components(separatedBy: "/").last ?? modelConfiguration.name
    }

    func load() async throws -> ModelContainer {
        while true {
            switch loadState {
            case .idle:
                return try await performLoad()
            case .loading:
                try await Task.sleep(for: .milliseconds(100))
            case .loaded(let modelContainer):
                return modelContainer
            }
        }
    }

    private func performLoad() async throws -> ModelContainer {
        loadState = .loading
        modelInfo = "Loading \(modelName)..."
        downloadProgress = 0.0
        
        let hub = HubApi()
        do {
            let _ = try await downloadModel(hub: hub, configuration: modelConfiguration) { [weak self] progress in
                Task { @MainActor in 
                    self?.downloadProgress = progress.fractionCompleted 
                }
            }
            let modelContainer = try await LLMModelFactory.shared.loadContainer(hub: hub, configuration: modelConfiguration) { _ in }
            let numParams = await modelContainer.perform { $0.model.numParameters() }

            self.modelInfo = "\(modelConfiguration.id) • \(numParams / 1_000_000)M"
            loadState = .loaded(modelContainer)
            return modelContainer
        } catch {
            loadState = .idle
            throw error
        }
    }

    private func generate(prompt: String, toolResult: String? = nil) async {
        if toolResult == nil {
            self.output = ""
            self.totalTokens = 0
            generationStartTime = Date.timeIntervalSinceReferenceDate
            ttftTimer?.invalidate()
            ttftTimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { [weak self] _ in
                guard let self = self else { return }
                Task { @MainActor in self.timeToFirstToken = (Date.timeIntervalSinceReferenceDate - self.generationStartTime) * 1000 }
            }
            chatHistory.append(.user(prompt))
        }

        if let toolResult { chatHistory.append(.tool(toolResult)) }

        var finalChat = chatHistory
        
        // Ensure system prompt exists
        let hasSystem = finalChat.contains { message in
            switch message {
            case .system: return true
            default: return false
            }
        }
        
        if !hasSystem {
            finalChat.insert(.system("You are a helpful assistant"), at: 0)
        }

        let userInput = UserInput(
            chat: finalChat, 
            tools: includeWeatherTool ? toolExecutor.allToolSchemas : nil,
            additionalContext: ["enable_thinking": enableThinking]
        )

        do {
            let modelContainer = try await load()
            let lmInput = try await modelContainer.prepare(input: userInput)
            self.promptLength = lmInput.text.tokens.size
            
            let stream = try await modelContainer.generate(input: lmInput, parameters: generateParameters)

            var fullResponse = ""
            var pendingToolCall: ToolCall?

            for await part in stream {
                if self.ttftTimer != nil {
                    self.ttftTimer?.invalidate()
                    self.ttftTimer = nil
                    self.firstTokenTime = Date.timeIntervalSinceReferenceDate
                }

                if let toolCall = part.toolCall {
                    pendingToolCall = toolCall
                    break
                } else if let chunk = part.chunk, !chunk.isEmpty {
                    fullResponse += chunk
                    self.output += chunk
                    self.totalTokens += 1
                    
                    let elapsed = Date.timeIntervalSinceReferenceDate - self.firstTokenTime
                    if elapsed > 0 {
                        self.tokensPerSecond = Double(self.totalTokens) / elapsed
                    }
                }
            }

            if let toolCall = pendingToolCall {
                let result = (try? await toolExecutor.execute(toolCall)) ?? "Error"
                await generate(prompt: prompt, toolResult: result)
            } else {
                self.chatHistory.append(.assistant(fullResponse))
            }
        } catch {
            output = "Error: \(error)"
        }
    }

    func generate() {
        guard !running && !prompt.isEmpty else { return }
        let currentPrompt = prompt
        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt)
            prompt = ""
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }
    
    func clearHistory() {
        chatHistory.removeAll()
        output = ""
    }
}
