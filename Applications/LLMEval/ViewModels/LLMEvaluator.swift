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
    
    // --- MEMORY / HISTORY ---
    var chatHistory: [Chat.Message] = []

    var downloadProgress: Double?
    var totalSize: String?

    var tokensPerSecond: Double = 0.0
    var timeToFirstToken: Double = 0.0
    var promptLength: Int = 0
    var totalTokens: Int = 0
    var totalTime: Double = 0.0

    var wasTruncated: Bool = false

    private var ttftTimer: Timer?
    private var generationStartTime: TimeInterval = 0
    private var generationTimer: Timer?
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
        modelInfo = "Downloading \(modelName)..."
        downloadProgress = 0.0
        Memory.cacheLimit = 20 * 1024 * 1024

        let hub = HubApi(downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first)

        do {
            let _ = try await downloadModel(hub: hub, configuration: modelConfiguration) { [weak self] progress in
                Task { @MainActor in self?.updateDownloadProgress(progress) }
            }
            let modelContainer = try await LLMModelFactory.shared.loadContainer(hub: hub, configuration: modelConfiguration) { _ in }
            let numParams = await modelContainer.perform { $0.model.numParameters() }

            self.prompt = ""
            self.modelInfo = formatModelInfo(name: modelConfiguration.name, parameters: numParams)
            loadState = .loaded(modelContainer)
            return modelContainer
        } catch {
            resetLoadingState()
            throw error
        }
    }

    private func updateDownloadProgress(_ progress: Progress) {
        modelInfo = "Downloading \(modelName) (\(Int(progress.fractionCompleted * 100))%)"
        downloadProgress = progress.fractionCompleted
        if progress.totalUnitCount > 0 {
            totalSize = "\(formatBytes(progress.completedUnitCount)) of \(formatBytes(progress.totalUnitCount))"
        }
    }

    private func resetLoadingState() {
        loadState = .idle
        downloadProgress = nil
    }

    private func formatModelInfo(name: String, parameters: Int) -> String {
        let paramMillions = parameters / (1024 * 1024)
        return "\(name.components(separatedBy: "/").last ?? name) • \(paramMillions >= 1000 ? String(format: "%.1fB", Double(paramMillions)/1000.0) : "\(paramMillions)M")"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
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
        if !finalChat.contains(where: { if case .system = $0 { return true }; return false }) {
            finalChat.insert(.system("You are a helpful assistant"), at: 0)
        }

        let userInput = UserInput(chat: finalChat, tools: includeWeatherTool ? toolExecutor.allToolSchemas : nil, additionalContext: ["enable_thinking": enableThinking])

        do {
            let modelContainer = try await load()
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let lmInput = try await modelContainer.prepare(input: userInput)
            self.promptLength = lmInput.text.tokens.size
            let start = Date.timeIntervalSinceReferenceDate
            let stream = try await modelContainer.generate(input: lmInput, parameters: generateParameters)

            var fullResponse = ""
            var pendingToolCall: ToolCall?

            for await part in stream {
                if self.ttftTimer != nil {
                    self.ttftTimer?.invalidate()
                    self.ttftTimer = nil
                    self.firstTokenTime = Date.timeIntervalSinceReferenceDate
                    self.timeToFirstToken = (self.firstTokenTime - start) * 1000
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
                        self.totalTime = elapsed
                    }
                }
            }

            if pendingToolCall == nil {
                self.chatHistory.append(.assistant(fullResponse))
            } else if let toolCall = pendingToolCall {
                await self.executeToolAndContinue(toolCall: toolCall, originalPrompt: prompt)
            }
        } catch {
            output = "Failed: \(error)"
        }
    }

    private func executeToolAndContinue(toolCall: ToolCall, originalPrompt: String) async {
        self.output += "\n\n[Executing tool: \(toolCall.function.name)...]\n\n"
        let result = (try? await toolExecutor.execute(toolCall)) ?? "Error executing tool"
        await generate(prompt: originalPrompt, toolResult: result)
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
