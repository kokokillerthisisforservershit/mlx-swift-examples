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
    // This keeps the conversation context alive
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

        let hub = HubApi(
            downloadBase: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        )

        do {
            let _ = try await downloadModel(
                hub: hub,
                configuration: modelConfiguration
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.updateDownloadProgress(progress)
                }
            }

            modelInfo = "Loading \(modelName)..."
            downloadProgress = nil
            totalSize = nil

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                hub: hub,
                configuration: modelConfiguration
            ) { _ in }

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

        if progress.totalUnitCount > 0 && progress.totalUnitCount < 100 {
            totalSize = "File \(progress.completedUnitCount + 1) of \(progress.totalUnitCount)"
        } else if progress.totalUnitCount > 0 {
            totalSize = "\(formatBytes(progress.completedUnitCount)) of \(formatBytes(progress.totalUnitCount))"
        } else {
            totalSize = nil
        }
    }

    private func resetLoadingState() {
        loadState = .idle
        downloadProgress = nil
        totalSize = nil
    }

    private func formatModelInfo(name: String, parameters: Int) -> String {
        let modelName = name.components(separatedBy: "/").last ?? name
        let paramMillions = parameters / (1024 * 1024)
        let paramString: String
        if paramMillions >= 1000 {
            let paramBillions = Double(paramMillions) / 1000.0
            paramString = String(format: "%.1fB", paramBillions)
        } else {
            paramString = "\(paramMillions)M"
        }
        return "\(modelName) • \(paramString) parameters"
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
            self.tokensPerSecond = 0.0
            self.promptLength = 0
            self.timeToFirstToken = 0.0
            self.totalTime = 0.0
            self.wasTruncated = false

            generationStartTime = Date.timeIntervalSinceReferenceDate
            ttftTimer?.invalidate()
            ttftTimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { [weak self] _ in
                guard let self = self else { return }
                Task { @MainActor in
                    let elapsed = Date.timeIntervalSinceReferenceDate - self.generationStartTime
                    self.timeToFirstToken = elapsed * 1000
                }
            }
            
            // Add user message to history
            chatHistory.append(.user(prompt))
        }

        if let toolResult {
            chatHistory.append(.tool(toolResult))
        }

        // Initialize with system prompt if history is empty or just started
        var finalChat = chatHistory
        if !finalChat.contains(where: { if case .system = $0 { return true }; return false }) {
            finalChat.insert(.system("You are a helpful assistant"), at: 0)
        }

        let userInput = UserInput(
            chat: finalChat,
            tools: includeWeatherTool ? toolExecutor.allToolSchemas : nil,
            additionalContext: ["enable_thinking": enableThinking]
        )

        do {
            let modelContainer = try await load()
            let parameters = generateParameters

            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let lmInput = try await modelContainer.prepare(input: userInput)
            let promptTokenCount = lmInput.text.tokens.size
            let start = Date.timeIntervalSinceReferenceDate
            let stream = try await modelContainer.generate(input: lmInput, parameters: parameters)

            var iterator = stream.makeAsyncIterator()
            var fullResponse = ""

            if let first = await iterator.next() {
                let firstTick = Date.timeIntervalSinceReferenceDate
                let promptTime = firstTick - start

                Task { @MainActor in
                    self.ttftTimer?.invalidate()
                    self.ttftTimer = nil
                    self.timeToFirstToken = promptTime * 1000
                    self.promptLength = promptTokenCount
                    self.firstTokenTime = Date.timeIntervalSinceReferenceDate
                    
                    self.generationTimer?.invalidate()
                    self.generationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                        guard let self = self else { return }
                        Task { @MainActor in
                            let elapsed = Date.timeIntervalSinceReferenceDate - self.firstTokenTime
                            if elapsed > 0 && self.totalTokens > 0 {
                                self.tokensPerSecond = Double(self.totalTokens) / elapsed
                                self.totalTime = elapsed
                            }
                        }
                    }
                }

                var generateTokens: Double = 0
                var pendingToolCall: ToolCall?

                func processPart(_ part: GenerateResult) {
                    if let toolCall = part.toolCall {
                        pendingToolCall = toolCall
                    } else if let chunk = part.chunk, !chunk.isEmpty {
                        fullResponse += chunk
                        Task { @MainActor in
                            self.output += chunk
                            self.
