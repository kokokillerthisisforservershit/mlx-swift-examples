// Copyright © 2026 Godless Architecture. All rights reserved.
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
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit-mlx"
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
            let modelDirectory = try await downloadModel(
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

            self.prompt = PresetPrompts.all[0].prompt
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
        }

        let chat: [Chat.Message] = [
            .system("You are a helpful assistant"),
            .user(prompt),
        ]

        var finalChat = chat
        if let toolResult {
            finalChat.append(.tool(toolResult))
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

                var generateTokens: Double = 1
                var pendingToolCall: ToolCall?

                if let toolCall = first.toolCall {
                    pendingToolCall = toolCall
                } else if let chunk = first.chunk, !chunk.isEmpty {
                    Task { @MainActor in
                        self.output += chunk
                        self.totalTokens += 1
                    }
                }

                if pendingToolCall == nil {
                    while let next = await iterator.next() {
                        if let toolCall = next.toolCall {
                            pendingToolCall = toolCall
                            break
                        }
                        if let chunk = next.chunk, !chunk.isEmpty {
                            Task { @MainActor in
                                self.output += chunk
                                self.totalTokens += 1
                            }
                            generateTokens += 1
                        }
                    }
                }

                let secondTick = Date.timeIntervalSinceReferenceDate
                let generateTime = secondTick - firstTick
                let generateTps = generateTokens / generateTime

                Task { @MainActor in
                    self.generationTimer?.invalidate()
                    self.generationTimer = nil
                    self.tokensPerSecond = generateTps
                    self.totalTime = generateTime
                    if self.totalTokens >= (parameters.maxTokens ?? Int.max) {
                        self.wasTruncated = true
                    }
                }

                if let toolCall = pendingToolCall {
                    await self.executeToolAndContinue(toolCall: toolCall, originalPrompt: prompt)
                }
            }
        } catch {
            ttftTimer?.invalidate()
            ttftTimer = nil
            generationTimer?.invalidate()
            generationTimer = nil
            output = "Failed: \(error)"
        }
    }

    private func executeToolAndContinue(toolCall: ToolCall, originalPrompt: String) async {
        self.output += "\n\n[Executing tool: \(toolCall.function.name)...]\n\n"
        let result: String
        do {
            result = try await toolExecutor.execute(toolCall)
        } catch {
            result = "Error executing tool: \(error.localizedDescription)"
        }
        await generate(prompt: originalPrompt, toolResult: result)
    }

    func generate() {
        guard !running else { return }
        let currentPrompt = prompt
        guard !currentPrompt.isEmpty else { return }

        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt)
            prompt = ""
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        ttftTimer?.invalidate()
        ttftTimer = nil
        generationTimer?.invalidate()
        generationTimer = nil
        running = false
    }
}
