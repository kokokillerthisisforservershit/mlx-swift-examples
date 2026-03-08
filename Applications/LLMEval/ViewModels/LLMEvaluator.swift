// Copyright © 2026 Godless Architecture. All rights reserved.



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

            // Prototype logic: Skip strict file verification to ensure loading
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
        let formatter = Byte
