// Copyright © 2025 Apple Inc.
// Band link modification © 2026

import SwiftUI

struct HeaderView: View {
    @Bindable var llm: LLMEvaluator
    @Binding var selectedDisplayStyle: ContentView.DisplayStyle

    @Environment(\.horizontalSizeClass) var horizontalSizeClass

    private let bandTikTok = URL(string: "https://www.tiktok.com/@godlessarchitecture")!

    var status: some View {
        // Model info with status
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Model")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Text(llm.modelInfo)
                    .font(.headline)
                    .lineLimit(1)
            }

            Spacer()

            // TikTok Button
            Link(destination: bandTikTok) {
                HStack(spacing: 6) {
                    Image(systemName: "music.note")
                    Text("Godless Architecture")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.black)
                .foregroundColor(.white)
                .cornerRadius(8)
            }

            if llm.running {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    var options: some View {
        HStack(spacing: 24) {
            Toggle("Tools", isOn: $llm.includeWeatherTool)
                .toggleStyle(.switch)
                .fixedSize()
                .help("Enable function calling with weather, math, and time tools")

            Toggle("Thinking", isOn: $llm.enableThinking)
                .toggleStyle(.switch)
                .fixedSize()
                .help("Enable thinking mode (supported by Qwen3)")
        }
    }

    var tokens: some View {
        // Max tokens slider
        VStack(alignment: .leading, spacing: 4) {
            Text("Max Tokens: \(llm.maxTokens)")
                .font(.caption)
                .foregroundStyle(.secondary)

            Slider(
                value: Binding(
                    get: { log2(Double(llm.maxTokens)) },
                    set: { llm.maxTokens = Int(pow(2, $0)) }
                ),
                in: 10 ... 15,  // 2^10 (1024) to 2^15 (32768)
                step: 1
            )
            .frame(width: 120)
            .help("Maximum number of tokens to generate (1024-32768)")
        }
    }

    var display: some View {
        Picker("Display", selection: $selectedDisplayStyle) {
            ForEach(ContentView.DisplayStyle.allCases, id: \.self) { option in
                Text(option.rawValue.capitalized)
                    .tag(option)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(maxWidth: 180)
    }

    var body: some View {
        if horizontalSizeClass == .compact {
            VStack {
                status
                DisclosureGroup("Controls") {
                    VStack {
                        options
                        HStack {
                            tokens
                            display
                        }
                    }
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 12) {
                status

                // Controls row
                HStack(spacing: 16) {
                    HStack(spacing: 24) {
                        options
                        tokens
                    }

                    Spacer()

                    display
                }
            }
            .padding(.bottom, 12)
        }
    }
}
