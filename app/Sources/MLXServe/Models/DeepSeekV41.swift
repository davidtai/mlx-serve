import Foundation

/// DeepSeek-V4.1-Flash's streaming repack (`deepseek_v41` with an EXL3 expert bank, `experts.bin`), which mlx-stream
/// serves: the trunk loads once and the routed experts stream from SSD. The launch flags and context sizes its served
/// runs were measured with; an MLX pack of the same arch (served in-tree) takes the defaults.
enum DeepSeekV41 {
    static let modelType = "deepseek_v41"
    static let expertBank = "experts.bin"

    /// The plugin admits its expert rows against this box ceiling: 112 GiB in decimal GB, which leaves macOS 16 GiB
    /// of a 128 GiB Mac. `--memory-ceiling-gb` sets the server's static GPU ceiling for every bill, so it rides only
    /// a streaming-repack launch.
    static let memoryCeilingGB = "120.259"
    static let wiredMarginGiB = "2"

    static func launchArgs(streamedRepack: Bool) -> [String] {
        streamedRepack ? ["--memory-ceiling-gb", memoryCeilingGB, "--wired-margin-gib", wiredMarginGiB] : []
    }

    /// The reply allowance the plugin bills beside a prompt: max_tokens 1,024 plus a 64-token pad.
    static let replyPad = 1_088

    /// `ctx_size` choices: prompts of 1K..512K plus the reply allowance, then the model's 1M limit (each one measured).
    /// The plugin bills every prompt up to `ctx_size` at load, so a larger choice admits fewer expert rows; none = 16K.
    static let ctxSizeChoices: [Int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map { $0 * 1024 + replyPad } + [1_048_576]

    static func contextPresets(streamedRepack: Bool) -> [Int] {
        streamedRepack ? ctxSizeChoices : ContextSizeDisplay.presets
    }

    /// True for a model directory whose config.json says `deepseek_v41` and which holds the EXL3 expert bank: the
    /// same rule the server's parseConfig routes to mlx-stream by.
    static func isStreamedRepack(atModelPath path: String) -> Bool {
        let dir = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              obj["model_type"] as? String == modelType else { return false }
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent(expertBank).path)
    }
}
