import XCTest
@testable import MLXCore

/// DeepSeek-V4.1-Flash's streaming repack (mlx-stream): the launch flags its served runs use and its own context-size
/// choices, both only for a `deepseek_v41` directory holding the EXL3 expert bank.
final class DeepSeekV41LaunchTests: XCTestCase {

    func testLaunchFlagsAreTheServedRunsFlagsAndOnlyForTheRepack() {
        XCTAssertEqual(DeepSeekV41.launchArgs(streamedRepack: true),
                       ["--memory-ceiling-gb", "120.259", "--wired-margin-gib", "2"])
        XCTAssertEqual(DeepSeekV41.launchArgs(streamedRepack: false), [])
    }

    func testLaunchFlagsParseAsTheServerRequires() {
        // main.zig: --memory-ceiling-gb takes decimal GB above 4; --wired-margin-gib an integer 2..32.
        let args = DeepSeekV41.launchArgs(streamedRepack: true)
        let ceiling = Double(args[1]) ?? 0
        XCTAssertEqual(ceiling, 112 * 1_073_741_824 / 1e9, accuracy: 0.001)
        XCTAssertTrue((2...32).contains(Int(args[3]) ?? 0))
    }

    func testTheRepackIsTheArchWithItsExpertBank() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dsv41-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertFalse(DeepSeekV41.isStreamedRepack(atModelPath: dir.path))
        try #"{"model_type": "deepseek_v41"}"#.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        XCTAssertFalse(DeepSeekV41.isStreamedRepack(atModelPath: dir.path), "an MLX pack of the arch is served in-tree")
        try Data().write(to: dir.appendingPathComponent("experts.bin"))
        XCTAssertTrue(DeepSeekV41.isStreamedRepack(atModelPath: dir.path))
    }

    func testContextChoicesAreTheSweepSizesPlusTheReplyPadAndTheModelLimit() {
        let sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map { $0 * 1024 + 1088 }
        XCTAssertEqual(DeepSeekV41.ctxSizeChoices, sweep + [1_048_576])
        XCTAssertEqual(DeepSeekV41.contextPresets(streamedRepack: true), DeepSeekV41.ctxSizeChoices)
        XCTAssertEqual(DeepSeekV41.contextPresets(streamedRepack: false), ContextSizeDisplay.presets)
    }
}
