const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.16.0's bundled libc++ fails to compile against the macOS 27 SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>);
    // fixed in 0.17.0, which is why the floor moved.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "mlx-serve requires Zig 0.17 (have {d}.{d}.{d}). Run ./scripts/fetch-zig.sh or grab it from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

/// stb_image_write's JPEG bit-packer relies on WRAPPING left shifts of a signed
/// int (`bitBuf <<= 8`), which is UB by the letter of C and which Zig's C
/// frontend TRAPS under UBSan — so a Debug build of any graph that encodes a
/// JPEG aborts inside libc. One flag list, used at every site that compiles it,
/// because the flag is about the C source and not about which graph it is in
/// (`zig build test` is Debug and ran 6 crashed tests without it).
const stb_write_flags: []const []const u8 = &.{ "-O2", "-fno-sanitize=undefined" };

pub fn build(b: *std.Build) void {
    // Pin LC_BUILD_VERSION minos to macOS 26.2 — the honest floor: the linked
    // libmlx is built at deployment target 26.2 (NAX kernels, scripts/
    // build-mlx.sh), so on older macOS the binary can't run anyway; failing at
    // the binary with a clear dyld version error beats "loading" and dying on
    // the dylib. Matches app LSMinimumSystemVersion + Package.swift. Guard:
    // tests/test_mlx_staged_nax.sh (binary minos check).
    // Host-gated: on a Linux box a semver 26.2 min would be read as a glibc
    // floor (glibc has no 26.2) and break the native Linux graph.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .macos) .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        } else .{},
    });
    const optimize = b.standardOptimizeOption(.{});

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (target.result.os.tag == .macos) {
        verifyBrewDeps(b);
        verifyMlxStage(b);
    }

    // Hermetic Latent2RGB/JPEG tests: the COMPILED artifact links no MLX and
    // no Homebrew webp, which is what lets a Linux Cloud Agent build it — on
    // Linux this is the only graph registered. On a Mac the step builds the
    // same hermetic artifact, but `verifyBrewDeps`/`verifyMlxStage` above run
    // at CONFIGURE time for every step, so it is not a way to build without a
    // staged mlx.
    // Native query — do not inherit the macOS 26.2 minos default_target.
    addPreviewTest(b, b.resolveTargetQuery(.{}));

    // Linux serve graph: same engine sources, macOS-only engines (ds4 Metal,
    // embedded llama.cpp, ANE objc) replaced by compile-time stubs, mlx +
    // mlx-c staged from the Linux Vulkan fork by scripts/build-mlx-linux.sh.
    // Gated on the TARGET so the Linux exe can be built on any host zig can
    // target; the host gate below still guards the macOS graph.
    if (target.result.os.tag == .linux) {
        addLinuxServe(b, target, optimize);
        return;
    }

    if (builtin.os.tag != .macos) return;

    // App version. Release builds pass it explicitly (app/build.sh computes the
    // next CalVer from the GitHub releases and stamps it into app/Info.plist;
    // the release workflow passes the tag). A plain `zig build` used to fall
    // back to a literal "0.1.0-dev", which then showed up as the version in
    // `--version` AND on the console page — so a dev build reported a version
    // that exists nowhere. Default to the checked-in Info.plist stamp instead:
    // one source of truth, already in the repo, and the same string the last
    // real build shipped. Same pattern as the engine pins below.
    const version = b.option([]const u8, "version", "Version string") orelse readAppVersion(b) orelse "0.0.0-dev";

    const mas = b.option(bool, "mas", "MAS build (no curl/model-pull subprocess)") orelse false;

    // Engine-version pins surfaced by `mlx-serve --version` (the macOS app spawns
    // it and parses the output — see src/version.zig). These are the versions
    // that have NO runtime query API (MLX + ggml report themselves at runtime):
    //   --mlx-c-version  pinned mlx-c submodule version; defaults from the
    //                    lib/mlx/.version stamp (written by scripts/build-mlx.sh)
    //   --ds4-commit     pinned ds4 submodule short commit (build.sh: `git rev-parse`)
    //   --llama-tag      llama.cpp release tag; defaults from lib/llama/.version
    //                    (written by scripts/fetch-llama.sh) so a plain dev build
    //                    still reports it. app/build.sh passes all three.
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxcPin(b) orelse "unknown";
    const ds4_commit = b.option([]const u8, "ds4-commit", "Pinned ds4 submodule short commit") orelse "unknown";
    const llama_tag = b.option([]const u8, "llama-tag", "llama.cpp release tag (bNNNN)") orelse readLlamaTag(b) orelse "unknown";

    const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
    // The slim host (docs/plugins.md): the MLX engine and the registered plugins; ds4, llama.cpp and the ANE bridge
    // are left out for their stubs. Only the server graph is slimmed; the test graph is always the full one.
    const slim = b.option(bool, "slim", "Slim host: the MLX engine and the registered plugins; no ds4, llama.cpp or ANE") orelse false;
    // The registry's plugin lines (src/plugins.zig): false builds the server without that plugin's files at all.
    const with_mlx_stream = b.option(bool, "mlx-stream", "Register the mlx-stream plugin (lib/mlx-stream: the DeepSeek-V4.1 arch)") orelse true;
    const core: CoreOptions = .{ .version = version, .mas = mas, .mlx_c_version = mlx_c_version, .ds4_commit = ds4_commit, .llama_tag = llama_tag, .git_sha = git_sha, .slow_tests = slowTests(b) };
    const build_options = core.add(b, !slim, with_mlx_stream);
    const test_options = core.add(b, true, with_mlx_stream);
    const shared = addShared(b, target, optimize);

    // ds4 Metal kernel sources embedded via @embedFile and exposed as a
    // named module so src/arch/ds4.zig can import them with `@import("ds4_metal_sources")`
    // without traversing the project root.
    const ds4_metal_sources = b.createModule(.{
        .root_source_file = b.path("lib/ds4_metal_sources.zig"),
        .target = target,
        .optimize = optimize,
    });
    const mlx_steel_sources = b.createModule(.{
        .root_source_file = b.path("lib/mlx_steel_sources.zig"),
        .target = target,
        .optimize = optimize,
    });
    const opencode2_plugin = b.createModule(.{
        .root_source_file = b.path("lib/opencode2_plugin.zig"),
        .target = target,
        .optimize = optimize,
    });
    const agent_skills = b.createModule(.{
        .root_source_file = b.path("skills/agent_skills.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "ds4_metal_sources", .module = ds4_metal_sources },
            .{ .name = "mlx_steel_sources", .module = mlx_steel_sources },
            .{ .name = "opencode2_plugin", .module = opencode2_plugin },
            .{ .name = "agent_skills", .module = agent_skills },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize, "") },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, "") },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize, "") },
        },
    });
    shared.importInto(mod);

    // Jinja2 template engine (from llama.cpp's common/jinja + nlohmann/json).
    // Pre-compiled as a static library with system clang++ (C++17 requires system libc++).
    // Rebuild with: cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o
    mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    // stb_image_write for PNG encoding (native image-generation endpoint)
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = stb_write_flags });
    mod.addIncludePath(b.path("lib"));

    // xatlas UV unwrapping + FQMS decimation (MIT, vendored) with C shims for the
    // Hunyuan3D texture paint stage. See lib/{xatlas,fqms} + src/{uvwrap,mesh_simplify}.zig.
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));
    mod.addCSourceFile(.{ .file = b.path("lib/fqms/fqms_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/fqms"));

    // ds4 inference engine for DSV4-Flash (Metal backend, macOS only). See
    // `lib/ds4/` submodule pinned at 9139e2a and `src/arch/ds4.zig`. Kernel
    // sources are embedded via `lib/ds4_metal_sources.zig` and extracted at
    // runtime to ~/.mlx-serve/ds4-metal/<hash>/.
    if (!slim) {
        addDs4Sources(b, mod);
        mod.addIncludePath(b.path("lib/ds4"));
    }
    if (with_mlx_stream) _ = addMlxStreamModule(b, mod, shared.sdk, target, optimize, .{});

    // ANE prefill-MLP offload (perf-plan-aug-17 P5): objc bridge to the
    // private AppleNeuralEngine framework (dlopen'd at runtime — the probe
    // returns unavailable on machines/OSes without it) + the per-layer MLP
    // MIL program builder. See lib/ane/ + src/ane.zig; provenance in NOTICE.
    // The slim host links the Linux graph's unavailable stubs instead.
    if (slim) mod.addCSourceFile(.{ .file = b.path("src/ane_stub.c"), .flags = &.{"-O2"} }) else addAneSources(b, mod);

    // llama.cpp libllama for generic GGUF models (Metal backend, macOS only).
    // Staged by `scripts/fetch-llama.sh` into lib/llama/ (a single self-contained
    // dylib + headers extracted from the pinned XCFramework). See src/arch/llama.zig.
    if (!slim) addLlamaLib(b, mod);

    // mlx + mlx-c: self-built from the pinned submodules (lib/mlx-src,
    // lib/mlxc-src) into lib/mlx by scripts/build-mlx.sh, with NAX kernels
    // enabled (the Homebrew bottle ships without them). MUST come before the
    // /opt/homebrew lib path so a leftover brew mlx-c can never win the link.
    addMlxLib(b, mod);
    // webp include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.linkFramework("IOSurface", .{});

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes (app bundling)
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);

    // The server graph's semantic check: nothing depends on this artifact's binary, so no code is generated.
    const check_exe = b.addExecutable(.{ .name = "mlx-serve-check", .root_module = mod });
    const check_step = b.step("check", "Check that the server graph compiles, without codegen (with -Dslim: the slim host)");
    check_step.dependOn(&check_exe.step);
    // The Linux graph's semantic check (stub engines; macOS-only plugins register nothing): its module without link
    // inputs, nothing emitted, Homebrew's webp headers in place of the system's; glibc 2.39 (arc4random_buf, as a
    // current distribution's).
    const linux_check = b.addExecutable(.{
        .name = "mlx-serve-linux-check",
        .root_module = linuxModule(b, b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu, .glibc_version = .{ .major = 2, .minor = 39, .patch = 0 } }), .Debug, version, "unknown", "/opt/homebrew/include"),
    });
    const check_linux = b.step("check-linux", "Check that the Linux server graph (stub engines) compiles, without codegen or a Linux MLX stage");
    check_linux.dependOn(&linux_check.step);

    // Unit tests — reuses the same module config (mlx-c, jinja_cpp, etc.)
    const test_deps: TestDeps = .{ .options = test_options, .shared = shared, .ds4_metal_sources = ds4_metal_sources, .mlx_steel_sources = mlx_steel_sources, .opencode2_plugin = opencode2_plugin, .agent_skills = agent_skills, .frameworks = macos_sdk_frameworks, .with_mlx_stream = with_mlx_stream };
    const test_mod = test_deps.module(b, b.path("src/tests.zig"), target, optimize);

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const test_build = b.step("test-build", "Compile unit tests without running them");
    test_build.dependOn(&b.addInstallArtifact(unit_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    if (qwen_preprocess_fixture) |fixture| {
        run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
    }
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Only the test root's module runs its `test` decls, so each shared module gets its own artifact.
    // mlx.zig's tests create arrays (the device): the full suite runs them, the hermetic lanes never do.
    // mlx-test gets its own module instance: linking MLX into the shared one would link it into the CPU lane.
    const mlx_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlx.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "log", .module = shared.log }},
    });
    addMlxLib(b, mlx_test_mod);
    const shared_tests = [_]*std.Build.Step.Compile{
        b.addTest(.{ .name = "log-test", .root_module = shared.log }),
        b.addTest(.{ .name = "io_util-test", .root_module = shared.io_util }),
        b.addTest(.{ .name = "mtp_acceptance-test", .root_module = shared.mtp_acceptance }),
        b.addTest(.{ .name = "mlx-test", .root_module = mlx_test_mod }),
    };
    for (shared_tests) |t| test_step.dependOn(&b.addRunArtifact(t).step);

    // The plugin SDK's tests (no MLX linked) and the conformance suite's CPU lane (docs/plugins.md): sdk.testing over
    // every registered plugin, rooted at the registry, linked like the unit tests; its last check is that no Metal
    // device was created.
    const registry_mod = test_deps.module(b, b.path("src/plugins.zig"), target, optimize);
    const sdk_tests = b.addTest(.{ .name = "sdk-test", .root_module = shared.sdk, .filters = if (test_filter) |f| &.{f} else &.{} });
    // Only the registry's own tests: the files a registered plugin reaches keep theirs in the unit tests.
    const conformance_tests = b.addTest(.{ .name = "conformance", .root_module = registry_mod, .filters = &.{"plugins "} });
    const conformance = b.step("conformance", "Run the SDK's tests and the plugin conformance suite (CPU lane, no device)");
    conformance.dependOn(&b.addRunArtifact(sdk_tests).step);
    conformance.dependOn(&b.addRunArtifact(conformance_tests).step);
    // The registry's compile-time refusals: each case compiles src/plugins_refusals.zig with one bad plugin line and
    // passes only on the compile error that names it.
    for (registry_refusals) |c| {
        const case_options = b.addOptions();
        case_options.addOption([]const u8, "name", c.case);
        const m = b.createModule(.{ .root_source_file = b.path("src/plugins_refusals.zig"), .target = target, .optimize = optimize, .imports = &.{
            .{ .name = "sdk", .module = shared.sdk },
            .{ .name = "build_options", .module = test_options.createModule() },
            .{ .name = "refusal_case", .module = case_options.createModule() },
        } });
        const obj = b.addObject(.{ .name = b.fmt("refusal-{s}", .{c.case}), .root_module = m });
        obj.expect_errors = .{ .contains = c.err };
        conformance.dependOn(&obj.step);
    }
    test_step.dependOn(conformance);
    const sdk_test_build = b.step("sdk-test-build", "Compile the SDK, conformance and shared-module tests without running them (mlx-test creates arrays)");
    for ([_]*std.Build.Step.Compile{ sdk_tests, conformance_tests } ++ shared_tests) |t| sdk_test_build.dependOn(&b.addInstallArtifact(t, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);

    // The mlx-stream plugin's own tests (its repo's src/tests.zig) and its conformance suite (src/conformance.zig),
    // built against this host: the plugin module is each test build's root, and the host files its harnesses reach
    // (src/sdk_test_host.zig) import it back as `mlx_stream`. `zig build test` runs both (the gated ones skip without
    // their environment).
    if (with_mlx_stream) {
        const pkg_tests = addMlxStreamTests(b, test_deps, "mlx-stream-test", "src/tests.zig", target, optimize, test_filter);
        const pkg_conformance = addMlxStreamTests(b, test_deps, "mlx-stream-conformance", "src/conformance.zig", target, optimize, "mlx-stream conformance");
        const pkg_test_build = b.step("mlx-stream-test-build", "Compile the mlx-stream plugin's tests and conformance suite without running them");
        for ([_]*std.Build.Step.Compile{ pkg_tests, pkg_conformance }) |t| pkg_test_build.dependOn(&b.addInstallArtifact(t, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const pkg_test = b.step("mlx-stream-test", "Run the mlx-stream plugin's tests against this host");
        pkg_test.dependOn(&b.addRunArtifact(pkg_tests).step);
        const pkg_conf = b.step("mlx-stream-conformance", "Run the mlx-stream plugin's conformance suite against this host (CPU lane, no device)");
        pkg_conf.dependOn(&b.addRunArtifact(pkg_conformance).step);
        conformance.dependOn(pkg_conf);
        test_step.dependOn(pkg_test);
    }

    // ── vz-agent: the Agent Sandbox's guest-side binary.
    //
    // A standalone static aarch64-linux-musl ELF (~200 KB) that the app injects
    // into the guest rootfs before boot, exactly like `/.vz-init`. It serves the
    // vsock exec protocol (`src/vz_agent.zig`), replacing the hvc1 console shell.
    //
    // It is NOT imported by main.zig — it links nothing but libc and never runs
    // on macOS. Its tests do, though: `serveConnection` is OS-agnostic, so the
    // whole request → spawn → stream → exit path is exercised over a socketpair
    // here on the host. Wire them into `zig build test` explicitly, since the
    // main test module's root never reaches this file.
    addVzAgent(b, target, optimize, test_step);

    // ── iOS on-device engine: a static library (libmlxserve.a) linking the
    //    MLX-only decode path. ds4 + llama.cpp are stubbed (build_options.ios =
    //    true). Two slices: `zig build ios-lib` (device, arm64-iphoneos) and
    //    `zig build ios-lib-sim` (arm64 iphonesimulator). Driven by the iPhone
    //    app project's build scripts (../mlx-iphone/scripts/build-zig-ios.sh),
    //    which supply the matching --sysroot and copy the artifact out of
    //    zig-out/ios/<sdk>/lib. `-Dios-include=<dir>` points at the iOS dist's
    //    include dir for third-party headers (webp); defaults to Homebrew's,
    //    whose versions are pinned identical by verifyBrewDeps.
    const ios_include = b.option([]const u8, "ios-include", "Include dir for webp/stb headers when cross-compiling the iOS lib") orelse "/opt/homebrew/include";
    addIosLib(b, version, ios_include, .{ .step = "ios-lib", .abi = .none, .sdk = "iphoneos" });
    addIosLib(b, version, ios_include, .{ .step = "ios-lib-sim", .abi = .simulator, .sdk = "iphonesimulator" });
}

/// Hermetic Latent2RGB + JPEG tests. No MLX, no Homebrew — the only `zig build`
/// graph that is valid on Linux (issue #208). The ARTIFACT is hermetic, the
/// STEP is not a way around a missing mlx: `verifyBrewDeps` + `verifyMlxStage`
/// run at configure time for every step, so on a Mac `lib/mlx/` must be staged
/// before this builds. UBSan is off for stb (its bit shifts trip the sanitizer).
fn addPreviewTest(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/preview.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = stb_write_flags });
    mod.addIncludePath(b.path("lib"));
    const tests = b.addTest(.{
        .name = "preview-test",
        .root_module = mod,
    });
    const run = b.addRunArtifact(tests);
    const step = b.step("preview-test", "Hermetic JPEG / Latent2RGB preview tests (no MLX)");
    step.dependOn(&run.step);
}

/// `zig build` on/for Linux: the full mlx-serve HTTP server against the
/// Linux MLX (Vulkan backend) + mlx-c pair staged into lib/mlx by
/// scripts/build-mlx-linux.sh. Mirrors the macOS graph minus everything
/// Apple-specific:
///   - no Metal/ds4/llama.cpp/ANE — stub engines (build_options.macos_engines
///     = false), so only MLX safetensors models are servable, exactly like
///     the iOS build;
///   - no IOKit/Metal frameworks, no metallib fingerprint (round_cost mixes
///     exe bytes only when git_sha is empty, and mixMlxArtifacts no-ops
///     without dyld);
///   - system libwebp instead of Homebrew;
///   - libjinja built for the host by the same staging script (the committed
///     libjinja.a holds Mach-O objects).
fn addLinuxServe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    verifyMlxStageLinux(b);

    const version = b.option([]const u8, "version", "Version string") orelse readAppVersion(b) orelse "0.0.0-dev";
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxcPin(b) orelse "unknown";
    const mod = linuxModule(b, target, optimize, version, mlx_c_version, "/usr/include");

    // Jinja2 template engine — same vendored sources as the macOS graph, built
    // as an ELF static lib by scripts/build-mlx-linux.sh (zig c++).
    mod.addObjectFile(b.path("lib/jinja_cpp/libjinja-linux.a"));

    // mlx (Vulkan fork) + mlx-c, staged in lib/mlx — same link shape as macOS.
    addMlxLib(b, mod);
    // ELF has no @loader_path: the Mach-O rpaths emitted above are inert here,
    // so the loader never finds libmlxc.so. Mirror them in $ORIGIN form.
    mod.addRPath(.{ .cwd_relative = "$ORIGIN/../../lib/mlx/lib" });
    mod.addRPath(.{ .cwd_relative = "$ORIGIN/../../../lib/mlx/lib" });

    // System libwebp for the vision pipeline (pkg-config resolves -lwebp).
    mod.linkSystemLibrary("webp", .{});

    // Bonjour/mDNS peer discovery (src/lan.zig) via Avahi's dns_sd compat lib
    // (Arch: avahi ships /usr/lib/libdns_sd.so; Debian: libavahi-compat-libdnssd-dev).
    mod.linkSystemLibrary("dns_sd", .{ .use_pkg_config = .no });

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);
}

/// The Linux server graph's module: its build options (stub engines), imports, include paths and portable C sources,
/// without the link inputs (the staged Linux mlx / mlx-c, libjinja-linux.a, libwebp, dns_sd), which `addLinuxServe` adds.
/// `check-linux` checks this module alone from the macOS host: nothing is emitted, so no Linux MLX stage is needed.
fn linuxModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, version: []const u8, mlx_c_version: []const u8, webp_include: []const u8) *std.Build.Module {
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption(bool, "mas", false);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "ds4_commit", "unknown");
    build_options.addOption([]const u8, "llama_tag", "unavailable (macOS-only engine)");
    build_options.addOption([]const u8, "git_sha", "");
    build_options.addOption(bool, "ios", false);
    build_options.addOption(bool, "macos_engines", false);
    build_options.addOption(bool, "embedded_engines", false);
    build_options.addOption(bool, "slow_tests", slowTests(b));
    // mlx-stream is macOS-only: the Linux graph registers no plugin and builds none of its files.
    build_options.addOption(bool, "plugin_mlx_stream", false);

    const opencode2_plugin = b.createModule(.{
        .root_source_file = b.path("lib/opencode2_plugin.zig"),
        .target = target,
        .optimize = optimize,
    });
    const agent_skills = b.createModule(.{
        .root_source_file = b.path("skills/agent_skills.zig"),
        .target = target,
        .optimize = optimize,
    });
    const mlx_steel_sources = b.createModule(.{
        .root_source_file = b.path("lib/mlx_steel_sources.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "mlx_steel_sources", .module = mlx_steel_sources },
            .{ .name = "opencode2_plugin", .module = opencode2_plugin },
            .{ .name = "agent_skills", .module = agent_skills },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize, "") },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, "") },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{webp_include}) }, .{ .cwd_relative = webp_include }, target, optimize, "") },
        },
    });
    addShared(b, target, optimize).importInto(mod);

    // Jinja2's headers (its ELF static lib, built by scripts/build-mlx-linux.sh, is a link input: addLinuxServe).
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image (JPEG/PNG decode) + stb_image_write (PNG encode), xatlas
    // (UV unwrap for Hunyuan3D texture paint) — portable C/C++.
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = stb_write_flags });
    mod.addIncludePath(b.path("lib"));
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));
    mod.addCSourceFile(.{ .file = b.path("lib/fqms/fqms_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/fqms"));

    // ANE offload C ABI → unavailable stubs on Linux (src/ane_stub.c); ane.zig
    // compiles unchanged and gates itself off via available() == false.
    mod.addCSourceFile(.{ .file = b.path("src/ane_stub.c"), .flags = &.{"-O2"} });
    return mod;
}

/// Linux counterpart of verifyMlxStage: fail loudly when scripts/
/// build-mlx-linux.sh has not staged the Linux mlx/mlx-c pair.
fn verifyMlxStageLinux(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.so", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlx.so", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged for Linux. Build it with:\n" ++
                "  git submodule update --init lib/mlxc-src && MLX_SOURCE=<staged Linux mlx tree> ./scripts/build-mlx-linux.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
    // The server embeds files from the opencode2 submodule
    // (lib/opencode2_plugin.zig @embedFile). A missing checkout surfaces as a
    // cryptic FileNotFound mid-compile, so check it at configure time.
    buildRootHandle(b).access(b.graph.io, "lib/opencode2-mlx-serve/LICENSE", .{}) catch {
        std.debug.print(
            "\n[mlx-serve] lib/opencode2-mlx-serve is not checked out. Run:\n" ++
                "  git submodule update --init lib/opencode2-mlx-serve\n\n",
            .{},
        );
        std.process.exit(1);
    };
}

/// `zig build vz-agent` → `zig-out/guest/vz-agent` (static aarch64 Linux ELF),
/// plus the host-side unit tests wired into `zig build test`.
fn addVzAgent(
    b: *std.Build,
    host_target: std.Build.ResolvedTarget,
    host_optimize: std.builtin.OptimizeMode,
    test_step: *std.Build.Step,
) void {
    // Guest binary. musl + static so it runs on ANY base image — the bundled
    // Debian rootfs for the App Store build, or a user-chosen alpine/slim image.
    const guest_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .musl,
    });
    const guest = b.addExecutable(.{
        .name = "vz-agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = guest_target,
            // Size, not speed: it shuttles bytes between a socket and a pipe.
            .optimize = .ReleaseSmall,
            .link_libc = true,
        }),
    });
    const install = b.addInstallArtifact(guest, .{
        .dest_dir = .{ .override = .{ .custom = "guest" } },
    });
    const step = b.step("vz-agent", "Build the Agent Sandbox guest binary (static aarch64-linux)");
    step.dependOn(&install.step);

    // Host-side tests of the same source.
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // A macOS-native build of the SAME source, which listens on a unix socket
    // instead of vsock. `GuestExecInteropTests` (Swift) drives it, so the host
    // frame driver and the guest agent are proven against each other without a
    // VM — the golden-byte tests alone can't catch a streaming bug.
    const host_agent = b.addExecutable(.{
        .name = "vz-agent-host",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    const host_step = b.step("vz-agent-host", "Build vz-agent natively (unix-socket mode, for interop tests)");
    host_step.dependOn(&b.addInstallArtifact(host_agent, .{}).step);
}

const IosSlice = struct { step: []const u8, abi: std.Target.Abi, sdk: []const u8 };

fn addIosLib(b: *std.Build, version: []const u8, ios_include: []const u8, slice: IosSlice) void {
    // Min 18.0 to match the MLX metallib (Metal 3.2). abi=.none → device,
    // abi=.simulator → iOS Simulator slice.
    const ios_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .ios,
        .os_version_min = .{ .semver = .{ .major = 18, .minor = 0, .patch = 0 } },
        .abi = slice.abi,
    });

    const ios_options = b.addOptions();
    ios_options.addOption([]const u8, "version", version);
    ios_options.addOption(bool, "ios", true);
    // Mirror the build options the shared engine sources read (server.zig,
    // scheduler.zig). iOS is sandboxed (no curl/model-pull subprocess), so
    // mas=true; ds4 + llama.cpp are stubbed here, so their version pins are
    // unreported. Without these the iOS lib fails to compile ("options has no
    // member named 'mas'/...").
    ios_options.addOption(bool, "mas", true);
    ios_options.addOption(bool, "macos_engines", false);
    ios_options.addOption(bool, "embedded_engines", false);
    ios_options.addOption([]const u8, "mlx_c_version", "unknown");
    ios_options.addOption([]const u8, "ds4_commit", "unknown");
    ios_options.addOption([]const u8, "llama_tag", "unknown");
    ios_options.addOption([]const u8, "git_sha", "");
    // mlx-stream is macOS-only: the iOS graphs register no plugin and build none of its files.
    ios_options.addOption(bool, "plugin_mlx_stream", false);

    const mod = b.createModule(.{
        .root_source_file = b.path("src/ios_lib.zig"),
        .target = ios_target,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = ios_options.createModule() },
        },
    });
    addShared(b, ios_target, .ReleaseFast).importInto(mod);

    // Apple cross-compiles don't auto-resolve the SDK's libc/frameworks from
    // --sysroot alone, so wire them explicitly (resolved per slice via xcrun).
    //
    // NO iOS SDK → register NOTHING (the `ios-lib` steps just don't exist in
    // this environment) instead of failing the whole configure: app/build.sh
    // pins DEVELOPER_DIR to the CommandLineTools for the macOS link, and CLT
    // ships no iOS SDKs — a @panic here aborted every macOS app build even
    // though nobody asked for an iOS step.
    var code: u8 = undefined;
    const sdk_path = b.runAllowFail(
        &.{ "xcrun", "--sdk", slice.sdk, "--show-sdk-path" },
        &code,
        .ignore, // silent when absent — CLT environments hit this on purpose
    ) catch return;
    const ios_sdk = std.mem.trim(u8, sdk_path, " \n\r\t");
    if (ios_sdk.len == 0) return;
    mod.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{ios_sdk}) });
    mod.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{ios_sdk}) });

    // Headers for the @import("jinja_c")/@import("stb") sites (jinja_wrapper.h,
    // stb_image.h, webp/decode.h). The matching static archives are linked by
    // Xcode at final app-link time.
    mod.addIncludePath(b.path("lib/jinja_cpp"));
    mod.addIncludePath(b.path("lib"));
    mod.addIncludePath(.{ .cwd_relative = ios_include });
    mod.addImport("jinja_c", addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), ios_target, .ReleaseFast, ios_sdk));
    mod.addImport("stb", addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), ios_target, .ReleaseFast, ios_sdk));
    mod.addImport("webp", addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{ios_include}) }, .{ .cwd_relative = ios_include }, ios_target, .ReleaseFast, ios_sdk));
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = stb_write_flags });
    // xatlas UV unwrapping (C++), used by the Hunyuan3D texture paint stage via
    // src/uvwrap.zig extern decls — compiled into the lib like the macOS exe.
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));
    mod.addCSourceFile(.{ .file = b.path("lib/fqms/fqms_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/fqms"));

    const lib = b.addLibrary(.{
        .name = "mlxserve",
        .root_module = mod,
        .linkage = .static,
    });
    lib.bundle_compiler_rt = true;

    const install = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .override = .{ .custom = b.fmt("ios/{s}/lib", .{slice.sdk}) } },
    });
    const step = b.step(slice.step, b.fmt("Build the iOS engine static lib ({s})", .{slice.sdk}));
    step.dependOn(&install.step);
    // The same graph's semantic check: nothing emitted.
    const check_lib = b.addLibrary(.{ .name = "mlxserve-check", .root_module = mod, .linkage = .static });
    const check = b.step(b.fmt("{s}-check", .{slice.step}), b.fmt("Check that the iOS engine graph ({s}) compiles, without codegen", .{slice.sdk}));
    check.dependOn(&check_lib.step);
}

/// Translates a single C header into an importable module (`@import("name")`
/// at the call site) via `addTranslateC`, replacing an inline `@cImport` —
/// removed as a language builtin in 0.17.0-dev.
fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ios_sdk: []const u8,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    // iOS cross-compile: addTranslateC (0.17's @cImport replacement) does NOT
    // inherit the parent module's SDK include search the way inline @cImport
    // used to, so a header that pulls in <stdio.h>/<inttypes.h> can't find the
    // Apple libc and translation fails. Wire the SDK's system include here.
    // Host builds pass "" (their toolchain resolves the system headers).
    if (ios_sdk.len > 0)
        translate.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{ios_sdk}) });
    return translate.createModule();
}

fn addDs4Sources(b: *std.Build, module: *std.Build.Module) void {
    // Match ds4's Makefile flags (lib/ds4/Makefile lines 10–11). We drop
    // `-mcpu=native` so the produced binary stays portable across Apple
    // Silicon generations — ds4 itself ships portable IR for its Metal
    // kernels, and the C host code is not perf-critical compared to the GPU
    // path. `-Wno-unused-parameter` + `-Wno-unused-variable` keep upstream's
    // warnings from breaking our build without patching the submodule.
    const c_flags = &[_][]const u8{
        "-O3",
        "-ffast-math",
        "-std=c99",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4.c"), .flags = c_flags });
    // ds4.c #includes ds4_distributed.h; the engine/session path links its impl.
    // ds4_gpu.h is implemented in ds4_metal.m; ds4_kvstore/web/help/agent.c and
    // ds4_gpu_args.c are CLI/server-only and not part of the library path
    // mlx-serve embeds (upstream Makefile CORE_OBJS is the authority).
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_distributed.c"), .flags = c_flags });
    // SSD weight-streaming (issue #39): ds4_ssd.c is a standalone TU (#includes
    // only ds4_ssd.h) implementing the streaming expert cache the engine_options
    // ssd_streaming_* fields drive. Added upstream after the previous pin.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_ssd.c"), .flags = c_flags });
    // Two-machine tensor parallelism + multi-GPU layer placement (pin 9139e2a):
    // ds4.c references ds4_tp_* and ds4_compute_layer_placement/ds4_layer_pack_print
    // unconditionally, so both TUs must link even though we never enable TP.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_tp.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_layer_pack.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_image.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_engram.c"), .flags = c_flags });
    // Our own shim: exports sizeof/offsetof of the real C structs so the
    // ds4_ffi.zig layout test catches mirror drift (mid-struct-insert class).
    module.addCSourceFile(.{ .file = b.path("src/ds4_layout_check.c"), .flags = c_flags });

    const objc_flags = &[_][]const u8{
        "-O3",
        "-ffast-math",
        "-fobjc-arc",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_metal.m"), .flags = objc_flags });
}

/// lib/mlx-stream: the DeepSeek-V4.1 arch as a plugin (docs/plugins.md), its own repo, a pinned submodule. It reaches
/// the host only through `sdk`. `-Dmlx-stream-dir=/abs/path` builds against a checkout instead of the submodule.
/// Returns the plugin's module, imported into `host` as `mlx_stream`.
fn addMlxStreamModule(b: *std.Build, host: *std.Build.Module, sdk: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, o: MlxStreamModule) *std.Build.Module {
    const root = mlxStreamRoot(b);
    const m = b.createModule(.{
        .root_source_file = if (root) |r| r.path(b, o.root) else missingMlxStream(b),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "sdk", .module = sdk }},
    });
    if (root) |r| addMlxStreamCSources(b, m, r, o.inject);
    host.addImport("mlx_stream", m);
    return m;
}

const MlxStreamModule = struct {
    /// The module's root file in the plugin checkout: src/root.zig, or a test build's root.
    root: []const u8 = "src/root.zig",
    /// The read pool's scripted-fault hooks (test builds only).
    inject: bool = false,
};

var mlx_stream_root: ?std.Build.LazyPath = null;
var mlx_stream_resolved = false;
/// The plugin checkout, or null when it is missing (no submodule checkout, no `-Dmlx-stream-dir`): then every graph
/// that registers the plugin fails to compile with one message that says what to do (`missingMlxStream`), and the
/// graphs that do not (`-Dmlx-stream=false`, vz-agent, --help) are unaffected.
fn mlxStreamRoot(b: *std.Build) ?std.Build.LazyPath {
    if (mlx_stream_resolved) return mlx_stream_root;
    mlx_stream_resolved = true;
    const dir = b.option([]const u8, "mlx-stream-dir", "mlx-stream checkout to build against (default: lib/mlx-stream)");
    const base = if (dir != null) std.Io.Dir.cwd() else buildRootHandle(b);
    base.access(b.graph.io, b.pathJoin(&.{ dir orelse "lib/mlx-stream", "src/root.zig" }), .{}) catch return null;
    mlx_stream_root = if (dir) |d| .{ .cwd_relative = d } else b.path("lib/mlx-stream");
    return mlx_stream_root;
}

fn missingMlxStream(b: *std.Build) std.Build.LazyPath {
    return b.addWriteFiles().add("mlx_stream_missing.zig",
        \\comptime {
        \\    @compileError("mlx-stream: no plugin checkout at lib/mlx-stream. Check out the submodule (git submodule update --init lib/mlx-stream), build against a checkout (-Dmlx-stream-dir=/abs/path), or build without the plugin (-Dmlx-stream=false)");
        \\}
        \\pub const plugin = @import("sdk").Plugin{ .name = "mlx-stream", .api = @import("sdk").api, .mlx = @import("sdk").mlx_pin, .provides = .{} };
        \\pub const testing = struct {};
        \\
    );
}

/// The plugin's C sources (csrc/): the lookahead read pool (pthreads, pread + memcpy into slot rows, never MLX) and
/// its MTLSharedEvent signal (non-ARC objc), and the MLX event / alloc shims, which include the staged MLX's private
/// headers, so they compile against this host's MLX and link libmlx. With `sdk.plugin_profile` on, the decode
/// profile's command-buffer timeline sources too.
fn addMlxStreamCSources(b: *std.Build, module: *std.Build.Module, root: std.Build.LazyPath, inject: bool) void {
    const profile = pluginProfile(b);
    const flags: []const []const u8 = if (inject)
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-DQ3LD_INJECT" }
    else if (profile)
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-DQ3LD_EVSIG" }
    else
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" };
    const objc: []const []const u8 = &.{ "-O2", "-Wall", "-Wextra", "-Werror", "-fno-objc-arc" };
    const cxx: []const []const u8 = &.{ "-std=c++20", "-O2", "-D_METAL_", "-DACCELERATE_NEW_LAPACK", "-fno-sanitize=all", "-Wall", "-Wno-unused-parameter", "-Wno-deprecated-declarations" };
    module.addCSourceFile(.{ .file = root.path(b, "csrc/q3_lookahead4_exl3.c"), .flags = flags });
    module.addCSourceFile(.{ .file = root.path(b, "csrc/q3_event_shim.mm"), .flags = objc });
    module.addIncludePath(root.path(b, "csrc"));
    module.addCSourceFile(.{ .file = root.path(b, "csrc/mlx_event_shim.cpp"), .flags = cxx });
    module.addCSourceFile(.{ .file = root.path(b, "csrc/mlx_alloc_shim.cpp"), .flags = cxx });
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addIncludePath(b.path("lib/mlx/include/metal_cpp"));
    module.addIncludePath(b.path("lib/mlxc-src"));
    module.linkSystemLibrary("mlx", .{ .use_pkg_config = .no });
    if (profile) {
        module.addCSourceFile(.{ .file = root.path(b, "csrc/dsv41_cb_timeline.mm"), .flags = objc });
        module.addCSourceFile(.{ .file = root.path(b, "csrc/dsv41_newbuffer_count.mm"), .flags = objc });
        module.addCSourceFile(.{ .file = root.path(b, "csrc/dsv41_tl_mlx.cpp"), .flags = cxx });
    }
}

/// A plugin test build (`root` in the plugin checkout): the plugin module is the test root, and the host files its
/// harnesses reach come in as `mlx_serve_host` (src/sdk_test_host.zig), whose module imports the plugin back as
/// `mlx_stream` (one plugin module per build).
fn addMlxStreamTests(b: *std.Build, d: TestDeps, name: []const u8, root: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, filter: ?[]const u8) *std.Build.Step.Compile {
    const bridge = d.moduleWith(b, b.path("src/sdk_test_host.zig"), target, optimize);
    const pkg = addMlxStreamModule(b, bridge, d.shared.sdk, target, optimize, .{ .root = root, .inject = true });
    pkg.addImport("mlx_serve_host", bridge);
    return b.addTest(.{ .name = name, .root_module = pkg, .filters = if (filter) |f| &.{f} else &.{} });
}

/// `-Dplugin-profile`: compile the registered plugins' profile probes in (`sdk.plugin_profile`); off in served builds.
var plugin_profile_opt: ?bool = null;
fn pluginProfile(b: *std.Build) bool {
    if (plugin_profile_opt) |v| return v;
    plugin_profile_opt = b.option(bool, "plugin-profile", "Compile the registered plugins' profile probes in (profile builds only)") orelse false;
    return plugin_profile_opt.?;
}

/// ANE prefill offload sources (lib/ane): the private-framework bridge and
/// the per-layer MLP program builder, both ARC objc. Runtime-probed —
/// compiling them in costs nothing on machines without the framework.
fn addAneSources(b: *std.Build, module: *std.Build.Module) void {
    const objc_flags = &[_][]const u8{
        "-O3",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_bridge.m"), .flags = objc_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_mlp.m"), .flags = objc_flags });
    module.addIncludePath(b.path("lib/ane"));
}

/// What the macOS unit-test graphs link and import: the unit tests and the conformance suite share it.
/// src/plugins_refusals.zig's cases and the compile error line each must end with. A negotiation refusal ends with the
/// host's MLX pin, so its line matches up to `/?/` from the registry's refusal site (plugins.zig:55).
const registry_refusals = [_]struct { case: []const u8, err: []const u8 }{
    .{ .case = "api_major", .err = "src/plugins.zig:55:50: error: plugin bad-api: ApiMajorMismatch (built against SDK 2.0 on MLX /?/)" },
    .{ .case = "mlx_pin", .err = "src/plugins.zig:55:50: error: plugin bad-mlx: MlxPinMismatch (built against SDK 1.0 on MLX v0.0.1; this host is SDK 1.0 on MLX v/?/)" },
    .{ .case = "mlx_pin_macos_only", .err = "src/plugins.zig:55:50: error: plugin mac-pin: MlxPinMismatch (built against SDK 1.0 on MLX v0.0.1;/?/)" },
    .{ .case = "duplicate", .err = "plugin twin: registered twice" },
    .{ .case = "source_no_claims", .err = "NoClaims: no claims" },
    .{ .case = "engine_wrong_claims", .err = "WrongClaims.claims: parameter *const sdk.peek.GroupPeek where the SDK has *const sdk.peek.ConfigPeek" },
    .{ .case = "arch_batches_owned_state", .err = ": batches_decode with owns_decode_state" },
    .{ .case = "arch_claim_unpaired", .err = ": claimProcess and releaseProcess come as a pair" },
    .{ .case = "name_not_json_safe", .err = "plugin name not JSON-safe: quo\"te" },
    .{ .case = "source_claims_not_fn", .err = "ClaimsNotFn.claims is not a function" },
    .{ .case = "engine_claims_param_count", .err = "ClaimsTwoParams.claims: takes a different parameter count than the SDK's" },
    .{ .case = "source_claims_returns", .err = "ClaimsReturnsBool.claims: returns bool where the SDK has ?sdk.peek.Priority" },
    .{ .case = "source_no_name", .err = "Nameless: no name" },
};

const TestDeps = struct {
    options: *std.Build.Step.Options,
    shared: Shared,
    ds4_metal_sources: *std.Build.Module,
    mlx_steel_sources: *std.Build.Module,
    opencode2_plugin: *std.Build.Module,
    agent_skills: *std.Build.Module,
    frameworks: ?[]const u8,
    /// The registry registers mlx-stream (`-Dmlx-stream`).
    with_mlx_stream: bool,

    fn module(d: TestDeps, b: *std.Build, root: std.Build.LazyPath, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
        const m = d.moduleWith(b, root, target, optimize);
        if (d.with_mlx_stream) _ = addMlxStreamModule(b, m, d.shared.sdk, target, optimize, .{ .inject = true });
        return m;
    }

    /// A test graph's module without the plugin: the caller imports `mlx_stream` (a plugin test build's own module).
    fn moduleWith(d: TestDeps, b: *std.Build, root: std.Build.LazyPath, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
        const m = b.createModule(.{
            .root_source_file = root,
            .target = target,
            .optimize = optimize,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "build_options", .module = d.options.createModule() },
                .{ .name = "ds4_metal_sources", .module = d.ds4_metal_sources },
                .{ .name = "mlx_steel_sources", .module = d.mlx_steel_sources },
                .{ .name = "opencode2_plugin", .module = d.opencode2_plugin },
                .{ .name = "agent_skills", .module = d.agent_skills },
                .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize, "") },
                .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, "") },
                .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize, "") },
            },
        });
        d.shared.importInto(m);
        m.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
        m.addIncludePath(b.path("lib/jinja_cpp"));
        m.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
        m.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = stb_write_flags });
        m.addIncludePath(b.path("lib"));
        m.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
        m.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
        m.addIncludePath(b.path("lib/xatlas"));
        m.addCSourceFile(.{ .file = b.path("lib/fqms/fqms_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
        m.addIncludePath(b.path("lib/fqms"));
        addDs4Sources(b, m);
        m.addIncludePath(b.path("lib/ds4"));
        addAneSources(b, m);
        addLlamaLib(b, m);
        m.linkSystemLibrary("c++", .{});
        addMlxLib(b, m);
        m.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
        m.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
        m.linkSystemLibrary("webp", .{});
        if (d.frameworks) |fw_path| m.addFrameworkPath(.{ .cwd_relative = fw_path });
        m.linkFramework("IOKit", .{});
        m.linkFramework("CoreFoundation", .{});
        m.linkFramework("Foundation", .{});
        m.linkFramework("Metal", .{});
        m.linkFramework("IOSurface", .{});
        return m;
    }
};

/// The build options a macOS graph's sources read. The server and the test graph differ only in
/// `embedded_engines` (the slim host's switch).
const CoreOptions = struct {
    version: []const u8,
    mas: bool,
    mlx_c_version: []const u8,
    ds4_commit: []const u8,
    llama_tag: []const u8,
    git_sha: []const u8,
    slow_tests: bool,

    fn add(c: CoreOptions, b: *std.Build, embedded_engines: bool, mlx_stream: bool) *std.Build.Step.Options {
        const o = b.addOptions();
        o.addOption([]const u8, "version", c.version);
        o.addOption(bool, "mas", c.mas);
        o.addOption([]const u8, "mlx_c_version", c.mlx_c_version);
        o.addOption([]const u8, "ds4_commit", c.ds4_commit);
        o.addOption([]const u8, "llama_tag", c.llama_tag);
        o.addOption([]const u8, "git_sha", c.git_sha);
        // false for the macOS exe/tests; the iOS static-lib step (`zig build ios-lib`)
        // builds its own options with ios=true so the engine swaps the macOS-only
        // ds4 + llama.cpp engines for no-op stubs (iOS serves MLX safetensors only).
        o.addOption(bool, "ios", false);
        // True on every macOS graph: the macOS-only sources (a macOS-only plugin's, the native
        // module archs) are compiled in. iOS static lib and Linux exe =
        // no: they get compile-time stubs (src/*_stub.zig) and src/ane_stub.c on Linux. The stub
        // selection reads this option, NOT `ios` — `ios` keeps its own meaning (low-mem policy,
        // sandboxing assumptions).
        o.addOption(bool, "macos_engines", true);
        // The embedded engines (ds4 Metal, libllama) are linked: the macOS exe and tests, not the
        // slim host, iOS or Linux, which select src/arch/*_stub.zig and src/ds4_ffi_stub.zig.
        o.addOption(bool, "embedded_engines", embedded_engines);
        // The registry registers mlx-stream (src/plugins.zig); the unit-test graph follows -Dmlx-stream too.
        o.addOption(bool, "plugin_mlx_stream", mlx_stream);
        // The corpus replay and benchmark tests run tens of seconds in Debug (#639), so
        // `zig build test` skips them and `zig build test -Dslow-tests` runs them.
        o.addOption(bool, "slow_tests", c.slow_tests);
        return o;
    }
};

/// The modules every graph shares by name (docs/plugins.md, PR 1): the MLX FFI, logging, the I/O helpers and the
/// plugin SDK over them. One instance per graph, so the host and its plugins see one set of types.
const Shared = struct {
    mlx: *std.Build.Module,
    log: *std.Build.Module,
    io_util: *std.Build.Module,
    /// The MTP acceptance modes (src/mtp_acceptance.zig), shared so the SDK can export them.
    mtp_acceptance: *std.Build.Module,
    sdk: *std.Build.Module,
    /// lib/mlx-serve-gguf and lib/sushi's EXL3 module, which reach mlx, log and io_util through `mlx_host` (the SDK).
    gguf: *std.Build.Module,
    exl3: *std.Build.Module,

    fn importInto(s: Shared, m: *std.Build.Module) void {
        m.addImport("mlx", s.mlx);
        m.addImport("log", s.log);
        m.addImport("io_util", s.io_util);
        m.addImport("mtp_acceptance", s.mtp_acceptance);
        m.addImport("sdk", s.sdk);
        m.addImport("mlx_serve_gguf", s.gguf);
        m.addImport("sushi_exl3", s.exl3);
    }
};

fn addShared(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Shared {
    const log = b.createModule(.{ .root_source_file = b.path("src/log.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const io_util = b.createModule(.{ .root_source_file = b.path("src/io_util.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const mtp_acceptance = b.createModule(.{ .root_source_file = b.path("src/mtp_acceptance.zig"), .target = target, .optimize = optimize });
    const mlx = b.createModule(.{
        .root_source_file = b.path("src/mlx.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "log", .module = log }},
    });
    const sdk_build = b.addOptions();
    sdk_build.addOption(bool, "plugin_profile", pluginProfile(b));
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/sdk.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx", .module = mlx }, .{ .name = "log", .module = log }, .{ .name = "io_util", .module = io_util }, .{ .name = "mtp_acceptance", .module = mtp_acceptance }, .{ .name = "sdk_build", .module = sdk_build.createModule() } },
    });
    return .{ .mlx = mlx, .log = log, .io_util = io_util, .mtp_acceptance = mtp_acceptance, .sdk = sdk, .gguf = engineModule(b, ggufRoot(b), sdk, target, optimize), .exl3 = engineModule(b, exl3Root(b), sdk, target, optimize) };
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// The llama.cpp tag staged by scripts/fetch-llama.sh (it writes LLAMA_TAG to
/// `lib/llama/.version`). Read at configure time so a plain `zig build` reports
/// the real tag without app/build.sh having to pass `--llama-tag`. Returns null
/// (→ "unknown") when llama hasn't been fetched yet.
/// `b.option` may be declared once; the macOS graphs and the Linux check share this answer.
var slow_tests_opt: ?bool = null;
fn slowTests(b: *std.Build) bool {
    if (slow_tests_opt) |v| return v;
    slow_tests_opt = b.option(bool, "slow-tests", "Also run the slow corpus-replay and benchmark tests") orelse false;
    return slow_tests_opt.?;
}

fn readLlamaTag(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/llama/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    return if (trimmed.len == 0) null else b.dupe(trimmed);
}

/// An engine module from its own repo (lib/mlx-serve-gguf, lib/sushi): it reaches mlx, log and io_util through
/// `mlx_host`, which is the SDK (it exposes all three), so every graph shares one instance of each.
fn engineModule(b: *std.Build, root: std.Build.LazyPath, host: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = root,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
}

/// `-Dgguf-dir=/abs/path` builds against a mlx-serve-gguf checkout instead of the submodule.
/// `b.option` may be declared once; the graphs share this answer.
var gguf_root: ?std.Build.LazyPath = null;
fn ggufRoot(b: *std.Build) std.Build.LazyPath {
    if (gguf_root) |r| return r;
    const dir = b.option([]const u8, "gguf-dir", "mlx-serve-gguf checkout to build against (default: lib/mlx-serve-gguf)");
    gguf_root = if (dir) |d| .{ .cwd_relative = b.pathJoin(&.{ d, "src/root.zig" }) } else b.path("lib/mlx-serve-gguf/src/root.zig");
    return gguf_root.?;
}

/// `-Dsushi-dir=/abs/path` builds against a sushi checkout instead of the submodule.
var exl3_root: ?std.Build.LazyPath = null;
fn exl3Root(b: *std.Build) std.Build.LazyPath {
    if (exl3_root) |r| return r;
    const dir = b.option([]const u8, "sushi-dir", "sushi checkout to build against (default: lib/sushi)");
    exl3_root = if (dir) |d| .{ .cwd_relative = b.pathJoin(&.{ d, "src/exl3/root.zig" }) } else b.path("lib/sushi/src/exl3/root.zig");
    return exl3_root.?;
}

fn addLlamaLib(b: *std.Build, module: *std.Build.Module) void {
    // Link the prebuilt libllama staged by scripts/fetch-llama.sh. The dylib's
    // install-name is @rpath/libllama.dylib; we add an rpath to its build-tree
    // location so `zig build run` / unit tests resolve it in dev. The app bundle
    // and CLI tarball rewrite that reference to @executable_path/... and re-sign
    // with the Developer ID (see release.yml / app/build.sh).
    module.addIncludePath(b.path("lib/llama/include"));
    module.addLibraryPath(b.path("lib/llama/lib"));
    // use_pkg_config = .no: a Homebrew `llama.cpp` install ships a llama.pc that
    // would otherwise hijack this link (pulling in /opt/homebrew's version + its
    // separate libggml). We want exactly the pinned dylib staged in lib/llama/lib.
    module.linkSystemLibrary("llama", .{ .use_pkg_config = .no });
    // @loader_path resolves against the BINARY's own location at launch,
    // not the launching process's cwd. A bare relative string here (e.g.
    // "lib/llama/lib") gets baked verbatim into LC_RPATH when `zig build`
    // runs with cwd == build root (b.build_root.path is null in that case,
    // so b.path() can't make it absolute) and dyld then resolves that
    // relative string against argv[0]'s cwd, breaking any launch from
    // outside the repo root. An absolute path avoids that but bakes in a
    // machine-specific location, so the build isn't relocatable. This is
    // relative to the binary itself, so it stays correct from any launch
    // cwd and survives copying the whole zig-out + lib tree elsewhere.
    //
    // This module backs two different binaries at two different depths
    // under the build root, so one entry can't serve both: the installed
    // exe lands at zig-out/bin/mlx-serve (@loader_path = zig-out/bin/, 2
    // levels up to root), while `zig build test` runs straight out of
    // .zig-cache/o/<hash>/test (@loader_path = that dir, 3 levels up to
    // root). dyld tries every LC_RPATH entry in order and silently skips
    // ones that don't resolve, so listing both depths here is safe — each
    // binary finds its own and ignores the other.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/llama/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/llama/lib" });

    // Our clean C shim over llama.h (src/llama_ffi.zig mirrors lib/llama_shim/llama_shim.h).
    // C11 for pthread_once-based one-time backend init.
    module.addIncludePath(b.path("lib/llama_shim"));
    module.addCSourceFile(.{
        .file = b.path("lib/llama_shim/llama_shim.c"),
        .flags = &.{ "-O2", "-std=c11", "-Wno-unused-parameter" },
    });
}

/// Link the self-built mlx + mlx-c staged in lib/mlx by scripts/build-mlx.sh
/// (pinned submodules lib/mlx-src + lib/mlxc-src, deployment target 26.2 so
/// MLX's NAX kernels are compiled in — the Homebrew bottle ships without them
/// and hard-wires is_nax_available() false even on M5). Install names are
/// @rpath/...; the build-tree rpath resolves them in dev, release.yml /
/// app/build.sh rewrite to @executable_path and re-sign for bundles.
/// Guard test: tests/test_mlx_staged_nax.sh.
fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // use_pkg_config = .no: a leftover Homebrew mlx-c must never hijack this
    // link — we want exactly the staged NAX-enabled pair (same class as the
    // llama.pc hijack above).
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // See addLlamaLib above: @loader_path is relative to the binary itself,
    // so this stays correct regardless of the launching process's cwd and
    // stays relocatable across machines. Two entries for the same reason —
    // the installed exe and the `zig build test` binary sit at different
    // depths under the build root.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Mirrors verifyBrewDeps: fail loudly with the fix, never
/// let the linker produce a confusing -lmlxc error (or silently pick up a
/// leftover brew copy from /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// The pinned mlx-c revision from lib/mlx/.version (written by
/// scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>"), surfaced in
/// `mlx-serve --version`. Returns null (→ "unknown") when not staged yet.
/// `CFBundleShortVersionString` out of the checked-in app/Info.plist — the one
/// place the current CalVer is committed (app/build.sh stamps it on every real
/// build). Read at configure time so a plain `zig build` reports the same
/// version the last shipped build did, instead of a made-up literal.
fn readAppVersion(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "app/Info.plist",
        b.allocator,
        .limited(64 * 1024),
    ) catch return null;
    const key = "<key>CFBundleShortVersionString</key>";
    const at = std.mem.indexOf(u8, bytes, key) orelse return null;
    const open = std.mem.indexOfPos(u8, bytes, at + key.len, "<string>") orelse return null;
    const start = open + "<string>".len;
    const end = std.mem.indexOfPos(u8, bytes, start, "</string>") orelse return null;
    const v = std.mem.trim(u8, bytes[start..end], " \t\r\n");
    return if (v.len > 0) b.dupe(v) else null;
}

fn readMlxcPin(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "mlxc=")) return b.dupe(tok["mlxc=".len..]);
    }
    return null;
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    // mlx + mlx-c are NOT brew deps anymore: they are pinned submodules built
    // by scripts/build-mlx.sh (see addMlxLib) so the NAX kernels ship enabled.
    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[mlx-serve] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[mlx-serve] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[mlx-serve] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}
