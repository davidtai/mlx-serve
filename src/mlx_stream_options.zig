//! mlx-stream's build options (G7): the DeepSeek-V4.1 profile timers, compiled out of every served build.
const BuildOption = @import("sdk/build_option.zig").BuildOption;

pub const options = [_]BuildOption{
    // The DSpark cycle's host split (src/dsv41_decode_timers.zig).
    .{ .name = "dsv41-decode-timers", .field = "dsv41_decode_timers", .description = "Compile the DSV4.1 DSpark cycle's phase timers in (profile builds only)" },
    // The prompt pass routed-call timers (src/dsv41_prefill_timers.zig).
    .{ .name = "dsv41-prefill-timers", .field = "dsv41_prefill_timers", .description = "Compile the DSV4.1 prompt pass routed-call timers in (profile builds only)" },
};
