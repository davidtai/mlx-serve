//! DeepSeek-V4.1's own config on the host side: its sidecar paths, its prompt-pass bill, the load's facts, and the
//! model settings it takes with the tier defaults they fall back to. The fields keep the names the host's
//! ModelConfig gave them, so every reader in the package reads them unchanged.

const std = @import("std");
const model = @import("model.zig");
const v41 = @import("deepseek_v41.zig");

/// The arch's construction-time numerics (`numeric_tier`).
pub const NumericTier = @import("model_settings.zig").NumericTier;

pub const Config = struct {
    /// The model directory (config.json, the resident shards, the expert bank) and the exported Engram token map
    /// beside it; owned by the host's config.
    expert_bank_dir: ?[]const u8 = null,
    engram_token_map_path: ?[]const u8 = null,
    /// The prompt pass's bill for the prefill admission.
    dsv41_prefill: ?v41.PrefillBill = null,
    num_hidden_layers: u32 = 0,
    /// The load's facts: the memory in use before the load (`--memory-baseline-gb`, else the preflight's reading),
    /// the decode and prompt slot rows per layer (`--expert-rows`, a harness's; null = the admission's fill), the
    /// residents past the page cache (null = on).
    memory_baseline_bytes: ?u64 = null,
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    nocache_weights: ?bool = null,
    /// The routed waves wait on the reads' events instead of the host (null = the tier's default).
    expert_event_gates: ?bool = null,
    /// The read pool threads' scheduling (null = off).
    expert_reader_sched: ?@import("model_settings.zig").ReaderSched = null,
    /// The numerics, chosen at construction (null = served).
    numeric_tier: ?NumericTier = null,
    /// The prompt pass layer by layer (null = the tier's default).
    layer_major_prefill: ?bool = null,
    /// The wide prefill read schedule (null = the tier's defaults below).
    expert_wide_feed: ?bool = null,
    expert_wide_seed: ?bool = null,
    expert_wide_hot_first: ?bool = null,
    expert_wide_depth: ?u8 = null,
    /// Wide-call experts of at most this many rows on the decode GEMV (null = none).
    expert_wide_cold_rows: ?u8 = null,
    /// The wide call's base-bank rows as one deferred call.
    expert_wide_defer_base: ?bool = null,
    /// P1: each layer's predicted seed read ahead during its attention.
    expert_wide_read_ahead: ?bool = null,
    /// P1b: the seed's deferred base call run as soon as the seed has landed.
    expert_wide_base_at_seed: ?bool = null,
    /// The decode read-ahead's speculative records per layer call (1..4; null = 2).
    expert_lookahead_budget: ?u8 = null,
    /// P1c: the seed's ranks grouped apart from the stream's, the base call after the last seed group.
    expert_wide_seed_aligned: ?bool = null,
    /// P1d: the base rows resident at the barrier drain first, in their own call (null = off).
    expert_wide_resident_first: ?bool = null,
    /// The input embedding read from its host rows from construction (null = on).
    embedding_host_rows: ?bool = null,

    /// The host's config, field for field.
    pub fn fromHost(c: *const model.ModelConfig) Config {
        var out: Config = .{};
        inline for (@typeInfo(Config).@"struct".field_names) |f| @field(out, f) = @field(c, f);
        return out;
    }

    /// A model directory's config as the host parses it, as this arch's config (its strings in `a`).
    pub fn load(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Config {
        return fromHost(&try model.parseConfig(io, a, model_dir));
    }

    /// The prefill routes as the module builds them: a setting when given, else the tier's default (the served
    /// tier: K16 layer-major, two wide groups in flight, the wide feed; the stock tier, whose prompt forwards are
    /// decode-width: none).
    pub fn dsv41LayerMajor(self: *const Config) bool {
        return self.layer_major_prefill orelse self.dsv41ServedTier();
    }

    /// The decode read-ahead's speculative records per layer call: 2 unless set.
    pub fn dsv41LookaheadBudget(self: *const Config) u8 {
        return self.expert_lookahead_budget orelse 2;
    }

    /// The served tier reads 3 groups ahead (P1's v1b: the SSD kept busy through the routed stage's drains).
    pub fn dsv41WideDepth(self: *const Config) u8 {
        return self.expert_wide_depth orelse if (self.dsv41ServedTier()) 5 else 1;
    }

    pub fn dsv41WideFeed(self: *const Config) bool {
        return self.expert_wide_feed orelse self.dsv41ServedTier();
    }

    /// The feed's halves: each its own setting, else the feed's value (the feed = seed + hot-first).
    pub fn dsv41WideSeed(self: *const Config) bool {
        return self.expert_wide_seed orelse self.dsv41WideFeed();
    }

    pub fn dsv41WideHotFirst(self: *const Config) bool {
        return self.expert_wide_hot_first orelse self.dsv41WideFeed();
    }

    /// The deferred base-bank call: the setting, else on for the served tier without cold rows.
    pub fn dsv41WideDeferBase(self: *const Config) bool {
        return self.expert_wide_defer_base orelse (self.dsv41ServedTier() and (self.expert_wide_cold_rows orelse 0) == 0);
    }

    /// P1's read-ahead: the setting, else on wherever the prompt pass is layer-major with the wide seed.
    pub fn dsv41WideReadAhead(self: *const Config) bool {
        return self.expert_wide_read_ahead orelse (self.dsv41LayerMajor() and self.dsv41WideSeed());
    }

    /// P1b's base call at the seed: the setting, else on wherever the wide seed and the deferred base call both are.
    pub fn dsv41WideBaseAtSeed(self: *const Config) bool {
        return self.expert_wide_base_at_seed orelse (self.dsv41WideSeed() and self.dsv41WideDeferBase());
    }

    /// P1c's seed-aligned groups: the setting, else on wherever the base call at the seed and the hottest-first order are.
    pub fn dsv41WideSeedAligned(self: *const Config) bool {
        return self.expert_wide_seed_aligned orelse (self.dsv41WideBaseAtSeed() and self.dsv41WideHotFirst());
    }

    /// P1d's resident-first base call: the setting, else off.
    pub fn dsv41WideResidentFirst(self: *const Config) bool {
        return self.expert_wide_resident_first orelse false;
    }

    fn dsv41ServedTier(self: *const Config) bool {
        return (self.numeric_tier orelse .served) == .served;
    }
};
