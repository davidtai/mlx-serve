//! G4, phase-specific admission: each kind bills its own named terms per phase, the host composes them, fills two row
//! counts and admits both phases before any allocation, and checks the constructed module once against the bill.
//! Pure host: no MLX, no device query, no global, so bills and fills run in the CPU lane and the preflight never
//! touches the device.

const std = @import("std");
const peek = @import("peek.zig");

pub const MemoryBill = struct {
    pub const Phase = enum { prompt, decode };
    /// One named term, decimal bytes per phase (0 where absent). Rows-free by construction: the persistent slot rows
    /// are `per_row` times the fill's rows. Every term is billed, a measured one at its declared bound.
    pub const Term = struct {
        name: []const u8,
        bytes: [2]u64,
        /// Held by the constructed module before any request (the construction check's terms); a phase's
        /// transients (waves, KV, the MLX cache, posted gathers) are not.
        at_construction: bool,
        /// Not derivable from headers: `bytes` is a declared bound, measured once at construction (`checkMeasured`).
        measured: bool = false,
    };

    terms: []const Term = &.{},
    /// One slot row on every routed layer (an expert_source's); 0 for a model without one.
    per_row: u64 = 0,

    /// The phase's terms, the baseline and the slot rows apart.
    pub fn fixed(b: MemoryBill, phase: Phase) u64 {
        var n: u64 = 0;
        for (b.terms) |t| n += t.bytes[@backingInt(phase)];
        return n;
    }

    /// The phase's total over the box baseline at `rows` slot rows per layer.
    pub fn total(b: MemoryBill, phase: Phase, baseline: u64, rows: u32) u64 {
        return baseline + b.fixed(phase) + rows * b.per_row;
    }

    /// What the process may hold above the baseline: the larger phase (a phase change frees before it grows, so
    /// there is no transition term).
    pub fn processBound(b: MemoryBill, rows: Rows) u64 {
        return @max(b.total(.prompt, 0, rows.prompt), b.total(.decode, 0, rows.decode));
    }

    /// What the constructed module holds before any request: the construction terms at their prompt bytes and the
    /// prompt rows.
    pub fn constructionBytes(b: MemoryBill, prompt_rows: u32) u64 {
        var n: u64 = prompt_rows * b.per_row;
        for (b.terms) |t| {
            if (t.at_construction) n += t.bytes[@backingInt(Phase.prompt)];
        }
        return n;
    }

    /// The kinds' parts as one bill, terms in order. One row source: the fill models a single row count, so two
    /// parts with slot rows are refused by name.
    pub fn compose(gpa: std.mem.Allocator, parts: []const MemoryBill) error{ TwoRowSources, OutOfMemory }!MemoryBill {
        var terms: std.ArrayList(Term) = .empty;
        errdefer terms.deinit(gpa);
        var per_row: u64 = 0;
        for (parts) |p| {
            if (p.per_row > 0 and per_row > 0) return error.TwoRowSources;
            try terms.appendSlice(gpa, p.terms);
            per_row += p.per_row;
        }
        return .{ .terms = try terms.toOwnedSlice(gpa), .per_row = per_row };
    }

    /// Frees what `compose` allocated.
    pub fn free(b: MemoryBill, gpa: std.mem.Allocator) void {
        gpa.free(b.terms);
    }
};

/// Slot rows per routed layer, per phase: prompt <= decode <= the layer's experts.
pub const Rows = struct { prompt: u32, decode: u32 };

/// What every kind's `bill` hook receives. Pure host: the routes the plugin installs (resolved once, through the
/// same resolver its install reads), the request shape, and the ceiling and the stop as arguments.
pub const BillRequest = struct {
    peek: *const peek.ConfigPeek,
    /// The plugin's own resolved routes (its type; the host never reads them).
    routes: *const anyopaque,
    prompt_tokens: u64,
    /// The request's generation cap. Each plugin bills its own KV reservation from it, by the same rule its prompt
    /// forward reserves with.
    max_tokens: u64,
    /// The GPU memory ceiling every plan fits under.
    ceiling: u64,
    /// What the admission keeps free under the ceiling (the guard's stop on our box).
    stop: u64,

    pub fn target(r: BillRequest) u64 {
        return r.ceiling -| r.stop;
    }
};

/// The fill: the most decode rows, then the most prompt rows (prompt <= decode <= `n_experts`), each phase's total
/// within `target`. Refused by name below `min_rows`, and for a bill without slot rows (nothing to fill).
pub fn fill(b: MemoryBill, baseline: u64, target: u64, n_experts: u32, min_rows: u32) error{ NoSlotRows, NativeBillDoesNotFit }!Rows {
    if (b.per_row == 0) return error.NoSlotRows;
    const most = struct {
        fn f(fixed: u64, t: u64, per_row: u64) u64 {
            return if (fixed >= t) 0 else (t - fixed) / per_row;
        }
    }.f;
    const decode = @min(most(baseline + b.fixed(.decode), target, b.per_row), n_experts);
    const prompt = @min(most(baseline + b.fixed(.prompt), target, b.per_row), decode);
    if (prompt < min_rows) return error.NativeBillDoesNotFit;
    return .{ .prompt = @intCast(prompt), .decode = @intCast(decode) };
}

/// Both phases within `target` at `rows`, once, before any slot bank or resident is allocated (forced rows are
/// checked here; the fill guarantees its own).
pub fn admit(b: MemoryBill, baseline: u64, rows: Rows, target: u64) error{ PromptOverTarget, DecodeOverTarget }!void {
    if (b.total(.prompt, baseline, rows.prompt) > target) return error.PromptOverTarget;
    if (b.total(.decode, baseline, rows.decode) > target) return error.DecodeOverTarget;
}

/// Once, at construction: the bill was taken at the rows the module built.
pub fn checkRows(billed: Rows, built: Rows) error{BillRowsMismatch}!void {
    if (billed.prompt != built.prompt or billed.decode != built.decode) return error.BillRowsMismatch;
}

/// Once, at construction: the footprint after the install within `tolerance` of the construction bytes.
pub fn checkConstruction(construction_bytes: u64, footprint: u64, tolerance: u64) error{ConstructionOverBill}!void {
    if (footprint > construction_bytes + tolerance) return error.ConstructionOverBill;
}

/// Once, at construction: a measured term's one measurement within its declared bound (the prompt phase's, the
/// phase in force at construction).
pub fn checkMeasured(t: MemoryBill.Term, measured: u64) error{ConstructionOverBill}!void {
    std.debug.assert(t.measured);
    if (measured > t.bytes[@backingInt(MemoryBill.Phase.prompt)]) return error.ConstructionOverBill;
}

const testing = std.testing;

const gb: u64 = 1_000_000_000;

fn fixtureBill() MemoryBill {
    const terms = comptime [_]MemoryBill.Term{
        .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true },
        .{ .name = "waves", .bytes = .{ 14 * gb, 2 * gb }, .at_construction = false },
        .{ .name = "kv", .bytes = .{ 1 * gb, 1 * gb }, .at_construction = false },
    };
    return .{ .terms = &terms, .per_row = gb / 4 };
}

test "sdk bill: phase totals, the fill's two counts and the admission refuse by phase name" {
    const b = fixtureBill();
    try testing.expectEqual(75 * gb, b.fixed(.prompt));
    try testing.expectEqual(62 * gb, b.fixed(.decode));
    try testing.expectEqual(10 * gb + 75 * gb + 4 * (gb / 4), b.total(.prompt, 10 * gb, 4));
    // target 110: prompt (110 - 85) / 0.25 = 100 rows, decode (110 - 72) / 0.25 = 152 rows, capped by 128 experts.
    const rows = try fill(b, 10 * gb, 110 * gb, 128, 16);
    try testing.expectEqual(Rows{ .prompt = 100, .decode = 128 }, rows);
    try admit(b, 10 * gb, rows, 110 * gb);
    try testing.expectError(error.PromptOverTarget, admit(b, 10 * gb, .{ .prompt = 101, .decode = 128 }, 110 * gb));
    try testing.expectError(error.DecodeOverTarget, admit(b, 10 * gb, .{ .prompt = 0, .decode = 128 }, 85 * gb));
    try testing.expectError(error.NativeBillDoesNotFit, fill(b, 10 * gb, 88 * gb, 128, 16));
    try testing.expectError(error.NoSlotRows, fill(.{ .terms = b.terms }, 10 * gb, 110 * gb, 128, 16));
    try testing.expectEqual(@max(75 * gb + 25 * gb, 62 * gb + 32 * gb), b.processBound(rows));
}

test "sdk bill: a measured term is billed at its bound: it fills exactly the rows of the same bytes as a plain term" {
    const plain = [_]MemoryBill.Term{ .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true }, .{ .name = "host_side", .bytes = .{ 900_000_000, 900_000_000 }, .at_construction = true } };
    const declared = [_]MemoryBill.Term{ .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true }, .{ .name = "host_side", .bytes = .{ 900_000_000, 900_000_000 }, .at_construction = true, .measured = true } };
    const a = try fill(.{ .terms = &plain, .per_row = gb / 2 }, 9 * gb, 118 * gb, 384, 16);
    try testing.expectEqual(a, try fill(.{ .terms = &declared, .per_row = gb / 2 }, 9 * gb, 118 * gb, 384, 16));
    try checkMeasured(declared[1], 330_000_000);
    try testing.expectError(error.ConstructionOverBill, checkMeasured(declared[1], 900_000_001));
}

test "sdk bill: the construction check: the bill at the built rows, the footprint within the construction terms" {
    const b = fixtureBill();
    // residents (the construction term) + 100 prompt rows; the waves and the KV come with a request
    try testing.expectEqual(60 * gb + 25 * gb, b.constructionBytes(100));
    try checkRows(.{ .prompt = 100, .decode = 128 }, .{ .prompt = 100, .decode = 128 });
    try testing.expectError(error.BillRowsMismatch, checkRows(.{ .prompt = 100, .decode = 128 }, .{ .prompt = 100, .decode = 127 }));
    try checkConstruction(b.constructionBytes(100), 85 * gb + 250_000_000, 250_000_000);
    try testing.expectError(error.ConstructionOverBill, checkConstruction(b.constructionBytes(100), 85 * gb + 250_000_001, 250_000_000));
}

test "sdk bill: the ceiling and the stop are arguments; an upstream default margin refuses what the pinned stop admits" {
    const b = fixtureBill();
    const base: BillRequest = .{ .peek = undefined, .routes = undefined, .prompt_tokens = 16384, .max_tokens = 1024, .ceiling = 120 * gb, .stop = 2 * gb };
    try testing.expectEqual(118 * gb, base.target());
    var upstream = base;
    upstream.stop = 8 * (1 << 30); // the 8 GiB wired-margin default
    const forced: Rows = .{ .prompt = 130, .decode = 160 };
    try admit(b, 10 * gb, forced, base.target());
    try testing.expectError(error.PromptOverTarget, admit(b, 10 * gb, forced, upstream.target()));
}

test "sdk bill: composition keeps every kind's terms in order and refuses a second row source" {
    const quant = [_]MemoryBill.Term{.{ .name = "residents", .bytes = .{ 3, 2 }, .at_construction = true }};
    const source = [_]MemoryBill.Term{.{ .name = "slot_transient", .bytes = .{ 5, 1 }, .at_construction = true }};
    const arch = [_]MemoryBill.Term{.{ .name = "waves", .bytes = .{ 7, 0 }, .at_construction = false }};
    const b = try MemoryBill.compose(testing.allocator, &.{ .{ .terms = &quant }, .{ .terms = &source, .per_row = 11 }, .{ .terms = &arch } });
    defer b.free(testing.allocator);
    try testing.expectEqual(@as(usize, 3), b.terms.len);
    try testing.expectEqualStrings("waves", b.terms[2].name);
    try testing.expectEqual(@as(u64, 15), b.fixed(.prompt));
    try testing.expectEqual(@as(u64, 3), b.fixed(.decode));
    try testing.expectEqual(@as(u64, 11), b.per_row);
    try testing.expectError(error.TwoRowSources, MemoryBill.compose(testing.allocator, &.{ .{ .terms = &source, .per_row = 11 }, .{ .terms = &arch, .per_row = 1 } }));
}
