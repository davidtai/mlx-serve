//! G5, the kernel registry: a plugin that brings Metal texts declares one manifest that pins every text by its
//! sha256, and the host builds and binds the set once per load. The registry machinery (kernel_set, kernel_routes,
//! kernel_trace) moves here, generic over the registry, with the quant kind's first registered consumer; no arm adds
//! a construction-time device self-check.

/// A kernel set's pin: what conformance and `/props` report, next to the binary's own sha256 the window pins.
pub const Pin = struct {
    /// sha256 (hex) of the manifest, which pins every text by its own sha256: moving the directory keeps it,
    /// changing a text does not.
    manifest_sha256: []const u8,
};
