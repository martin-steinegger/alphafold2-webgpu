# AFWebGPU engineering invariants

- Never change reference tensors or tolerances merely to make a failing test pass.
- Never fall back to the CPU for AlphaFold neural-network operations in production paths.
- Every GPU kernel must have a deterministic differential test against an independent reference implementation.
- Do not materialize tensors with `O(L^3)` storage.
- Route all GPU allocations through the shared allocator so peak memory remains measurable.
- Target Chrome on macOS Apple Silicon first while keeping standards-compliant WebGPU/WGSL.
- Establish numerical correctness before optimizing performance.
- Keep shape, dtype, and byte-size validation at public boundaries.

