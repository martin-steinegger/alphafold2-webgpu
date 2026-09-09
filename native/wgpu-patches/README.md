# Patches applied to the wgpu checkout

Each applies to a wgpu tree with `git apply`, against the commit named in its
header. Build with them by pointing the addon at the patched tree:

    cd /path/to/wgpu && git apply /path/to/native/wgpu-patches/*.patch
    AFWEBGPU_WGPU_PATH=/path/to/wgpu native/wgpu-backend/build.sh

`open()` then reports `wgpu=<that path>`, so a measurement says which tree
produced it.

## 0001, enable subgroups

naga implements every subgroup builtin and rejects only the directive, so a
kernel written for Dawn is a parse error there and one written for naga is a
parse error on Dawn. Thirty changed lines in one file move `subgroups` from
`UnimplementedEnableExtension` to `ImplementedEnableExtension` and map it to
the `SUBGROUP` capability that already exists.

Measured against wgpu trunk 726f0ab: stock naga rejects `enable subgroups;`
with "not yet supported"; patched naga validates the same shader.

It halves the directive difference rather than removing it. Our Dawn kernels
also carry `enable subgroup_size_control;`, which gates `@subgroup_size` and
has no naga equivalent, and wgpu never attaches
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` at all. So
`src/runtime/dialect.ts` keeps its directive half until that is answered too.
