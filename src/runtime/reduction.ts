/**
 * How a row-LayerNorm kernel divides its work, however this device can.
 *
 * Row LayerNorm is about a tenth of the trunk — 11.3% of an extra-MSA block
 * and 9.9% of a main Evoformer one — and every one of them reduces a row
 * twice, once for the mean and once for the variance. Three arrangements are
 * offered, and which a device gets is what it can run rather than what is
 * fastest in the abstract, because the fastest needs the most from it.
 *
 * rows-per-subgroup gives each subgroup a whole row and puts eight of them
 * in a workgroup, so a reduction is one instruction and there is no barrier
 * and no workgroup memory at all. It is the arrangement the Pallas kernel in
 * ColabFold's AlphaFold reaches by handing one program a block of 64 rows: the
 * gain is not the barriers, it is that a workgroup covers many rows. Measured
 * against the tree on a Blackwell over the shapes the Evoformer normalizes:
 * 0.0306 ms to 0.0196 at 29,972 rows of 256 channels, 0.0271 to 0.0175 at
 * 25,600 of 256, 0.0069 to 0.0054 at 3,481 of 128, and 0.0443 to 0.0124 at
 * 60,416 of 64. The narrow rows gain most, having had one channel a lane to
 * reduce and six barriers to do it in.
 *
 * One row to a workgroup of 32 — the same barrier-free reduction, one row at a
 * time — is 0.0256 and 0.0411 on those two, so it is the rows a workgroup
 * covers that pays and not the barriers, and it is not offered.
 *
 * subgroup-rows needs the subgroup width pinned, so a device that has
 * subgroups but cannot be held to 32 lanes falls to subgroup-workgroup: one
 * row a workgroup still, but the tree replaced by one subgroup instruction and
 * a single barrier to join the subgroups. That is 0.0229 and 0.0408 on the
 * same two shapes. A device with no subgroups at all keeps the tree.
 *
 * Two arrangements that look promising and are not, over the same shapes:
 * caching the row in workgroup memory so the source is read once instead of
 * three times is slower everywhere, because the re-reads already hit cache;
 * and accumulating both moments in one pass, so the variance comes from
 * E[x^2] - mean^2, is 3% to 5% faster and four to twenty-five times less
 * accurate, which is not a trade this model makes.
 */
import { supportsSubgroupSize } from "./subgroups.js";
import { dialect } from "./dialect.js";

/** Rows one workgroup covers where a subgroup can own one, and its width. */
const SUBGROUP_ROWS = 8;
const SUBGROUP_LANES = 32;
const SUBGROUP_ROWS_WORKGROUP = SUBGROUP_ROWS * SUBGROUP_LANES;

/** Invocations a workgroup gets where one of them covers a single row. */
const ROW_WORKGROUP = 64;

export interface RowNormalizeLayout {
  /** Rows one workgroup covers. The dispatch grid divides the count by this. */
  readonly rowsPerWorkgroup: number;
  /** enable directives, which must precede everything else in the module. */
  readonly enables: string;
  /** Module-scope declarations the reduction needs. */
  readonly declarations: string;
  /** The entry point's attributes, workgroup size included. */
  readonly attributes: string;
  /** Built-ins main has to declare, ready to append to its parameter list. */
  readonly builtins: string;
  /**
   * Opens main, given an expression for the row count.
   *
   * Defines norm_row, which is always a row this dispatch may read — a
   * workgroup covering rows past the end clamps to zero rather than branching
   * away, so the reductions stay in uniform control flow — along with
   * norm_live, which says whether the row is a real one and must guard every
   * store, and norm_lane and norm_stride, which the channel loops walk by.
   */
  open(rows: string): string;
  /** Sums source across the row, into a new let named target. */
  sum(source: string, target: string): string;
}

/** One row a subgroup, eight subgroups a workgroup. */
function subgroupRows(subgroupEnable: string): RowNormalizeLayout {
  return {
    rowsPerWorkgroup: SUBGROUP_ROWS,
    enables: `${subgroupEnable}enable subgroup_size_control;\n`,
    declarations: "",
    attributes: `@compute @workgroup_size(${SUBGROUP_ROWS_WORKGROUP})`
      + ` @subgroup_size(${SUBGROUP_LANES})`,
    builtins: ", @builtin(subgroup_invocation_id) norm_lane: u32",
    open: (rows: string): string => `
  let norm_index = (group.x + group.y * GRID_WIDTH) * ${SUBGROUP_ROWS}u
    + local.x / ${SUBGROUP_LANES}u;
  let norm_live = norm_index < ${rows};
  let norm_row = select(0u, norm_index, norm_live);
  let norm_stride = ${SUBGROUP_LANES}u;`,
    sum: (source: string, target: string): string => `
  let ${target} = subgroupAdd(${source});`,
  };
}

/** One row a workgroup, its subgroups reduced in one instruction each. */
function subgroupWorkgroup(subgroupEnable: string): RowNormalizeLayout {
  let reductions = 0;
  return {
    rowsPerWorkgroup: 1,
    enables: subgroupEnable,
    declarations: `var<workgroup> norm_partial: array<f32, ${ROW_WORKGROUP}>;`,
    attributes: `@compute @workgroup_size(${ROW_WORKGROUP})`,
    builtins: ", @builtin(subgroup_size) norm_width: u32,"
      + " @builtin(subgroup_invocation_id) norm_subgroup_lane: u32",
    open: (rows: string): string => `
  let norm_row = group.x + group.y * GRID_WIDTH;
  if (norm_row >= ${rows}) { return; }
  let norm_live = true;
  let norm_lane = local.x;
  let norm_stride = ${ROW_WORKGROUP}u;`,
    sum: (source: string, target: string): string => {
      const joined = `norm_joined_${reductions += 1}`;
      // Every subgroup width WebGPU permits divides the workgroup, so the join
      // walks exactly the subgroups the workgroup holds.
      return `
  let ${joined} = subgroupAdd(${source});
  if (norm_subgroup_lane == 0u) { norm_partial[local.x / norm_width] = ${joined}; }
  workgroupBarrier();
  if (local.x == 0u) {
    var norm_total = 0.0;
    for (var s = 0u; s < ${ROW_WORKGROUP}u / norm_width; s += 1u) { norm_total += norm_partial[s]; }
    norm_partial[0] = norm_total;
  }
  workgroupBarrier();
  let ${target} = norm_partial[0];`;
    },
  };
}

/** One row a workgroup, reduced by a halving tree, which anything can run. */
function treeWorkgroup(): RowNormalizeLayout {
  return {
    rowsPerWorkgroup: 1,
    enables: "",
    declarations: `var<workgroup> norm_partial: array<f32, ${ROW_WORKGROUP}>;`,
    attributes: `@compute @workgroup_size(${ROW_WORKGROUP})`,
    builtins: "",
    open: (rows: string): string => `
  let norm_row = group.x + group.y * GRID_WIDTH;
  if (norm_row >= ${rows}) { return; }
  let norm_live = true;
  let norm_lane = local.x;
  let norm_stride = ${ROW_WORKGROUP}u;`,
    sum: (source: string, target: string): string => `
  norm_partial[local.x] = ${source};
  workgroupBarrier();
  for (var stride = ${ROW_WORKGROUP / 2}u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { norm_partial[local.x] += norm_partial[local.x + stride]; }
    workgroupBarrier();
  }
  let ${target} = norm_partial[0];`,
  };
}

/**
 * The layout a device gets.
 *
 * Each call returns its own, because a layout names a variable per reduction
 * and two reductions in one shader may not share a name.
 */
export function rowNormalizeLayout(device: GPUDevice | undefined): RowNormalizeLayout {
  if (device?.features?.has("subgroups" as GPUFeatureName) !== true) return treeWorkgroup();
  const { subgroupEnable } = dialect(device);
  return supportsSubgroupSize(device, SUBGROUP_LANES)
    ? subgroupRows(subgroupEnable) : subgroupWorkgroup(subgroupEnable);
}

/** Rows one workgroup of this layout covers, for sizing a dispatch grid. */
export function rowsPerNormalizeWorkgroup(device: GPUDevice | undefined): number {
  return rowNormalizeLayout(device).rowsPerWorkgroup;
}
