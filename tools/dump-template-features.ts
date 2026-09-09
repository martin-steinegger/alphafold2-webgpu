/**
 * Writes one template's features to disk, for differential testing.
 *
 * tools/capture_alphafold_template_reference.py reads the same structure with
 * AlphaFold's own code and compares, so the two halves of the template input
 * path — reading a structure and turning it into features — are each checked
 * against the implementation they are ports of.
 *
 *   npx tsx tools/dump-template-features.ts <structure> <query> <output-dir> [chain]
 */
import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseStructure } from "../src/input/structure.js";
import { templateFeatures } from "../src/input/template-features.js";
import { templateAngleFeatures, templateTorsions } from "../src/input/template-torsions.js";

const [structurePath, queryArgument, outputArgument, chainArgument] = process.argv.slice(2);
if (structurePath === undefined || queryArgument === undefined || outputArgument === undefined) {
  throw new Error("usage: dump-template-features <structure> <query|@file> <output-dir> [chain]");
}
const query = queryArgument.startsWith("@")
  ? readFileSync(resolve(queryArgument.slice(1)), "utf8").replace(/\s+/g, "").toUpperCase()
  : queryArgument.replace(/\s+/g, "").toUpperCase();
const output = resolve(outputArgument);
await mkdir(output, { recursive: true });

const structure = parseStructure(readFileSync(resolve(structurePath), "utf8"));
const chain = chainArgument === undefined
  ? structure.chains[0]!
  : structure.chains.find((candidate) => candidate.id === chainArgument);
if (chain === undefined) throw new Error(`structure has no chain ${chainArgument}`);

const features = templateFeatures(query, chain);
const torsions = templateTorsions(features);
const angles = templateAngleFeatures(features);

const tensors: Record<string, Float32Array | Int32Array> = {
  chainAtomPositions: chain.atomPositions,
  chainAtomMask: chain.atomMask,
  aatype: features.aatype,
  atomPositions: features.atomPositions,
  atomMask: features.atomMask,
  pseudoBeta: features.pseudoBeta,
  pseudoBetaMask: features.pseudoBetaMask,
  backboneMask: features.backboneMask,
  queryToTemplate: features.alignment.queryToTemplate,
  torsionSinCos: torsions.sinCos,
  alternativeTorsionSinCos: torsions.alternativeSinCos,
  torsionMask: torsions.mask,
  angleFeatures: angles.features,
  angleRowMask: angles.rowMask,
};
const shapes: Record<string, readonly number[]> = {
  chainAtomPositions: [chain.sequence.length, 37, 3],
  chainAtomMask: [chain.sequence.length, 37],
  aatype: [query.length],
  atomPositions: [query.length, 37, 3],
  atomMask: [query.length, 37],
  pseudoBeta: [query.length, 3],
  pseudoBetaMask: [query.length],
  backboneMask: [query.length],
  queryToTemplate: [query.length],
  torsionSinCos: [query.length, 7, 2],
  alternativeTorsionSinCos: [query.length, 7, 2],
  torsionMask: [query.length, 7],
  angleFeatures: [query.length, 57],
  angleRowMask: [query.length],
};
for (const [name, tensor] of Object.entries(tensors)) {
  await writeFile(resolve(output, `${name}.bin`), Buffer.from(
    tensor.buffer, tensor.byteOffset, tensor.byteLength,
  ));
}
await writeFile(resolve(output, "template.json"), `${JSON.stringify({
  query, chain: chain.id, chainSequence: chain.sequence, format: structure.format,
  chains: structure.chains.map((candidate) => ({ id: candidate.id, residues: candidate.sequence.length })),
  alignment: {
    alignedResidues: features.alignment.alignedResidues,
    identicalResidues: features.alignment.identicalResidues,
    coverage: features.alignment.coverage,
    identity: features.alignment.identity,
    score: features.alignment.score,
  },
  dtypes: Object.fromEntries(Object.entries(tensors).map(
    ([name, tensor]) => [name, tensor instanceof Int32Array ? "int32" : "float32"],
  )),
  shapes,
}, null, 2)}\n`);
console.log(`${chain.sequence.length} template residues, ${features.alignment.alignedResidues} aligned `
  + `(${(features.alignment.coverage * 100).toFixed(1)}% coverage, `
  + `${(features.alignment.identity * 100).toFixed(1)}% identity) -> ${output}`);
