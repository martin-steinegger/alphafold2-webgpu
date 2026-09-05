import { recycleFeatureSource, type RecycleFeatureSource } from "./a3m-features.js";
import type { MonomerRecycleFeatures, MonomerTemplateFeatures } from "../model/monomer.js";
import type { PairwiseAlignment } from "./pairwise-align.js";
import { parseStructure, type StructureChain } from "./structure.js";
import { templateFeatures } from "./template-features.js";
import { templateAngleFeatures } from "./template-torsions.js";

/**
 * Everything a prediction needs to know about one uploaded template.
 *
 * The alignment and the chain come back alongside the features because they
 * are what someone needs to see before trusting the result: which chain was
 * used, how much of the query it covered, and how close it was.
 */
export interface PreparedTemplate {
  readonly features: MonomerTemplateFeatures;
  readonly chain: StructureChain;
  readonly alignment: PairwiseAlignment;
}

export interface TemplateOptions {
  /** Which chain to use; the first one when not given. */
  readonly chainId?: string;
  /** Refuse a template covering less of the query than this. */
  readonly minimumCoverage?: number;
}

/** The chains a structure offers, for a picker. */
export function templateChains(structureText: string): readonly StructureChain[] {
  return parseStructure(structureText).chains;
}

export function prepareTemplate(
  structureText: string, query: string, options: TemplateOptions = {},
): PreparedTemplate {
  const { chains } = parseStructure(structureText);
  const chain = options.chainId === undefined
    ? chains[0]! : chains.find((candidate) => candidate.id === options.chainId);
  if (chain === undefined) {
    throw new Error(
      `This structure has no chain ${options.chainId}. It has `
      + `${chains.map((candidate) => candidate.id).join(", ")}.`,
    );
  }
  const features = templateFeatures(query, chain, options.minimumCoverage === undefined
    ? {} : { minimumCoverage: options.minimumCoverage });
  const angles = templateAngleFeatures(features);
  return {
    chain, alignment: features.alignment,
    features: {
      pair: {
        pseudoBeta: features.pseudoBeta, pseudoBetaMask: features.pseudoBetaMask,
        backboneMask: features.backboneMask, aatype: features.aatype,
      },
      angleFeatures: angles.features, rowMask: angles.rowMask,
    },
  };
}

/**
 * Attaches a template to a feature source.
 *
 * The model reads it from the first recycle, since a template does not change
 * between them, so only the first set needs it.
 */
export function withTemplate(
  source: RecycleFeatureSource<MonomerRecycleFeatures>, template: MonomerTemplateFeatures,
): RecycleFeatureSource<MonomerRecycleFeatures> {
  return recycleFeatureSource(source.length, function* recycles() {
    let first = true;
    for (const features of source) {
      yield first ? { ...features, template } : features;
      first = false;
    }
  });
}
