const AMINO_ACIDS = /^[ARNDCQEGHILKMFPSTWYVX]+$/;

export interface ParsedSequenceExpression {
  readonly chains: readonly string[];
  readonly sequence: string;
  readonly multimer: boolean;
}

/** Parse the single ColabFold-style sequence field used for monomers and complexes. */
export function parseSequenceExpression(value: string): ParsedSequenceExpression {
  const normalized = value.replace(/\s+/g, "").toUpperCase();
  const chains = normalized.split(":");
  if (chains.some((chain) => chain.length === 0 || !AMINO_ACIDS.test(chain))) {
    throw new Error("Sequence must contain valid amino-acid chains separated by single colons");
  }
  const sequence = chains.join("");
  if (!Number.isSafeInteger(sequence.length) || sequence.length === 0) {
    throw new RangeError("sequence length is invalid");
  }
  return { chains, sequence, multimer: chains.length > 1 };
}
