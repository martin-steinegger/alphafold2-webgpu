import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { alignSequences } from "../src/input/pairwise-align.js";
import { ATOM_CA, ATOM_CB, ATOM_TYPE_COUNT, ATOM_TYPES, residueLetter } from "../src/input/residue-atoms.js";
import { parseStructure } from "../src/input/structure.js";
import { templateFeatures } from "../src/input/template-features.js";
import { templateAngleFeatures, templateTorsions } from "../src/input/template-torsions.js";

const FRAGMENT = readFileSync(
  resolve(import.meta.dirname, "fixtures/template/ubiquitin-fragment.pdb"), "utf8",
);
const FRAGMENT_SEQUENCE = "MQIFVKTLTGKT";

describe("pairwise alignment", () => {
  it("aligns a sequence to itself", () => {
    const alignment = alignSequences(FRAGMENT_SEQUENCE, FRAGMENT_SEQUENCE);
    expect(Array.from(alignment.queryToTemplate)).toEqual([...FRAGMENT_SEQUENCE].map((_, index) => index));
    expect(alignment.coverage).toBe(1);
    expect(alignment.identity).toBe(1);
  });

  it("places a short template inside a longer query without paying for the overhang", () => {
    // Free terminal gaps: the template is a fragment, and it belongs where it
    // matches rather than stretched across the whole query.
    const alignment = alignSequences(`AAAA${FRAGMENT_SEQUENCE}AAAA`, FRAGMENT_SEQUENCE);
    expect(alignment.alignedResidues).toBe(FRAGMENT_SEQUENCE.length);
    expect(alignment.identity).toBe(1);
    expect(alignment.queryToTemplate[4]).toBe(0);
    expect(alignment.queryToTemplate[0]).toBe(-1);
    expect(alignment.queryToTemplate[alignment.queryToTemplate.length - 1]).toBe(-1);
  });

  it("opens one gap rather than many for an insertion in the query", () => {
    const query = `${FRAGMENT_SEQUENCE.slice(0, 6)}GGGGG${FRAGMENT_SEQUENCE.slice(6)}`;
    const alignment = alignSequences(query, FRAGMENT_SEQUENCE);
    expect(alignment.alignedResidues).toBe(FRAGMENT_SEQUENCE.length);
    for (let position = 6; position < 11; position += 1) {
      expect(alignment.queryToTemplate[position]).toBe(-1);
    }
    expect(alignment.queryToTemplate[11]).toBe(6);
  });

  it("reports no alignment for an empty sequence", () => {
    expect(alignSequences("ACDE", "").alignedResidues).toBe(0);
    expect(alignSequences("", "ACDE").queryToTemplate.length).toBe(0);
  });
});

describe("structure reading", () => {
  it("reads a PDB fragment into residues and atom37 coordinates", () => {
    const { chains, format } = parseStructure(FRAGMENT);
    expect(format).toBe("pdb");
    expect(chains).toHaveLength(1);
    const chain = chains[0]!;
    expect(chain.id).toBe("A");
    expect(chain.sequence).toBe(FRAGMENT_SEQUENCE);
    expect(chain.atomPositions.length).toBe(FRAGMENT_SEQUENCE.length * ATOM_TYPE_COUNT * 3);
    // Ubiquitin's first residue is a methionine whose N sits at 27.340, 24.430, 2.614.
    Array.from(chain.atomPositions.subarray(0, 3)).forEach((value, axis) => {
      expect(value).toBeCloseTo([27.34, 24.43, 2.614][axis]!, 5);
    });
    expect(chain.atomMask[0]).toBe(1);
    // Glycine is the only residue without a CB, and there is none in this fragment.
    expect(chain.atomMask[ATOM_CB]).toBe(1);
    expect(chain.residueNumbers[0]).toBe(1);
  });

  it("reads the same chain out of mmCIF", () => {
    const rows = [...FRAGMENT.split("\n")].filter((line) => line.startsWith("ATOM")).map((line) => {
      const atom = line.slice(12, 16).trim(); const residue = line.slice(17, 20);
      const number = line.slice(22, 26).trim();
      return `ATOM ${atom} ${residue} A ${number} ${line.slice(30, 38).trim()} `
        + `${line.slice(38, 46).trim()} ${line.slice(46, 54).trim()} 1`;
    });
    const cif = ["data_test", "loop_", "_atom_site.group_PDB", "_atom_site.label_atom_id",
      "_atom_site.label_comp_id", "_atom_site.auth_asym_id", "_atom_site.auth_seq_id",
      "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
      "_atom_site.pdbx_PDB_model_num", ...rows, "#"].join("\n");
    const { chains, format } = parseStructure(cif);
    expect(format).toBe("mmcif");
    expect(chains[0]!.sequence).toBe(FRAGMENT_SEQUENCE);
  });

  it("keeps a modified residue and drops a ligand", () => {
    expect(residueLetter("MSE")).toBe("M");
    expect(residueLetter("HOH")).toBeUndefined();
    const withLigand = `${FRAGMENT}\nHETATM  999  O   HOH A 900      10.000  10.000  10.000  1.00  0.00           O`;
    expect(parseStructure(withLigand).chains).toHaveLength(1);
  });

  it("refuses a file with no protein in it", () => {
    expect(() => parseStructure("HETATM    1  O   HOH A   1       0.000   0.000   0.000")).toThrow(/no protein chain/);
  });

  it("names the 37 atoms in AlphaFold's order", () => {
    expect(ATOM_TYPES).toHaveLength(37);
    expect(ATOM_TYPES.slice(0, 5)).toEqual(["N", "CA", "C", "CB", "O"]);
    expect(ATOM_CA).toBe(1);
  });
});

describe("template features", () => {
  const chain = parseStructure(FRAGMENT).chains[0]!;

  it("indexes the template by query position and gaps the rest", () => {
    const query = `AAA${FRAGMENT_SEQUENCE}`;
    const features = templateFeatures(query, chain);
    expect(features.length).toBe(query.length);
    // The three query residues with no template behind them are gaps: aatype
    // 21, no coordinates, and every mask over them zero.
    for (let position = 0; position < 3; position += 1) {
      expect(features.aatype[position]).toBe(21);
      expect(features.pseudoBetaMask[position]).toBe(0);
      expect(features.backboneMask[position]).toBe(0);
      expect(features.atomMask.subarray(position * ATOM_TYPE_COUNT, (position + 1) * ATOM_TYPE_COUNT))
        .toEqual(new Float32Array(ATOM_TYPE_COUNT));
    }
    // Methionine is index 12 in ARNDCQEGHILKMFPSTWYV.
    expect(features.aatype[3]).toBe(12);
    expect(features.backboneMask[3]).toBe(1);
    expect(Array.from(features.pseudoBeta.subarray(9, 12)))
      .toEqual(Array.from(chain.atomPositions.subarray(ATOM_CB * 3, ATOM_CB * 3 + 3)));
  });

  it("refuses a template that covers less of the query than asked", () => {
    expect(() => templateFeatures(`${"A".repeat(100)}${FRAGMENT_SEQUENCE}`, chain, { minimumCoverage: .5 }))
      .toThrow(/covers 11% of the query/);
  });
});

describe("template torsions", () => {
  const chain = parseStructure(FRAGMENT).chains[0]!;
  const features = templateFeatures(FRAGMENT_SEQUENCE, chain);

  it("gives every defined angle unit length and pins the undefined ones", () => {
    const torsions = templateTorsions(features);
    for (let residue = 0; residue < features.length; residue += 1) {
      for (let torsion = 0; torsion < 7; torsion += 1) {
        const sine = torsions.sinCos[(residue * 7 + torsion) * 2]!;
        const cosine = torsions.sinCos[(residue * 7 + torsion) * 2 + 1]!;
        expect(Math.hypot(sine, cosine)).toBeCloseTo(1, 5);
        if (torsions.mask[residue * 7 + torsion] === 0) {
          // AlphaFold's own placeholder, which is what an undefined angle gets
          // instead of the noise a degenerate frame would produce.
          expect([sine, cosine]).toEqual([1, 0]);
        }
      }
    }
    // The first residue has no residue before it, so pre-omega and phi are
    // undefined; psi only needs this residue's own backbone.
    expect(torsions.mask[0]).toBe(0);
    expect(torsions.mask[1]).toBe(0);
    expect(torsions.mask[2]).toBe(1);
  });

  it("lays the 57 angle channels out as AlphaFold concatenates them", () => {
    const { features: angles, rowMask } = templateAngleFeatures(features);
    expect(angles.length).toBe(features.length * 57);
    const torsions = templateTorsions(features);
    for (let residue = 0; residue < features.length; residue += 1) {
      const row = residue * 57;
      // One-hot aatype, then 14 torsion values, 14 alternatives and 7 masks.
      expect(angles[row + features.aatype[residue]!]).toBe(1);
      expect(angles.slice(row + 22, row + 36))
        .toEqual(torsions.sinCos.slice(residue * 14, residue * 14 + 14));
      expect(angles.slice(row + 50, row + 57))
        .toEqual(torsions.mask.slice(residue * 7, residue * 7 + 7));
      // The row's MSA mask is the psi mask, which needs one residue's backbone.
      expect(rowMask[residue]).toBe(torsions.mask[residue * 7 + 2]);
    }
  });
});
