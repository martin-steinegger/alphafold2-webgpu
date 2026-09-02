import { describe, expect, it } from "vitest";
import { readZipArchive } from "./zip-reader.js";
import {
  citations, packageResults, predictedAlignedErrorJson, resultPackageEntries, zipArchive, type ResultPackage,
} from "../web/result-package.js";

async function readArchive(blob: Blob): Promise<Map<string, Uint8Array>> {
  return readZipArchive(new Uint8Array(await blob.arrayBuffer()));
}

const text = (bytes: Uint8Array | undefined): string => new TextDecoder().decode(bytes);

function samplePackage(overrides: Partial<ResultPackage> = {}): ResultPackage {
  const sequence = "MKTAYIA";
  const length = sequence.length;
  const pae = new Float32Array(length * length);
  for (let index = 0; index < pae.length; index += 1) pae[index] = (index % 11) + 0.125;
  return {
    jobName: "test_abc12",
    sequence,
    chainLengths: [4, 3],
    confidence: {
      plddt: Float32Array.from({ length }, (_, index) => 60 + index),
      meanPlddt: 63, ptm: 0.5, iptm: 0.25,
      predictedAlignedError: pae, maxPredictedAlignedError: 31.75,
    },
    pdb: "REMARK   1 TEST\nTER\nEND\n",
    scoresJson: "{\"ptm\":0.5}",
    a3m: ">query\nMKTAYIA\n>hit\nMKTAYIA\n",
    depth: 2,
    images: [{ suffix: "plddt", png: Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 13, 10, 26, 10]) }],
    settings: { recycles: 3, seed: 0 },
    log: "Starting prediction…\nFinished.",
    usedMmseqs2: true,
    multimer: true,
    ...overrides,
  };
}

describe("zipArchive", () => {
  it("writes entries an unzip implementation can read back", async () => {
    const files = await readArchive(await zipArchive([
      { name: "folder/a.txt", data: "hello ".repeat(200) },
      { name: "folder/b.bin", data: Uint8Array.from([1, 2, 3, 4]), store: true },
      { name: "folder/empty.txt", data: "" },
    ]));
    expect([...files.keys()]).toEqual(["folder/a.txt", "folder/b.bin", "folder/empty.txt"]);
    expect(text(files.get("folder/a.txt"))).toBe("hello ".repeat(200));
    expect(Array.from(files.get("folder/b.bin")!)).toEqual([1, 2, 3, 4]);
    expect(files.get("folder/empty.txt")!.length).toBe(0);
  });

  it("compresses repetitive text well below its stored size", async () => {
    const data = "MKTAYIAKQRQISFVKSHFSRQ".repeat(500);
    const compressed = await zipArchive([{ name: "a.a3m", data }]);
    const stored = await zipArchive([{ name: "a.a3m", data, store: true }]);
    expect(compressed.size).toBeLessThan(stored.size / 4);
  });
});

describe("resultPackageEntries", () => {
  it("packages structure, scores, plots, alignment, settings and citations", async () => {
    const result = samplePackage();
    const files = await readArchive(await packageResults(result));
    expect([...files.keys()]).toEqual([
      "test_abc12/test_abc12_unrelaxed_model_1.pdb",
      "test_abc12/test_abc12_scores.json",
      "test_abc12/test_abc12_predicted_aligned_error_v1.json",
      "test_abc12/test_abc12_plddt.png",
      "test_abc12/test_abc12.a3m",
      "test_abc12/test_abc12.csv",
      "test_abc12/config.json",
      "test_abc12/log.txt",
      "test_abc12/cite.bib",
    ]);
    expect(text(files.get("test_abc12/test_abc12_unrelaxed_model_1.pdb"))).toBe(result.pdb);
    expect(text(files.get("test_abc12/test_abc12.csv"))).toBe("id,sequence\ntest_abc12,MKTA:YIA\n");
    expect(text(files.get("test_abc12/log.txt"))).toBe("Starting prediction…\nFinished.\n");
    expect(JSON.parse(text(files.get("test_abc12/config.json")))).toEqual({ recycles: 3, seed: 0 });
  });

  it("keeps already-compressed plots stored verbatim", async () => {
    const png = Uint8Array.from({ length: 64 }, (_, index) => (index * 37) % 251);
    const files = await readArchive(await packageResults(samplePackage({ images: [{ suffix: "pae", png }] })));
    expect(Array.from(files.get("test_abc12/test_abc12_pae.png")!)).toEqual(Array.from(png));
  });

  it("writes the alignment error in AlphaFold-DB's shape", () => {
    const result = samplePackage();
    const parsed = JSON.parse(predictedAlignedErrorJson(result.confidence, result.sequence.length)) as
      readonly { predicted_aligned_error: number[][]; max_predicted_aligned_error: number }[];
    expect(parsed).toHaveLength(1);
    expect(parsed[0]!.predicted_aligned_error).toHaveLength(7);
    expect(parsed[0]!.predicted_aligned_error[0]).toHaveLength(7);
    expect(parsed[0]!.predicted_aligned_error[0]![1]).toBe(1.13);
    expect(parsed[0]!.max_predicted_aligned_error).toBe(31.75);
  });

  it("cites only the methods a run actually used", () => {
    const monomerLocal = citations({ multimer: false, usedMmseqs2: false });
    expect(monomerLocal).toContain("jumper2021highly");
    expect(monomerLocal).not.toContain("evans2021protein");
    expect(monomerLocal).not.toContain("mirdita2022colabfold");
    const complexRemote = citations({ multimer: true, usedMmseqs2: true });
    expect(complexRemote).toContain("evans2021protein");
    expect(complexRemote).toContain("mirdita2022colabfold");
    expect(complexRemote).toContain("steinegger2017mmseqs2");
  });

  it("names every entry inside the job folder", () => {
    for (const entry of resultPackageEntries(samplePackage())) expect(entry.name.startsWith("test_abc12/")).toBe(true);
  });
});
