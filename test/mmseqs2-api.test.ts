import { describe, expect, it, vi } from "vitest";
import { parseA3m } from "../src/input/a3m.js";
import {
  assembleComplexA3m, extractMmseqs2A3m, generateMmseqs2ComplexMsa,
  generateMmseqs2Msa, readTarFiles,
} from "../src/input/mmseqs2-api.js";

function tar(files: Readonly<Record<string, string>>): Uint8Array {
  const chunks: Uint8Array[] = [];
  for (const [name, value] of Object.entries(files)) {
    const header = new Uint8Array(512); const encodedName = new TextEncoder().encode(name);
    header.set(encodedName, 0);
    header.set(new TextEncoder().encode(value.length.toString(8).padStart(11, "0") + "\0"), 124);
    header[156] = 48;
    const data = new TextEncoder().encode(value); const padded = new Uint8Array(Math.ceil(data.length / 512) * 512); padded.set(data);
    chunks.push(header, padded);
  }
  chunks.push(new Uint8Array(1024));
  const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0; for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output;
}

const resultTar = tar({
  "uniref.a3m": ">101\nACDE\n>hit1\nAC-E\n\0",
  "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nACDE\n>env1\nAcCDE\n\0",
});

describe("MMseqs2 API", () => {
  it("reads files and combines the ColabFold UniRef and environmental A3Ms", () => {
    expect([...readTarFiles(resultTar).keys()]).toEqual(["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]);
    expect(extractMmseqs2A3m(resultTar)).toBe(">101\nACDE\n>hit1\nAC-E\n>101\nACDE\n>env1\nAcCDE\n");
  });

  it("submits, polls, downloads, and validates a generated MSA", async () => {
    const requests: { url: string; init?: RequestInit }[] = [];
    const responses = [
      new Response(JSON.stringify({ status: "PENDING", id: "ticket-1" })),
      new Response(JSON.stringify({ status: "RUNNING" })),
      new Response(JSON.stringify({ status: "COMPLETE" })),
      new Response(new Uint8Array([1, 2, 3])),
    ];
    const fetchImplementation = vi.fn(async (input: URL | RequestInfo, init?: RequestInit) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) });
      return responses.shift()!;
    }) as typeof fetch;
    const phases: string[] = [];
    const result = await generateMmseqs2Msa("ACDE", {
      fetchImplementation, wait: async () => {}, decompress: async () => resultTar,
      onProgress: (progress) => phases.push(progress.phase),
    });
    expect(result.ticket).toBe("ticket-1"); expect(result.depth).toBe(4);
    expect(requests.map((entry) => entry.url)).toEqual([
      "https://api.colabfold.com/ticket/msa", "https://api.colabfold.com/ticket/ticket-1",
      "https://api.colabfold.com/ticket/ticket-1", "https://api.colabfold.com/result/download/ticket-1",
    ]);
    expect(String(requests[0]!.init?.body)).toContain("mode=env");
    expect(phases).toEqual(["submitting", "queued", "running", "downloading", "complete"]);
  });

  it("assembles paired rows densely and unpaired rows block-diagonally", () => {
    const assembled = assembleComplexA3m(
      ["AC", "GG"], ["AC", "GG"],
      [">101\nAC\n>uA\nA-\n", ">102\nGG\n>uB\nG-\n"],
      [">101\nAC\n>p\n-C\n", ">102\nGG\n>p\n-G\n"],
    );
    expect(parseA3m(assembled.a3m).sequences).toEqual(["ACGG", "-C-G", "A---", "--G-"]);
    expect(assembled.depth).toBe(4);
    expect([...assembled.mask]).toEqual([
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 0, 0,
      0, 0, 1, 1,
    ]);
  });

  it("uses a dense unpaired alignment for repeated homomer chains", () => {
    const assembled = assembleComplexA3m(
      ["AC", "AC", "AC"], ["AC"], [">101\nAC\n>hit\nA-\n"],
    );
    expect(parseA3m(assembled.a3m).sequences).toEqual(["ACACAC", "A-A-A-"]);
    expect([...assembled.mask]).toEqual(new Array(12).fill(1));
  });

  it("runs ColabFold unpaired and greedy-paired searches for heteromers", async () => {
    const unpairedTar = tar({
      "uniref.a3m": ">101\nAC\n>uA\nA-\n\0>102\nGG\n>uB\nG-\n\0",
    });
    const pairedTar = tar({
      "pair.a3m": ">101\nAC\n>p\n-C\n\0>102\nGG\n>p\n-G\n\0",
    });
    const submissions: { endpoint: string; body: string }[] = [];
    const fetchImplementation = vi.fn(async (input: URL | RequestInfo, init?: RequestInit) => {
      const url = String(input);
      if (init?.method === "POST" && (url.endsWith("ticket/msa") || url.endsWith("ticket/pair"))) {
        const endpoint = url.endsWith("pair") ? "pair" : "msa";
        submissions.push({ endpoint, body: String(init?.body) });
        return new Response(JSON.stringify({ status: "PENDING", id: endpoint }));
      }
      if (url.includes("/ticket/")) return new Response(JSON.stringify({ status: "COMPLETE" }));
      if (url.endsWith("/msa")) return new Response(Uint8Array.of(1));
      if (url.endsWith("/pair")) return new Response(Uint8Array.of(2));
      throw new Error(`unexpected request ${url}`);
    }) as typeof fetch;
    const result = await generateMmseqs2ComplexMsa(["AC", "GG"], {
      useEnvironmental: false, fetchImplementation, wait: async () => {},
      decompress: async (archive) => new Uint8Array(archive)[0] === 1 ? unpairedTar : pairedTar,
    });
    expect(result.unpairedTicket).toBe("msa");
    expect(result.pairedTicket).toBe("pair");
    expect(result.depth).toBe(4);
    expect(submissions.map((entry) => entry.endpoint).sort()).toEqual(["msa", "pair"]);
    expect(submissions.find((entry) => entry.endpoint === "msa")!.body).toContain("mode=all");
    expect(submissions.find((entry) => entry.endpoint === "pair")!.body).toContain("mode=pairgreedy");
    expect(submissions.every((entry) => entry.body.includes("%3E101") && entry.body.includes("%3E102"))).toBe(true);
  });
});
