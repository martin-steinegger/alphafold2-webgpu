import { parseA3m } from "./a3m.js";

const DEFAULT_API_URL = "https://api.colabfold.com";
const QUERY_ID = 101;
const TRANSIENT_SUBMISSION = new Set(["UNKNOWN", "RATELIMIT"]);
const TRANSIENT_JOB = new Set(["UNKNOWN", "PENDING", "RUNNING", "RATELIMIT"]);

export type Mmseqs2MsaPhase = "submitting" | "queued" | "running" | "downloading" | "complete" | "retrying";

export interface Mmseqs2MsaProgress {
  readonly phase: Mmseqs2MsaPhase;
  readonly status: string;
  readonly search?: "monomer" | "unpaired" | "paired";
  readonly ticket?: string;
  readonly elapsedMilliseconds: number;
}

export interface Mmseqs2MsaResult {
  readonly a3m: string;
  readonly ticket: string;
  readonly depth: number;
  readonly elapsedMilliseconds: number;
}

export interface Mmseqs2ComplexMsaResult {
  readonly a3m: string;
  readonly mask: Float32Array;
  readonly depth: number;
  readonly unpairedTicket: string;
  readonly pairedTicket?: string;
  readonly elapsedMilliseconds: number;
}

export interface Mmseqs2MsaOptions {
  readonly apiUrl?: string;
  readonly useEnvironmental?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: Mmseqs2MsaProgress) => void;
  /** Test hook; applications should use the browser fetch implementation. */
  readonly fetchImplementation?: typeof fetch;
  /** Test hook for removing poll delays. */
  readonly wait?: (milliseconds: number, signal?: AbortSignal) => Promise<void>;
  /** Test hook for supplying an already decompressed tar archive. */
  readonly decompress?: (archive: ArrayBuffer) => Promise<Uint8Array>;
}

export interface Mmseqs2ComplexMsaOptions extends Mmseqs2MsaOptions {
  readonly pairingStrategy?: "greedy" | "complete";
}

interface TicketResponse { readonly status?: unknown; readonly id?: unknown; }

function normalizedSequence(sequence: string): string {
  const value = sequence.replace(/\s+/g, "").toUpperCase();
  if (!/^[ARNDCQEGHILKMFPSTWYVX]+$/.test(value)) {
    throw new Error("Sequence must contain only standard amino-acid letters or X");
  }
  return value;
}

function abortError(): Error {
  return new DOMException("MMseqs2 search was cancelled", "AbortError");
}

async function waitWithAbort(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted === true) throw abortError();
  await new Promise<void>((resolve, reject) => {
    const onAbort = (): void => { clearTimeout(timer); reject(abortError()); };
    const timer = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve(); }, milliseconds);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function parseTicket(value: TicketResponse): { status: string; id?: string } {
  const status = typeof value.status === "string" ? value.status.toUpperCase() : "ERROR";
  return { status, ...(typeof value.id === "string" && value.id !== "" ? { id: value.id } : {}) };
}

async function request(fetchImplementation: typeof fetch, url: URL, init: RequestInit, label: string): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const response = await fetchImplementation(url, init);
      if (response.ok) return response;
      if (![429, 500, 502, 503, 504].includes(response.status)) {
        lastError = new Error(`${label} failed: HTTP ${response.status}`);
        break;
      }
      lastError = new Error(`${label} failed: HTTP ${response.status}`);
    } catch (error) {
      if (init.signal?.aborted === true) throw abortError();
      lastError = error;
    }
    await waitWithAbort(500 * 2 ** attempt, init.signal ?? undefined);
  }
  throw lastError instanceof Error ? lastError : new Error(`${label} failed`);
}

function tarString(bytes: Uint8Array, start: number, length: number): string {
  const end = bytes.indexOf(0, start);
  return new TextDecoder().decode(bytes.subarray(start, end < 0 || end > start + length ? start + length : end));
}

function tarOctal(bytes: Uint8Array, start: number, length: number): number {
  const value = tarString(bytes, start, length).trim().replace(/\0/g, "");
  const parsed = Number.parseInt(value || "0", 8);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error("MMseqs2 result contains an invalid tar entry size");
  return parsed;
}

/** Reads regular files from the small POSIX tar archive returned by the ColabFold API. */
export function readTarFiles(bytes: Uint8Array): ReadonlyMap<string, Uint8Array> {
  const files = new Map<string, Uint8Array>();
  for (let offset = 0; offset + 512 <= bytes.byteLength;) {
    const header = bytes.subarray(offset, offset + 512);
    if (header.every((value) => value === 0)) break;
    const name = tarString(header, 0, 100);
    const prefix = tarString(header, 345, 155);
    const path = prefix === "" ? name : `${prefix}/${name}`;
    const size = tarOctal(header, 124, 12);
    const dataStart = offset + 512; const dataEnd = dataStart + size;
    if (path === "" || dataEnd > bytes.byteLength) throw new Error("MMseqs2 result contains a truncated tar entry");
    const type = header[156];
    if (type === 0 || type === 48) files.set(path.replace(/^\.\//, ""), bytes.slice(dataStart, dataEnd));
    offset = dataStart + Math.ceil(size / 512) * 512;
  }
  return files;
}

function queryBlock(contents: string, queryId: number): string {
  for (const block of contents.replace(/\r/g, "").split(/\0+/)) {
    const trimmed = block.trim();
    if (new RegExp(`^>${queryId}(?:\\s|$)`).test(trimmed)) return trimmed;
  }
  throw new Error(`MMseqs2 result does not contain query ${queryId}`);
}

function fileBySuffix(files: ReadonlyMap<string, Uint8Array>, suffix: string): Uint8Array | undefined {
  return [...files].find(([path]) => path === suffix || path.endsWith(`/${suffix}`))?.[1];
}

function extractQueryA3ms(
  tarBytes: Uint8Array,
  queryCount: number,
  useEnvironmental: boolean,
  paired: boolean,
): readonly string[] {
  const files = readTarFiles(tarBytes);
  if (paired) {
    const pair = fileBySuffix(files, "pair.a3m");
    if (pair === undefined) throw new Error("MMseqs2 pairing result is missing pair.a3m");
    const contents = new TextDecoder().decode(pair);
    return Array.from({ length: queryCount }, (_, index) => `${queryBlock(contents, QUERY_ID + index)}\n`);
  }
  const uniref = fileBySuffix(files, "uniref.a3m");
  if (uniref === undefined) throw new Error("MMseqs2 result is missing uniref.a3m");
  const unirefContents = new TextDecoder().decode(uniref);
  let environmentalContents: string | undefined;
  if (useEnvironmental) {
    const environmental = fileBySuffix(files, "bfd.mgnify30.metaeuk30.smag30.a3m");
    if (environmental === undefined) throw new Error("MMseqs2 result is missing the environmental A3M");
    environmentalContents = new TextDecoder().decode(environmental);
  }
  return Array.from({ length: queryCount }, (_, index) => {
    const blocks = [queryBlock(unirefContents, QUERY_ID + index)];
    if (environmentalContents !== undefined) blocks.push(queryBlock(environmentalContents, QUERY_ID + index));
    return `${blocks.join("\n")}\n`;
  });
}

/** Extracts and combines the UniRef and environmental A3Ms exactly as ColabFold does. */
export function extractMmseqs2A3m(tarBytes: Uint8Array, useEnvironmental = true): string {
  return extractQueryA3ms(tarBytes, 1, useEnvironmental, false)[0]!;
}

async function decompressGzip(archive: ArrayBuffer): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") throw new Error("This browser cannot decompress the MMseqs2 result");
  const stream = new Blob([archive]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

interface Mmseqs2JobResult { readonly archive: Uint8Array; readonly ticket: string; }

async function runMmseqs2Job(
  sequences: readonly string[],
  endpoint: "msa" | "pair",
  mode: string,
  search: "monomer" | "unpaired" | "paired",
  options: Mmseqs2MsaOptions,
): Promise<Mmseqs2JobResult> {
  const fetchImplementation = options.fetchImplementation ?? fetch;
  const wait = options.wait ?? waitWithAbort;
  const decompress = options.decompress ?? decompressGzip;
  const apiUrl = new URL(options.apiUrl ?? DEFAULT_API_URL); if (!apiUrl.pathname.endsWith("/")) apiUrl.pathname += "/";
  const start = performance.now();
  const report = (phase: Mmseqs2MsaPhase, status: string, ticket?: string): void => options.onProgress?.({
    phase, status, search, ...(ticket === undefined ? {} : { ticket }), elapsedMilliseconds: performance.now() - start,
  });
  const signal = options.signal;
  const query = sequences.map((sequence, index) => `>${QUERY_ID + index}\n${sequence}\n`).join("");
  const body = new URLSearchParams({ q: query, mode });
  let ticket: string | undefined;
  let status = "UNKNOWN";
  while (ticket === undefined) {
    report("submitting", "SUBMIT");
    const response = await request(fetchImplementation, new URL(`ticket/${endpoint}`, apiUrl), {
      method: "POST", body, ...(signal === undefined ? {} : { signal }),
      headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
    }, "MMseqs2 submission");
    const submitted = parseTicket(await response.json() as TicketResponse); status = submitted.status; ticket = submitted.id;
    if (TRANSIENT_SUBMISSION.has(status)) {
      ticket = undefined; report("retrying", status); await wait(5_000 + Math.floor(Math.random() * 5_000), signal); continue;
    }
    if (status === "ERROR") throw new Error("MMseqs2 rejected the sequence or is temporarily unavailable");
    if (status === "MAINTENANCE") throw new Error("The MMseqs2 API is undergoing maintenance; try again later");
    if (ticket === undefined) throw new Error(`MMseqs2 returned ${status} without a ticket`);
  }
  while (TRANSIENT_JOB.has(status)) {
    report(status === "RUNNING" ? "running" : "queued", status, ticket);
    await wait(5_000 + Math.floor(Math.random() * 5_000), signal);
    const response = await request(fetchImplementation, new URL(`ticket/${encodeURIComponent(ticket)}`, apiUrl),
      signal === undefined ? {} : { signal }, "MMseqs2 status");
    status = parseTicket(await response.json() as TicketResponse).status;
  }
  if (status !== "COMPLETE") throw new Error(`MMseqs2 search ended with status ${status}`);
  report("downloading", status, ticket);
  const response = await request(fetchImplementation,
    new URL(`result/download/${encodeURIComponent(ticket)}`, apiUrl),
    signal === undefined ? {} : { signal }, "MMseqs2 result download");
  const archive = await decompress(await response.arrayBuffer());
  report("complete", status, ticket);
  return { archive, ticket };
}

/** Generates a monomer MSA through the public ColabFold MMseqs2 API. */
export async function generateMmseqs2Msa(sequenceValue: string,
  options: Mmseqs2MsaOptions = {}): Promise<Mmseqs2MsaResult> {
  const sequence = normalizedSequence(sequenceValue);
  const useEnvironmental = options.useEnvironmental ?? true;
  const start = performance.now();
  const job = await runMmseqs2Job(
    [sequence], "msa", useEnvironmental ? "env" : "all", "monomer", options,
  );
  const a3m = extractMmseqs2A3m(job.archive, useEnvironmental);
  const alignment = parseA3m(a3m);
  if (alignment.query !== sequence) throw new Error("MMseqs2 returned an A3M for a different query sequence");
  const elapsedMilliseconds = performance.now() - start;
  return { a3m, ticket: job.ticket, depth: alignment.depth, elapsedMilliseconds };
}

interface ComplexMsaAssembly {
  readonly a3m: string;
  readonly mask: Float32Array;
  readonly depth: number;
}

function serializedRow(sequence: string, deletions: readonly number[]): string {
  if (sequence.length !== deletions.length) throw new RangeError("MSA sequence and deletion row lengths differ");
  let output = "";
  for (let index = 0; index < sequence.length; index += 1) {
    const deletion = deletions[index]!;
    if (!Number.isSafeInteger(deletion) || deletion < 0) throw new RangeError("MSA deletion counts must be non-negative");
    output += "a".repeat(deletion) + sequence[index]!;
  }
  return output;
}

/** Merge per-entity paired/unpaired A3Ms using AlphaFold-Multimer's dense/block-diagonal layout. */
export function assembleComplexA3m(
  chainsValue: readonly string[],
  uniqueSequences: readonly string[],
  unpairedA3ms: readonly string[],
  pairedA3ms?: readonly string[],
): ComplexMsaAssembly {
  const chains = chainsValue.map(normalizedSequence);
  if (chains.length < 2) throw new RangeError("complex MSA assembly requires at least two chains");
  if (uniqueSequences.length === 0 || unpairedA3ms.length !== uniqueSequences.length
    || (pairedA3ms !== undefined && pairedA3ms.length !== uniqueSequences.length)) {
    throw new RangeError("complex MSA entity arrays have inconsistent lengths");
  }
  const entityForChain = chains.map((chain) => uniqueSequences.indexOf(chain));
  if (entityForChain.some((entity) => entity < 0)) throw new Error("a complex chain is absent from the unique entities");
  const unpaired = unpairedA3ms.map(parseA3m);
  const paired = pairedA3ms?.map(parseA3m);
  for (let entity = 0; entity < uniqueSequences.length; entity += 1) {
    if (unpaired[entity]!.query !== uniqueSequences[entity]
      || (paired !== undefined && paired[entity]!.query !== uniqueSequences[entity])) {
      throw new Error("MMseqs2 returned an A3M for a different complex chain");
    }
  }
  if (paired !== undefined && !paired.every((alignment) => alignment.depth === paired[0]!.depth)) {
    throw new Error("paired MMseqs2 A3Ms do not contain aligned row counts");
  }
  const length = chains.reduce((sum, chain) => sum + chain.length, 0);
  const sequences: string[] = [];
  const deletions: number[][] = [];
  const masks: number[][] = [];
  const descriptions: string[] = [];
  const append = (description: string, sequenceParts: readonly string[], deletionParts: readonly (readonly number[])[],
    maskParts: readonly (readonly number[])[]): void => {
    const sequence = sequenceParts.join("");
    const deletion = deletionParts.flat();
    const mask = maskParts.flat();
    if (sequence.length !== length || deletion.length !== length || mask.length !== length) {
      throw new RangeError("assembled complex MSA row has the wrong total length");
    }
    descriptions.push(description); sequences.push(sequence); deletions.push(deletion); masks.push(mask);
  };

  const pairedSequenceSets = uniqueSequences.map(() => new Set<string>());
  if (paired !== undefined) {
    for (let row = 0; row < paired[0]!.depth; row += 1) {
      for (let entity = 0; entity < uniqueSequences.length; entity += 1) {
        pairedSequenceSets[entity]!.add(paired[entity]!.sequences[row]!);
      }
      append(`paired_${row}`,
        entityForChain.map((entity) => paired[entity]!.sequences[row]!),
        entityForChain.map((entity) => paired[entity]!.deletionMatrix[row]!),
        chains.map((chain) => new Array(chain.length).fill(1)));
    }
  }

  for (let entity = 0; entity < uniqueSequences.length; entity += 1) {
    const alignment = unpaired[entity]!;
    for (let row = 0; row < alignment.depth; row += 1) {
      if (paired !== undefined && pairedSequenceSets[entity]!.has(alignment.sequences[row]!)) continue;
      append(`unpaired_${entity}_${row}`,
        chains.map((chain, chainIndex) => entityForChain[chainIndex] === entity
          ? alignment.sequences[row]! : "-".repeat(chain.length)),
        chains.map((chain, chainIndex) => entityForChain[chainIndex] === entity
          ? alignment.deletionMatrix[row]! : new Array(chain.length).fill(0)),
        chains.map((chain, chainIndex) => new Array(chain.length).fill(entityForChain[chainIndex] === entity ? 1 : 0)));
    }
  }
  if (sequences.length === 0 || sequences[0] !== chains.join("")) {
    throw new Error("assembled complex MSA has no complete query row");
  }
  const a3m = descriptions.map((description, row) =>
    `>${description}\n${serializedRow(sequences[row]!, deletions[row]!)}\n`).join("");
  return { a3m, mask: Float32Array.from(masks.flat()), depth: sequences.length };
}

/** Generate ColabFold-compatible unpaired and greedy-paired MSAs for a complex. */
export async function generateMmseqs2ComplexMsa(
  chainsValue: readonly string[],
  options: Mmseqs2ComplexMsaOptions = {},
): Promise<Mmseqs2ComplexMsaResult> {
  const chains = chainsValue.map(normalizedSequence);
  if (chains.length < 2) throw new RangeError("complex MMseqs2 search requires at least two chains");
  const uniqueSequences = [...new Set(chains)];
  const useEnvironmental = options.useEnvironmental ?? true;
  const pairingStrategy = options.pairingStrategy ?? "greedy";
  const start = performance.now();
  const unpairedPromise = runMmseqs2Job(
    uniqueSequences, "msa", useEnvironmental ? "env" : "all", "unpaired", options,
  );
  const pairedPromise = uniqueSequences.length > 1
    ? runMmseqs2Job(uniqueSequences, "pair", `pair${pairingStrategy}`, "paired", options)
    : undefined;
  const [unpairedJob, pairedJob] = await Promise.all([unpairedPromise, pairedPromise]);
  const unpairedA3ms = extractQueryA3ms(unpairedJob.archive, uniqueSequences.length, useEnvironmental, false);
  const pairedA3ms = pairedJob === undefined ? undefined
    : extractQueryA3ms(pairedJob.archive, uniqueSequences.length, false, true);
  const assembled = assembleComplexA3m(chains, uniqueSequences, unpairedA3ms, pairedA3ms);
  return {
    ...assembled,
    unpairedTicket: unpairedJob.ticket,
    ...(pairedJob === undefined ? {} : { pairedTicket: pairedJob.ticket }),
    elapsedMilliseconds: performance.now() - start,
  };
}
