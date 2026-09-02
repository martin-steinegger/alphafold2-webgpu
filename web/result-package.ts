/**
 * Packages a finished prediction the way ColabFold does: one archive holding
 * the structure, the confidence scores, the plots, the alignment, the settings
 * that produced them and the citations for the methods involved.
 *
 * The archive is written here rather than by a library so the page keeps its
 * "no third-party code at run time" property. Entries are deflated with the
 * platform's own `CompressionStream` where it exists and stored verbatim
 * where it does not, which every unzip implementation reads either way.
 *
 * Nothing in this module touches the DOM, so the page's canvases are rendered
 * to PNG by the caller and handed over as bytes.
 */

/** One file in the archive. Directories are implied by slashes in `name`. */
export interface ZipEntry {
  readonly name: string;
  readonly data: Uint8Array | string;
  /** Already-compressed payloads (PNG) gain nothing from a second pass. */
  readonly store?: boolean;
}

const encoder = new TextEncoder();

/**
 * `BlobPart` is typed against a non-shared buffer, which a `Uint8Array` only
 * promises in its type parameter; every array here owns an ordinary buffer.
 */
const blobPart = (bytes: Uint8Array): BlobPart => bytes as unknown as BlobPart;

let crcTable: Uint32Array | undefined;

function crc32(data: Uint8Array): number {
  if (crcTable === undefined) {
    crcTable = new Uint32Array(256);
    for (let index = 0; index < 256; index += 1) {
      let value = index;
      for (let bit = 0; bit < 8; bit += 1) value = (value & 1) === 1 ? (value >>> 1) ^ 0xedb88320 : value >>> 1;
      crcTable[index] = value >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let index = 0; index < data.length; index += 1) {
    crc = (crc >>> 8) ^ crcTable[(crc ^ data[index]!) & 0xff]!;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

async function deflateRaw(data: Uint8Array): Promise<Uint8Array | undefined> {
  if (typeof CompressionStream === "undefined" || data.length === 0) return undefined;
  try {
    const stream = new Blob([blobPart(data)]).stream().pipeThrough(new CompressionStream("deflate-raw"));
    const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
    return compressed.length < data.length ? compressed : undefined;
  } catch {
    // A browser without "deflate-raw" stores the entry instead.
    return undefined;
  }
}

function dosDateTime(date: Date): { readonly time: number; readonly date: number } {
  const year = Math.max(1980, date.getFullYear());
  return {
    time: (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1),
    date: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
  };
}

function header(fields: readonly (readonly [number, number])[]): Uint8Array {
  const size = fields.reduce((sum, [, width]) => sum + width, 0);
  const bytes = new Uint8Array(size);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const [value, width] of fields) {
    if (width === 2) view.setUint16(offset, value, true); else view.setUint32(offset, value, true);
    offset += width;
  }
  return bytes;
}

/**
 * Writes the entries as a single ZIP archive.
 *
 * Sizes and checksums are known before anything is written, so the archive
 * needs no data descriptors and stays readable by streaming readers.
 */
export async function zipArchive(entries: readonly ZipEntry[], now: Date = new Date()): Promise<Blob> {
  const { time, date } = dosDateTime(now);
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = encoder.encode(entry.name);
    const data = typeof entry.data === "string" ? encoder.encode(entry.data) : entry.data;
    const compressed = entry.store === true ? undefined : await deflateRaw(data);
    const payload = compressed ?? data;
    const method = compressed === undefined ? 0 : 8;
    const crc = crc32(data);
    // Bit 11 marks the name as UTF-8; version 20 covers deflate and stored entries.
    const common = [[20, 2], [0x0800, 2], [method, 2], [time, 2], [date, 2],
      [crc, 4], [payload.length, 4], [data.length, 4], [name.length, 2]] as const;
    parts.push(header([[0x04034b50, 4], ...common, [0, 2]]), name, payload);
    // Extra, comment and disk fields stay empty; the attribute fields default
    // to a regular file, and the last word points at the local header.
    central.push(header([[0x02014b50, 4], [20, 2], ...common,
      [0, 2], [0, 2], [0, 2], [0, 2], [0, 4], [offset, 4]]), name);
    offset += 30 + name.length + payload.length;
  }
  const centralSize = central.reduce((sum, entry) => sum + entry.length, 0);
  const end = header([[0x06054b50, 4], [0, 2], [0, 2], [entries.length, 2], [entries.length, 2],
    [centralSize, 4], [offset, 4], [0, 2]]);
  return new Blob([...parts, ...central, end].map(blobPart), { type: "application/zip" });
}

/** The confidence figures the archive reports, whichever model produced them. */
export interface PackagedConfidence {
  readonly plddt: Float32Array;
  readonly meanPlddt: number;
  readonly ptm: number;
  readonly iptm?: number;
  readonly predictedAlignedError: Float32Array;
  readonly maxPredictedAlignedError: number;
}

export interface ResultPackage {
  readonly jobName: string;
  readonly sequence: string;
  readonly chainLengths: readonly number[];
  readonly confidence: PackagedConfidence;
  readonly pdb: string;
  readonly scoresJson: string;
  readonly a3m: string;
  readonly depth: number;
  /** Rendered plots, keyed by the ColabFold suffix they carry (`plddt`, `pae`, `coverage`). */
  readonly images: readonly { readonly suffix: string; readonly png: Uint8Array }[];
  readonly settings: Readonly<Record<string, unknown>>;
  readonly log: string;
  /** Whether the alignment came from the MMseqs2 server, which changes the citations. */
  readonly usedMmseqs2: boolean;
  readonly multimer: boolean;
}

const round = (value: number, decimals: number): number => Number(value.toFixed(decimals));

/** The alignment-error matrix in AlphaFold-DB's published shape, as ColabFold writes it. */
export function predictedAlignedErrorJson(confidence: PackagedConfidence, length: number): string {
  const rows: number[][] = [];
  for (let row = 0; row < length; row += 1) {
    const values = new Array<number>(length);
    for (let column = 0; column < length; column += 1) {
      values[column] = round(confidence.predictedAlignedError[row * length + column]!, 2);
    }
    rows.push(values);
  }
  return JSON.stringify([{
    predicted_aligned_error: rows,
    max_predicted_aligned_error: round(confidence.maxPredictedAlignedError, 2),
  }]);
}

const ALPHAFOLD_CITATION = `@article{jumper2021highly,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael
    and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\\v{Z}}{\\'i}dek, Augustin
    and Potapenko, Anna and others},
  journal = {Nature},
  title = {Highly accurate protein structure prediction with {AlphaFold}},
  year = {2021},
  volume = {596},
  pages = {583--589},
  doi = {10.1038/s41586-021-03819-2}
}`;

const MULTIMER_CITATION = `@article{evans2021protein,
  author = {Evans, Richard and O'Neill, Michael and Pritzel, Alexander and Antropova, Natasha
    and Senior, Andrew and Green, Tim and {\\v{Z}}{\\'i}dek, Augustin and Bates, Russ and Blackwell, Sam
    and Yim, Jason and others},
  journal = {bioRxiv},
  title = {Protein complex prediction with {AlphaFold-Multimer}},
  year = {2021},
  doi = {10.1101/2021.10.04.463034}
}`;

const COLABFOLD_CITATION = `@article{mirdita2022colabfold,
  author = {Mirdita, Milot and Sch{\\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim
    and Ovchinnikov, Sergey and Steinegger, Martin},
  journal = {Nature Methods},
  title = {{ColabFold}: making protein folding accessible to all},
  year = {2022},
  volume = {19},
  pages = {679--682},
  doi = {10.1038/s41592-022-01488-1}
}`;

const MMSEQS2_CITATION = `@article{steinegger2017mmseqs2,
  author = {Steinegger, Martin and S{\\"o}ding, Johannes},
  journal = {Nature Biotechnology},
  title = {{MMseqs2} enables sensitive protein sequence searching for the analysis of massive data sets},
  year = {2017},
  volume = {35},
  pages = {1026--1028},
  doi = {10.1038/nbt.3988}
}`;

/** BibTeX for every method that contributed to this particular prediction. */
export function citations(options: { readonly multimer: boolean; readonly usedMmseqs2: boolean }): string {
  const entries = [ALPHAFOLD_CITATION];
  if (options.multimer) entries.push(MULTIMER_CITATION);
  if (options.usedMmseqs2) entries.push(COLABFOLD_CITATION, MMSEQS2_CITATION);
  return `${entries.join("\n\n")}\n`;
}

function queriesCsv(jobName: string, chainLengths: readonly number[], sequence: string): string {
  const chains: string[] = [];
  let start = 0;
  for (const length of chainLengths) { chains.push(sequence.slice(start, start + length)); start += length; }
  return `id,sequence\n${jobName},${chains.join(":")}\n`;
}

/**
 * Lays the prediction out as archive entries, under a folder named after the
 * job so unzipping in a downloads directory leaves one tidy result behind.
 */
export function resultPackageEntries(result: ResultPackage): ZipEntry[] {
  const folder = `${result.jobName}/`;
  const entries: ZipEntry[] = [
    { name: `${folder}${result.jobName}_unrelaxed_model_1.pdb`, data: result.pdb },
    { name: `${folder}${result.jobName}_scores.json`, data: result.scoresJson },
    {
      name: `${folder}${result.jobName}_predicted_aligned_error_v1.json`,
      data: predictedAlignedErrorJson(result.confidence, result.sequence.length),
    },
  ];
  for (const image of result.images) {
    entries.push({ name: `${folder}${result.jobName}_${image.suffix}.png`, data: image.png, store: true });
  }
  entries.push(
    { name: `${folder}${result.jobName}.a3m`, data: result.a3m },
    { name: `${folder}${result.jobName}.csv`, data: queriesCsv(result.jobName, result.chainLengths, result.sequence) },
    { name: `${folder}config.json`, data: `${JSON.stringify(result.settings, null, 2)}\n` },
    { name: `${folder}log.txt`, data: result.log.endsWith("\n") ? result.log : `${result.log}\n` },
    { name: `${folder}cite.bib`, data: citations(result) },
  );
  return entries;
}

/** Builds the downloadable archive for a finished prediction. */
export async function packageResults(result: ResultPackage): Promise<Blob> {
  return zipArchive(resultPackageEntries(result));
}
