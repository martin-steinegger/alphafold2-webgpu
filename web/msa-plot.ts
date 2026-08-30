import { parseA3m } from "../src/input/a3m.js";

export interface MsaCoverageData {
  readonly sequences: readonly string[];
  readonly identities: Float32Array;
  readonly order: Uint32Array;
  readonly coverage: Uint32Array;
  readonly depth: number;
  readonly length: number;
}

export function analyzeMsa(a3m: string): MsaCoverageData {
  const alignment = parseA3m(a3m); const { sequences, depth, length } = alignment;
  const identities = new Float32Array(depth); const coverage = new Uint32Array(length);
  for (let row = 0; row < depth; row += 1) {
    let matches = 0; const sequence = sequences[row]!;
    for (let position = 0; position < length; position += 1) {
      if (sequence[position] !== "-") coverage[position] = coverage[position]! + 1;
      if (sequence[position] === sequences[0]![position]) matches += 1;
    }
    identities[row] = matches / length;
  }
  const sorted = Array.from({ length: depth }, (_, index) => index)
    .sort((left, right) => identities[left]! - identities[right]! || left - right);
  return { sequences, identities, order: Uint32Array.from(sorted), coverage, depth, length };
}

function identityColor(value: number): readonly [number, number, number] {
  const anchors = [
    [222, 67, 63], [244, 153, 54], [237, 214, 72], [55, 183, 180], [45, 81, 190],
  ] as const;
  const scaled = Math.max(0, Math.min(1, value)) * (anchors.length - 1);
  const left = Math.min(anchors.length - 2, Math.floor(scaled)); const fraction = scaled - left;
  const interpolate = (channel: 0 | 1 | 2): number => Math.round(
    anchors[left]![channel] + (anchors[left + 1]![channel] - anchors[left]![channel]) * fraction,
  );
  return [interpolate(0), interpolate(1), interpolate(2)];
}

export function drawMsaCoverage(canvas: HTMLCanvasElement, a3m: string): MsaCoverageData {
  const data = analyzeMsa(a3m); const context = canvas.getContext("2d");
  if (context === null) return data;
  const { width, height } = canvas; const left = 56; const right = 82; const top = 18; const bottom = 38;
  const plotWidth = width - left - right; const plotHeight = height - top - bottom;
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);

  const sampledRows = Math.min(data.depth, 2_048);
  const image = new ImageData(data.length, sampledRows);
  image.data.fill(255);
  for (let outputRow = 0; outputRow < sampledRows; outputRow += 1) {
    const rank = sampledRows === 1 ? data.depth - 1
      : Math.round((sampledRows - 1 - outputRow) / (sampledRows - 1) * (data.depth - 1));
    const source = data.order[rank]!; const identity = data.identities[source]!; const color = identityColor(identity);
    for (let position = 0; position < data.length; position += 1) {
      if (data.sequences[source]![position] === "-") continue;
      const pixel = (outputRow * data.length + position) * 4;
      image.data[pixel] = color[0]; image.data[pixel + 1] = color[1]; image.data[pixel + 2] = color[2];
    }
  }
  const temporary = document.createElement("canvas"); temporary.width = data.length; temporary.height = sampledRows;
  temporary.getContext("2d")?.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false; context.drawImage(temporary, left, top, plotWidth, plotHeight);

  context.strokeStyle = "#111"; context.lineWidth = 1.5; context.beginPath();
  for (let position = 0; position < data.length; position += 1) {
    const x = left + (position + .5) / data.length * plotWidth;
    const y = top + (1 - data.coverage[position]! / data.depth) * plotHeight;
    if (position === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }
  context.stroke(); context.strokeStyle = "#777"; context.lineWidth = 1; context.strokeRect(left, top, plotWidth, plotHeight);
  context.fillStyle = "#666"; context.font = "11px Roboto Mono"; context.textAlign = "center";
  context.fillText("Positions", left + plotWidth / 2, height - 8);
  context.save(); context.translate(13, top + plotHeight / 2); context.rotate(-Math.PI / 2);
  context.fillText("Sequences", 0, 0); context.restore();
  context.textAlign = "right"; context.fillText(String(data.depth), left - 7, top + 5); context.fillText("0", left - 7, top + plotHeight);
  context.textAlign = "center"; context.fillText("1", left, height - 22); context.fillText(String(data.length), left + plotWidth, height - 22);

  const colorX = width - 52; const colorWidth = 13;
  for (let pixel = 0; pixel < plotHeight; pixel += 1) {
    const identity = 1 - pixel / Math.max(1, plotHeight - 1); const color = identityColor(identity);
    context.fillStyle = `rgb(${color.join(",")})`; context.fillRect(colorX, top + pixel, colorWidth, 1);
  }
  context.strokeStyle = "#777"; context.strokeRect(colorX, top, colorWidth, plotHeight);
  context.fillStyle = "#666"; context.textAlign = "left";
  context.fillText("100%", colorX + 18, top + 5); context.fillText("50%", colorX + 18, top + plotHeight / 2 + 4);
  context.fillText("0%", colorX + 18, top + plotHeight);
  return data;
}
