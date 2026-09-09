import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

/**
 * No backticks in comments.
 *
 * Inside a WGSL template a backtick ends the string, and TypeScript reports
 * that as TS1005: ',' expected at the column after it, which says nothing
 * about the cause. Quoting a name the markdown way is a habit that costs a
 * diagnosis every time it lands in a shader, and buys nothing anywhere else,
 * so the rule is the same everywhere and there is nothing to remember.
 *
 * Finding comments needs a scanner rather than a regex: a line starting with
 * an asterisk is usually a doc comment but is sometimes a multiplication
 * continuing an expression, and stripping that line's backtick once broke the
 * attention shader. So strings, templates with their nested substitutions, and
 * regex literals are all skipped properly.
 */

const ROOTS = ["src", "tools", "web", "test"];

function sourceFiles(directory: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) found.push(...sourceFiles(path));
    else if (entry.endsWith(".ts") && !entry.endsWith(".d.ts")) found.push(path);
  }
  return found;
}

/** Half-open [start, end) spans of every comment in a TypeScript source. */
export function commentSpans(text: string): [number, number][] {
  const spans: [number, number][] = [];
  const length = text.length;
  // Template nesting: -1 is template text, anything else is the brace depth
  // inside a substitution, where code (and so comments) can appear again.
  const nesting: number[] = [];
  let index = 0;
  let previous = "";
  const lineComment = (): void => {
    const end = text.indexOf("\n", index);
    spans.push([index, end < 0 ? length : end]);
    index = end < 0 ? length : end;
  };
  const blockComment = (): void => {
    const end = text.indexOf("*/", index + 2);
    spans.push([index, end < 0 ? length : end + 2]);
    index = end < 0 ? length : end + 2;
  };
  const quoted = (quote: string): void => {
    index += 1;
    while (index < length && text[index] !== quote) index += text[index] === "\\" ? 2 : 1;
    index += 1;
  };
  while (index < length) {
    const character = text[index]!;
    if (character === "/" && text[index + 1] === "/") { lineComment(); continue; }
    if (character === "/" && text[index + 1] === "*") { blockComment(); continue; }
    if (character === "/" && !")]}".includes(previous) && !/[\w$]/.test(previous)) {
      index += 1; // a regular expression, which may hold slashes and quotes
      while (index < length && text[index] !== "/") {
        if (text[index] === "\\") index += 1;
        else if (text[index] === "[") {
          while (index < length && text[index] !== "]") index += text[index] === "\\" ? 2 : 1;
        } else if (text[index] === "\n") break;
        index += 1;
      }
      index += 1; previous = "/"; continue;
    }
    if (character === "'" || character === '"') { quoted(character); previous = character; continue; }
    if (character === "`") {
      nesting.push(-1); index += 1;
      while (index < length && nesting.length > 0) {
        const inner = text[index]!;
        const depth = nesting[nesting.length - 1]!;
        if (depth === -1) {
          if (inner === "\\") { index += 2; continue; }
          if (inner === "`") { nesting.pop(); index += 1; continue; }
          if (inner === "$" && text[index + 1] === "{") { nesting.push(0); index += 2; continue; }
          index += 1; continue;
        }
        if (inner === "{") { nesting[nesting.length - 1] = depth + 1; index += 1; continue; }
        if (inner === "}") {
          if (depth === 0) nesting.pop(); else nesting[nesting.length - 1] = depth - 1;
          index += 1; continue;
        }
        if (inner === "`") { nesting.push(-1); index += 1; continue; }
        if (inner === "'" || inner === '"') { quoted(inner); continue; }
        if (inner === "/" && text[index + 1] === "/") { lineComment(); continue; }
        if (inner === "/" && text[index + 1] === "*") { blockComment(); continue; }
        index += 1;
      }
      previous = "`"; continue;
    }
    if (!/\s/.test(character)) previous = character;
    index += 1;
  }
  return spans;
}

describe("comments", () => {
  it("hold no backtick anywhere in the tree", () => {
    const offences: string[] = [];
    for (const path of ROOTS.flatMap(sourceFiles)) {
      const text = readFileSync(path, "utf8");
      if (!text.includes("`")) continue;
      for (const [start, end] of commentSpans(text)) {
        const comment = text.slice(start, end);
        if (!comment.includes("`")) continue;
        const line = text.slice(0, start).split("\n").length;
        offences.push(`${path}:${line} ${comment.split("\n")[0]!.trim()}`);
      }
    }
    expect(offences.join("\n")).toBe("");
  });

  it("finds the backtick that ends a shader early", () => {
    const source = [
      "const SHADER = `",
      "@compute @workgroup_size(64)",
      "fn main() {}",
      "`;",
      "// See `clusteringOverrides` for the shape.",
    ].join("\n");
    const found = commentSpans(source).map(([a, b]) => source.slice(a, b));
    expect(found).toEqual(["// See `clusteringOverrides` for the shape."]);
  });

  it("does not mistake a multiplication for a doc comment", () => {
    // The shape that broke the attention shader: a continuation line opening
    // with an asterisk, ending the template on the same line.
    const source = "const s = `x`\n      * weights[a] + weights[b]`;\n";
    expect(commentSpans(source)).toEqual([]);
  });

  it("skips backticks inside strings, templates and regexes", () => {
    const source = "const a = \"`\"; const b = `${\"`\"}`; const c = /[`]/; // plain\n";
    expect(commentSpans(source).map(([a, b]) => source.slice(a, b))).toEqual(["// plain"]);
  });
});
