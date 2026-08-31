import { describe, expect, it } from "vitest";
import {
  jaxFoldIn, jaxPaddingConsistentUniform, jaxSplit, multimerMsaKeys, type JaxKey,
} from "../src/input/jax-prng.js";

describe("JAX Threefry2x32 compatibility", () => {
  it("matches JAX partitionable split and fold_in", () => {
    const root: JaxKey = [0, 0];
    expect(jaxSplit(root, 2)).toEqual([
      [1797259609, 2579123966], [928981903, 3453687069],
    ]);
    expect(jaxFoldIn(root, 1)).toEqual([928981903, 3453687069]);
  });

  it("matches AlphaFold Multimer sampling keys and scalar uniforms", () => {
    const keys = multimerMsaKeys([0, 0]);
    expect(keys.nextRoot).toEqual([1797259609, 2579123966]);
    expect(keys.sample).toEqual([3764384500, 1946563400]);
    expect(keys.maskPosition).toEqual([4222303052, 2537446787]);
    expect(keys.maskGumbel).toEqual([887112546, 2608837638]);
    expect(Array.from({ length: 6 }, (_, index) =>
      jaxPaddingConsistentUniform(keys.sample, [index]))).toEqual([
      0.38760459423065186, 0.5710806846618652, 0.3175692558288574,
      0.7691905498504639, 0.0032564401626586914, 0.11287987232208252,
    ]);
    expect(jaxPaddingConsistentUniform(keys.maskPosition, [0, 0])).toBe(0.17804670333862305);
  });
});
