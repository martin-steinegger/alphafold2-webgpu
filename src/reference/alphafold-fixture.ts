import type { AttentionWeights } from "../evoformer/attention.js";
import type {
  AttentionModuleWeights,
  EvoformerBlockWeights,
  EvoformerPairBlockWeights,
  ExtraMsaBlockWeights,
  RowAttentionModuleWeights,
  TemplatePairBlockWeights,
  TriangleAttentionModuleWeights,
} from "../evoformer/block.js";
import type { InputEmbedderWeights } from "../evoformer/input-embedder.js";
import type { OuterProductMeanWeights } from "../evoformer/outer-product-mean.js";
import type { QueryOnlyTemplateWeights } from "../evoformer/template.js";
import type { MultimerMockTemplateWeights } from "../evoformer/multimer-template.js";
import type { TransitionWeights } from "../evoformer/transition.js";
import type { TriangleMultiplicationWeights } from "../triangle/types.js";
import type { StructureModuleWeights } from "../structure/module.js";
import {
  adaptMultimerInvariantPointAttentionWeights,
  type MultimerInvariantPointAttentionWeights,
} from "../structure/ipa.js";
import type { ResidueGeometryTables } from "../structure/geometry.js";
import type { PredictedAlignedErrorWeights, PredictedLddtWeights } from "../heads/confidence.js";
import type { QueryOnlyFeatureTables } from "../input/query-only-features.js";
import type { BinaryTensorManifest } from "./tensor-store.js";

export interface TensorStore {
  readonly manifest: BinaryTensorManifest;
  tensor(name: string): Promise<Float32Array>;
  shape(name: string): readonly number[];
  /** Retain the shard holding `name` in its stored form so `read` can decode from it synchronously. */
  ensureLoaded(name: string): Promise<void>;
  /** Decode `name`, or block `block` of its `blocks` stacked blocks, from a retained shard. */
  read(name: string, block?: number, blocks?: number): Float32Array;
}

/**
 * A weight object whose tensors are decoded from the stored shards on every
 * access. The Evoformer stacks are 94% of the model; holding them decoded
 * would keep a float32 copy of the whole model on the host for the session,
 * where a compressed model is a quarter of that. A block's tensors are read
 * once when its weights are packed for upload, so the float32 values exist
 * only for that moment. The getters are enumerable, and objects are composed
 * with `mergeLazy` rather than spread so they never materialize by accident.
 */
function lazyWeights<T extends object>(fields: { readonly [K in keyof T]: () => T[K] }): T {
  const target = {} as T;
  for (const key of Object.keys(fields) as (keyof T)[]) {
    Object.defineProperty(target, key, { get: fields[key], enumerable: true });
  }
  return target;
}

function mergeLazy<A extends object, B extends object>(first: A, second: B): A & B {
  const target = {} as A & B;
  Object.defineProperties(target, Object.getOwnPropertyDescriptors(first));
  Object.defineProperties(target, Object.getOwnPropertyDescriptors(second));
  return target;
}

type ParameterMap = Readonly<Record<string, Readonly<Record<string, string>>>>;

interface ModelManifest {
  readonly evoformerStack: { readonly blocks: number; readonly parameters: ParameterMap };
  readonly extraMsaStack: { readonly blocks: number; readonly parameters: ParameterMap };
  readonly embedding: { readonly parameters: ParameterMap };
  readonly templateEmbedding: { readonly parameters: ParameterMap };
  readonly multimerTemplate?: { readonly templates: number; readonly parameters: ParameterMap };
  readonly structureModule: { readonly parameters: ParameterMap };
  readonly confidenceHeads: {
    readonly parameters: { readonly predictedLddt: ParameterMap; readonly predictedAlignedError: ParameterMap };
  };
}

function transpose(input: Float32Array, rows: number, columns: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column]!;
    }
  }
  return output;
}

export class AlphaFoldFixture {
  readonly store: TensorStore;
  readonly manifest: ModelManifest;

  private constructor(store: TensorStore) {
    this.store = store;
    this.manifest = store.manifest as unknown as ModelManifest;
  }

  static fromStore(store: TensorStore): AlphaFoldFixture { return new AlphaFoldFixture(store); }

  tensor(name: string): Promise<Float32Array> { return this.store.tensor(name); }
  shape(name: string): readonly number[] { return this.store.shape(name); }

  async #parameter(
    parameters: ParameterMap,
    module: string,
    name: string,
    block?: number,
    blocks?: number,
  ): Promise<Float32Array> {
    const tensorName = parameters[module]?.[name];
    if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
    const value = await this.store.tensor(tensorName);
    if (block === undefined) return value;
    if (blocks === undefined) throw new Error("stacked parameter requires a block count");
    const size = value.length / blocks;
    return value.subarray(block * size, (block + 1) * size);
  }

  /** A thunk decoding block `block` of a stacked parameter from its retained shard. */
  async #lazyParameter(
    parameters: ParameterMap, module: string, name: string, block: number, blocks: number,
  ): Promise<() => Float32Array> {
    const tensorName = parameters[module]?.[name];
    if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
    await this.store.ensureLoaded(tensorName);
    return () => this.store.read(tensorName, block, blocks);
  }

  #parameterShape(parameters: ParameterMap, module: string, name: string, stacked: boolean): readonly number[] {
    const tensorName = parameters[module]?.[name];
    if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
    const shape = this.store.shape(tensorName);
    return stacked ? shape.slice(1) : shape;
  }

  async #attention(
    parameters: ParameterMap, root: string, block: number, blocks: number,
  ): Promise<AttentionModuleWeights> {
    const parameter = (module: string, name: string): Promise<() => Float32Array> =>
      this.#lazyParameter(parameters, module, name, block, blocks);
    const attentionRoot = `${root}/attention`;
    const weights = lazyWeights<AttentionWeights>({
      queryNormScale: await parameter(`${root}/query_norm`, "scale"),
      queryNormOffset: await parameter(`${root}/query_norm`, "offset"),
      queryWeight: await parameter(attentionRoot, "query_w"),
      keyWeight: await parameter(attentionRoot, "key_w"),
      valueWeight: await parameter(attentionRoot, "value_w"),
      gatingWeight: await parameter(attentionRoot, "gating_w"),
      gatingBias: await parameter(attentionRoot, "gating_b"),
      outputWeight: await parameter(attentionRoot, "output_w"),
      outputBias: await parameter(attentionRoot, "output_b"),
    });
    return {
      heads: this.#parameterShape(parameters, attentionRoot, "gating_b", true)[0]!,
      attention: weights,
    };
  }

  async #triangleAttention(
    parameters: ParameterMap, root: string, block: number, blocks: number,
  ): Promise<TriangleAttentionModuleWeights> {
    const result = await this.#attention(parameters, root, block, blocks);
    return mergeLazy(result, lazyWeights<Pick<TriangleAttentionModuleWeights, "pairProjectionWeight">>({
      pairProjectionWeight: await this.#lazyParameter(parameters, root, "feat_2d_weights", block, blocks),
    }));
  }

  async #transition(
    parameters: ParameterMap, root: string, block: number, blocks: number,
  ): Promise<TransitionWeights> {
    const parameter = (module: string, name: string): Promise<() => Float32Array> =>
      this.#lazyParameter(parameters, module, name, block, blocks);
    return lazyWeights<TransitionWeights>({
      layerNormScale: await parameter(`${root}/input_layer_norm`, "scale"),
      layerNormOffset: await parameter(`${root}/input_layer_norm`, "offset"),
      firstWeight: await parameter(`${root}/transition1`, "weights"),
      firstBias: await parameter(`${root}/transition1`, "bias"),
      secondWeight: await parameter(`${root}/transition2`, "weights"),
      secondBias: await parameter(`${root}/transition2`, "bias"),
    });
  }

  async #triangle(
    parameters: ParameterMap, root: string, channels: number, block: number, blocks: number,
  ): Promise<TriangleMultiplicationWeights> {
    const parameter = (module: string, name: string): Promise<() => Float32Array> =>
      this.#lazyParameter(parameters, module, name, block, blocks);
    const hidden = this.#parameterShape(parameters, `${root}/left_projection`, "bias", true)[0]!;
    const projection = async (
      module: string, inputChannels: number, outputChannels: number,
    ): Promise<() => Float32Array> => {
      const stored = await parameter(`${root}/${module}`, "weights");
      return () => transpose(stored(), inputChannels, outputChannels);
    };
    return lazyWeights<TriangleMultiplicationWeights>({
      layerNormInWeight: await parameter(`${root}/layer_norm_input`, "scale"),
      layerNormInBias: await parameter(`${root}/layer_norm_input`, "offset"),
      linearAPWeight: await projection("left_projection", channels, hidden),
      linearAPBias: await parameter(`${root}/left_projection`, "bias"),
      linearAGWeight: await projection("left_gate", channels, hidden),
      linearAGBias: await parameter(`${root}/left_gate`, "bias"),
      linearBPWeight: await projection("right_projection", channels, hidden),
      linearBPBias: await parameter(`${root}/right_projection`, "bias"),
      linearBGWeight: await projection("right_gate", channels, hidden),
      linearBGBias: await parameter(`${root}/right_gate`, "bias"),
      layerNormOutWeight: await parameter(`${root}/center_layer_norm`, "scale"),
      layerNormOutBias: await parameter(`${root}/center_layer_norm`, "offset"),
      linearZWeight: await projection("output_projection", hidden, channels),
      linearZBias: await parameter(`${root}/output_projection`, "bias"),
      linearGWeight: await projection("gating_linear", channels, channels),
      linearGBias: await parameter(`${root}/gating_linear`, "bias"),
    });
  }

  async #outerProductMean(
    parameters: ParameterMap, block: number, blocks: number,
  ): Promise<OuterProductMeanWeights> {
    const parameter = (module: string, name: string): Promise<() => Float32Array> =>
      this.#lazyParameter(parameters, module, name, block, blocks);
    return lazyWeights<OuterProductMeanWeights>({
      layerNormScale: await parameter("outer_product_mean/layer_norm_input", "scale"),
      layerNormOffset: await parameter("outer_product_mean/layer_norm_input", "offset"),
      leftWeight: await parameter("outer_product_mean/left_projection", "weights"),
      leftBias: await parameter("outer_product_mean/left_projection", "bias"),
      rightWeight: await parameter("outer_product_mean/right_projection", "weights"),
      rightBias: await parameter("outer_product_mean/right_projection", "bias"),
      outputWeight: await parameter("outer_product_mean", "output_w"),
      outputBias: await parameter("outer_product_mean", "output_b"),
    });
  }

  async mainStackWeights(pairChannels = 128): Promise<readonly EvoformerBlockWeights[]> {
    const { parameters, blocks } = this.manifest.evoformerStack;
    const result: EvoformerBlockWeights[] = [];
    for (let block = 0; block < blocks; block += 1) {
      const row = await this.#rowAttention(parameters, block, blocks);
      result.push({
        msaRowAttention: row,
        msaColumnAttention: await this.#attention(parameters, "msa_column_attention", block, blocks),
        msaTransition: await this.#transition(parameters, "msa_transition", block, blocks),
        outerProductMean: await this.#outerProductMean(parameters, block, blocks),
        triangleMultiplicationOutgoing: await this.#triangle(
          parameters, "triangle_multiplication_outgoing", pairChannels, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          parameters, "triangle_multiplication_incoming", pairChannels, block, blocks,
        ),
        triangleAttentionStarting: await this.#triangleAttention(
          parameters, "triangle_attention_starting_node", block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          parameters, "triangle_attention_ending_node", block, blocks,
        ),
        pairTransition: await this.#transition(parameters, "pair_transition", block, blocks),
      });
    }
    return result;
  }

  async #rowAttention(parameters: ParameterMap, block: number, blocks: number): Promise<RowAttentionModuleWeights> {
    const base = await this.#attention(parameters, "msa_row_attention_with_pair_bias", block, blocks);
    return mergeLazy(base, lazyWeights<Omit<RowAttentionModuleWeights, keyof AttentionModuleWeights>>({
      pairLayerNormScale: await this.#lazyParameter(
        parameters, "msa_row_attention_with_pair_bias/feat_2d_norm", "scale", block, blocks,
      ),
      pairLayerNormOffset: await this.#lazyParameter(
        parameters, "msa_row_attention_with_pair_bias/feat_2d_norm", "offset", block, blocks,
      ),
      pairProjectionWeight: await this.#lazyParameter(
        parameters, "msa_row_attention_with_pair_bias", "feat_2d_weights", block, blocks,
      ),
    }));
  }

  async extraPairStackWeights(pairChannels = 128): Promise<readonly EvoformerPairBlockWeights[]> {
    const { parameters, blocks } = this.manifest.extraMsaStack;
    const result: EvoformerPairBlockWeights[] = [];
    for (let block = 0; block < blocks; block += 1) {
      result.push({
        outerProductMean: await this.#outerProductMean(parameters, block, blocks),
        triangleMultiplicationOutgoing: await this.#triangle(
          parameters, "triangle_multiplication_outgoing", pairChannels, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          parameters, "triangle_multiplication_incoming", pairChannels, block, blocks,
        ),
        triangleAttentionStarting: await this.#triangleAttention(
          parameters, "triangle_attention_starting_node", block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          parameters, "triangle_attention_ending_node", block, blocks,
        ),
        pairTransition: await this.#transition(parameters, "pair_transition", block, blocks),
      });
    }
    return result;
  }

  async extraStackWeights(pairChannels = 128): Promise<readonly ExtraMsaBlockWeights[]> {
    const { parameters, blocks } = this.manifest.extraMsaStack;
    const pairWeights = await this.extraPairStackWeights(pairChannels);
    const result: ExtraMsaBlockWeights[] = [];
    for (let block = 0; block < blocks; block += 1) {
      const root = "msa_column_global_attention";
      const attention = `${root}/attention`;
      const parameter = (module: string, name: string) => this.#lazyParameter(parameters, module, name, block, blocks);
      type GlobalTensors = Omit<ExtraMsaBlockWeights["msaColumnGlobalAttention"], "heads">;
      const globalTensors = lazyWeights<GlobalTensors>({
        queryNormScale: await parameter(`${root}/query_norm`, "scale"),
        queryNormOffset: await parameter(`${root}/query_norm`, "offset"),
        queryWeight: await parameter(attention, "query_w"), keyWeight: await parameter(attention, "key_w"),
        valueWeight: await parameter(attention, "value_w"), gatingWeight: await parameter(attention, "gating_w"),
        gatingBias: await parameter(attention, "gating_b"), outputWeight: await parameter(attention, "output_w"),
        outputBias: await parameter(attention, "output_b"),
      });
      result.push({
        ...pairWeights[block]!,
        msaRowAttention: await this.#rowAttention(parameters, block, blocks),
        msaColumnGlobalAttention: mergeLazy(globalTensors, {
          heads: this.#parameterShape(parameters, attention, "gating_b", true)[0]!,
        }),
        msaTransition: await this.#transition(parameters, "msa_transition", block, blocks),
      });
    }
    return result;
  }

  async embeddingWeights(): Promise<InputEmbedderWeights> {
    const p = this.manifest.embedding.parameters;
    const parameter = (module: string, name: string): Promise<Float32Array> => this.#parameter(p, module, name);
    return {
      preprocess1dWeight: await parameter("preprocess_1d", "weights"),
      preprocess1dBias: await parameter("preprocess_1d", "bias"),
      preprocessMsaWeight: await parameter("preprocess_msa", "weights"),
      preprocessMsaBias: await parameter("preprocess_msa", "bias"),
      leftSingleWeight: await parameter("left_single", "weights"),
      leftSingleBias: await parameter("left_single", "bias"),
      rightSingleWeight: await parameter("right_single", "weights"),
      rightSingleBias: await parameter("right_single", "bias"),
      previousPositionWeight: await parameter("prev_pos_linear", "weights"),
      previousPositionBias: await parameter("prev_pos_linear", "bias"),
      previousMsaNormScale: await parameter("prev_msa_first_row_norm", "scale"),
      previousMsaNormOffset: await parameter("prev_msa_first_row_norm", "offset"),
      previousPairNormScale: await parameter("prev_pair_norm", "scale"),
      previousPairNormOffset: await parameter("prev_pair_norm", "offset"),
      relativePositionWeight: await parameter("pair_activiations", "weights"),
      relativePositionBias: await parameter("pair_activiations", "bias"),
      extraMsaWeight: await parameter("extra_msa_activations", "weights"),
      extraMsaBias: await parameter("extra_msa_activations", "bias"),
    };
  }

  async multimerEmbeddingWeights(): Promise<InputEmbedderWeights> {
    const p = this.manifest.embedding.parameters;
    const parameter = (module: string, name: string): Promise<Float32Array> => this.#parameter(p, module, name);
    return {
      preprocess1dWeight: await parameter("preprocess_1d", "weights"),
      preprocess1dBias: await parameter("preprocess_1d", "bias"),
      preprocessMsaWeight: await parameter("preprocess_msa", "weights"),
      preprocessMsaBias: await parameter("preprocess_msa", "bias"),
      leftSingleWeight: await parameter("left_single", "weights"),
      leftSingleBias: await parameter("left_single", "bias"),
      rightSingleWeight: await parameter("right_single", "weights"),
      rightSingleBias: await parameter("right_single", "bias"),
      previousPositionWeight: await parameter("prev_pos_linear", "weights"),
      previousPositionBias: await parameter("prev_pos_linear", "bias"),
      previousMsaNormScale: await parameter("prev_msa_first_row_norm", "scale"),
      previousMsaNormOffset: await parameter("prev_msa_first_row_norm", "offset"),
      previousPairNormScale: await parameter("prev_pair_norm", "scale"),
      previousPairNormOffset: await parameter("prev_pair_norm", "offset"),
      relativePositionWeight: await parameter("~_relative_encoding/position_activations", "weights"),
      relativePositionBias: await parameter("~_relative_encoding/position_activations", "bias"),
      extraMsaWeight: await parameter("extra_msa_activations", "weights"),
      extraMsaBias: await parameter("extra_msa_activations", "bias"),
    };
  }

  async multimerTemplateWeights(): Promise<MultimerMockTemplateWeights> {
    const section = this.manifest.multimerTemplate;
    if (section === undefined) throw new Error("Multimer model bundle is missing mock-template parameters");
    const p = section.parameters;
    const parameter = (module: string, name: string): Promise<Float32Array> => this.#parameter(p, module, name);
    const root = "single_template_embedding/template_embedding_iteration";
    const blockWeights: TemplatePairBlockWeights[] = [];
    for (let block = 0; block < 2; block += 1) {
      const starting = await this.#triangleAttention(p, `${root}/triangle_attention_starting_node`, block, 2);
      const ending = await this.#triangleAttention(p, `${root}/triangle_attention_ending_node`, block, 2);
      blockWeights.push({
        triangleAttentionStarting: starting,
        triangleAttentionEnding: ending,
        triangleMultiplicationOutgoing: await this.#triangle(
          p, `${root}/triangle_multiplication_outgoing`, 64, block, 2,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          p, `${root}/triangle_multiplication_incoming`, 64, block, 2,
        ),
        pairTransition: await this.#transition(p, `${root}/pair_transition`, block, 2),
      });
    }
    const pairInputBias = new Float32Array(64);
    for (let module = 0; module <= 8; module += 1) {
      const bias = await parameter(`single_template_embedding/template_pair_embedding_${module}`, "bias");
      for (let channel = 0; channel < 64; channel += 1) {
        pairInputBias[channel] = pairInputBias[channel]! + bias[channel]!;
      }
    }
    for (const module of [2, 3]) {
      const weight = await parameter(`single_template_embedding/template_pair_embedding_${module}`, "weights");
      for (let channel = 0; channel < 64; channel += 1) {
        pairInputBias[channel] = pairInputBias[channel]! + weight[channel]!;
      }
    }
    return {
      queryNormScale: await parameter("single_template_embedding/query_embedding_norm", "scale"),
      queryNormOffset: await parameter("single_template_embedding/query_embedding_norm", "offset"),
      pairInputWeight: await parameter("single_template_embedding/template_pair_embedding_8", "weights"),
      pairInputBias, blockWeights,
      outputNormScale: await parameter("single_template_embedding/output_layer_norm", "scale"),
      outputNormOffset: await parameter("single_template_embedding/output_layer_norm", "offset"),
      outputWeight: await parameter("output_linear", "weights"),
      outputBias: await parameter("output_linear", "bias"),
      msaInputWeight: await parameter("template_single_embedding", "weights"),
      msaInputBias: await parameter("template_single_embedding", "bias"),
      msaOutputWeight: await parameter("template_projection", "weights"),
      msaOutputBias: await parameter("template_projection", "bias"),
      templateRows: section.templates,
    };
  }

  async templateWeights(): Promise<QueryOnlyTemplateWeights> {
    const p = this.manifest.templateEmbedding.parameters;
    const blocks = 2;
    const root = "single_template_embedding/template_pair_stack/__layer_stack_no_state";
    const blockWeights: TemplatePairBlockWeights[] = [];
    for (let block = 0; block < blocks; block += 1) {
      blockWeights.push({
        triangleAttentionStarting: await this.#triangleAttention(
          p, `${root}/triangle_attention_starting_node`, block, blocks,
        ),
        triangleAttentionEnding: await this.#triangleAttention(
          p, `${root}/triangle_attention_ending_node`, block, blocks,
        ),
        triangleMultiplicationOutgoing: await this.#triangle(
          p, `${root}/triangle_multiplication_outgoing`, 64, block, blocks,
        ),
        triangleMultiplicationIncoming: await this.#triangle(
          p, `${root}/triangle_multiplication_incoming`, 64, block, blocks,
        ),
        pairTransition: await this.#transition(p, `${root}/pair_transition`, block, blocks),
      });
    }
    return {
      embeddingBias: await this.#parameter(p, "single_template_embedding/embedding2d", "bias"),
      blockWeights,
      outputNormScale: await this.#parameter(p, "single_template_embedding/output_layer_norm", "scale"),
      outputNormOffset: await this.#parameter(p, "single_template_embedding/output_layer_norm", "offset"),
      valueWeight: await this.#parameter(p, "attention", "value_w"),
      outputWeight: await this.#parameter(p, "attention", "output_w"),
      outputBias: await this.#parameter(p, "attention", "output_b"),
      heads: this.#parameterShape(p, "attention", "value_w", false)[1]!,
    };
  }

  async structureWeights(): Promise<StructureModuleWeights> {
    const p = this.manifest.structureModule.parameters;
    const parameter = (module: string, name: string): Promise<Float32Array> => this.#parameter(p, module, name);
    const root = "fold_iteration";
    const ipa = `${root}/invariant_point_attention`;
    const sidechain = `${root}/rigid_sidechain`;
    return {
      initialize: {
        singleProjectionWeight: await this.#parameter(this.manifest.embedding.parameters, "single_activations", "weights"),
        singleProjectionBias: await this.#parameter(this.manifest.embedding.parameters, "single_activations", "bias"),
        singleNormScale: await parameter("single_layer_norm", "scale"),
        singleNormOffset: await parameter("single_layer_norm", "offset"),
        initialProjectionWeight: await parameter("initial_projection", "weights"),
        initialProjectionBias: await parameter("initial_projection", "bias"),
      },
      ipa: {
        pairNormScale: await parameter("pair_layer_norm", "scale"),
        pairNormOffset: await parameter("pair_layer_norm", "offset"),
        queryScalarWeight: await parameter(`${ipa}/q_scalar`, "weights"),
        queryScalarBias: await parameter(`${ipa}/q_scalar`, "bias"),
        keyValueScalarWeight: await parameter(`${ipa}/kv_scalar`, "weights"),
        keyValueScalarBias: await parameter(`${ipa}/kv_scalar`, "bias"),
        queryPointWeight: await parameter(`${ipa}/q_point_local`, "weights"),
        queryPointBias: await parameter(`${ipa}/q_point_local`, "bias"),
        keyValuePointWeight: await parameter(`${ipa}/kv_point_local`, "weights"),
        keyValuePointBias: await parameter(`${ipa}/kv_point_local`, "bias"),
        trainablePointWeights: await parameter(ipa, "trainable_point_weights"),
        attention2dWeight: await parameter(`${ipa}/attention_2d`, "weights"),
        attention2dBias: await parameter(`${ipa}/attention_2d`, "bias"),
        outputWeight: await parameter(`${ipa}/output_projection`, "weights"),
        outputBias: await parameter(`${ipa}/output_projection`, "bias"),
      },
      postAttention: {
        attentionNormScale: await parameter(`${root}/attention_layer_norm`, "scale"),
        attentionNormOffset: await parameter(`${root}/attention_layer_norm`, "offset"),
        transitionWeights: [await parameter(`${root}/transition`, "weights"),
          await parameter(`${root}/transition_1`, "weights"), await parameter(`${root}/transition_2`, "weights")],
        transitionBiases: [await parameter(`${root}/transition`, "bias"),
          await parameter(`${root}/transition_1`, "bias"), await parameter(`${root}/transition_2`, "bias")],
        transitionNormScale: await parameter(`${root}/transition_layer_norm`, "scale"),
        transitionNormOffset: await parameter(`${root}/transition_layer_norm`, "offset"),
        affineWeight: await parameter(`${root}/affine_update`, "weights"),
        affineBias: await parameter(`${root}/affine_update`, "bias"),
      },
      sidechain: {
        inputWeight: await parameter(`${sidechain}/input_projection`, "weights"),
        inputBias: await parameter(`${sidechain}/input_projection`, "bias"),
        initialInputWeight: await parameter(`${sidechain}/input_projection_1`, "weights"),
        initialInputBias: await parameter(`${sidechain}/input_projection_1`, "bias"),
        residual1Weights: [await parameter(`${sidechain}/resblock1`, "weights"),
          await parameter(`${sidechain}/resblock2`, "weights")],
        residual1Biases: [await parameter(`${sidechain}/resblock1`, "bias"),
          await parameter(`${sidechain}/resblock2`, "bias")],
        residual2Weights: [await parameter(`${sidechain}/resblock1_1`, "weights"),
          await parameter(`${sidechain}/resblock2_1`, "weights")],
        residual2Biases: [await parameter(`${sidechain}/resblock1_1`, "bias"),
          await parameter(`${sidechain}/resblock2_1`, "bias")],
        angleWeight: await parameter(`${sidechain}/unnormalized_angles`, "weights"),
        angleBias: await parameter(`${sidechain}/unnormalized_angles`, "bias"),
      },
    };
  }

  async multimerStructureWeights(): Promise<StructureModuleWeights> {
    const p = this.manifest.structureModule.parameters;
    const parameter = (module: string, name: string): Promise<Float32Array> => this.#parameter(p, module, name);
    const root = "fold_iteration";
    const ipa = `${root}/invariant_point_attention`;
    const sidechain = `${root}/rigid_sidechain`;
    const nativeIpa: MultimerInvariantPointAttentionWeights = {
      pairNormScale: await parameter("pair_layer_norm", "scale"),
      pairNormOffset: await parameter("pair_layer_norm", "offset"),
      queryScalarWeight: await parameter(`${ipa}/q_scalar_projection`, "weights"),
      keyScalarWeight: await parameter(`${ipa}/k_scalar_projection`, "weights"),
      valueScalarWeight: await parameter(`${ipa}/v_scalar_projection`, "weights"),
      queryPointWeight: await parameter(`${ipa}/q_point_projection/point_projection`, "weights"),
      queryPointBias: await parameter(`${ipa}/q_point_projection/point_projection`, "bias"),
      keyPointWeight: await parameter(`${ipa}/k_point_projection/point_projection`, "weights"),
      keyPointBias: await parameter(`${ipa}/k_point_projection/point_projection`, "bias"),
      valuePointWeight: await parameter(`${ipa}/v_point_projection/point_projection`, "weights"),
      valuePointBias: await parameter(`${ipa}/v_point_projection/point_projection`, "bias"),
      trainablePointWeights: await parameter(ipa, "trainable_point_weights"),
      attention2dWeight: await parameter(`${ipa}/attention_2d`, "weights"),
      attention2dBias: await parameter(`${ipa}/attention_2d`, "bias"),
      outputWeight: await parameter(`${ipa}/output_projection`, "weights"),
      outputBias: await parameter(`${ipa}/output_projection`, "bias"),
    };
    return {
      initialize: {
        singleProjectionWeight: await this.#parameter(this.manifest.embedding.parameters, "single_activations", "weights"),
        singleProjectionBias: await this.#parameter(this.manifest.embedding.parameters, "single_activations", "bias"),
        singleNormScale: await parameter("single_layer_norm", "scale"),
        singleNormOffset: await parameter("single_layer_norm", "offset"),
        initialProjectionWeight: await parameter("initial_projection", "weights"),
        initialProjectionBias: await parameter("initial_projection", "bias"),
      },
      ipa: adaptMultimerInvariantPointAttentionWeights(nativeIpa, 384, 128, 12, 16, 16, 4, 8),
      postAttention: {
        attentionNormScale: await parameter(`${root}/attention_layer_norm`, "scale"),
        attentionNormOffset: await parameter(`${root}/attention_layer_norm`, "offset"),
        transitionWeights: [await parameter(`${root}/transition`, "weights"),
          await parameter(`${root}/transition_1`, "weights"), await parameter(`${root}/transition_2`, "weights")],
        transitionBiases: [await parameter(`${root}/transition`, "bias"),
          await parameter(`${root}/transition_1`, "bias"), await parameter(`${root}/transition_2`, "bias")],
        transitionNormScale: await parameter(`${root}/transition_layer_norm`, "scale"),
        transitionNormOffset: await parameter(`${root}/transition_layer_norm`, "offset"),
        affineWeight: await parameter(`${root}/quat_rigid/rigid`, "weights"),
        affineBias: await parameter(`${root}/quat_rigid/rigid`, "bias"),
      },
      sidechain: {
        inputWeight: await parameter(`${sidechain}/input_projection`, "weights"),
        inputBias: await parameter(`${sidechain}/input_projection`, "bias"),
        initialInputWeight: await parameter(`${sidechain}/input_projection_1`, "weights"),
        initialInputBias: await parameter(`${sidechain}/input_projection_1`, "bias"),
        residual1Weights: [await parameter(`${sidechain}/resblock1`, "weights"),
          await parameter(`${sidechain}/resblock2`, "weights")],
        residual1Biases: [await parameter(`${sidechain}/resblock1`, "bias"),
          await parameter(`${sidechain}/resblock2`, "bias")],
        residual2Weights: [await parameter(`${sidechain}/resblock1_1`, "weights"),
          await parameter(`${sidechain}/resblock2_1`, "weights")],
        residual2Biases: [await parameter(`${sidechain}/resblock1_1`, "bias"),
          await parameter(`${sidechain}/resblock2_1`, "bias")],
        angleWeight: await parameter(`${sidechain}/unnormalized_angles`, "weights"),
        angleBias: await parameter(`${sidechain}/unnormalized_angles`, "bias"),
      },
    };
  }

  async confidenceWeights(): Promise<{
    readonly lddt: PredictedLddtWeights;
    readonly pae: PredictedAlignedErrorWeights;
  }> {
    const lp = this.manifest.confidenceHeads.parameters.predictedLddt;
    const pp = this.manifest.confidenceHeads.parameters.predictedAlignedError;
    const parameter = (map: ParameterMap, module: string, name: string) => this.#parameter(map, module, name);
    return {
      lddt: {
        normScale: await parameter(lp, "input_layer_norm", "scale"),
        normOffset: await parameter(lp, "input_layer_norm", "offset"),
        act0Weight: await parameter(lp, "act_0", "weights"), act0Bias: await parameter(lp, "act_0", "bias"),
        act1Weight: await parameter(lp, "act_1", "weights"), act1Bias: await parameter(lp, "act_1", "bias"),
        logitsWeight: await parameter(lp, "logits", "weights"), logitsBias: await parameter(lp, "logits", "bias"),
      },
      pae: {
        logitsWeight: await parameter(pp, "logits", "weights"), logitsBias: await parameter(pp, "logits", "bias"),
      },
    };
  }

  async geometryTables(): Promise<ResidueGeometryTables> {
    return {
      defaultFrames: await this.tensor("geometryDefaultFrames"),
      atom14ToGroup: await this.tensor("geometryAtom14ToGroup"),
      atom14Positions: await this.tensor("geometryAtom14Positions"),
      atom14Mask: await this.tensor("geometryAtom14Mask"),
    };
  }

  async queryOnlyFeatureTables(): Promise<QueryOnlyFeatureTables> {
    return {
      atom37ToAtom14: await this.tensor("geometryAtom37ToAtom14"),
      atom37Mask: await this.tensor("geometryAtom37Mask"),
    };
  }
}
