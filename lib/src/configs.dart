// ignore_for_file: non_constant_identifier_names

import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/devices.dart';
import 'package:transformers/src/utils/hub.dart';


/// Loads a config from the specified path.
/// @param {string} pretrained_model_name_or_path The path to the config directory.
/// @param {PretrainedOptions} options Additional options for loading the config.
/// @returns {Promise<Object>} A promise that resolves with information about the loaded config.
Future<Object> loadConfig(String pretrained_model_name_or_path, PretrainedOptions options) async {
  return await getModelJSON(pretrained_model_name_or_path, 'config.json', true, options);
}

///
/// @param {PretrainedConfig} config
/// @returns {Object} The normalized configuration.
Map<String, dynamic> getNormalizedConfig(Map<String, dynamic> config) {
  final mapping = {};

  Map<String, dynamic> init_normalized_config = {};
  switch (config['model_type']) {
  // Sub-configs
    case 'llava':
    case 'paligemma':
    case 'gemma3':
    case 'florence2':
    case 'llava_onevision':
    case 'idefics3':
    case 'ultravox':
    case 'smolvlm':
      init_normalized_config = getNormalizedConfig(config['text_config']);
      break;
    case 'moondream1':
      init_normalized_config = getNormalizedConfig(config['phi_config']);
      break;
    case 'musicgen':
      init_normalized_config = getNormalizedConfig(config['decoder']);
      break;
    case 'multi_modality':
      init_normalized_config = getNormalizedConfig(config['language_config']);
      break;

  // Decoder-only models
    case 'gpt2':
    case 'gptj':
    case 'jais':
    case 'codegen':
    case 'gpt_bigcode':
      mapping['num_heads'] = 'n_head';
      mapping['num_layers'] = 'n_layer';
      mapping['hidden_size'] = 'n_embd';
      break;
    case 'gpt_neox':
    case 'stablelm':
    case 'opt':
    case 'falcon':
      mapping['num_heads'] = 'num_attention_heads';
      mapping['num_layers'] = 'num_hidden_layers';
      mapping['hidden_size'] = 'hidden_size';
      break;
    case 'llama':
    case 'olmo':
    case 'olmo2':
    case 'mobilellm':
    case 'granite':
    case 'cohere':
    case 'mistral':
    case 'starcoder2':
    case 'qwen2':
    case 'qwen2_vl':
    case 'phi':
    case 'phi3':
    case 'phi3_v':
      mapping['num_heads'] = 'num_key_value_heads';
      mapping['num_layers'] = 'num_hidden_layers';
      mapping['hidden_size'] = 'hidden_size';
      mapping['num_attention_heads'] = 'num_attention_heads';
      break;
    case 'qwen3':
    case 'gemma':
    case 'gemma2':
    case 'gemma3_text':
    case 'glm':
    case 'helium':
      mapping['num_heads'] = 'num_key_value_heads';
      mapping['num_layers'] = 'num_hidden_layers';
      mapping['dim_kv'] = 'head_dim';
      break;
    case 'openelm':
      mapping['num_heads'] = 'num_kv_heads';
      mapping['num_layers'] = 'num_transformer_layers';
      mapping['dim_kv'] = 'head_dim';
      break;
    case 'gpt_neo':
    case 'donut-swin':
      mapping['num_heads'] = 'num_heads';
      mapping['num_layers'] = 'num_layers';
      mapping['hidden_size'] = 'hidden_size';
      break;
    case 'bloom':
      mapping['num_heads'] = 'n_head';
      mapping['num_layers'] = 'n_layer';
      mapping['hidden_size'] = 'hidden_size';
      break;
    case 'mpt':
      mapping['num_heads'] = 'n_heads';
      mapping['num_layers'] = 'n_layers';
      mapping['hidden_size'] = 'd_model';
      break;
    case 'exaone':
      mapping['num_heads'] = 'num_key_value_heads';
      mapping['num_layers'] = 'num_layers';
      mapping['dim_kv'] = 'head_dim';
      mapping['num_attention_heads'] = 'num_attention_heads';
      break;

  // Encoder-decoder models
    case 't5':
    case 'mt5':
    case 'longt5':
      mapping['num_decoder_layers'] = 'num_decoder_layers';
      mapping['num_decoder_heads'] = 'num_heads';
      mapping['decoder_dim_kv'] = 'd_kv';
      mapping['num_encoder_layers'] = 'num_layers';
      mapping['num_encoder_heads'] = 'num_heads';
      mapping['encoder_dim_kv'] = 'd_kv';
      break;
    case 'bart':
    case 'mbart':
    case 'marian':
    case 'whisper':
    case 'lite-whisper':
    case 'm2m_100':
    case 'blenderbot':
    case 'blenderbot-small':
    case 'florence2_language':
      mapping['num_decoder_layers'] = 'decoder_layers';
      mapping['num_decoder_heads'] = 'decoder_attention_heads';
      mapping['decoder_hidden_size'] = 'd_model';
      mapping['num_encoder_layers'] = 'encoder_layers';
      mapping['num_encoder_heads'] = 'encoder_attention_heads';
      mapping['encoder_hidden_size'] = 'd_model';
      break;
    case 'speecht5':
      mapping['num_decoder_layers'] = 'decoder_layers';
      mapping['num_decoder_heads'] = 'decoder_attention_heads';
      mapping['decoder_hidden_size'] = 'hidden_size';
      mapping['num_encoder_layers'] = 'encoder_layers';
      mapping['num_encoder_heads'] = 'encoder_attention_heads';
      mapping['encoder_hidden_size'] = 'hidden_size';
      break;
    case 'trocr':
      mapping['num_encoder_layers'] = mapping['num_decoder_layers'] = 'decoder_layers';
      mapping['num_encoder_heads'] = mapping['num_decoder_heads'] = 'decoder_attention_heads';
      mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'd_model';
      break;
    case 'musicgen_decoder':
      mapping['num_encoder_layers'] = mapping['num_decoder_layers'] = 'num_hidden_layers';
      mapping['num_encoder_heads'] = mapping['num_decoder_heads'] = 'num_attention_heads';
      mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'hidden_size';
      break;
    case 'moonshine':
      mapping['num_decoder_layers'] = 'decoder_num_hidden_layers';
      mapping['num_decoder_heads'] = 'decoder_num_key_value_heads';
      mapping['num_encoder_layers'] = 'encoder_num_hidden_layers';
      mapping['num_encoder_heads'] = 'encoder_num_key_value_heads';
      mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'hidden_size';
      break;
    case 'vision-encoder-decoder':
      final decoderConfig = getNormalizedConfig(config['decoder']);

      final add_encoder_pkv = decoderConfig.containsKey('num_decoder_layers');
      final result = pick(config, ['model_type', 'is_encoder_decoder']);
      if (add_encoder_pkv) {
        // Decoder is part of an encoder-decoder model
        result['num_decoder_layers'] = decoderConfig['num_decoder_layers'];
        result['num_decoder_heads'] = decoderConfig['num_decoder_heads'];
        result['decoder_hidden_size'] = decoderConfig['decoder_hidden_size'];

        result['num_encoder_layers'] = decoderConfig['num_encoder_layers'];
        result['num_encoder_heads'] = decoderConfig['num_encoder_heads'];
        result['encoder_hidden_size'] = decoderConfig['encoder_hidden_size'];
      } else {
        // Decoder is a decoder-only model
        result['num_layers'] = decoderConfig['num_layers'];
        result['num_heads'] = decoderConfig['num_heads'];
        result['hidden_size'] = decoderConfig['hidden_size'];
      }
      return result;

  }

  // NOTE: If `num_attention_heads` is not set, it is assumed to be equal to `num_heads`
  final normalized_config = {
    ...init_normalized_config,
    ...pick(config, ['model_type', 'multi_query', 'is_encoder_decoder']),
  };
  for (final key in mapping.keys) {
    normalized_config[key] = config[mapping[key]];
  }
  return normalized_config;
}

/// Base class for all configuration classes. For more information, see the corresponding
/// [Python documentation](https://huggingface.co/docs/transformers/main/en/main_classes/configuration#transformers.PretrainedConfig).
class PretrainedConfig {
  // NOTE: Typo in original

  String? model_type;

  bool is_encoder_decoder = false;

  late int max_position_embeddings;

  /** @type {TransformersJSConfig} */
  // 'transformers.js_config';

  late Map<String, dynamic> normalized_config;

  /// Create a new PreTrainedTokenizer instance.
  /// @param {Object} configJSON The JSON of the config.
  PretrainedConfig(Map<String, dynamic> configJSON) {
    // Object.assign(this, configJSON);
    max_position_embeddings = configJSON['max_position_embeddings'];
    normalized_config = getNormalizedConfig(configJSON);
  }

  /// Loads a pre-trained config from the given `pretrained_model_name_or_path`.
  ///
  /// @param {string} pretrained_model_name_or_path The path to the pre-trained config.
  /// @param {PretrainedOptions} options Additional options for loading the config.
  /// @throws {Error} Throws an error if the config.json is not found in the `pretrained_model_name_or_path`.
  ///
  /// @returns {Promise<PretrainedConfig>} A new instance of the `PretrainedConfig` class.
  static Future<PretrainedConfig> from_pretrained(String pretrained_model_name_or_path, [PretrainedOptions? options]) async {
    final progress_callback = options?.progress_callback;
    dynamic config = options?.config;
    final cache_dir = options?.cache_dir;
    final local_files_only = options?.local_files_only ?? false;
    final revision = options?.revision ?? 'main';

    if (config != null && config is! PretrainedConfig) {
      config = PretrainedConfig(config);
    }

    final Map<String, dynamic>  data = config ?? await loadConfig(pretrained_model_name_or_path, PretrainedOptions(
      progress_callback: progress_callback,
      config: config,
      cache_dir: cache_dir,
      local_files_only: local_files_only,
      revision: revision,
    ));
    return PretrainedConfig(data);
  }
}

/// Helper class which is used to instantiate pretrained configs with the `from_pretrained` function.
///
/// @example
/// const config = await AutoConfig.from_pretrained('Xenova/bert-base-uncased');
class AutoConfig {
  /// @type {typeof PretrainedConfig.from_pretrained}
  static Future<PretrainedConfig> from_pretrained(String pretrained_model_name_or_path, [PretrainedOptions? options]) async {
    return PretrainedConfig.from_pretrained(pretrained_model_name_or_path, options);
  }
}

/// Device-specific configuration options.
/// @typedef {Omit<TransformersJSConfig, "device" | "device_config">} DeviceConfig
class DeviceConfig {
  /// The data type of the key-value cache.
  dynamic kv_cache_dtype;

  /// Override the free dimensions of the model.
  /// See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
  /// for more information.
  Map<String, int>? free_dimension_overrides;

  /// The default data type to use for the model.
  dynamic dtype;

  /// Whether to load the model using the external data format (used for models >= 2GB in size).
  ExternalData use_external_data_format = ExternalDataBoolean(false);
}

/// Transformers.js-specific configuration, possibly present in config.json under the key `transformers.js_config`.
/// @typedef {Object} TransformersJSConfig
/// @property {Record<import('./utils/devices.js').DeviceType, DeviceConfig>} [device_config] Device-specific configurations.
/// @property {import('./utils/tensor.js').DataType|Record<import('./utils/dtypes.js').DataType, import('./utils/tensor.js').DataType>} [kv_cache_dtype] The data type of the key-value cache.
/// @property {Record<string, number>} [free_dimension_overrides] Override the free dimensions of the model.
/// See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
/// for more information.
/// @property {import('./utils/devices.js').DeviceType} [device] The default device to use for the model.
/// @property {import('./utils/dtypes.js').DataType|Record<string, import('./utils/dtypes.js').DataType>} [dtype] The default data type to use for the model.
/// @property {import('./utils/hub.js').ExternalData|Record<string, import('./utils/hub.js').ExternalData>} [use_external_data_format=false] Whether to load the model using the external data format (used for models >= 2GB in size).
class TransformersJSConfig extends DeviceConfig {
  /// Device-specific configurations.
  Map<DeviceType, DeviceConfig>? device_config;

  /// The default device to use for the model.
  DeviceType? device;
}
