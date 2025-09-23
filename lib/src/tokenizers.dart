// ignore_for_file: non_constant_identifier_names

import 'dart:convert';
import 'dart:math';

import "package:unorm_dart/unorm_dart.dart" as unorm;
import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/data_structures.dart';
import 'package:transformers/src/utils/hub.dart';

import 'configs.dart';

/// @typedef {Object} TokenizerProperties Additional tokenizer-specific properties.
/// @property {boolean} [legacy=false] Whether or not the `legacy` behavior of the tokenizer should be used.
/// @typedef {import('./utils/hub.js').PretrainedOptions & TokenizerProperties} PretrainedTokenizerOptions
class PretrainedTokenizerOptions extends PretrainedOptions {
  /// Whether or not the `legacy` behavior of the tokenizer should be used.
  bool? legacy = false;

  PretrainedTokenizerOptions({
    super.progress_callback,
    super.config,
    super.cache_dir,
    super.local_files_only,
    super.revision,
    this.legacy,
  });
}

/// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
/// @param {string} text The text to clean up.
/// @returns {string} The cleaned up text.
String clean_up_tokenization(String text) {
  // Clean up a list of simple English tokenization artifacts
  // like spaces before punctuations and abbreviated forms
  return text.replaceAll(RegExp(r' \.'), '.')
      .replaceAll(RegExp(r' \?'), '?')
      .replaceAll(RegExp(r' \!'), '!')
      .replaceAll(RegExp(r' ,'), ',')
      .replaceAll(RegExp(r" \' "), "'")
      .replaceAll(RegExp(r" n\'t"), "n't")
      .replaceAll(RegExp(r" \'m"), "'m")
      .replaceAll(RegExp(r" \'s"), "'s")
      .replaceAll(RegExp(r" \'ve"), "'ve")
      .replaceAll(RegExp(r" \'re"), "'re");
}

/// Helper function to fuse consecutive unknown tokens.
/// @param {string[]} arr The list of input tokens
/// @param {Map<string, any>} tokens_to_ids The mapping from tokens to token ids.
/// @param {number} unk_token_id The value to fuse on.
/// @private
List<String> fuse_unk(List<String> arr, Map<String, dynamic> tokens_to_ids, int unk_token_id) {
  final List<String> fused = [];
  int i = 0;
  while (i < arr.length) {
    fused.add(arr[i]);
    if ((tokens_to_ids[arr[i]] ?? unk_token_id) != unk_token_id) {
      ++i;
      continue;
    }

    while (++i < arr.length && (tokens_to_ids[arr[i]] ?? unk_token_id) == unk_token_id) {
      if (tokens_to_ids[fused.last] != unk_token_id) {
        fused[fused.length - 1] += arr[i];
      }
    }
  }

  return fused;
}

/// Split a string on whitespace.
/// @param {string} text The text to split.
/// @returns {string[]} The split string.
List<String> whitespace_split(String text) {
  return RegExp(r'\S+').allMatches(text).map((x) => x.group(0)!).toList();
}

/// Loads a tokenizer from the specified path.
/// @param {string} pretrained_model_name_or_path The path to the tokenizer directory.
/// @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
/// @returns {Promise<any[]>} A promise that resolves with information about the loaded tokenizer.
Future<(Map<String, dynamic> tokenizerJSON, Map<String, dynamic> tokenizerConfig)> loadTokenizer(
  String pretrained_model_name_or_path,
  PretrainedTokenizerOptions options,
) async {
  final info = await Future.wait([
    getModelJSON(pretrained_model_name_or_path, 'tokenizer.json', true, options),
    getModelJSON(pretrained_model_name_or_path, 'tokenizer_config.json', true, options),
  ]);

  // Override legacy option if `options.legacy` is not null
  if (options.legacy != null) {
    info[1]['legacy'] = options.legacy;
  }
  return (info.first, info.last);
}

/// Represent a token added by the user on top of the existing Model vocabulary.
/// AddedToken can be configured to specify the behavior they should have in various situations like:
///   - Whether they should only match single words
///   - Whether to include any whitespace on its left or right
class AddedToken {
  late String content;
  late int id;
  late bool single_word;
  late bool lstrip;
  late bool rstrip;
  late bool special;
  bool? normalized;

  /// Creates a new instance of AddedToken.
  /// @param {Object} config Added token configuration object.
  /// @param {string} config.content The content of the added token.
  /// @param {number} config.id The id of the added token.
  /// @param {boolean} [config.single_word=false] Whether this token must be a single word or can break words.
  /// @param {boolean} [config.lstrip=false] Whether this token should strip whitespaces on its left.
  /// @param {boolean} [config.rstrip=false] Whether this token should strip whitespaces on its right.
  /// @param {boolean} [config.normalized=false] Whether this token should be normalized.
  /// @param {boolean} [config.special=false] Whether this token is special.
  AddedToken(Map<String, dynamic> config) {
    content = config['content'];
    id = config['id'];
    single_word = config['single_word'] ?? false;
    lstrip = config['lstrip'] ?? false;
    rstrip = config['rstrip'] ?? false;
    special = config['special'] ?? false;
    normalized = config['normalized'];
  }
}

/// Abstract base class for tokenizer models.
class TokenizerModel {
  Map<String, dynamic> config;

  List<String> vocab = [];

  /// A mapping of tokens to ids.
  Map<String, int> tokens_to_ids = {};

  int? unk_token_id;

  String? unk_token;

  String? end_of_word_suffix;

  /// Whether to fuse unknown tokens when encoding. Defaults to false.
  bool should_fuse_unk = false;

  /// Creates a new instance of TokenizerModel.
  /// @param {Object} config The configuration object for the TokenizerModel.
  TokenizerModel(this.config) {
    should_fuse_unk = config['fuse_unk'] ?? false;
  }

  /// Instantiates a new TokenizerModel instance based on the configuration object provided.
  /// @param {Object} config The configuration object for the TokenizerModel.
  /// @param {...*} args Optional arguments to pass to the specific TokenizerModel constructor.
  /// @returns {TokenizerModel} A new instance of a TokenizerModel.
  /// @throws Will throw an error if the TokenizerModel type in the config is not recognized.
  static TokenizerModel fromConfig(Map<String, dynamic> config, Map<String, dynamic> tokenizerConfig) {
    switch (config['type']) {
      case 'WordPiece':
        return WordPieceTokenizer(config);
      case 'Unigram':
        return Unigram(config, tokenizerConfig);
      case 'BPE':
        return BPE(config);

      default:
        // Some older tokenizers, like `google-t5/t5-small`, `openai-community/gpt2`, and `distilbert/distilbert-base-uncased`, do not have a `type` field.
        // In this case, we can infer the tokenizer type based on the structure of the `vocab` field and other properties.
        if (config['vocab'] != null) {
          if (config['vocab'] is List) {
            // config.vocab is of type `[string, number][]`
            return Unigram(config, tokenizerConfig);
          } else if (config['continuing_subword_prefix'] != null && config['unk_token'] != null) {
            if (config['merges'] != null) {
              return BPE(config);
            } else {
              return WordPieceTokenizer(config);
            }
          } else {
            return LegacyTokenizerModel(config, tokenizerConfig);
          }
        }
        throw ArgumentError.value(config['type'], 'config.type', 'Unknown TokenizerModel type: ${config['type']}');
    }
  }

  /// Internal function to call the TokenizerModel instance.
  /// @param {string[]} tokens The tokens to encode.
  /// @returns {string[]} The encoded tokens.
  List<String> call(List<String> tokens) {
    tokens = encode(tokens);
    if (should_fuse_unk) {
      // Fuse unknown tokens
      tokens = fuse_unk(tokens, tokens_to_ids, unk_token_id!);
    }
    return tokens;
  }

  /// Encodes a list of tokens into a list of token IDs.
  /// @param {string[]} tokens The tokens to encode.
  /// @returns {string[]} The encoded tokens.
  /// @throws Will throw an error if not implemented in a subclass.
  List<String> encode(List<String> tokens) {
    throw UnimplementedError('encode should be implemented in subclass.');
  }

  /// Converts a list of tokens into a list of token IDs.
  /// @param {string[]} tokens The tokens to convert.
  /// @returns {number[]} The converted token IDs.
  List<int> convert_tokens_to_ids(List<String> tokens) {
    return tokens.map((t) => tokens_to_ids[t] ?? unk_token_id!).toList();
  }

  /// Converts a list of token IDs into a list of tokens.
  /// @param {number[]|bigint[]} ids The token IDs to convert.
  /// @returns {string[]} The converted tokens.
  List<String> convert_ids_to_tokens(List<int> ids) {
    return ids.map((i) => vocab.elementAtOrNull(i) ?? unk_token!).toList();
  }
}

/// A subclass of TokenizerModel that uses WordPiece encoding to encode tokens.
/// @extends TokenizerModel
class WordPieceTokenizer extends TokenizerModel {
  // late Map<String, int> tokens_to_ids;
  // late int unk_token_id;
  // late String unk_token;
  late int max_input_chars_per_word;
  // late List<String> vocab;

  /// @param {Object} config The configuration object.
  /// @param {Object} config.vocab A mapping of tokens to ids.
  /// @param {string} config.unk_token The unknown token string.
  /// @param {string} config.continuing_subword_prefix The prefix to use for continuing subwords.
  /// @param {number} [config.max_input_chars_per_word=100] The maximum number of characters per word.
  WordPieceTokenizer(super.config) {
    /**
     * A mapping of tokens to ids.
     * @type {Map<string, number>}
     */
    // tokens_to_ids = objectToMap(config['vocab']);
    tokens_to_ids = config['vocab'];

    /**
     * The id of the unknown token.
     * @type {number}
     */
    unk_token_id = tokens_to_ids[config['unk_token']];

    /**
     * The unknown token string.
     * @type {string}
     */
    unk_token = config['unk_token'];

    /**
     * The maximum number of characters allowed per word.
     * @type {number}
     */
    max_input_chars_per_word = config['max_input_chars_per_word'] ?? 100;

    /**
     * An array of tokens.
     * @type {string[]}
     */
    vocab = List.filled(tokens_to_ids.length, '');
    for (final e in tokens_to_ids.entries) {
      vocab[e.value] = e.key;
    }
  }

  /// Encodes an array of tokens using WordPiece encoding.
  /// @param {string[]} tokens The tokens to encode.
  /// @returns {string[]} An array of encoded tokens.
  @override
  List<String> encode(List<String> tokens) {
    final List<String> outputTokens = [];
    for (final token in tokens) {
      final chars = token.split('');
      if (chars.length > max_input_chars_per_word) {
        outputTokens.add(unk_token!);
        continue;
      }

      bool isUnknown = false;
      int start = 0;
      final List<String> subTokens = [];

      while (start < chars.length) {
        int end = chars.length;
        String? currentSubstring;
        while (start < end) {
          String substr = chars.sublist(start, end).join('');

          if (start > 0) {
            substr = config['continuing_subword_prefix'] + substr;
          }
          if (tokens_to_ids.containsKey(substr)) {
            currentSubstring = substr;
            break;
          }

          --end;
        }
        if (currentSubstring == null) {
          isUnknown = true;
          break;
        }
        subTokens.add(currentSubstring);
        start = end;
      }
      if (isUnknown) {
        outputTokens.add(unk_token!);
      } else {
        outputTokens.addAll(subTokens);
      }
    }

    return outputTokens;
  }
}

/// Class representing a Unigram tokenizer model.
/// @extends TokenizerModel
class Unigram extends TokenizerModel {
  late List<double> scores;

  late String bos_token;

  int? bos_token_id;

  String? eos_token;

  int? eos_token_id;

  late double minScore;

  late double unk_score;

  final CharTrie trie = CharTrie();

  late bool fuse_unk;

  /// Create a new Unigram tokenizer model.
  /// @param {Object} config The configuration object for the Unigram model.
  /// @param {number} config.unk_id The ID of the unknown token
  /// @param {[string, number][]} config.vocab A 2D array representing a mapping of tokens to scores.
  /// @param {Object} moreConfig Additional configuration object for the Unigram model.
  Unigram(super.config, Map<String, dynamic> moreConfig) {
    final vocabSize = config['vocab'].length;

    vocab = List.filled(vocabSize, '');
    scores = List.filled(vocabSize, 0);
    for (int i = 0; i < vocabSize; ++i) {
      final [token, score] = config['vocab'][i];
      vocab[i] = token;
      scores[i] = score;
    }

    unk_token_id = config['unk_id'];
    unk_token = vocab[config['unk_id']];

    tokens_to_ids = Map.fromEntries(vocab.indexed.map((e) => MapEntry(e.$2, e.$1)));
    bos_token = ' '; // beginning of a sentence token

    bos_token_id = tokens_to_ids[bos_token]; // NOTE: may be undefined
    eos_token = moreConfig['eos_token'];

    eos_token_id = tokens_to_ids[eos_token];
    unk_token = vocab[unk_token_id!];

    minScore = scores.reduce(min);

    unk_score = minScore - 10.0;
    scores[unk_token_id!] = unk_score;

    trie.extend(vocab);

    // NOTE: `fuse_unk` is hardcoded to true for Unigram models
    // See: https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/models/unigram/model.rs#L119
    fuse_unk = true;
  }

  /// Populates lattice nodes.
  /// @param {TokenLattice} lattice The token lattice to populate with nodes.
  void populateNodes(TokenLattice lattice) {
    final chars = lattice.chars;
    const mblen = 1;
    int beginPos = 0;
    while (beginPos < chars.length) {
      bool hasSingleNode = false;

      final tokens = [];
      final sliced = chars.sublist(beginPos).join('');
      final prefixedTokens = trie.commonPrefixSearch(sliced);
      for (final token in prefixedTokens) {
        tokens.add(token);
        final tokenId = tokens_to_ids[token]!;
        final tokenScore = scores[tokenId];
        final n = len(token);
        lattice.insert(beginPos, n, tokenScore, tokenId);
        if (!hasSingleNode && n == mblen) {
          hasSingleNode = true;
        }
      }
      if (!hasSingleNode) {
        lattice.insert(beginPos, mblen, unk_score, unk_token_id!);
      }
      beginPos += mblen;
    }
  }

  /// Encodes an array of tokens into an array of subtokens using the unigram model.
  ///
  /// @param {string} normalized The normalized string.
  /// @returns {string[]} An array of subtokens obtained by encoding the input tokens using the unigram model.
  List<String> tokenize(String normalized) {
    final lattice = TokenLattice(normalized, bos_token_id, eos_token_id);
    populateNodes(lattice);
    return lattice.tokens();
  }

  /// Encodes an array of tokens using Unigram encoding.
  /// @param {string[]} tokens The tokens to encode.
  /// @returns {string[]} An array of encoded tokens.
  @override
  List<String> encode(List<String> tokens) {
    final List<String> toReturn = [];
    for (final token in tokens) {
      final tokenized = tokenize(token);
      toReturn.addAll(tokenized);
    }
    return toReturn;
  }
}

// /**
//  * Returns list of utf-8 byte and a mapping to unicode strings.
//  * Specifically avoids mapping to whitespace/control characters the BPE code barfs on.
//  * @returns {Object} Object with utf-8 byte keys and unicode string values.
//  */
// const BYTES_TO_UNICODE = (() => {
// // Returns list of utf-8 byte and a mapping to unicode strings.
// // We specifically avoids mapping to whitespace/control characters
// // the bpe code barfs on.
//
// const bs = [
// ...Array.from({ length: "~".charCodeAt(0) - "!".charCodeAt(0) + 1 }, (_, i) => i + "!".charCodeAt(0)),
// ...Array.from({ length: "¬".charCodeAt(0) - "¡".charCodeAt(0) + 1 }, (_, i) => i + "¡".charCodeAt(0)),
// ...Array.from({ length: "ÿ".charCodeAt(0) - "®".charCodeAt(0) + 1 }, (_, i) => i + "®".charCodeAt(0)),
// ];
//     const cs = bs.slice();
// let n = 0;
// for (let b = 0; b < 256; ++b) {
// if (!bs.includes(b)) {
// bs.push(b);
// cs.push(256 + n);
// n += 1;
// }
// }
// const ccs = cs.map(n => String.fromCharCode(n));
// return Object.fromEntries(bs.map((b, i) => [b, ccs[i]]));
// })();
//
// const UNICODE_TO_BYTES = reverseDictionary(BYTES_TO_UNICODE);

/**
 * @typedef {Object} BPENode
 * @property {string} token The token associated with the node
 * @property {number} bias A positional bias for the node.
 * @property {number} [score] The score of the node.
 * @property {BPENode} [prev] The previous node in the linked list.
 * @property {BPENode} [next] The next node in the linked list.
 */
class BPENode {
  String token;
  double bias;
  double? score;
  BPENode? prev;
  BPENode? next;

  bool deleted = false;

  BPENode({
    required this.token,
    required this.bias,
    this.score,
    this.prev,
    this.next,
  });

  BPENode clone() {
    final node = BPENode(
      token: token,
      bias: bias,
      score: score,
      prev: prev,
      next: next,
    );
    node.deleted = deleted;
    return node;
  }
}

/// BPE class for encoding text into Byte-Pair-Encoding (BPE) tokens.
/// @extends TokenizerModel
class BPE extends TokenizerModel {
  late List<List<String>> merges;

  late Map<String, int> bpe_ranks;

  String? continuing_subword_suffix;

  late bool byte_fallback;

  late bool ignore_merges;

  /// The maximum length we should cache in a model.
  /// Strings that are too long have minimal chances to cache hit anyway
  int max_length_to_cache = 256;

  /// The default capacity for a `BPE`'s internal cache.
  int cache_capacity = 10000;

  late LRUCache cache = LRUCache<String, List<String>>(cache_capacity);

  /// Create a BPE instance.
  /// @param {Object} config The configuration object for BPE.
  /// @param {Object} config.vocab A mapping of tokens to ids.
  /// @param {string[]|[string, string][]} config.merges An array of BPE merges as strings.
  /// @param {string} config.unk_token The unknown token used for out of vocabulary words.
  /// @param {string} config.end_of_word_suffix The suffix to place at the end of each word.
  /// @param {string} [config.continuing_subword_suffix] The suffix to insert between words.
  /// @param {boolean} [config.byte_fallback=false] Whether to use spm byte-fallback trick (defaults to False)
  /// @param {boolean} [config.ignore_merges=false] Whether or not to match tokens with the vocab before using merges.
  BPE(super.config) {
    /** @type {Map<string, number>} */
    tokens_to_ids = config['vocab'];

    unk_token_id = tokens_to_ids[config['unk_token']];
    unk_token = config['unk_token'];

    vocab = List.filled(tokens_to_ids.length, '');
    for (final e in tokens_to_ids.entries) {
      vocab[e.value] = e.key;
    }

    // Tokenizers >= 0.20.0 serializes BPE merges as a [string, string][] instead of a string[],
    // which resolves the ambiguity for merges containing spaces.
    final use_new_merge_format = config['merges'][0] is List;

    /** @type {[string, string][]} */
    merges = use_new_merge_format
        ? /** @type {[string, string][]} */config['merges']
        : (/** @type {string[]} */config['merges'] as List<String>).map((x) => /** @type {[string, string]} */x.split(' ').take(2)).toList();
    bpe_ranks = Map.fromEntries(merges.indexed.map((e) => MapEntry(jsonEncode(e.$2), e.$1)));

    end_of_word_suffix = config['end_of_word_suffix'];

    // NOTE: `continuing_subword_suffix` is custom (to support `BlenderbotSmallTokenizer`)
    continuing_subword_suffix = config['continuing_subword_suffix'];

    byte_fallback = config['byte_fallback'] ?? false;

    ignore_merges = config['ignore_merges'] ?? false;
  }

  /// Clears the cache.
  void clear_cache() {
    cache.clear();
  }

  /// Apply Byte-Pair-Encoding (BPE) to a given token. Efficient heap-based priority
  /// queue implementation adapted from https://github.com/belladoreai/llama-tokenizer-js.
  /// @param {string} token The token to encode.
  /// @returns {string[]} The BPE encoded tokens.
  List<String> bpe(String token) {
    if (token.isEmpty) return [];

    final cached = cache[token];
    if (cached != null) return cached;

    final word = token.split('');
    if (end_of_word_suffix?.isNotEmpty == true) {
      word[word.length - 1] += end_of_word_suffix!;
    }

    List<String> result = [];
    if (word.length > 1) {
      // Create a priority queue to store the nodes that will be merged.
      // The comparator function compares the scores of the nodes.
      final queue = PriorityQueue<BPENode>((a, b) => a.score! < b.score!);

      // Construct a doubly-linked list of nodes that will be inserted into the priority queue,
      // starting with the individual characters. We also populate each node with a positional
      // bias to break ties in the priority queue.
      BPENode startingNode = BPENode(
        token: word[0],
        bias: 0,
        prev: null,
        next: null,
      );

      BPENode previousNode = startingNode;
      for (int i = 1; i < word.length; ++i) {
        final currentNode = BPENode(
          bias: i / word.length, // Add fractional component to break ties
          token: word[i],
          prev: previousNode,
          next: null,
        );
        previousNode.next = currentNode;
        _add_node(queue, previousNode);
        previousNode = currentNode;
      }

      while (queue.isNotEmpty()) {
        // Get the next node with the highest priority
        final node = queue.pop();
        final next = node.next;

        // Check that this merge is still possible
        if (node.deleted || next == null || next.deleted) continue;

        // Here, we mark the current node (left side of the merge) and the next node (right side of the merge) as deleted.
        // This is because they will both be replaced by a new node representing the merge result.
        node.deleted = true;
        next.deleted = true;
        final prev = node.prev;

        // Next, we fix the node that comes before the current node (i.e., left side of the merge).
        if (prev != null) {

          // Make a shallow copy of the previous node
          final newPreviousNode = prev.clone();

          // Mark the old previous node as deleted. This avoids erroneous merges later,
          // because there may still be references to this node in the priority queue.
          prev.deleted = true;
          node.prev = newPreviousNode;

          // Update the reference of the previous node, by pointing its previous node to this new previous node.
          if (newPreviousNode.prev != null) {
            newPreviousNode.prev!.next = newPreviousNode;
          } else {
            // If the previous of the previous node does not exist, it means that
            // `newPreviousNode` must be the new `startingNode`.
            startingNode = newPreviousNode;
          }
        }

        // Create a new node which represents the result of the merge.
        final merged = BPENode(
          token: node.token + next.token,
          bias: node.bias,
          prev: node.prev,
          next: next.next,
        );

        // We now consider where we can add the new merged node to the priority queue:
        // 1. prev <-> merged
        if (merged.prev != null) {
          merged.prev!.next = merged;
          _add_node(queue, merged.prev!);
        } else {
          // If `merged.prev` does not exist, then `merged` must be the new `startingNode`.
          startingNode = merged;
        }

        // 2. merged <-> next
        if (merged.next != null) {
          merged.next!.prev = merged;
          _add_node(queue, merged);
        }
      }

      // Traverse the linked list, starting from the `startingNode`, and collect the tokens.
      for (BPENode? currentNode = startingNode; currentNode != null; currentNode = currentNode.next) {
        result.add(currentNode.token);
      }
    } else {
      result = word;
    }

    // Possibly append suffix
    if (continuing_subword_suffix?.isNotEmpty == true) {
      // Do not append suffix to the last token
      for (int i = 0; i < result.length - 1; ++i) {
        result[i] += continuing_subword_suffix!;
      }
    }

    if (token.length < max_length_to_cache) {
      // Save the result to the cache
      cache[token] = result;
    }

    return result;
  }


  /// Helper function to add a node to the priority queue.
  /// @param {PriorityQueue} queue
  /// @param {BPENode} node
  /// @private
  void _add_node(PriorityQueue<BPENode> queue, BPENode node) {
    // `score` is a measure of the merge priority: lower means higher priority
    // We use the BPE rank as a measure of priority (i.e., the local of the merge in the merges list)
    // We also add a fractional component to the score to break ties (with the earlier character having higher priority)
    final rank = bpe_ranks[jsonEncode([node.token, node.next!.token])];
    if (rank != null) {
      node.score = rank + node.bias;
      queue.push([node]);
    }
  }

  /// Encodes the input sequence of tokens using the BPE algorithm and returns the resulting subword tokens.
  /// @param {string[]} tokens The input sequence of tokens to encode.
  /// @returns {string[]} The resulting subword tokens after applying the BPE algorithm to the input sequence of tokens.
  List<String> encode(List<String> tokens) {
    final List<String> outputTokens = [];

    for (final token in tokens) {
      if (ignore_merges && tokens_to_ids.containsKey(token)) {
        outputTokens.add(token);
        continue;
      }

      final bpe_token_list = bpe(token);

      for (final t in bpe_token_list) {
        if (tokens_to_ids.containsKey(t)) {
          outputTokens.add(t);
        } else if (byte_fallback) {
          final byteTokens = utf8.encode(t)
              .map((x) => '<0x${x.toRadixString(16).toUpperCase().padLeft(2, '0')}>');
          if (byteTokens.every((x) => tokens_to_ids.containsKey(x))) {
            // Ensure the byte tokens are actually in the vocabulary, otherwise
            // we fall back to the unknown token. For more information, see
            // https://github.com/huggingface/transformers/issues/28096.
            outputTokens.addAll(byteTokens);
          } else {
            outputTokens.add(unk_token!);
          }
        } else {
          outputTokens.add(unk_token!);
        }
      }
    }

    return outputTokens;
  }

}

/// Legacy tokenizer class for tokenizers with only a vocabulary.
class LegacyTokenizerModel extends TokenizerModel {
  late String bos_token;

  int? bos_token_id;

  String? eos_token;

  int? eos_token_id;

  String? pad_token;

  int? pad_token_id;

  /// Create a LegacyTokenizerModel instance.
  /// @param {Object} config The configuration object for LegacyTokenizerModel.
  /// @param {Object} config.vocab A (possibly nested) mapping of tokens to ids.
  /// @param {Object} moreConfig Additional configuration object for the LegacyTokenizerModel model.
  LegacyTokenizerModel(super.config, Map<String, dynamic> moreConfig) {
    /**@type {Map<string, number>} */
    tokens_to_ids = moreConfig['target_lang'] ? config['vocab'][moreConfig['target_lang']] : config['vocab'];

    bos_token = moreConfig['bos_token'];
    bos_token_id = tokens_to_ids[bos_token];

    eos_token = moreConfig['eos_token'];
    eos_token_id = tokens_to_ids[eos_token];

    pad_token = moreConfig['pad_token'];
    pad_token_id = tokens_to_ids[pad_token];

    unk_token = moreConfig['unk_token'];
    unk_token_id = tokens_to_ids[unk_token];

    vocab = List.filled(tokens_to_ids.length, '');
    for (final e in tokens_to_ids.entries) {
      vocab[e.value] = e.key;
    }
  }

  @override
  List<String> encode(List<String> tokens) {
    return tokens;
  }
}

/// A base class for text normalization.
class Normalizer {
  Map<String, dynamic> config;

  /// @param {Object} config The configuration object for the normalizer.
  Normalizer(this.config);

  /// Factory method for creating normalizers from config objects.
  /// @param {Object} config The configuration object for the normalizer.
  /// @returns {Normalizer} A Normalizer object.
  /// @throws {Error} If an unknown Normalizer type is specified in the config.
  static Normalizer? fromConfig(Map<String, dynamic>? config) {
    if (config == null) return null;

    switch (config['type']) {
      // case 'BertNormalizer':
      //   return new BertNormalizer(config);
      case 'Precompiled':
        return Precompiled(config);
      // case 'Sequence':
      //   return new NormalizerSequence(config);
      // case 'Replace':
      //   return new Replace(config);
      // case 'NFC':
      //   return new NFC(config);
      // case 'NFD':
      //   return new NFD(config);
      // case 'NFKC':
      //   return new NFKC(config);
      // case 'NFKD':
      //   return new NFKD(config);
      // case 'Strip':
      //   return new StripNormalizer(config);
      // case 'StripAccents':
      //   return new StripAccents(config);
      // case 'Lowercase':
      //   return new Lowercase(config);
      // case 'Prepend':
      //   return new Prepend(config);
      default:
        throw ArgumentError.value(config['type'], 'config.type', 'Unknown Normalizer type: ${config['type']}');
    }
  }

  /// Normalize the input text.
  /// @abstract
  /// @param {string} text The text to normalize.
  /// @returns {string} The normalized text.
  /// @throws {Error} If this method is not implemented in a subclass.
  String normalize(String text) {
    throw UnimplementedError('normalize should be implemented in subclass.');
  }

  /// Alias for {@link Normalizer#normalize}.
  /// @param {string} text The text to normalize.
  /// @returns {string} The normalized text.
  String call(String text) {
    return normalize(text);
  }
}

/// A callable class representing a pre-tokenizer used in tokenization. Subclasses
/// should implement the `pre_tokenize_text` method to define the specific pre-tokenization logic.
class PreTokenizer {
  Map<String, dynamic> config;

  PreTokenizer(this.config);

  /// Factory method that returns an instance of a subclass of `PreTokenizer` based on the provided configuration.
  ///
  /// @static
  /// @param {Object} config A configuration object for the pre-tokenizer.
  /// @returns {PreTokenizer} An instance of a subclass of `PreTokenizer`.
  /// @throws {Error} If the provided configuration object does not correspond to any known pre-tokenizer.
  static PreTokenizer? fromConfig(Map<String, dynamic>? config) {
    if (config == null) return null;

    switch (config['type']) {
      // case 'BertPreTokenizer':
      //   return new BertPreTokenizer(config);
      case 'Sequence':
        return PreTokenizerSequence(config);
      // case 'Whitespace':
      //   return new WhitespacePreTokenizer(config);
      case 'WhitespaceSplit':
        return WhitespaceSplit(config);
      case 'Metaspace':
        return MetaspacePreTokenizer(config);

      // case 'ByteLevel':
      //   return new ByteLevelPreTokenizer(config);
      // case 'Split':
      //   return new SplitPreTokenizer(config);
      // case 'Punctuation':
      //   return new PunctuationPreTokenizer(config);
      // case 'Digits':
      //   return new DigitsPreTokenizer(config);
      // case 'Replace':
      //   return new ReplacePreTokenizer(config);
      default:
        throw ArgumentError.value(config['type'], 'config.type', 'Unknown PreTokenizer type: ${config['type']}');
    }
  }

  /// Method that should be implemented by subclasses to define the specific pre-tokenization logic.
  ///
  /// @abstract
  /// @param {string} text The text to pre-tokenize.
  /// @param {Object} [options] Additional options for the pre-tokenization logic.
  /// @returns {string[]} The pre-tokenized text.
  /// @throws {Error} If the method is not implemented in the subclass.
  List<String> pre_tokenize_text(String text, [Map<String, dynamic>? options]) {
    throw UnimplementedError("pre_tokenize_text should be implemented in subclass.");
  }

  /// Tokenizes the given text into pre-tokens.
  /// @param {string|string[]} text The text or array of texts to pre-tokenize.
  /// @param {Object} [options] Additional options for the pre-tokenization logic.
  /// @returns {string[]} An array of pre-tokens.
  List<String> pre_tokenize(dynamic text, [Map<String, dynamic>? options]) {
    return switch(text) {
      String() => pre_tokenize_text(text, options),
      List<String> text => text.map((t) => pre_tokenize_text(t, options)).expand((e) => e).toList(),
      _ => throw ArgumentError.value(text, 'text', 'text must be of type string or List<String>'),
    };
  }

  /// Alias for {@link PreTokenizer#pre_tokenize}.
  /// @param {string|string[]} text The text or array of texts to pre-tokenize.
  /// @param {Object} [options] Additional options for the pre-tokenization logic.
  /// @returns {string[]} An array of pre-tokens.
  List<String> call(dynamic text, [Map<String, dynamic>? options]) {
    return pre_tokenize(text, options);
  }
}

class PostProcessedOutput {
  List<String> tokens;

  List<int> token_type_ids;

  PostProcessedOutput(this.tokens, this.token_type_ids);
}

class PostProcessor {
  /// The configuration for the post-processor.
  Map<String, dynamic> config;

  PostProcessor(this.config);

  /// Factory method to create a PostProcessor object from a configuration object.
  ///
  /// @param {Object} config Configuration object representing a PostProcessor.
  /// @returns {PostProcessor} A PostProcessor object created from the given configuration.
  /// @throws {Error} If an unknown PostProcessor type is encountered.
  static PostProcessor? fromConfig(Map<String, dynamic>? config) {
    if (config == null) return null;

    switch (config['type']) {
      case 'TemplateProcessing':
        return TemplateProcessing(config);

      // case 'ByteLevel':
      //   return new ByteLevelPostProcessor(config);
      //
      // case 'RobertaProcessing':
      //   return new RobertaProcessing(config);
      // case 'BertProcessing':
      //   return new BertProcessing(config);
      //
      // case 'Sequence':
      //   return new PostProcessorSequence(config);
      default:
        throw ArgumentError.value(config['type'], 'config.type', 'Unknown PostProcessor type: ${config['type']}');
    }
  }

  /// Method to be implemented in subclass to apply post-processing on the given tokens.
  ///
  /// @param {Array} tokens The input tokens to be post-processed.
  /// @param {...*} args Additional arguments required by the post-processing logic.
  /// @returns {PostProcessedOutput} The post-processed tokens.
  /// @throws {Error} If the method is not implemented in subclass.
  PostProcessedOutput post_process(List<String>? tokens, [List<String>? tokens2, bool? add_special_tokens]) {
    throw UnimplementedError('post_process should be implemented in subclass.');
  }

  /// Alias for {@link PostProcessor#post_process}.
  /// @param {Array} tokens The text or array of texts to post-process.
  /// @param {...*} args Additional arguments required by the post-processing logic.
  /// @returns {PostProcessedOutput} The post-processed tokens.
  PostProcessedOutput call(List<String>? tokens, [List<String>? tokens2, bool? add_special_tokens]) {
    return post_process(tokens, tokens2, add_special_tokens);
  }
}

/// Post processor that replaces special tokens in a template with actual tokens.
/// @extends PostProcessor
class TemplateProcessing extends PostProcessor {
  late List<Map<String, dynamic>> single;

  late List<Map<String, dynamic>> pair;

  /// Creates a new instance of `TemplateProcessing`.
  /// @param {Object} config The configuration options for the post processor.
  /// @param {Array} config.single The template for a single sequence of tokens.
  /// @param {Array} config.pair The template for a pair of sequences of tokens.
  TemplateProcessing(super.config) {
    single = List<Map<String, dynamic>>.from(config['single']);
    pair = List<Map<String, dynamic>>.from(config['pair']);
  }

  /// Replaces special tokens in the template with actual tokens.
  /// @param {string[]} tokens The list of tokens for the first sequence.
  /// @param {string[]} [tokens_pair=null] The list of tokens for the second sequence (optional).
  /// @returns {PostProcessedOutput} An object containing the list of tokens with the special tokens replaced with actual tokens.
  @override
  PostProcessedOutput post_process(List<String>? tokens, [List<String>? tokens_pair, bool? add_special_tokens = true]) {
    final type = tokens_pair == null ? single : pair;

    List<String> processedTokens = [];
    List<int> types = [];

    for (final item in type) {
      if (item.containsKey('SpecialToken')) {
        if (add_special_tokens!) {
          processedTokens.add(item['SpecialToken']['id']);
          types.add(item['SpecialToken']['type_id']);
        }
      } else if (item.containsKey('Sequence')) {
        if (item['Sequence']['id'] == 'A') {
          processedTokens.addAll(tokens ?? []);
          types.addAll(List.filled(tokens?.length ?? 0, item['Sequence']['type_id']));
        } else if (item['Sequence']['id'] == 'B') {
          processedTokens.addAll(tokens_pair ?? []);
          types.addAll(List.filled(tokens_pair?.length ?? 0, item['Sequence']['type_id']));
        }
      }
    }

    return PostProcessedOutput(processedTokens, types);
  }
}

/// The base class for token decoders.
class Decoder {
  /// The configuration object
  Map<String, dynamic> config;

  List<AddedToken> added_tokens = [];

  String? end_of_word_suffix;

  String? trim_offsets;

  /// Creates an instance of `Decoder`.
  Decoder(this.config) {
    trim_offsets = config['trim_offsets'];
  }

  /// Creates a decoder instance based on the provided configuration.
  ///
  /// @param {Object} config The configuration object.
  /// @returns {Decoder} A decoder instance.
  /// @throws {Error} If an unknown decoder type is provided.
  static Decoder? fromConfig(Map<String, dynamic>? config) {
    if (config == null) return null;

    switch (config['type']) {
      // case 'WordPiece':
      //   return new WordPieceDecoder(config);
      case 'Metaspace':
        return MetaspaceDecoder(config);
      // case 'ByteLevel':
      //   return new ByteLevelDecoder(config);
      //
      // case 'Replace':
      //   return new ReplaceDecoder(config);
      // case 'ByteFallback':
      //   return new ByteFallback(config);
      // case 'Fuse':
      //   return new FuseDecoder(config);
      // case 'Strip':
      //   return new StripDecoder(config);
      //
      // case 'Sequence':
      //   return new DecoderSequence(config);
      //
      // case 'CTC':
      //   return new CTCDecoder(config);
      // case 'BPEDecoder':
      //   return new BPEDecoder(config);
      default:
        throw ArgumentError.value(config['type'], 'config.type', 'Unknown Decoder type: ${config['type']}');
    }
  }

  /// Calls the `decode` method.
  ///
  /// @param {string[]} tokens The list of tokens.
  /// @returns {string} The decoded string.
  String call(List<String> tokens) {
    return decode(tokens);
  }

  /// Decodes a list of tokens.
  /// @param {string[]} tokens The list of tokens.
  /// @returns {string} The decoded string.
  String decode(List<String> tokens) {
    return decode_chain(tokens).join('');
  }

  /// Apply the decoder to a list of tokens.
  ///
  /// @param {string[]} tokens The list of tokens.
  /// @returns {string[]} The decoded list of tokens.
  /// @throws {Error} If the `decode_chain` method is not implemented in the subclass.
  List<String> decode_chain(List<String> tokens) {
    throw UnimplementedError('`decode_chain` should be implemented in subclass.');
  }
}

/// This PreTokenizer replaces spaces with the given replacement character, adds a prefix space if requested,
/// and returns a list of tokens.
/// @extends PreTokenizer
class MetaspacePreTokenizer extends PreTokenizer {
  late bool addPrefixSpace;
  late String replacement;
  late String strRep;
  late String prepend_scheme;

  /// @param {Object} config The configuration object for the MetaspacePreTokenizer.
  /// @param {boolean} config.add_prefix_space Whether to add a prefix space to the first token.
  /// @param {string} config.replacement The character to replace spaces with.
  /// @param {string} [config.str_rep=config.replacement] An optional string representation of the replacement character.
  /// @param {'first'|'never'|'always'} [config.prepend_scheme='always'] The metaspace prepending scheme.
  MetaspacePreTokenizer(super.config) {
    addPrefixSpace = config['add_prefix_space'];
    replacement = config['replacement'];
    strRep = config['str_rep'] ?? replacement;
    prepend_scheme = config['prepend_scheme'] ?? 'always';
  }

  /// This method takes a string, replaces spaces with the replacement character,
  /// adds a prefix space if requested, and returns a new list of tokens.
  /// @param {string} text The text to pre-tokenize.
  /// @param {Object} [options] The options for the pre-tokenization.
  /// @param {number} [options.section_index] The index of the section to pre-tokenize.
  /// @returns {string[]} A new list of pre-tokenized tokens.
  @override
  List<String> pre_tokenize_text(String text, [Map<String, dynamic>? options]) {
    final int section_index = options!['section_index'];
    String normalized = text.replaceAll(' ', strRep);

    if (
      // We add a prefix space if:
      //  (1) The addPrefixSpace option is enabled and the normalized
      //      token does not already start with the replacement character.
      (addPrefixSpace && !normalized.startsWith(replacement))

      // and (2) either:
      //  (a) prepend_scheme is 'always'
      //  (b) prepend_scheme is 'first' and this is the first section
      && (
        prepend_scheme == 'always' ||
        (prepend_scheme == 'first' && section_index == 0)
      )
    ) {
      normalized = strRep + normalized;
    }

    return [normalized];
  }
}

/// MetaspaceDecoder class extends the Decoder class and decodes Metaspace tokenization.
/// @extends Decoder
class MetaspaceDecoder extends Decoder {
  /// Whether to add a prefix space to the decoded string.
  late bool addPrefixSpace;

  /// The string to replace spaces with.
  late String replacement;

  /// Constructs a new MetaspaceDecoder object.
  /// @param {Object} config The configuration object for the MetaspaceDecoder.
  /// @param {boolean} config.add_prefix_space Whether to add a prefix space to the decoded string.
  /// @param {string} config.replacement The string to replace spaces with.
  MetaspaceDecoder(super.config) {
    addPrefixSpace = config['add_prefix_space'];
    replacement = config['replacement'];
  }

  /// @type {Decoder['decode_chain']}
  @override
  List<String> decode_chain(List<String> tokens) {
    final List<String> result = [];
    for (int i = 0; i < tokens.length; ++i) {
      String normalized = tokens[i].replaceAll(replacement, ' ');
      if (addPrefixSpace && i == 0 && normalized.startsWith(' ')) {
        normalized = normalized.substring(1);
      }
      result.add(normalized);
    }
    return result;
  }
}

/// A normalizer that applies a precompiled charsmap.
/// This is useful for applying complex normalizations in C++ and exposing them to JavaScript.
/// @extends Normalizer
/// @param {Object} config The configuration object for the Precompiled normalizer.
/// @param {Object} config.precompiled_charsmap The precompiled charsmap object.
class Precompiled extends Normalizer {
  dynamic charsmap;

  /// Create a new instance of Precompiled normalizer.
  /// @param {Object} config The configuration object.
  /// @param {any} config.precompiled_charsmap Precompiled chars mapping.
  Precompiled(Map<String, dynamic> config) : super(config) {
    charsmap = config['precompiled_charsmap'];
  }

  /// Normalizes the given text by applying the precompiled charsmap.
  /// @param {string} text The text to normalize.
  /// @returns {string} The normalized text.
  @override
  String normalize(String text) {
    // As stated in the sentencepiece normalization docs (https://github.com/google/sentencepiece/blob/master/doc/normalization.md#use-pre-defined-normalization-rule),
    // there are 5 pre-defined normalization rules:
    //  1. nmt_nfkc: NFKC normalization with some additional normalization around spaces. (default)
    //  2. nfkc: original NFKC normalization.
    //  3. nmt_nfkc_cf: nmt_nfkc + Unicode case folding (mostly lower casing)
    //  4. nfkc_cf: nfkc + Unicode case folding.
    //  5. identity: no normalization
    //
    // For now, we only implement the default (nmt_nfkc).
    // See https://raw.githubusercontent.com/google/sentencepiece/master/data/nmt_nfkc.tsv for the full list of rules.
    // TODO: detect when a different `this.charsmap` is used.

    // Remove control characters
    text = text.replaceAll(
      RegExp(r'[\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]', multiLine: true),
      '',
    );
    // Replace certain characters with a space
    text = text.replaceAll(
      RegExp(r'[\u0009\u000A\u000C\u000D\u00A0\u1680\u2000-\u200F\u2028\u2029\u202F\u205F\u2581\u3000\uFEFF\uFFFD]', multiLine: true),
      '\u0020',
    );

    if (text.contains('\uFF5E')) {
      // To match the sentencepiece implementation 100%, we must handle a very strange edge-case.
      // For some reason, the "Fullwidth Tilde" character (\uFF5E) should not be converted to the standard Tilde character (\u007E).
      // However, NFKC normalization does do this conversion. As a result, we split the string on the Fullwidth Tilde character,
      // perform NFKC normalization on each substring, and then join them back together with the Fullwidth Tilde character.
      final parts = text.split('\uFF5E');
      text = parts.map((part) => unorm.nfkc(part)).join('\uFF5E');
    } else {
      text = unorm.nfkc(text);
    }

    return text;
  }
}

/// A pre-tokenizer that applies a sequence of pre-tokenizers to the input text.
/// @extends PreTokenizer
class PreTokenizerSequence extends PreTokenizer {
  late List<PreTokenizer> tokenizers;

  /// Creates an instance of PreTokenizerSequence.
  /// @param {Object} config The configuration object for the pre-tokenizer sequence.
  /// @param {Object[]} config.pretokenizers An array of pre-tokenizer configurations.
  PreTokenizerSequence(Map<String, dynamic> config) : super(config) {
    tokenizers = List<Map<String, dynamic>>.from(config['pretokenizers']).map((x) => PreTokenizer.fromConfig(x)!).toList();
  }

  /// Applies each pre-tokenizer in the sequence to the input text in turn.
  /// @param {string} text The text to pre-tokenize.
  /// @param {Object} [options] Additional options for the pre-tokenization logic.
  /// @returns {string[]} The pre-tokenized text.
  @override
  List<String> pre_tokenize_text(String text, [Map<String, dynamic>? options]) {
    // Use reduce to apply each tokenizer to the text
    return tokenizers.fold([text], (preTokenizedText, tokenizer) {
      return tokenizer.pre_tokenize(preTokenizedText, options);
    });
  }
}

/// Splits a string of text by whitespace characters into individual tokens.
/// @extends PreTokenizer
class WhitespaceSplit extends PreTokenizer {
  /// Creates an instance of WhitespaceSplit.
  /// @param {Object} config The configuration object for the pre-tokenizer.
  WhitespaceSplit(super.config);

  /// Pre-tokenizes the input text by splitting it on whitespace characters.
  /// @param {string} text The text to be pre-tokenized.
  /// @param {Object} [options] Additional options for the pre-tokenization logic.
  /// @returns {string[]} An array of tokens produced by splitting the input text on whitespace.
  @override
  List<String> pre_tokenize_text(String text, [Map<String, dynamic>? options]) {
    return whitespace_split(text);
  }
}

/// @typedef {Object} EncodingSingle
/// @property {number[]} input_ids List of token ids to be fed to a model.
/// @property {number[]} attention_mask List of token type ids to be fed to a model
/// @property {number[]} [token_type_ids] List of indices specifying which tokens should be attended to by the model
class EncodingSingle {
  /// List of token ids to be fed to a model.
  List<int> input_ids;

  /// List of indices specifying which tokens should be attended to by the model
  List<int> attention_mask;

  /// List of token type ids to be fed to a model
  List<int> token_type_ids;

  EncodingSingle({
    this.input_ids = const [],
    this.attention_mask = const [],
    this.token_type_ids = const [],
  });

  @override
  String toString() => jsonEncode({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids,
  });
}

class BatchEncoding {
  /// List of token ids to be fed to a model.
  List<List<int>> input_ids;

  /// List of indices specifying which tokens should be attended to by the model
  List<List<int>> attention_mask;

  /// List of token type ids to be fed to a model
  List<List<int>> token_type_ids;

  BatchEncoding({
    this.input_ids = const [],
    this.attention_mask = const [],
    this.token_type_ids = const [],
  });

  @override
  String toString() => jsonEncode({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids,
  });
}

class PreTrainedTokenizer {
  bool return_token_type_ids = false;

  String padding_side = 'right';

  late Map<String, dynamic> _tokenizer_config;

  Normalizer? normalizer;

  PreTokenizer? pre_tokenizer;

  late TokenizerModel model;

  PostProcessor? post_processor;

  Decoder? decoder;

  List<String> special_tokens = [];

  List<num> all_special_ids = [];

  List<AddedToken> added_tokens = [];

  List<String> additional_special_tokens = [];

  late DictionarySplitter added_tokens_splitter;

  Map<String, AddedToken> added_tokens_map = {};

  String? mask_token;

  int? mask_token_id;

  String? pad_token;

  int? pad_token_id;

  String? sep_token;

  int? sep_token_id;

  String? unk_token;

  int? unk_token_id;

  String? bos_token;

  int? bos_token_id;

  String? eos_token;

  int? eos_token_id;

  late num model_max_length;

  /// Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
  bool? remove_space;

  late bool clean_up_tokenization_spaces;

  late bool do_lowercase_and_remove_accent;

  bool legacy = false;

  late dynamic chat_template;

  Map _compiled_template_cache = {};

  PreTrainedTokenizer(Map<String, dynamic> tokenizerJSON, Map<String, dynamic> tokenizerConfig) {
    _tokenizer_config = tokenizerConfig;

    // Construct parts of the tokenizer from the JSON
    normalizer = Normalizer.fromConfig(tokenizerJSON['normalizer']);
    pre_tokenizer = PreTokenizer.fromConfig(tokenizerJSON['pre_tokenizer']);
    model = TokenizerModel.fromConfig(tokenizerJSON['model'], tokenizerConfig);
    post_processor = PostProcessor.fromConfig(tokenizerJSON['post_processor']);
    decoder = Decoder.fromConfig(tokenizerJSON['decoder']);

    for (final addedToken in tokenizerJSON['added_tokens']) {
      final token = AddedToken(addedToken);
      added_tokens.add(token);

      model.tokens_to_ids[token.content] = token.id;
      model.vocab[token.id] = token.content;

      if (token.special) {
        special_tokens.add(token.content);
        all_special_ids.add(token.id);
      }
    }

    // Update additional_special_tokens
    additional_special_tokens = tokenizerConfig['additional_special_tokens'] ?? [];
    special_tokens.addAll(additional_special_tokens);
    special_tokens = special_tokens.toSet().toList(); // Remove duplicates

    if (decoder != null) {
      // Slight hack, but it prevents code duplication:
      decoder!.added_tokens = added_tokens;

      // Another slight hack to add `end_of_word_suffix` (if present) to the decoder
      // This is needed for cases where BPE model and ByteLevel decoder are used
      // For more information, see https://github.com/huggingface/transformers.js/issues/74
      // TODO: save this to the decoder when exporting?
      decoder!.end_of_word_suffix = model.end_of_word_suffix;
    }

    added_tokens_splitter = DictionarySplitter(
      added_tokens.map((x) => x.content),
    );

    added_tokens_map = Map.fromEntries(added_tokens.map((x) => MapEntry(x.content, x)));

    // Set mask token if present (otherwise will be undefined, which is fine)
    mask_token = getToken(['mask_token']);
    mask_token_id = model.tokens_to_ids[mask_token];

    pad_token = getToken(['pad_token', 'eos_token']);
    pad_token_id = model.tokens_to_ids[pad_token];

    sep_token = getToken(['sep_token']);
    sep_token_id = model.tokens_to_ids[sep_token];

    unk_token = getToken(['unk_token']);
    unk_token_id = model.tokens_to_ids[unk_token];

    bos_token = getToken(['bos_token']);
    bos_token_id = model.tokens_to_ids[bos_token];

    eos_token = getToken(['eos_token']);
    eos_token_id = model.tokens_to_ids[eos_token];

    model_max_length = tokenizerConfig['model_max_length'];

    remove_space = tokenizerConfig['remove_space'];

    clean_up_tokenization_spaces = tokenizerConfig['clean_up_tokenization_spaces'] ?? true;
    do_lowercase_and_remove_accent = tokenizerConfig['do_lowercase_and_remove_accent'] ?? false;

    if ((tokenizerConfig['padding_side'] as String?)?.isNotEmpty == true) {
      padding_side = tokenizerConfig['padding_side'];
    }

    chat_template = tokenizerConfig['chat_template'];
    if (chat_template is List) {
      // Chat templates are stored as lists of dicts with fixed key names,
      // we reconstruct that into a single dict while loading them.
      Map<String, dynamic> chat_template = {};
      for (final ct in this.chat_template) {
        final name = ct['name'];
        final template = ct['name'];
        if (name !is String || template !is String) {
          throw Exception('Chat template must be a list of objects with "name" and "template" properties');
        }
        chat_template[name] = template;
      }
      this.chat_template = chat_template;
    }
  }

  /// Returns the value of the first matching key in the tokenizer config object.
  /// @param {...string} keys One or more keys to search for in the tokenizer config object.
  /// @returns {string|null} The value associated with the first matching key, or null if no match is found.
  /// @throws {Error} If an object is found for a matching key and its __type property is not "AddedToken".
  /// @private
  String? getToken(List<String> keys) {
    for (final key in keys) {
      final item = _tokenizer_config[key];

      if (item == null) continue;

      if (item is Object) {
        // if (item.__type == 'AddedToken') {
        //   return item.content;
        // } else {
        //   throw Exception('Unknown token: $item');
        // }
        throw Exception('Unknown token: $item');
      } else {
        return item;
      }
    }
    return null;
  }

  /// Loads a pre-trained tokenizer from the given `pretrained_model_name_or_path`.
  ///
  /// @param {string} pretrained_model_name_or_path The path to the pre-trained tokenizer.
  /// @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
  ///
  /// @throws {Error} Throws an error if the tokenizer.json or tokenizer_config.json files are not found in the `pretrained_model_name_or_path`.
  /// @returns {Promise<PreTrainedTokenizer>} A new instance of the `PreTrainedTokenizer` class.
  static Future<PreTrainedTokenizer> from_pretrained(pretrained_model_name_or_path, PretrainedTokenizerOptions options) async {
    final info = await loadTokenizer(pretrained_model_name_or_path, options);

    return PreTrainedTokenizer(info.$1, info.$2);
  }

  /// Encode/tokenize the given text(s).
  /// @param {string|string[]} text The text to tokenize.
  /// @param {Object} options An optional object containing the following properties:
  /// @param {string|string[]} [options.text_pair=null] Optional second sequence to be encoded. If set, must be the same type as text.
  /// @param {boolean|'max_length'} [options.padding=false] Whether to pad the input sequences.
  /// @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
  /// @param {boolean} [options.truncation=null] Whether to truncate the input sequences.
  /// @param {number} [options.max_length=null] Maximum length of the returned list and optionally padding length.
  /// @param {boolean} [options.return_tensor=true] Whether to return the results as Tensors or arrays.
  /// @param {boolean} [options.return_token_type_ids=null] Whether to return the token type ids.
  /// @returns {BatchEncoding} Object to be passed to the model.
  BatchEncoding call(List<String> text, {
    List<String>? text_pair,
    bool padding = false,
    bool add_special_tokens = true,
    bool? truncation,
    num? max_length,
    bool return_tensor = true,
    bool? return_token_type_ids,
  }) {
    List<EncodingSingle> encodedTokens = [];

    if (text.isEmpty) {
      throw ArgumentError('text array must be non-empty');
    }

    if (text_pair != null) {
      if (text.length != text_pair.length) {
        throw ArgumentError('text and text_pair must have the same length');
      }

      encodedTokens = text.indexed.map((e) {
        final (i, t) = e;

        return _encode_plus(
          t,
          text_pair: text_pair[i],
          add_special_tokens: add_special_tokens,
          return_token_type_ids: return_token_type_ids,
        );
      }).toList();
    } else {
      encodedTokens = text.map((x) => _encode_plus(
        x,
        add_special_tokens: add_special_tokens,
        return_token_type_ids: return_token_type_ids,
      )).toList();
    }

    // At this point, `encodedTokens` is batched, of shape [batch_size, tokens].
    // However, array may be jagged. So, we may need pad to max_length.
    if (max_length == null) {
      max_length = model_max_length;
    } else if (truncation == null) {
      if (padding == true) {
        print('`max_length` is ignored when `padding: true` and there is no truncation strategy. To pad to max length, use `padding: \'max_length\'`.');
        max_length = model_max_length;
      } else if (padding == false) {
        print('Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation: true` to explicitly truncate examples to max length.');
        truncation = true;
      }
    }

    // padding: 'max_length' doesn't require any additional calculation
    // but padding: true has to calculate max_length from the sequences
    if (padding == true) {
      max_length = min(encodedTokens.map((x) => x.input_ids.length).reduce(max), max_length ?? double.infinity);
    }

    // Ensure it is less than model max length
    max_length = min(max_length, model_max_length ?? double.infinity);

    if (padding || truncation == true) {
      // Perform padding and/or truncation
      for (int i = 0; i < encodedTokens.length; ++i) {
        if (encodedTokens[i].input_ids.length == max_length) {
          continue;
        } else if (encodedTokens[i].input_ids.length > max_length) {
          // possibly truncate
          if (truncation == true) {
            encodedTokens[i].input_ids = encodedTokens[i].input_ids.sublist(0, max_length.toInt());
            encodedTokens[i].attention_mask = encodedTokens[i].attention_mask.sublist(0, max_length.toInt());
            encodedTokens[i].token_type_ids = encodedTokens[i].token_type_ids.sublist(0, max_length.toInt());
          }
        } else { // t.length < max_length
          // possibly pad
          if (padding) {
            final padding_amount = max(0, max_length.toInt() - encodedTokens[i].input_ids.length);

            if (padding_side == 'right') {
              encodedTokens[i].input_ids = encodedTokens[i].input_ids + List.filled(padding_amount, pad_token_id!);
              encodedTokens[i].attention_mask = encodedTokens[i].attention_mask + List.filled(padding_amount, 0);
              encodedTokens[i].token_type_ids = encodedTokens[i].token_type_ids + List.filled(padding_amount, 0);
            } else {
              encodedTokens[i].input_ids = List.filled(padding_amount, pad_token_id!) + encodedTokens[i].input_ids;
              encodedTokens[i].attention_mask = List.filled(padding_amount, 0) + encodedTokens[i].attention_mask;
              encodedTokens[i].token_type_ids = List.filled(padding_amount, 0) + encodedTokens[i].token_type_ids;
            }
          }
        }
      }
    }

    final BatchEncoding result = BatchEncoding();

    if (return_tensor && false) {
      if (!(padding && truncation == true)) {
        // Not, guaranteed that all items have same length, so
        // we perform additional check
        if (
          encodedTokens.any((x) {
            if (x.attention_mask.length != encodedTokens.first.attention_mask.length) {
              return true;
            }
            if (x.input_ids.length != encodedTokens.first.input_ids.length) {
              return true;
            }
            if (x.token_type_ids.length != encodedTokens.first.token_type_ids.length) {
              return true;
            }

            return false;
          })
        ) {
          throw Exception("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=true' and 'truncation=true' to have batched tensors with the same length.");
        }
      }

      // Now we actually convert to tensor
      // NOTE: In the same way as the python library, we return a batched tensor, regardless of
      // whether we have a single input or multiple inputs.
      final dims = [encodedTokens.length, encodedTokens[0].input_ids.length];

      // TODO: Need to figure out library for tensor
      // for (const key of Object.keys(encodedTokens[0])) {
      //   result[key] = new Tensor('int64',
      //   BigInt64Array.from(encodedTokens.flatMap(x => x[key]).map(BigInt)),
      //   dims
      //   );
      // }
    } else {
      result.input_ids = encodedTokens.map((x) => x.input_ids).toList();
      result.attention_mask = encodedTokens.map((x) => x.attention_mask).toList();
      result.token_type_ids = encodedTokens.map((x) => x.token_type_ids).toList();

      // // If not returning a tensor, we match the input type
      // if (!isBatched) {
      //   // Input was not batched, so we unwrap
      //   for (const key of Object.keys(result)) {
      //     result[key] = result[key][0];
      //   }
      // }

      // for (const key of Object.keys(encodedTokens[0])) {
      //   result[key] = encodedTokens.map(x => x[key]);
      // }
      // result = BatchEncoding(
      //   input_ids: encodedTokens.map((x) => x.input_ids).toList(),
      //   attention_mask: encodedTokens.map((x) => x.attention_mask).toList(),
      //   token_type_ids: encodedTokens.map((x) => x.token_type_ids).toList(),
      // );
    }

    return result;
  }

  /// Encodes a single text using the preprocessor pipeline of the tokenizer.
  ///
  /// @param {string|null} text The text to encode.
  /// @returns {string[]|null} The encoded tokens.
  List<String>? _encode_text(String? text) {
    if (text == null) return null;

    // Actual function which does encoding, for a single text
    // First, we take care of special tokens. Needed to avoid issues arising from
    // normalization and/or pretokenization (which may not preserve special tokens)
    final sections = added_tokens_splitter.split(text);

    // Process left/right stripping of added tokens
    for (int i = 0; i < sections.length; ++i) {
      final addedToken = added_tokens_map[sections[i]];
      if (addedToken != null) {
        if (addedToken.lstrip && i > 0) {
          sections[i - 1] = sections[i - 1].trimRight();
        }
        if (addedToken.rstrip && i < sections.length - 1) {
          sections[i + 1] = sections[i + 1].trimLeft();
        }
      }
    }

    final tokens = sections.indexed.map<List<String>>((e) {
      var (section_index, x) = e;

      if (x.isEmpty) return [];
      if (added_tokens_map.containsKey(x)) return [x]; // Return added tokens unchanged

      if (remove_space == true) {
        x = x.trim().split(RegExp(r'\s+')).join(' ');
      }
      /// TODO
      // if (do_lowercase_and_remove_accent) {
      //   x = lowercase_and_remove_accent(x);
      // }

      if (normalizer != null) {
        x = normalizer!(x);
      }

      // If, after normalization, this section is empty (e.g., trimming whitespace),
      // we return an empty array
      if (x.isEmpty) {
        return [];
      }

      final sectionTokens = pre_tokenizer != null ? pre_tokenizer!(x, {
        'section_index': section_index,
      }) : [x];

      final List<String> tokens = model(sectionTokens);

      return tokens;
    }).expand((e) => e).toList();

    return tokens;
  }

  /// Encodes a single text or a pair of texts using the model's tokenizer.
  ///
  /// @param {string} text The text to encode.
  /// @param {Object} options An optional object containing the following properties:
  /// @param {string} [options.text_pair=null] The optional second text to encode.
  /// @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
  /// @param {boolean} [options.return_token_type_ids=null] Whether to return token_type_ids.
  /// @returns {EncodingSingle} An object containing the encoded text.
  /// @private
  EncodingSingle _encode_plus(String text, {
    String? text_pair,
    bool add_special_tokens = true,
    bool? return_token_type_ids,
  }) {
    final (tokens, token_type_ids) = _tokenize_helper(text, pair: text_pair, add_special_tokens: add_special_tokens);

    final input_ids = model.convert_tokens_to_ids(tokens);

    final result = EncodingSingle(
      input_ids: input_ids,
      attention_mask: List.filled(input_ids.length, 1),
    );
    if ((return_token_type_ids ?? this.return_token_type_ids) && token_type_ids != null) {
      result.token_type_ids = token_type_ids;
    }
    return result;
  }

  /// Internal helper function to tokenize a text, and optionally a pair of texts.
  /// @param {string} text The text to tokenize.
  /// @param {Object} options An optional object containing the following properties:
  /// @param {string} [options.pair=null] The optional second text to tokenize.
  /// @param {boolean} [options.add_special_tokens=false] Whether or not to add the special tokens associated with the corresponding model.
  /// @returns {{tokens: string[], token_type_ids?: number[]}} An object containing the tokens and optionally the token type IDs.
  (List<String>, List<int>?) _tokenize_helper(String text, {
    String? pair,
    bool add_special_tokens = false,
  }) {
    final tokens = _encode_text(text);
    final tokens2 = _encode_text(pair);

    if (post_processor != null) {
      final post_processed = post_processor!(tokens, tokens2, add_special_tokens);
      return (post_processed.tokens, post_processed.token_type_ids);
    }

    return ((tokens ?? []) + (tokens2 ?? []), null);
  }

  /// Converts a string into a sequence of tokens.
  /// @param {string} text The sequence to be encoded.
  /// @param {Object} options An optional object containing the following properties:
  /// @param {string} [options.pair] A second sequence to be encoded with the first.
  /// @param {boolean} [options.add_special_tokens=false] Whether or not to add the special tokens associated with the corresponding model.
  /// @returns {string[]} The list of tokens.
  List<String> tokenize(String text, {
    String? pair,
    bool add_special_tokens = false,
  }) {
    return _tokenize_helper(text, pair: pair, add_special_tokens: add_special_tokens).$1;
  }

  /// Encodes a single text or a pair of texts using the model's tokenizer.
  ///
  /// @param {string} text The text to encode.
  /// @param {Object} options An optional object containing the following properties:
  /// @param {string} [options.text_pair=null] The optional second text to encode.
  /// @param {boolean} [options.add_special_tokens=true] Whether or not to add the special tokens associated with the corresponding model.
  /// @param {boolean} [options.return_token_type_ids=null] Whether to return token_type_ids.
  /// @returns {number[]} An array of token IDs representing the encoded text(s).
  List<num> encode(String text, {
    String? text_pair,
    bool add_special_tokens = true,
    bool? return_token_type_ids,
  }) {
    return _encode_plus(
      text,
      text_pair: text_pair,
      add_special_tokens: add_special_tokens,
      return_token_type_ids: return_token_type_ids,
    ).input_ids;
  }

  /// Decode a batch of tokenized sequences.
  /// @param {number[][]|Tensor} batch List/Tensor of tokenized input sequences.
  /// @param {Object} decode_args (Optional) Object with decoding arguments.
  /// @returns {string[]} List of decoded sequences.
  List<String> batch_decode(List<List<int>> batch, {
    bool skip_special_tokens = false,
    bool clean_up_tokenization_spaces = true,
  }) {
    // if (batch instanceof Tensor) {
    //   batch = batch.tolist();
    // }
    return batch.map((x) => decode(
      x,
      skip_special_tokens: skip_special_tokens,
      clean_up_tokenization_spaces: clean_up_tokenization_spaces,
    )).toList();
  }

  /// Decodes a sequence of token IDs back to a string.
  ///
  /// @param {number[]|bigint[]|Tensor} token_ids List/Tensor of token IDs to decode.
  /// @param {Object} [decode_args={}]
  /// @param {boolean} [decode_args.skip_special_tokens=false] If true, special tokens are removed from the output string.
  /// @param {boolean} [decode_args.clean_up_tokenization_spaces=true] If true, spaces before punctuations and abbreviated forms are removed.
  ///
  /// @returns {string} The decoded string.
  /// @throws {Error} If `token_ids` is not a non-empty array of integers.
  String decode(List<int> token_ids, {
    bool skip_special_tokens = false,
    bool clean_up_tokenization_spaces = true,
  }) {
    // if (token_ids instanceof Tensor) {
    //   token_ids = prepareTensorForDecode(token_ids);
    // }

    if (token_ids.isEmpty) {
      throw ArgumentError('token_ids must be a non-empty array of integers.');
    }

    return decode_single(
      token_ids,
      skip_special_tokens: skip_special_tokens,
      clean_up_tokenization_spaces: clean_up_tokenization_spaces,
    );
  }

  /// Decode a single list of token ids to a string.
  /// @param {number[]|bigint[]} token_ids List of token ids to decode
  /// @param {Object} decode_args Optional arguments for decoding
  /// @param {boolean} [decode_args.skip_special_tokens=false] Whether to skip special tokens during decoding
  /// @param {boolean} [decode_args.clean_up_tokenization_spaces=null] Whether to clean up tokenization spaces during decoding.
  /// If null, the value is set to `this.decoder.cleanup` if it exists, falling back to `this.clean_up_tokenization_spaces` if it exists, falling back to `true`.
  /// @returns {string} The decoded string
  String decode_single(List<int> token_ids, {
    bool skip_special_tokens = false,
    bool? clean_up_tokenization_spaces,
  }) {
    var tokens = model.convert_ids_to_tokens(token_ids);
    if (skip_special_tokens) {
      tokens = tokens.where((x) => !special_tokens.contains(x)).toList();
    }

    // If `this.decoder` is null, we just join tokens with a space:
    // https://github.com/huggingface/tokenizers/blob/8edec536a737cb04494b454805be16c020abb14f/tokenizers/src/tokenizer/mod.rs#L835
    /** @type {string} */
    String decoded = decoder != null ? decoder!(tokens) : tokens.join(' ');

    // Slight hack, but prevents having to pass `skip_special_tokens` to
    // each call to `decode`, which would lead to code duplication.
    if (decoder != null && decoder!.end_of_word_suffix != null) {
      decoded = decoded.replaceAll(decoder!.end_of_word_suffix!, ' ');
      if (skip_special_tokens) {
        decoded = decoded.trim();
      }
    }

    if (clean_up_tokenization_spaces ?? this.clean_up_tokenization_spaces) {
      decoded = clean_up_tokenization(decoded);
    }

    return decoded;
  }
}

/// Helper class which is used to instantiate pretrained tokenizers with the `from_pretrained` function.
/// The chosen tokenizer class is determined by the type specified in the tokenizer config.
///
/// @example
/// const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
class AutoTokenizer {
  static final TOKENIZER_CLASS_MAPPING = {
    // 'T5Tokenizer': T5Tokenizer,
    // 'DistilBertTokenizer': DistilBertTokenizer,
    // 'CamembertTokenizer': CamembertTokenizer,
    // 'DebertaTokenizer': DebertaTokenizer,
    // 'DebertaV2Tokenizer': DebertaV2Tokenizer,
    // 'BertTokenizer': BertTokenizer,
    // 'HerbertTokenizer': HerbertTokenizer,
    // 'ConvBertTokenizer': ConvBertTokenizer,
    // 'RoFormerTokenizer': RoFormerTokenizer,
    // 'XLMTokenizer': XLMTokenizer,
    // 'ElectraTokenizer': ElectraTokenizer,
    // 'MobileBertTokenizer': MobileBertTokenizer,
    // 'SqueezeBertTokenizer': SqueezeBertTokenizer,
    // 'AlbertTokenizer': AlbertTokenizer,
    // 'GPT2Tokenizer': GPT2Tokenizer,
    // 'BartTokenizer': BartTokenizer,
    // 'MBartTokenizer': MBartTokenizer,
    // 'MBart50Tokenizer': MBart50Tokenizer,
    // 'RobertaTokenizer': RobertaTokenizer,
    // 'WhisperTokenizer': WhisperTokenizer,
    // 'CodeGenTokenizer': CodeGenTokenizer,
    // 'CLIPTokenizer': CLIPTokenizer,
    // 'SiglipTokenizer': SiglipTokenizer,
    // 'MarianTokenizer': MarianTokenizer,
    // 'BloomTokenizer': BloomTokenizer,
    // 'NllbTokenizer': NllbTokenizer,
    // 'M2M100Tokenizer': M2M100Tokenizer,
    // 'LlamaTokenizer': LlamaTokenizer,
    // 'CodeLlamaTokenizer': CodeLlamaTokenizer,
    // 'XLMRobertaTokenizer': XLMRobertaTokenizer,
    // 'MPNetTokenizer': MPNetTokenizer,
    // 'FalconTokenizer': FalconTokenizer,
    // 'GPTNeoXTokenizer': GPTNeoXTokenizer,
    // 'EsmTokenizer': EsmTokenizer,
    // 'Wav2Vec2CTCTokenizer': Wav2Vec2CTCTokenizer,
    // 'BlenderbotTokenizer': BlenderbotTokenizer,
    // 'BlenderbotSmallTokenizer': BlenderbotSmallTokenizer,
    // 'SpeechT5Tokenizer': SpeechT5Tokenizer,
    // 'NougatTokenizer': NougatTokenizer,
    // 'VitsTokenizer': VitsTokenizer,
    // 'Qwen2Tokenizer': Qwen2Tokenizer,
    // 'GemmaTokenizer': GemmaTokenizer,
    // 'Grok1Tokenizer': Grok1Tokenizer,
    // 'CohereTokenizer': CohereTokenizer,
    // 'MgpstrTokenizer': MgpstrTokenizer,

    // Base case:
    'PreTrainedTokenizer': (Map<String, dynamic> tokenizerJSON, Map<String, dynamic> tokenizerConfig) => PreTrainedTokenizer(tokenizerJSON, tokenizerConfig),
  };


  /// Instantiate one of the tokenizer classes of the library from a pretrained model.
  ///
  /// The tokenizer class to instantiate is selected based on the `tokenizer_class` property of the config object
  /// (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
  ///
  /// @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
  /// - A string, the *model id* of a pretrained tokenizer hosted inside a model repo on huggingface.co.
  ///   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
  ///   user or organization name, like `dbmdz/bert-base-german-cased`.
  /// - A path to a *directory* containing tokenizer files, e.g., `./my_model_directory/`.
  /// @param {PretrainedTokenizerOptions} options Additional options for loading the tokenizer.
  ///
  /// @returns {Promise<PreTrainedTokenizer>} A new instance of the PreTrainedTokenizer class.
  static Future<PreTrainedTokenizer> from_pretrained(String pretrained_model_name_or_path, {
    ProgressCallback? progress_callback,
    PretrainedConfig? config,
    String? cache_dir,
    bool local_files_only = false,
    String revision = 'main',
    bool? legacy,
  }) async {
    final (tokenizerJSON, tokenizerConfig) = await loadTokenizer(
        pretrained_model_name_or_path,
        PretrainedTokenizerOptions(
      progress_callback: progress_callback,
      config: config,
      cache_dir: cache_dir,
      local_files_only: local_files_only,
      revision: revision,
      legacy: legacy,
    ));

    // Some tokenizers are saved with the "Fast" suffix, so we remove that if present.
    final tokenizerName = (tokenizerConfig['tokenizer_class'] as String?)?.replaceAll(RegExp(r'Fast$'), '') ?? 'PreTrainedTokenizer';

    var cls = TOKENIZER_CLASS_MAPPING[tokenizerName];
    if (cls == null) {
      print('Unknown tokenizer class "$tokenizerName", attempting to construct from base class.');
      cls = TOKENIZER_CLASS_MAPPING['PreTrainedTokenizer'];
    }
    return cls!(tokenizerJSON, tokenizerConfig);
  }
}
