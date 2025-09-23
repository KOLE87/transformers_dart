import 'dart:math';

/// Efficient Heap-based Implementation of a Priority Queue.
/// It uses an array-based binary heap, where the root is at index `0`, and the
/// children of node `i` are located at indices `2i + 1` and `2i + 2`, respectively.
///
/// Adapted from the following sources:
/// - https://stackoverflow.com/a/42919752/13989043 (original)
/// - https://github.com/belladoreai/llama-tokenizer-js (minor improvements)
class PriorityQueue<T> {
  final List<T> _heap = [];

  late bool Function(T a, T b) _comparator;

  late double _maxSize;

  /// Create a new PriorityQueue.
  /// @param {function(any, any): boolean} comparator Comparator function to determine priority. Defaults to a MaxHeap.
  PriorityQueue(bool Function(T a, T b) comparator, [double? maxSize]) {
    _comparator = comparator;
    _maxSize = maxSize ?? double.infinity;
  }

  /// The size of the queue
  get size => _heap.length;

  /// Check if the queue is empty.
  /// @returns {boolean} `true` if the queue is empty, `false` otherwise.
  bool isEmpty() => _heap.isEmpty;

  /// Check if the queue is not empty.
  bool isNotEmpty() => _heap.isNotEmpty;

  /// Return the element with the highest priority in the queue.
  /// @returns {any} The highest priority element in the queue.
  T peek() => _heap[0];

  /// Add one or more elements to the queue.
  /// @param  {...any} values The values to push into the queue.
  /// @returns {number} The new size of the queue.
  int push(List<T> values) => extend(values);

  /// Add multiple elements to the queue.
  /// @param {any[]} values The values to push into the queue.
  /// @returns {number} The new size of the queue.
  int extend(List<T> values) {
    for (final value in values) {
      if (size < _maxSize) {
        _heap.add(value);
        _siftUp();
      } else {
        // Get index of value with the lowest priority
        final smallest = _smallest();

        // If the new value has higher priority than the smallest value in the heap
        // then replace the smallest value with the new value and update the heap
        if (_comparator(value, _heap[smallest])) {
          _heap[smallest] = value;
          _siftUpFrom(smallest);
        }
      }
    }
    return size;
  }

  /// Remove and return the element with the highest priority in the queue.
  /// @returns {any} The element with the highest priority in the queue.
  T pop() {
    final poppedValue = peek();
    final bottom = this.size - 1;
    if (bottom > 0) {
      _swap(0, bottom);
    }
    _heap.removeLast();
    _siftDown();
    return poppedValue;
  }

  /// Replace the element with the highest priority in the queue with a new value.
  /// @param {*} value The new value.
  /// @returns {*} The replaced value.
  T replace(T value) {
    final replacedValue = peek();
    _heap[0] = value;
    _siftDown();
    return replacedValue;
  }

  /// Compute the index for the parent of the node at index `i`.
  /// @param {number} i The index of the node to get the parent of.
  /// @returns {number} The index of the parent node.
  /// @private
  int _parent(int i) => ((i + 1) >>> 1) - 1;

  /// Compute the index for the left child of the node at index `i`.
  /// @param {number} i The index of the node to get the left child of.
  /// @returns {number} The index of the left child.
  /// @private
  int _left(int i) => (i << 1) + 1;

  /// Compute the index for the right child of the node at index `i`.
  /// @param {number} i The index of the node to get the right child of.
  /// @returns {number} The index of the right child.
  /// @private
  int _right(int i) => (i + 1) << 1;

  /// Check if the element at index `i` is greater than the element at index `j`.
  /// @param {number} i The index of the first element to compare.
  /// @param {number} j The index of the second element to compare.
  /// @returns {boolean} `true` if the element at index `i` is greater than the element at index `j`, `false` otherwise.
  /// @private
  bool _greater(int i, int j) => _comparator(this._heap[i], this._heap[j]);

  /// Swap the elements at indices `i` and `j`.
  /// @param {number} i The index of the first element to swap.
  /// @param {number} j The index of the second element to swap.
  /// @private
  void _swap(int i, int j) {
    final temp = _heap[i];
    _heap[i] = _heap[j];
    _heap[j] = temp;
  }

  /// Maintain the heap property by updating positions in the heap,
  /// starting at the last element and moving up the heap.
  /// @private
  void _siftUp() {
    _siftUpFrom(size - 1);
  }

  /// Helper function to sift up from a given node.
  /// @param {number} node The index of the node to start sifting up from.
  void _siftUpFrom(int node) {
    while (node > 0 && _greater(node, _parent(node))) {
      _swap(node, _parent(node));
      node = _parent(node);
    }
  }

  /// Maintain the heap property by updating positions in the heap,
  /// starting at the first element and moving down the heap.
  /// @private
  void _siftDown() {
    int node = 0;
    while (
      (_left(node) < size && _greater(_left(node), node)) ||
      (_right(node) < size && _greater(_right(node), node))
    ) {
      final maxChild = (_right(node) < size && _greater(_right(node), _left(node)))
        ? _right(node) : _left(node);
      _swap(node, maxChild);
      node = maxChild;
    }
  }

  /// Get the index of the smallest element in the heap. Since we use an array-based heap,
  /// the index can be computed without needing to traverse the heap.
  /// @private
  int _smallest() => (pow(2, (log(size) / size(2)).floor()) - 1).toInt();
}

/// A trie structure to efficiently store and search for strings.
class CharTrie {
  CharTrieNode root = CharTrieNode.defaultNode();

  CharTrie();

  /// Adds one or more `texts` to the trie.
  /// @param {string[]} texts The strings to add to the trie.
  void extend(List<String> texts) {
    for (final text in texts) {
      push(text);
    }
  }

  /// Adds text to the trie.
  /// @param {string} text The string to add to the trie.
  void push(String text) {
    CharTrieNode node = root;
    for (final char in text.split('')) {
      CharTrieNode? child = node.children[char];
      if (child == null) {
        child = CharTrieNode.defaultNode();
        node.children[char] = child;
      }
      node = child;
    }
    node.isLeaf = true;
  }

  /// Searches the trie for all strings with a common prefix of `text`.
  /// @param {string} text The common prefix to search for.
  /// @yields {string} Each string in the trie that has `text` as a prefix.
  Iterable<String> commonPrefixSearch(String text) sync* {
    CharTrieNode? node = root;

    String prefix = "";
    for (final char in text.split('')) {
      prefix += char;
      node = node!.children[char];
      if (node == null) return;

      if (node.isLeaf) {
        yield prefix;
      }
    }
  }
}

/// Represents a node in a character trie.
class CharTrieNode {
  bool isLeaf;

  Map<String, CharTrieNode> children;

  /// Create a new CharTrieNode.
  /// @param {boolean} isLeaf Whether the node is a leaf node or not.
  /// @param {Map<string, CharTrieNode>} children A map containing the node's children, where the key is a character and the value is a `CharTrieNode`.
  CharTrieNode(this.isLeaf, this.children);

  /// Returns a new `CharTrieNode` instance with default values.
  /// @returns {CharTrieNode} A new `CharTrieNode` instance with `isLeaf` set to `false` and an empty `children` map.
  static CharTrieNode defaultNode() {
    return CharTrieNode(false, <String, CharTrieNode>{});
  }
}

/// A lattice data structure to be used for tokenization.
class TokenLattice {
  int? bosTokenId;

  int? eosTokenId;

  late List<String> chars;

  late int len;

  late List<TokenLatticeNode> nodes;

  late List<List<TokenLatticeNode>> beginNodes;

  late List<List<TokenLatticeNode>> endNodes;

  /// Creates a new TokenLattice instance.
  ///
  /// @param {string} sentence The input sentence to be tokenized.
  /// @param {number} bosTokenId The beginning-of-sequence token ID.
  /// @param {number} eosTokenId The end-of-sequence token ID.
  TokenLattice(String sentence, this.bosTokenId, this.eosTokenId) {
    chars = sentence.split('');
    len = chars.length;
    nodes = [];
    beginNodes = List.generate(len + 1, (_) => []);
    endNodes = List.generate(len + 1, (_) => []);

    final bos = TokenLatticeNode(bosTokenId, 0, 0, 0, 0.0);
    final eos = TokenLatticeNode(eosTokenId, 1, len, 0, 0.0);
    nodes.add(bos.clone());
    nodes.add(eos.clone());
    beginNodes[len].add(eos);
    endNodes[0].add(bos);
  }

  /// Inserts a new token node into the token lattice.
  ///
  /// @param {number} pos The starting position of the token.
  /// @param {number} length The length of the token.
  /// @param {number} score The score of the token.
  /// @param {number} tokenId The token ID of the token.
  void insert(int pos, int length, double score, int tokenId) {
    final nodeId = nodes.length;
    final node = TokenLatticeNode(tokenId, nodeId, pos, length, score);
    beginNodes[pos].add(node);
    endNodes[pos + length].add(node);
    nodes.add(node);
  }

  /// Implements the Viterbi algorithm to compute the most likely sequence of tokens.
  ///
  /// @returns {TokenLatticeNode[]} The most likely sequence of tokens.
  List<TokenLatticeNode> viterbi() {
    final len = this.len;
    int pos = 0;
    while (pos <= len) {
      if (beginNodes[pos].isEmpty) return [];

      for (final rnode in beginNodes[pos]) {
        rnode.prev = null;
        double bestScore = 0.0;
        TokenLatticeNode? bestNode;

        for (final lnode in endNodes[pos]) {
          final score = lnode.backtraceScore + rnode.score;
          if (bestNode == null || score > bestScore) {
            bestNode = lnode.clone();
            bestScore = score;
          }
        }

        if (bestNode != null) {
          rnode.prev = bestNode;
          rnode.backtraceScore = bestScore;
        } else {
          return [];
        }
      }
      ++pos;
    }

    final List<TokenLatticeNode> results = [];
    final root = beginNodes[len][0];
    final prev = root.prev;
    if (prev == null) return [];

    TokenLatticeNode node = prev.clone();
    while (node.prev != null) {
      results.add(node.clone());
      final n = node.clone();
      node = n.prev!.clone();
    }

    return results.reversed.toList();
  }

  /// @param {TokenLatticeNode} node
  /// @returns {string} The array of nodes representing the most likely sequence of tokens.
  String piece(TokenLatticeNode node) {
    return chars.sublist(node.pos, node.pos + node.length).join('');
  }

  /// @returns {string[]} The most likely sequence of tokens.
  List<String> tokens() {
    final nodes = viterbi();
    return nodes.map((x) => piece(x)).toList();
  }

  /// @returns {number[]} The most likely sequence of token ids.
  List<int?> tokenIds() {
    final nodes = viterbi();
    return nodes.map((x) => x.tokenId).toList();
  }
}

class TokenLatticeNode {
  int? tokenId;

  int nodeId;

  int pos;

  int length;

  double score;

  TokenLatticeNode? prev;

  double backtraceScore = 0.0;

  /// Represents a node in a token lattice for a given sentence.
  /// @param {number} tokenId The ID of the token associated with this node.
  /// @param {number} nodeId The ID of this node.
  /// @param {number} pos The starting position of the token in the sentence.
  /// @param {number} length The length of the token.
  /// @param {number} score The score associated with the token.
  TokenLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score);

  /// Returns a clone of this node.
  /// @returns {TokenLatticeNode} A clone of this node.
  TokenLatticeNode clone() {
    final n = TokenLatticeNode(tokenId, nodeId, pos, length, score);
    n.prev = prev;
    n.backtraceScore = backtraceScore;
    return n;
  }
}

class DictionarySplitterNode {
  Map<String, DictionarySplitterNode> dictionaryMap = {};

  String? end;

  void operator []=(String char, DictionarySplitterNode node) {
    dictionaryMap[char] = node;
  }

  DictionarySplitterNode? operator [](String char) {
    return dictionaryMap[char];
  }

  @override
  String toString() {
    return '{ end = $end; dictionaryMap = $dictionaryMap }';
  }
}

/// A data structure which uses a trie to split a string into tokens based on a dictionary.
/// It can also use a regular expression to preprocess the input text before splitting.
///
/// NOTE: To ensure multi-byte characters are handled correctly, we operate at byte-level instead of character-level.
class DictionarySplitter {
  late DictionarySplitterNode trie;

  /// @param {string[]} dictionary The dictionary of words to use for splitting.
  DictionarySplitter(Iterable<String> dictionary) {
    trie = _buildTrie(dictionary);
  }

  /// Builds a trie from the given dictionary.
  /// @param {string[]} dictionary The dictionary of words to build the trie from.
  /// @returns {Object} The root node of the trie.
  /// @private
  DictionarySplitterNode _buildTrie(Iterable<String> dictionary) {
    final trie = DictionarySplitterNode();
    for (final word in dictionary) {
      DictionarySplitterNode node = trie;
      for (int i = 0; i < word.length; ++i) {
        node = (node[word[i]] ??= DictionarySplitterNode());
      }
      node.end = word;
    }
    return trie;
  }

  /// Splits the input text into tokens based on the dictionary.
  /// @param {string} text The input text to split.
  /// @returns {string[]} An array of tokens.
  List<String> split(String text) {
    final List<String> result = [];
    final n = text.length;
    int start = 0;
    int i = 0;

    while (i < n) {
      DictionarySplitterNode? node = trie;
      String? match;
      int j = i;

      while (j < n && node != null) {
        try {
          node = node[text[j]];
        } catch (_) {
          break;
        }

        final end = node?.end;
        if (end != null) {
          // Always keep the last (i.e., longest) match.
          match = end;
        }
        ++j;
      }

      if (match != null) {
        if (i > start) {
          result.add(text.substring(start, i));
        }
        result.add(match);
        i += match.length;
        start = i;
      } else {
        ++i;
      }
    }

    if (start < n) {
      result.add(text.substring(start));
    }

    return result;
  }
}

/// A simple Least Recently Used (LRU) cache implementation in JavaScript.
/// This cache stores key-value pairs and evicts the least recently used item
/// when the capacity is exceeded.
class LRUCache<K, V> {
  int capacity;

  final Map<K, V> cache = {};

  /// Creates an LRUCache instance.
  /// @param {number} capacity The maximum number of items the cache can hold.
  LRUCache(this.capacity);

  /// Retrieves the value associated with the given key and marks the key as recently used.
  /// @param {any} key The key to retrieve.
  /// @returns {any} The value associated with the key, or undefined if the key does not exist.
  // V? get(K key) {
  V? operator [](K key) {
    final value = cache[key];
    if (value == null) return null;

    cache.remove(key);
    cache[key] = value;
    return value;
  }

  /// Inserts or updates the key-value pair in the cache.
  /// If the key already exists, it is updated and marked as recently used.
  /// If the cache exceeds its capacity, the least recently used item is evicted.
  /// @param {any} key The key to add or update.
  /// @param {any} value The value to associate with the key.
  // put(K key, V value) {
  operator []=(K key, V value) {
    if (cache.containsKey(key)) {
      cache.remove(key);
    }
    cache[key] = value;
    if (cache.length > capacity) {
      cache.remove(cache.keys.first);
    }
  }

  /// Clears the cache.
  clear() {
    cache.clear();
  }
}
