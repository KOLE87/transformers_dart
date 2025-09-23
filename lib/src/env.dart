const VERSION = '0.0.1';

class Backends {
  final onnx = {};
}

class env {
  static final version = VERSION;

  /////////////////// Backends settings ///////////////////
  // NOTE: These will be populated later by the backends themselves.
  static final backends = Backends;

  /////////////////// Model settings ///////////////////
  static final allowRemoteModels = true;
  static final remoteHost = 'https://huggingface.co/';
  static final remotePathTemplate = '{model}/resolve/{revision}/';

  // final allowLocalModels = !(IS_BROWSER_ENV || IS_WEBWORKER_ENV);
  static final allowLocalModels = true;
  // static final localModelPath = localModelPath;
  static final localModelPath = '';
  // static final useFS = IS_FS_AVAILABLE;
  static final useFS = true;

  /////////////////// Cache settings ///////////////////
  // static final useBrowserCache = IS_WEB_CACHE_AVAILABLE && !IS_DENO_RUNTIME;
  static final useBrowserCache = true;

  // static final useFSCache = IS_FS_AVAILABLE;
  static final useFSCache = true;
  // static final cacheDir = DEFAULT_CACHE_DIR;
  static final cacheDir = true;

  static final useCustomCache = false;
  static final customCache = null;
}
