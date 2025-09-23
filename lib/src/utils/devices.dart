// ignore_for_file: non_constant_identifier_names

/// The mapping of devices supported by Transformers.dart
class DEVICE_TYPES {
  /// Auto-detect based on device and environment
  static final String auto = 'auto';
  /// Auto-detect GPU
  static final String gpu = 'gpu';
  /// CPU
  static final String cpu = 'cpu';
  /// WebAssembly
  static final String wasm = 'wasm';
  /// WebGPU
  static final String webgpu = 'webgpu';
  /// CUDA
  static final String cuda = 'cuda';
  /// DirectML
  static final String dml = 'dml';

  /// WebNN (default)
  static final String webnn = 'webnn';

  /// WebNN NPU
  static final String  webnnNpu = 'webnn-npu';

  /// WebNN GPU
  static final String  webnnGpu = 'webnn-gpu';

  /// WebNN CPU
  static final String  webnnCpu = 'webnn-cpu';
}

/// @typedef {keyof typeof DEVICE_TYPES} DeviceType
/// The list of devices supported by Transformers.dart
sealed class DeviceType {}

class DeviceTypeAuto extends DeviceType {}
class DeviceTypeGpu extends DeviceType {}
class DeviceTypeCpu extends DeviceType {}
class DeviceTypeWasm extends DeviceType {}
class DeviceTypeWebgpu extends DeviceType {}
class DeviceTypeCuda extends DeviceType {}
class DeviceTypeDml extends DeviceType {}
class DeviceTypeWebnn extends DeviceType {}
class DeviceTypeWebnnNpu extends DeviceType {}
class DeviceTypeWebnnGpu extends DeviceType {}
class DeviceTypeWebnnCpu extends DeviceType {}
