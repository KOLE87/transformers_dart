import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'init_transformers_method_channel.dart';

abstract class TransformersPlatform extends PlatformInterface {
  /// Constructs a TransformersPlatform.
  TransformersPlatform() : super(token: _token);

  static final Object _token = Object();

  static TransformersPlatform _instance = MethodChannelTransformers();

  /// The default instance of [TransformersPlatform] to use.
  ///
  /// Defaults to [MethodChannelTransformers].
  static TransformersPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [TransformersPlatform] when
  /// they register themselves.
  static set instance(TransformersPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
