import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'init_transformers_platform_interface.dart';

/// An implementation of [TransformersPlatform] that uses method channels.
class MethodChannelTransformers extends TransformersPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('transformers');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
