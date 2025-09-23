
import 'init_transformers_platform_interface.dart';

class Transformers {
  Future<String?> getPlatformVersion() {
    return TransformersPlatform.instance.getPlatformVersion();
  }
}
