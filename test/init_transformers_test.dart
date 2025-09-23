import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/init_transformers.dart';
import 'package:transformers/init_transformers_platform_interface.dart';
import 'package:transformers/init_transformers_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockTransformersPlatform
    with MockPlatformInterfaceMixin
    implements TransformersPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final TransformersPlatform initialPlatform = TransformersPlatform.instance;

  test('$MethodChannelTransformers is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelTransformers>());
  });

  test('getPlatformVersion', () async {
    Transformers transformersPlugin = Transformers();
    MockTransformersPlatform fakePlatform = MockTransformersPlatform();
    TransformersPlatform.instance = fakePlatform;

    expect(await transformersPlugin.getPlatformVersion(), '42');
  });
}
