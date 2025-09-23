#ifndef FLUTTER_PLUGIN_TRANSFORMERS_PLUGIN_H_
#define FLUTTER_PLUGIN_TRANSFORMERS_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace transformers {

class TransformersPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  TransformersPlugin();

  virtual ~TransformersPlugin();

  // Disallow copy and assign.
  TransformersPlugin(const TransformersPlugin&) = delete;
  TransformersPlugin& operator=(const TransformersPlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace transformers

#endif  // FLUTTER_PLUGIN_TRANSFORMERS_PLUGIN_H_
