#include "include/transformers/transformers_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "transformers_plugin.h"

void TransformersPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  transformers::TransformersPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
