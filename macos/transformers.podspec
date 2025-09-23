#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint transformers.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'transformers'
  s.version          = '0.0.1'
  s.summary          = 'State-of-the-art Machine Learning for Dart. Run 🤗 Transformers cross-platform on your device, with no need for a server!'
  s.description      = <<-DESC
State-of-the-art Machine Learning for Dart. Run 🤗 Transformers cross-platform on your device, with no need for a server!
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'

  # If your plugin requires a privacy manifest, for example if it collects user
  # data, update the PrivacyInfo.xcprivacy file to describe your plugin's
  # privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  # s.resource_bundles = {'transformers_privacy' => ['Resources/PrivacyInfo.xcprivacy']}

  s.dependency 'FlutterMacOS'

  s.platform = :osx, '10.11'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
