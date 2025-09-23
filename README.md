# transformers

🚧 **THIS IS CURRENTLY A WORK IN PROGRESS** 🚧

State-of-the-art Machine Learning for Dart. Run 🤗 Transformers cross-platform on your device, with no need for a server!

This repo is based off of [transformers.js](https://github.com/huggingface/transformers.js), however, I believe it would be even more beneficial to update this based off of the python [transformers](https://github.com/huggingface/transformers). This will require more heavy lifting but hopefully the tradeoff is a more one-to-one translation to dart.

Currently I have only tested this in Windows and Android. More manually testing should be done for the other platforms.

Web is not currently supported (also due to [huggingface_hub](https://github.com/NathanKolbas/huggingface_hub_dart) needing to be updated to support web) but the plan for this is to use [transformers.js](https://github.com/huggingface/transformers.js) for web and the dart implementation for for all other platforms. This is due to the differences in filesystem.

There is still a lot of work to be done here. Currently, just the tokenizer is working for `xlm-roberta-base`. More work is need to get the inference up and running. Luckily, that is planned out.

Inference is currently under active development. Instead of holding off until everything is completed, I am hoping others will still benefit from having access to the tokenizer. 

## Version

This library is based off of commit [a5847c9fb6ca410df6fc35ee584140f867840150](https://github.com/huggingface/transformers.js/tree/a5847c9fb6ca410df6fc35ee584140f867840150) from the official [transformers.js](https://github.com/huggingface/transformers.js) library.

## Supported Devices

In theory, this library should work across all platforms except for the Web do to no file storage. Please see each section to know which platform has been tested.

### Windows

✔️ Tested and works.

### MacOS

❓ Not tested yet.

### Linux

❓ Not tested yet.

### Android

✔️ Tested and works.

### iOS

❓ Not tested yet.

### Web

❌ Not yet implemented.
