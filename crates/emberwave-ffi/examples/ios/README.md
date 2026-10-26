# emberwave on iPhone and iPad: the benchmark app

`EmberwaveBench` times emberwave on the device's own GPU, through Metal, with the C API of the XCFramework, as an app that uses emberwave would:

- **decode into a Metal texture**: a frame's bytes to the picture in a texture, the path of a player (`emberwave_decoder_decode_texture`);
- **decode into Y, Cb, Cr textures**: the same frames to their planes in three textures, for a player that converts the colours itself (`emberwave_decoder_decode_plane_textures`);
- **decode on the app's queue**: the same frames into a texture on the app's own command queue, where the call does not wait (`emberwave_context_create_metal`, `emberwave_decoder_submit_texture`), timed until the app's next command buffer, a copy of one pixel standing in for its draw, has read the picture;
- **decode to planes in memory**: the same frames to Y, Cb and Cr planes in CPU memory (`emberwave_decoder_decode_yuv`);
- **encode from memory**: pictures in CPU memory to frames within the byte budget, on the GPU (`emberwave_encoder_encode`).

It uses the pictures and budgets of the [browser benchmark](https://with-developer.github.io/emberwave/bench.html), 1080p, 1440p and 4K, so that the two compare: the browser's goes through WebGPU in Safari, this one through Metal directly. It shows the last decoded picture, drawn from the texture with the app's own Metal code. It also checks the textures, read back through Metal on queues of its own: the picture against the RGB decode, the planes against those decoded to memory, and a decode on the app's queue read by the app's next command buffer there.

## Running it on an iPhone

You need:
- a Mac with Xcode 16 or later, recent enough for the Mac's macOS and the phone's iOS. A newer macOS refuses to open an older Xcode: update it from the App Store.
- an Apple ID;
- an iPhone or iPad with iOS 16 or later;
- a USB cable that carries data. Xcode pairs with a phone over a cable the first time. Afterwards it reaches the phone over the network too, if *Connect via network* is on in *Window > Devices and Simulators*.

Nothing is published: the app is signed for your own devices only.

1. **The XCFramework.** From the root of the repository, fetch that of the release:

   ```sh
   crates/emberwave-ffi/examples/ios/fetch_xcframework.sh
   ```

   or build it, with `rustup target add aarch64-apple-ios aarch64-apple-ios-sim` and `scripts/build_xcframework.sh`. The app is written for the C API of your checkout, and the script warns when the release's differs: then check out the release's tag, or build the XCFramework.
2. **Open** `crates/emberwave-ffi/examples/ios/EmberwaveBench.xcodeproj` in Xcode: `open -a Xcode crates/emberwave-ffi/examples/ios/EmberwaveBench.xcodeproj`, as another editor may have taken over `.xcodeproj` files.
3. **Sign.** Select the project, the target `EmberwaveBench`, *Signing & Capabilities*, and choose your Apple ID under *Team* (a free *Personal Team* will do). If Xcode says the bundle identifier is taken, change `io.github.with-developer.emberwave-bench` to something of your own. Until the phone is connected, a Personal Team reports that it has no devices: the next step registers the phone.
4. **The phone.** Connect it, trust the Mac on it, and turn on *Settings > Privacy & Security > Developer Mode* if Xcode asks for it (the phone restarts). Choose the phone as the run destination at the top of Xcode's window.
5. **Run** (Command-R). The scheme builds the Release configuration with Metal's API validation off, so that the timings are those of the app as it would ship. With a free account, the first start asks you to trust the developer: *Settings > General > VPN & Device Management*.
6. **Benchmark.** In the app, tap *Run* and wait about two minutes with the app in front; the screen stays on meanwhile. Then *Copy the results*.

**If Xcode keeps asking to unlock the phone.** Xcode's run needs the phone's developer services. It may keep saying "iPhone needs to be unlocked to enable development services" although the phone is unlocked. Installing the app does not need those services. You can build and install it from a terminal at the root of the repository, then start it from the home screen, which also leaves the debugger out:

```sh
xcrun devicectl list devices   # the phone's identifier
xcodebuild -project crates/emberwave-ffi/examples/ios/EmberwaveBench.xcodeproj \
    -scheme EmberwaveBench -configuration Release \
    -destination 'platform=iOS,id=IDENTIFIER' -derivedDataPath target/ios-device \
    -allowProvisioningUpdates build
xcrun devicectl device install app --device IDENTIFIER \
    target/ios-device/Build/Products/Release-iphoneos/EmberwaveBench.app
```

The signing of step 3 applies, and the first start asks you to trust the developer as in step 5.

The project builds for iPhones and iPads, and for the iOS Simulator on Macs with Apple silicon only: the XCFramework has no library for the Simulator on Intel Macs.

For steadier numbers, let the phone cool down first, close other apps, keep it plugged in, and leave Low Power Mode off, which slows the GPU down; the report gives the thermal state at the end and whether Low Power Mode was on. A run during which the app left the front, where iOS stops or slows its GPU work, is marked not valid. Stopping the app in Xcode and starting it again from the home screen leaves the debugger out.

## In CI

`scripts/check_ios_app.sh` builds the app for iPhones, which checks that it links, and for the iOS Simulator, where it runs it with `-autorun`: a shorter benchmark of every size and the three checks, printed to the log. CI runs it on a macOS runner after the XCFramework is built (see the CI job `XCFramework`).
