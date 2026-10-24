# emberwave on iPhone and iPad: the benchmark app

`EmberwaveBench` times emberwave on the device's own GPU, through Metal, with the C API of the XCFramework, as an app that uses emberwave would:

- **decode into a Metal texture**: a frame's bytes to the picture in a texture, the path of a player (`emberwave_decoder_decode_texture`);
- **decode to planes in memory**: the same frames to Y, Cb and Cr planes in CPU memory (`emberwave_decoder_decode_yuv`);
- **encode from memory**: pictures in CPU memory to frames within the byte budget, on the GPU (`emberwave_encoder_encode`).

It uses the pictures and budgets of the [browser benchmark](https://with-developer.github.io/emberwave/bench.html), 1080p, 1440p and 4K, so that the two compare: the browser's goes through WebGPU in Safari, this one through Metal directly. It shows the last decoded picture, drawn from the texture with the app's own Metal code, and checks that a texture read back through Metal holds what the RGB decode gives.

## Running it on an iPhone

You need a Mac with Xcode, an Apple ID and a cable or the same Wi-Fi as the phone. Nothing is published: the app is signed for your own devices only.

1. **The XCFramework.** From the root of the repository, fetch that of the release:

   ```sh
   crates/emberwave-ffi/examples/ios/fetch_xcframework.sh
   ```

   or build it, with `rustup target add aarch64-apple-ios aarch64-apple-ios-sim` and `scripts/build_xcframework.sh`.
2. **Open** `crates/emberwave-ffi/examples/ios/EmberwaveBench.xcodeproj` in Xcode.
3. **Sign.** Select the project, the target `EmberwaveBench`, *Signing & Capabilities*, and choose your Apple ID under *Team* (a free *Personal Team* will do). If Xcode says the bundle identifier is taken, change `io.github.with-developer.emberwave-bench` to something of your own.
4. **The phone.** Connect it, and turn on *Settings > Privacy & Security > Developer Mode* if Xcode asks for it (the phone restarts). Choose the phone as the run destination at the top of Xcode's window.
5. **Run** (Command-R). The scheme builds the Release configuration with Metal's API validation off, so that the timings are those of the app as it would ship. With a free account, the first start asks you to trust the developer: *Settings > General > VPN & Device Management*.
6. **Benchmark.** In the app, tap *Run* and wait about a minute with the app in front; the screen stays on meanwhile. Then *Copy the results*.

The project builds for iPhones and iPads, and for the iOS Simulator on Macs with Apple silicon only: the XCFramework has no library for the Simulator on Intel Macs.

For steadier numbers, let the phone cool down first, close other apps, and keep it plugged in; the report gives the thermal state at the end. Stopping the app in Xcode and starting it again from the home screen leaves the debugger out.

## In CI

`scripts/check_ios_app.sh` builds the app for iPhones, which checks that it links, and for the iOS Simulator, where it runs it with `-autorun`: a shorter benchmark of every size and the texture check, printed to the log. CI runs it on a macOS runner after the XCFramework is built (see the CI job `XCFramework`).
