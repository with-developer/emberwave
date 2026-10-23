# emberwave C API

`emberwave-ffi` is the C interface of [emberwave](https://github.com/with-developer/emberwave): it decodes PyroWave video on the GPU, through Vulkan, Metal or Direct3D 12, for programs in C, C++ or any language that can call C functions. One header, `include/emberwave.h`, and one library.

## Getting it

Each [release](https://github.com/with-developer/emberwave/releases) has an archive `emberwave-ffi-VERSION-TARGET` for x86_64 Linux, Apple Silicon macOS and x86_64 Windows:

| Path | Content |
|---|---|
| `include/emberwave.h` | The header |
| `lib/libemberwave_ffi.so`, `lib/libemberwave_ffi.dylib`, or `lib/emberwave_ffi.dll` with its import library `lib/emberwave_ffi.dll.lib` | The shared library |
| `lib/libemberwave_ffi.a` or `lib/emberwave_ffi.lib` | The static library |
| `lib/native-static-libs.txt` | The system libraries a program linked to the static library needs as well |

To build it yourself, with Rust 1.87 or newer:

```sh
cargo build --release -p emberwave-ffi        # target/release: the shared library
cargo rustc --release -p emberwave-ffi --lib --crate-type staticlib \
    -- --print native-static-libs              # the static library, and what it needs
```

## Linking

The commands below assume an archive unpacked into `emberwave/` and a program `app.c`.

The shared library names itself `@rpath/libemberwave_ffi.dylib` on macOS and `libemberwave_ffi.so` on Linux, so a program looks for it on its run path. These commands set the run path to a `lib` directory next to the program and put the library there:

```sh
# macOS
cc app.c -I emberwave/include -L emberwave/lib -lemberwave_ffi -Wl,-rpath,@executable_path/lib
# Linux
cc app.c -I emberwave/include -L emberwave/lib -lemberwave_ffi -Wl,-rpath,'$ORIGIN/lib'
# both
mkdir -p lib && cp emberwave/lib/libemberwave_ffi.* lib/
```

On Windows, link the import library and keep the DLL next to the program (from a Visual Studio developer prompt):

```bat
cl app.c /I emberwave\include emberwave\lib\emberwave_ffi.dll.lib
copy emberwave\lib\emberwave_ffi.dll .
```

To link the static library instead, name its file, as `-lemberwave_ffi` would pick the shared one, and add the system libraries it needs:

```sh
cc app.c -I emberwave/include emberwave/lib/libemberwave_ffi.a $(cat emberwave/lib/native-static-libs.txt)
```

On Windows that is `emberwave\lib\emberwave_ffi.lib` with the libraries named in `emberwave\lib\native-static-libs.txt`.

## Use

```c
#include "emberwave.h"

EmberwaveContext *context;   /* one per GPU; compiles the shaders */
EmberwaveDecoder *decoder;   /* one per stream: picture size and chroma format */

emberwave_context_create(NULL, &context);
emberwave_decoder_create(context, 1920, 1080, EMBERWAVE_CHROMA_420, &decoder);

/* For every packet that arrives: */
emberwave_decoder_push_packet(decoder, packet, packet_size);
if (emberwave_decoder_is_ready(decoder, false)) {
    emberwave_decoder_decode_yuv(decoder, y, cb, cr);   /* or _decode_rgb */
}

emberwave_decoder_destroy(decoder);
emberwave_context_destroy(context);
```

[`examples/c/decode.c`](https://github.com/with-developer/emberwave/blob/main/crates/emberwave-ffi/examples/c/decode.c) in the repository is a complete program: it decodes a PWV1 file to a Y4M file. Its [`CMakeLists.txt`](https://github.com/with-developer/emberwave/blob/main/crates/emberwave-ffi/examples/c/CMakeLists.txt) links it to the shared library of a cargo build, from the root of the repository:

```sh
cargo build -p emberwave-ffi
cmake -S crates/emberwave-ffi/examples/c -B target/c-example -DEMBERWAVE_LIB_DIR=target/debug
cmake --build target/c-example
target/c-example/emberwave_decode testdata/noise_192x136_420.pwv1 out.y4m
```

`scripts/check_c_example.sh` does that for every test vector and compares the output with upstream's decoder; CI runs it on Linux, macOS and Windows.

## Rules

- **Errors.** Functions that can fail return an `EmberwaveStatus`. After a failure, `emberwave_last_error()` describes it, for the thread that made the call, until that thread's next failure.
- **Objects.** `*_create` makes a context or a decoder, `*_destroy` frees it, and destroying `NULL` does nothing. A decoder may outlive its context.
- **Threads.** A context may be used by several threads at once, for instance to create decoders. A decoder may move between threads, but only one thread may use it at a time.
- **Memory.** The decoder writes into memory the caller provides: planes with a stride of at least their width (`emberwave_decoder_info` gives the sizes), or RGB with a stride of at least three times the width. It checks the arguments before it decodes, so a bad one does not use up the frame.
- **Waiting.** `emberwave_decoder_decode_yuv` and `emberwave_decoder_decode_rgb` wait for the GPU and copy the picture to the CPU. Keeping the picture on the GPU, shared with the caller's graphics API, is not part of this interface yet.
- **Panics.** A panic inside emberwave is caught and reported as `EMBERWAVE_STATUS_INTERNAL`; destroy the object it happened in.
- **Packets.** `emberwave_decoder_push_packet` takes one packet or several back to back, in any order. Packets of a newer frame discard an unfinished older one; packets of older frames are ignored. `emberwave_decoder_is_ready` with `allow_partial_frame` accepts a frame whose coarse bands are complete, for a picture despite lost packets.
- **Colours.** With a `NULL` colour, `emberwave_decoder_decode_rgb` converts the way the frame's start of frame header says. Upstream's Metal encoder writes BT.709, full range and centred chroma there whatever the pictures are, so pass what you know, such as the colours of a PWV1 file's header from `emberwave_probe_pwv1`.

The header documents every function; `ARCHITECTURE.md` in the repository explains the design (decision D27).
