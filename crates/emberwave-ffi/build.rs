//! Names the shared library so that it can move with the application that
//! uses it. By default the macOS library carries the absolute path it was
//! built at, and the Linux one no name at all, which makes programs record
//! the path they were linked against. With these names they look it up by
//! file name: through the run path (`@rpath`) on macOS, and the usual search
//! (run path, `LD_LIBRARY_PATH`, system directories) on Linux.

fn main() {
    // Without this, a change to any file of the package would run the script
    // again and rebuild the crate.
    println!("cargo::rerun-if-changed=build.rs");
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            println!(
                "cargo::rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libemberwave_ffi.dylib"
            );
        }
        Ok("linux") => {
            println!("cargo::rustc-cdylib-link-arg=-Wl,-soname,libemberwave_ffi.so");
        }
        _ => {}
    }
}
