//! The committed C header is what cbindgen makes of the source.
//!
//! `EMBERWAVE_BLESS=1 cargo test -p emberwave-ffi --test header` writes it
//! anew after a change to the API.

use std::path::Path;

#[test]
fn the_header_is_up_to_date() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("cbindgen.toml is readable");
    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen reads src/lib.rs");
    let mut generated = Vec::new();
    bindings.write(&mut generated);
    let generated = String::from_utf8(generated).expect("the header is UTF-8");

    let path = crate_dir.join("include/emberwave.h");
    if std::env::var_os("EMBERWAVE_BLESS").is_some() {
        std::fs::write(&path, &generated).expect("include/emberwave.h is writable");
        return;
    }
    // A Windows checkout may have turned the line ends into CRLF.
    let committed = std::fs::read_to_string(&path)
        .expect("include/emberwave.h exists")
        .replace("\r\n", "\n");
    assert!(
        committed == generated,
        "include/emberwave.h is out of date: run \
         EMBERWAVE_BLESS=1 cargo test -p emberwave-ffi --test header"
    );
}
