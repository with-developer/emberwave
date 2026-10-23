//! C FFI for emberwave. This crate is a placeholder: the C API is a SHOULD
//! item that starts after milestone M5.

/// Returns the version of the underlying `emberwave-core` crate.
pub fn version() -> &'static str {
    emberwave_core::VERSION
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_matches_core() {
        assert_eq!(super::version(), emberwave_core::VERSION);
    }
}
