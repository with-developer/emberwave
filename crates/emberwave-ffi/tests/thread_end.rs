//! The C API called while a thread ends, from the destructor of a thread-local
//! that runs after emberwave's own. Its record of the last error is gone by
//! then: the calls go on without it, rather than abort the process (#126).

use std::ffi::CStr;
use std::ptr;

use emberwave_ffi::*;

/// Calls emberwave from its destructor.
struct Late;

impl Drop for Late {
    fn drop(&mut self) {
        // SAFETY: always a C string, empty once the record is gone.
        let _ = unsafe { CStr::from_ptr(emberwave_last_error()) };
        // SAFETY: a NULL decoder, which fails the call: it would record why.
        let status = unsafe { emberwave_decoder_push_packet(ptr::null_mut(), ptr::null(), 0) };
        assert_ne!(status, EmberwaveStatus::Ok);
    }
}

thread_local! {
    static LATE: Late = const { Late };
}

#[test]
fn calls_from_a_late_thread_local_destructor_do_not_abort() {
    std::thread::spawn(|| {
        // Made first, so that its destructor runs after emberwave's where
        // the destructors run in reverse order, as on macOS and with glibc.
        LATE.with(|_| {});
        // SAFETY: a NULL decoder, which fails the call: emberwave records why.
        let status = unsafe { emberwave_decoder_push_packet(ptr::null_mut(), ptr::null(), 0) };
        assert_ne!(status, EmberwaveStatus::Ok);
    })
    .join()
    .unwrap();
}
