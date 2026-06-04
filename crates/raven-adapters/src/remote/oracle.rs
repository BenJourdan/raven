//! Query oracle handles for future IPC and network graph backends.
//!
//! Remote oracle handles are expected to own per-trial scratch storage while
//! sharing a client and cache with other handles.
