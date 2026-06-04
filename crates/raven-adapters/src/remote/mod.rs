//! Building blocks for future IPC and network-backed graph adapters.
//!
//! The remote backend is intentionally a skeleton for now. It is split into
//! client, cache, and oracle concerns so transport/runtime choices can be made
//! without changing the in-process adapter API.

pub mod cache;
pub mod client;
pub mod oracle;
