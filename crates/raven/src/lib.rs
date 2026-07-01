//! Ergonomic in-memory Raven API.
//!
//! `Raven` owns the graph storage, incremental clustering state, and query
//! workspace. Users update weighted undirected edges and query node subsets
//! without coordinating the lower-level graph adapter and clustering data
//! structure separately.
//!
//! ```no_run
//! use raven::{Raven, RavenConfig};
//!
//! let mut config = RavenConfig::new(2);
//! config.coreset_size = 3;
//! config.sampling_seeds = 2;
//!
//! let mut index = Raven::new(config)?;
//! index.update_edge(1, 2, 1.0)?;
//! index.update_edge(2, 3, 1.0)?;
//!
//! let result = index.query(&[1, 2, 3])?;
//! println!("{:?}", result.labels);
//! # Ok::<(), raven::RavenError>(())
//! ```

mod config;
mod consensus;
mod error;
mod index;
mod query;

pub use config::RavenConfig;
pub use consensus::{ConsensusResult, TrialWeighting};
pub use error::{RavenError, Result};
pub use index::{EdgeUpdate, EdgeUpdateStats, Raven};
pub use query::QueryResult;

const ARITY: usize = 8;
