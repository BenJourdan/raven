use std::{error::Error, fmt};

use raven_adapters::in_memory::InMemoryIndexError;

/// Result alias used by the public Raven API.
pub type Result<T> = std::result::Result<T, RavenError>;

/// Errors returned by the public Raven API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RavenError {
    /// Invalid static configuration.
    InvalidConfig(String),
    /// Invalid user input supplied to an operation.
    InvalidInput(String),
    /// Invalid edge weight.
    InvalidWeight(String),
    /// Error from the in-memory graph/index backend.
    Index(String),
    /// Internal invariant failure or unexpected lower-level output shape.
    UnexpectedOutput(String),
}

impl fmt::Display for RavenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid Raven config: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid Raven input: {message}"),
            Self::InvalidWeight(message) => write!(f, "invalid edge weight: {message}"),
            Self::Index(message) => write!(f, "Raven index error: {message}"),
            Self::UnexpectedOutput(message) => write!(f, "unexpected Raven output: {message}"),
        }
    }
}

impl Error for RavenError {}

impl From<InMemoryIndexError> for RavenError {
    fn from(value: InMemoryIndexError) -> Self {
        Self::Index(value.to_string())
    }
}
