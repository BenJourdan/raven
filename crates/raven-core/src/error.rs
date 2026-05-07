use std::fmt;

/// Error types for dynamic coreset operations and oracles.
#[derive(Debug)]
pub enum AlgorithmicError<E> {
    OracleError(E),
    DataStructureError(DynamicCoresetError),
}

#[derive(Debug)]
pub enum OracleError<E> {
    GraphError(E),
    CoresetError(E),
}

impl<E> fmt::Display for OracleError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OracleError::GraphError(e) => write!(f, "Graph oracle error: {}", e),
            OracleError::CoresetError(e) => write!(f, "Coreset oracle error: {}", e),
        }
    }
}

// Error type for dynamic coreset operations.
#[derive(Debug)]
pub enum DynamicCoresetError {
    NoData,
    InvalidEdge(String, String),
    NodeNotFound(String),
    NodeAlreadyExists(String),
    NoSelfLoopsAllowed(String),
}

/// Returned when the reciprocal of a finite positive value overflows to `+inf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalOverflow;

impl fmt::Display for DynamicCoresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicCoresetError::NoData => write!(f, "No data in the dynamic coreset"),
            DynamicCoresetError::InvalidEdge(u, v) => {
                write!(f, "Invalid edge between {} and {}", u, v)
            }
            DynamicCoresetError::NodeNotFound(u) => write!(f, "Node not found: {}", u),
            DynamicCoresetError::NodeAlreadyExists(u) => write!(f, "Node already exists: {}", u),
            DynamicCoresetError::NoSelfLoopsAllowed(u) => {
                write!(f, "Self loops not allowed: {}", u)
            }
        }
    }
}

impl std::error::Error for DynamicCoresetError {}
impl std::error::Error for ReciprocalOverflow {}
impl std::error::Error for OracleError<DynamicCoresetError> {}

impl fmt::Display for ReciprocalOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reciprocal overflowed to infinity")
    }
}
