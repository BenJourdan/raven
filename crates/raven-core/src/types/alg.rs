use std::sync::Arc;
use faer::sparse::SparseRowMatRef;


/// Type alias for a clustering algorithm function.
/// The function takes a mutable ref to 
/// - a sparse matrix
/// - a usize parameter (expected number of clusters)
/// and returns a tuple of:
/// - a vector of usize cluster assignments for nodes
/// - a usize for the actual number of clusters found
pub type AlgType = Arc<
    dyn Fn(SparseRowMatRef<usize, f64>, usize) 
    -> 
    (Vec<usize>, usize) + Send + Sync + 'static
>;
