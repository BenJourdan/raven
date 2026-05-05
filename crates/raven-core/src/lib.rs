use std::sync::Arc;


pub mod types;
pub mod alg;
pub mod error;

use types::{
    PartitionType,
    PartitionOutput
};

/// A helper type for the neighbour oracle function.
/// Given a batch of node identifies, return a slice of
/// their neighbours and the corresponding edge weights.
trait GraphBatchNeighbours<V, S, E>:
    for<'a>  Fn(&'a [V]) -> Result<&'a[&'a[(V, S)]], error::OracleError<E>>
{
    
}
/// Implement the trait for any function that matches the signature.
impl <F, V, S, E> GraphBatchNeighbours<V, S, E> for F
where
    F: for<'a> Fn(&'a [V]) -> Result<&'a[&'a[(V, S)]], error::OracleError<E>>
{   
}

// In future, it should be possible to do the above with a trait alias:
// type GraphBatchNeighbours<V, S, E> = for<'a> Fn(&'a [V]) -> Result<&'a[&'a[(V, S)]], error::OracleError<E>>;

/// A helper type for the coreset neighbour oracle function.
/// Given a batch of node identifies in the coreset,
/// return a slice of their neighbours in the coreset
/// and the corresponding edge weights.
/// This ignores any edges to nodes outside the coreset.
trait CoresetNeighbours<V, S, E>:
    for<'a>  Fn(&'a [V]) -> Result<&'a[&'a[(V, S)]], error::OracleError<E>>
{
}
/// Implement the trait for any function that matches the signature.
impl <F, V, S, E> CoresetNeighbours<V, S, E> for F
where
    F: for<'a> Fn(&'a [V]) -> Result<&'a[&'a[(V, S)]], error::OracleError<E>>
{
} 


/// A trait for dynamic clustering algorithms.
pub trait DynamicClusteringAlg<V,S> {
    /// Apply a batch of node updates to the data structure.
    fn apply_node_ops<G, E>(
        &mut self, 
        diffs: &[(V, Option<S>)], 
        graph_oracle: &G) -> ()
    where
        G: GraphBatchNeighbours<V, S, E> + ?Sized; // ?Sized allows for dynamically sized types.

    /// Query the current clustering with a partition type.
    fn query<G, C, E>(
        &mut self,
        partition: PartitionType<V>,
        graph_oracle: &G,
        coreset_oracle: &C,
    ) -> PartitionOutput<V>
    where
        G: GraphBatchNeighbours<V, S, E> + ?Sized,
        C: CoresetNeighbours<V, S, E> + ?Sized;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_work(){
        alg::TreeData::<4, types::Strict<f64>>{
                timestamp: vec![],
                volume: vec![],
                size: vec![],
                f_delta: vec![],
                h_b: vec![],
                h_s: vec![],
        };

        let x = types::Strict::<f64>::new(1.0).unwrap();
        let y: Option<types::Strict<f64>> = Some(x);
    }

}
