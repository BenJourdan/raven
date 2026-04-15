
use tokio::sync::mpsc::{Sender, Receiver};
use std::sync::Arc;

use typed_floats::StrictlyPositiveFinite;

mod float_newtypes;
mod int_newtypes;


/// Struct to hold the operations 
#[derive(PartialEq, PartialOrd, Debug)]
pub struct NodeOps<V, T> {
    pub created: (Vec<V>, Vec<T>),
    pub modified: (Vec<V>, Vec<T>),
    pub deleted: (Vec<V>, Vec<T>),
}


impl <V, T> NodeOps<V, T> 
    where StrictlyPositiveFinite: From<T>
{

}

pub trait Adapter{
    type V : Clone + PartialEq + PartialOrd + std::fmt::Debug;
    type T : Clone + PartialEq + PartialOrd + std::fmt::Debug;

    fn update(&self, ops: NodeOps<Self::V, Self::T>) -> ();
    fn query(&self, query: &[Self::V]) -> Vec<Self::V>;
    fn graph_oracle_query(&self, query: &[Self::V]) -> Vec<Vec<(Self::V, Self::T)>>;
}

pub struct DynamicClustering< const ARITY: usize, V, T>{
    // Map stable unique node Ids to tree indices
    pub node_to_tree_map: FxHashMap<V, TreeIndex>,
    // and the reverse map:
    pub tree_to_node_map: FxHashMap<TreeIndex, V>,

    // degree priority queue
    pub degrees: PriorityQueue<V, NodeDegree>,

    // struct to hold tree data
    pub tree_data: TreeData<ARITY>,

    // sigma shift to set
    pub sigma: Float,

    // For lazy query time updates
    pub timestamp: usize,

    pub update_set: FxHashSet<TreeIndex>,

    pub coreset_size: usize,
    pub sampling_seeds: usize,

    pub num_clusters: usize,
    pub cluster_alg: AlgType,
    pub prop_name: String,
}

pub struct Engine<A, C> {
    adapter: A,
    core: C,
}

impl <A, C> Engine<A, C>
    where A: Adapter<V = String, T = f64>,
{

}



#[cfg(test)]
mod tests {
    use super::*;


}
