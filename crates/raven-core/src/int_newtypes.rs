use std::{fmt, ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};

// Used to refer to a node in a tree (stored in a vec)
// The root node is at index 0 etc
#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub struct TreeIndex(pub usize);

impl From<usize> for TreeIndex {
    fn from(index: usize) -> Self {
        TreeIndex(index)
    }
}

impl<T> Index<TreeIndex> for Vec<T> {
    type Output = T;
    fn index(&self, index: TreeIndex) -> &Self::Output {
        &self[index.0]
    }
}
impl<T> IndexMut<TreeIndex> for Vec<T> {
    fn index_mut(&mut self, index: TreeIndex) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl<T> Index<TreeIndex> for [T] {
    type Output = T;

    fn index(&self, index: TreeIndex) -> &Self::Output {
        &self[index.0]
    }
}

impl<T> IndexMut<TreeIndex> for [T] {
    fn index_mut(&mut self, index: TreeIndex) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl Add for TreeIndex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        TreeIndex(self.0 + rhs.0)
    }
}
impl Sub for TreeIndex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        TreeIndex(self.0 - rhs.0)
    }
}
impl Div<usize> for TreeIndex {
    type Output = Self;
    fn div(self, rhs: usize) -> Self {
        TreeIndex(self.0 / rhs)
    }
}

// Unique identifier for each node in the graph.
#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug, Ord, PartialOrd)]
pub struct NodeIdentity(pub usize);

impl fmt::Display for NodeIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}