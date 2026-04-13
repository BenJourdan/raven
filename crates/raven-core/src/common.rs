use std::fmt;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use std::iter::Sum;

use typed_floats::{StrictlyPositiveFinite, PositiveFinite};



pub const FP_EPSILON32: StrictlyPositiveFinite<f32> = unsafe{StrictlyPositiveFinite::<f32>::new_unchecked(1e-6)};
pub const FP_EPSILON64: StrictlyPositiveFinite<f64> = unsafe{StrictlyPositiveFinite::<f64>::new_unchecked(1e-12)};

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

// MARK: Float Newtypes

macro_rules! newtypes {
    ($($name:ident),*) => {
        $(
            #[repr(transparent)]
            #[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
            pub struct $name(pub Float);
            unsafe impl ReprAsF64 for $name {}
        )*
    };
}

newtypes!(
    EdgeWeight,
    NodeDegree,
    Contribution,
    SmoothedContribution,
    Volume,
    FDelta,
    SmoothingTermDelta,
    HB,
    HS
);

pub trait QuackLikeAFloat {
    fn into_float(self) -> Float;
    fn from_float(x: Float) -> Self;
}

/// Marker for types that are guaranteed to have the same representation as `f64`
/// (via a transparent wrapper chain). This lets us provide zero-copy casts.
pub unsafe trait ReprAsF64 {}
unsafe impl ReprAsF64 for f64 {}
unsafe impl ReprAsF64 for Float {}

/// Zero-copy reinterpret a vector of `T` as a vector of `U`.
/// Both `T` and `U` must be transparent wrappers over the same base representation (f64).
pub fn reinterpret_vec<T: ReprAsF64, U: ReprAsF64>(v: Vec<T>) -> Vec<U> {
    use std::mem::{align_of, size_of};
    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());
    let len = v.len();
    let cap = v.capacity();
    let ptr = v.as_ptr();
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(ptr as *mut U, len, cap) }
}

/// Zero-copy reinterpret a slice of `T` as a slice of `U`.
pub fn reinterpret_slice<T: ReprAsF64, U: ReprAsF64>(s: &[T]) -> &[U] {
    use std::mem::{align_of, size_of};
    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const U, s.len()) }
}

pub fn convert<T: QuackLikeAFloat, U: QuackLikeAFloat>(x: T) -> U {
    U::from_float(x.into_float())
}

macro_rules! impl_quack_like_a_float {
    ($($t:ident),*) => {
        $(

            impl From<Float> for $t {
                #[inline(always)]
                fn from(x: Float) -> Self {
                    $t(x)
                }
            }

            impl $t {
                #[inline(always)]
                pub fn new(x: Float) -> Self {
                    $t(x)
                }

                #[inline(always)]
                pub fn inv(self) -> Self {
                    $t(Float::from(1.0) / self.0)
                }

                #[inline(always)]
                pub fn zero() -> Self {
                    $t(Float::from(0.0))
                }
            }

            impl QuackLikeAFloat for $t {
                #[inline(always)]
                fn into_float(self) -> Float {
                    self.0
                }
                #[inline(always)]
                fn from_float(x: Float) -> Self {
                    $t(x)
                }
            }

            impl Add for $t {
                type Output = Self;
                #[inline(always)]
                fn add(self, rhs: Self) -> Self {
                    Self(self.0 + rhs.0)
                }
            }

            impl Div for $t {
                type Output = Self;
                #[inline(always)]
                fn div(self, rhs: Self) -> Self {
                    Self(self.0 / rhs.0)
                }
            }

            impl Mul<Float> for $t {
                type Output = Self;
                #[inline(always)]
                fn mul(self, rhs: Float) -> Self {
                    Self(self.0 * rhs)
                }
            }

            impl Sub for $t {
                type Output = Self;
                #[inline(always)]
                fn sub(self, rhs: Self) -> Self {
                    Self(self.0 - rhs.0)
                }
            }

            impl Neg for $t {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    Self(-self.0)
                }
            }

            impl AddAssign for $t {
                #[inline(always)]
                fn add_assign(&mut self, rhs: Self) {
                    self.0 += rhs.0;
                }
            }

            impl SubAssign for $t {
                #[inline(always)]
                fn sub_assign(&mut self, rhs: Self) {
                    self.0 -= rhs.0;
                }
            }

            impl MulAssign<Float> for $t {
                #[inline(always)]
                fn mul_assign(&mut self, rhs: Float) {
                    self.0 *= rhs;
                }
            }

            impl Sum for $t {
                #[inline(always)]
                fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                    Self(iter.map(|x| x.0).sum())
                }
            }

            impl<'a> Sum<&'a $t> for $t {
                #[inline(always)]
                fn sum<I: Iterator<Item = &'a $t>>(iter: I) -> Self {
                    Self(iter.map(|x| x.0).sum())
                }
            }

            impl fmt::Display for $t {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self.0)
                }
            }


        )*
    };
}

impl_quack_like_a_float!(
    EdgeWeight,
    NodeDegree,
    Contribution,
    SmoothedContribution,
    Volume,
    FDelta,
    SmoothingTermDelta,
    HB,
    HS
);

// MARK: Edge Deletion Result Enum:
#[derive(Debug, Clone)]
pub enum EdgeDeletionResult {
    BothNodesStillConnected,
    OneNodeDisconnected(String),
    BothNodesDisconnected(String, String),
}

// MARK: Error type:

#[derive(Debug)]
pub enum DynamicCoresetError {
    NoData,
    InvalidEdge(String, String),
    NodeNotFound(String),
    NodeAlreadyExists(String),
    NoSelfLoopsAllowed(String),
}

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

// MARK: Sampling Stats:

#[derive(Debug, Clone, Copy)]
pub struct SamplingStats {
    pub num_samples: usize,
    pub num_clippings: usize,
}

impl SamplingStats {
    pub fn new() -> Self {
        SamplingStats {
            num_samples: 0,
            num_clippings: 0,
        }
    }
    pub fn clipping_fraction(&self) -> Float {
        if self.num_samples == 0 {
            Float::from(0.0)
        } else {
            Float::from(self.num_clippings as Float_Dtype)
                / Float::from(self.num_samples as Float_Dtype)
        }
    }
}

impl Add for SamplingStats {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        SamplingStats {
            num_samples: self.num_samples + rhs.num_samples,
            num_clippings: self.num_clippings + rhs.num_clippings,
        }
    }
}

impl AddAssign for SamplingStats {
    fn add_assign(&mut self, rhs: Self) {
        self.num_samples += rhs.num_samples;
        self.num_clippings += rhs.num_clippings;
    }
}

// Power of two trait:

pub trait PowerOfTwo {}
pub struct ConstPow2<const N: usize>;

macro_rules! impl_power_of_two_up_to_1024 {
    ( $( $pow:expr ),* ) => {
        $(
            impl PowerOfTwo for ConstPow2<$pow> {}
        )*
    };
}
impl_power_of_two_up_to_1024! {
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
}
