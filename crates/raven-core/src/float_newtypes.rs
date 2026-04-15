use std::fmt;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use std::hash::Hash;

use std::iter::Sum;

use typed_floats::{StrictlyPositiveFinite, PositiveFinite, StrictlyPositive, NonNaNFinite};



pub const FP_EPSILON32: StrictlyPositiveFinite<f32> = unsafe{StrictlyPositiveFinite::<f32>::new_unchecked(1e-6)};
pub const FP_EPSILON64: StrictlyPositiveFinite<f64> = unsafe{StrictlyPositiveFinite::<f64>::new_unchecked(1e-12)};

/// Type alias for StrictlyPositiveFinite<T>
type Strict<T> = StrictlyPositiveFinite<T>;
/// Type alias for PositiveFinite<T>
type NonStrict<T> = PositiveFinite<T>;
type Positive<T> = StrictlyPositive<T>;
type Signed<T> = NonNaNFinite<T>;



/// Sealed trait to restrict the float scalar types we can use in our newtypes.
mod private{
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalOverflow;

pub trait FloatScalar: private::Sealed + Sized{
}

pub trait StrictCarrierOps: StrictBounds + Sized {
    type Scalar;

    fn recip(self) -> Positive<Self::Scalar>;
    fn try_recip_finite(self) -> Result<Self, ReciprocalOverflow>;
    unsafe fn recip_finite_unchecked(self) -> Self;
}

macro_rules! impl_strict_carrier_ops {
    ($($t:ty),* $(,)?) => {
        $(
            impl StrictCarrierOps for Strict<$t> {
                type Scalar = $t;

                fn recip(self) -> Positive::<$t> {
                    let y = (1.0 as $t) / self.get();
                    // SAFETY: self is finite and > 0, so y is also > 0.
                    unsafe { Positive::<$t>::new_unchecked(y) }
                }

                fn try_recip_finite(self) -> Result<Self, ReciprocalOverflow> {
                    let y = (1.0 as $t) / self.get();
                    if y.is_finite() {
                        // SAFETY: checked finite above, positivity follows from self > 0.
                        Ok(unsafe { Strict::<$t>::new_unchecked(y) })
                    } else {
                        Err(ReciprocalOverflow)
                    }
                }

                unsafe fn recip_finite_unchecked(self) -> Self {
                    let y = (1.0 as $t) / self.get();
                    debug_assert!(y.is_finite());
                    unsafe { Strict::<$t>::new_unchecked(y) }
                }
            }
        )*
    };
}

impl_strict_carrier_ops!(f32, f64);


pub trait NonStrictCarrierOps: NonStrictBounds + Sized{
    type Scalar;

    fn zero() -> Self;
}

macro_rules! impl_non_strict_carrier_ops {
    ($($t:ty),* $(,)?) => {
        $(
            impl NonStrictCarrierOps for NonStrict<$t> {
                type Scalar = $t;
                fn zero() -> Self {
                    // Default is 0.0
                    Self::default() 
                }
            }
        )*
    };
}

impl_non_strict_carrier_ops!(f32, f64);


// Trait to indicate that a type is a transparent wrapper over its inner type, 
// which must be Strict<T> or NonStrict<T> for some T: FloatScalar. 
// This allows us to safely reinterpret between different newtypes without copying.
pub unsafe trait TransparentOver {
    type Inner;
}


unsafe impl<T: FloatScalar> TransparentOver for Strict<T> {
    type Inner = Strict<T>;
}

unsafe impl<T: FloatScalar> TransparentOver for NonStrict<T> {
    type Inner = NonStrict<T>;
}


/// Zero-copy reinterpret a vector of `T` as a vector of `U`.
/// Both `T` and `U` must be transparent wrappers over the same base representation (f64).
pub fn reinterpret_vec<T, U>(v: Vec<T>) -> Vec<U> 
    where 
        T: TransparentOver,
        U: TransparentOver<Inner = T::Inner>
{
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
pub fn reinterpret_slice<T, U>(s: &[T]) -> &[U] 
    where 
        T: TransparentOver,
        U: TransparentOver<Inner = T::Inner>
{
    use std::mem::{align_of, size_of};
    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const U, s.len()) }
}


// Traits to setup the star topology for our newtypes that enables zero-copy reinterpretation between them.
pub trait WrapsCarrierFloat{
    type Inner;
    fn into_float(self) -> Self::Inner;
    fn from_float(x: Self::Inner) -> Self;

}

pub fn convert<T, U>(x: T) -> U 
    where
        T: WrapsCarrierFloat,
        U: WrapsCarrierFloat<Inner = T::Inner>
{
    U::from_float(x.into_float())
}


// Helper traits to enforce strict vs non-strict bounds on the newtypes.
// Reduces trait bound boiler plate since we can replace
// where 
//    T: FloatScalar,
//    Strict<T>: Eq + PartialEq + PartialOrd + Ord + std::hash::Hash
// with just
// where T: FloatScalar, U: StrictBounds<T>
// or even just:
// where Strict<T>: StrictBounds<T>
pub trait StrictBounds:
    Eq + PartialEq + PartialOrd + Ord + Hash + fmt::Display
{}

impl StrictBounds for Strict<f32> {}
impl StrictBounds for Strict<f64> {}

pub trait NonStrictBounds:
    Eq + PartialEq + PartialOrd + Ord + Hash + fmt::Display
{}

impl NonStrictBounds for NonStrict<f32> {}
impl NonStrictBounds for NonStrict<f64> {}



macro_rules! impl_wrapper_common {
    ($name:ident, $carrier:ident, $bounds:ident) => {
        #[repr(transparent)]
        #[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
        pub struct $name<T>(pub $carrier<T>)
        where
            $carrier<T>: $bounds;

        unsafe impl<T> TransparentOver for $name<T>
        where
            $carrier<T>: $bounds,
        {
            type Inner = $carrier<T>;
        }

        impl<T> WrapsCarrierFloat for $name<T>
        where
            $carrier<T>: $bounds,
        {
            type Inner = $carrier<T>;

            fn into_float(self) -> Self::Inner {
                self.0
            }

            fn from_float(x: Self::Inner) -> Self {
                Self(x)
            }
        }

        impl<T> From<$carrier<T>> for $name<T>
        where
            $carrier<T>: $bounds,
        {
            fn from(x: $carrier<T>) -> Self {
                Self(x)
            }
        }

        // Implement common functions
        impl<T> $name<T>
        where
            $carrier<T>: $bounds,
        {
                pub fn new(x: $carrier<T>) -> Self {
                    Self(x)
                }
                
        }

        impl<T> fmt::Display for $name<T>
        where
            $carrier<T>: $bounds,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl<T> Add for $name<T>
        where
            $carrier<T>: $bounds + Add<Output = $carrier<T>>,
        {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl<T> Mul for $name<T>
        where
            $carrier<T>: $bounds + Mul<Output = $carrier<T>>,
        {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

         impl<T> Sub for $name<T>
        where
            $carrier<T>: $bounds + Sub<Output = $carrier<T>>,
        {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
    };
}


// Declare the strict newtypes
macro_rules! newtypes_strict {
    ($($name:ident),*) => {
        $(  

            // Declare the newtype and implement the common traits            
            impl_wrapper_common!($name, Strict, StrictBounds);

            // Implement the strict carrier ops for the newtype by delegating to the inner Strict<T> type.
            impl<T> $name<T>
            where
                Strict<T>: StrictCarrierOps<Scalar = T>,
            {
                pub fn recip(self) -> Positive<T> {
                    self.0.recip()
                }

                pub fn try_recip_finite(self) -> Result<Self, ReciprocalOverflow> {
                    self.0.try_recip_finite().map(Self)
                }

                pub unsafe fn recip_finite_unchecked(self) -> Self {
                    unsafe { Self(self.0.recip_finite_unchecked()) }
                }


            }

        )*
    };
}

/// Declare the non-strict newtypes
macro_rules! newtypes_non_strict {
    ($($name:ident),*) => {
        $(  
            // Declare the newtype and implement the common traits
            impl_wrapper_common!($name, NonStrict, NonStrictBounds);

            impl<T> $name<T>
            where
                NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
            {
                pub fn zero() -> Self{
                    Self(NonStrict::<T>::zero())
                }    
            }

        )*
    };
}

// Use the macros to declare the newtypes:
newtypes_strict!(
    EdgeWeight,
    NodeDegree,
    Volume
);

newtypes_non_strict!(
    Contribution,
    SmoothedContribution,
    FDelta,
    SmoothingTermDelta,
    HB,
    HS
);







macro_rules! impl_quack_like_a_float {
    ($($t:ident),*) => {
        $(



            impl Add for $t {
                type Output = Self;
                #[inline(always)]
                fn add(self, rhs: Self) -> Self {
                    Self(self.0 + rhs.0)
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
