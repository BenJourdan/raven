use std::{fmt, hash::Hash, num::NonZeroUsize};

use super::{NonStrict, Strict};

#[derive(Debug, Clone, Copy)]
pub enum InvalidNumber {
    NaN,
    Zero,
    Negative,
    Positive,
    Infinite,
}

impl std::fmt::Display for InvalidNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvalidNumber::NaN => write!(f, "NaN is not a valid number"),
            InvalidNumber::Zero => write!(f, "Zero is not a valid number"),
            InvalidNumber::Negative => write!(f, "Negative numbers are not valid"),
            InvalidNumber::Positive => write!(f, "Positive numbers are not valid"),
            InvalidNumber::Infinite => write!(f, "Infinite values are not valid"),
        }
    }
}
impl std::error::Error for InvalidNumber {}

// adapt from typed-float's error type for invalid numbers, so we don't leak their error type:
impl From<typed_floats::InvalidNumber> for InvalidNumber {
    fn from(err: typed_floats::InvalidNumber) -> Self {
        match err {
            typed_floats::InvalidNumber::NaN => InvalidNumber::NaN,
            typed_floats::InvalidNumber::Zero => InvalidNumber::Zero,
            typed_floats::InvalidNumber::Negative => InvalidNumber::Negative,
            typed_floats::InvalidNumber::Positive => InvalidNumber::Positive,
            typed_floats::InvalidNumber::Infinite => InvalidNumber::Infinite,
        }
    }
}

/// Operations and standard bounds for the strict positive finite carrier family.
pub trait StrictCarrierOps: Ord + Hash + fmt::Display + Sized {
    type Scalar;

    fn from_positive_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber>;
    /// # Safety
    /// The caller must ensure that `x` is a valid positive finite number.
    unsafe fn from_positive_scalar_unchecked(x: Self::Scalar) -> Self;
    fn into_scalar(self) -> Self::Scalar;
    fn from_non_zero_usize(x: NonZeroUsize) -> Self;
    fn one() -> Self;
}

macro_rules! impl_strict_carrier_ops {
    ($($t:ty),* $(,)?) => {
        $(
            impl StrictCarrierOps for Strict<$t> {
                type Scalar = $t;

                fn from_positive_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber> {
                    Strict::<$t>::new(x).map_err(|e| e.into())
                }

                unsafe fn from_positive_scalar_unchecked(x: Self::Scalar) -> Self {

                    #[cfg(debug_assertions)]
                    {
                        Strict::<$t>::new(x).expect("Invalid number passed to from_positive_scalar_unchecked")
                    }
                    #[cfg(not(debug_assertions))]
                    {
                        unsafe { Strict::<$t>::new_unchecked(x) }
                    }

                }

                fn into_scalar(self) -> Self::Scalar {
                    self.get()
                }

                fn from_non_zero_usize(x: NonZeroUsize) -> Self {
                    let val = x.get() as $t;
                    // SAFETY: The value is guaranteed to be positive and finite.
                    unsafe { Strict::<$t>::new_unchecked(val) }
                }

                fn one() -> Self {
                    unsafe { Strict::<$t>::new_unchecked(1.0) }
                }
            }
        )*
    };
}

impl_strict_carrier_ops!(f32, f64);

/// Operations and standard bounds for the non-strict positive finite carrier family.
pub trait NonStrictCarrierOps: Ord + Hash + fmt::Display + Sized {
    type Scalar;

    fn from_non_negative_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber>;
    /// # Safety
    /// The caller must ensure that `x` is a valid non-negative finite number.
    unsafe fn from_non_negative_scalar_unchecked(x: Self::Scalar) -> Self;
    fn into_scalar(self) -> Self::Scalar;
    fn zero() -> Self;
    fn from_usize(x: usize) -> Self;
}

macro_rules! impl_non_strict_carrier_ops {
    ($($t:ty),* $(,)?) => {
        $(
            impl NonStrictCarrierOps for NonStrict<$t> {
                type Scalar = $t;

                fn from_non_negative_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber> {
                    NonStrict::<$t>::new(x).map_err(|e| e.into())
                }

                unsafe fn from_non_negative_scalar_unchecked(x: Self::Scalar) -> Self {
                    #[cfg(debug_assertions)]
                    {
                        NonStrict::<$t>::new(x).expect("Invalid number passed to from_non_negative_scalar_unchecked")
                    }
                    #[cfg(not(debug_assertions))]
                    {
                        unsafe { NonStrict::<$t>::new_unchecked(x) }
                    }
                }

                fn into_scalar(self) -> Self::Scalar {
                    self.get()
                }
                fn zero() -> Self {
                    unsafe { NonStrict::<$t>::new_unchecked(0.0) }
                }

                fn from_usize(x: usize) -> Self {
                    let val = x as $t;
                    // SAFETY: The value is guaranteed to be non-negative and finite.
                    unsafe { NonStrict::<$t>::new_unchecked(val) }
                }
            }
        )*
    };
}

impl_non_strict_carrier_ops!(f32, f64);
