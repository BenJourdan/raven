use std::num::NonZeroUsize;

use crate::error::ReciprocalOverflow;

use super::{NonStrict, NonStrictBounds, Positive, Strict, StrictBounds};


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


/// Operations that are valid for the strict positive finite carrier family.
pub trait StrictCarrierOps: StrictBounds + Sized {
    type Scalar;

    fn from_positive_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber>;
    /// SAFETY: The caller must ensure that `x` is a valid positive finite number.
    unsafe fn from_positive_scalar_unchecked(x: Self:: Scalar) -> Self;
    fn into_scalar(self) -> Self::Scalar;
    fn from_non_zero_usize(x: NonZeroUsize) -> Self;
    fn recip(self) -> Positive<Self::Scalar>;
    fn try_recip_finite(self) -> Result<Self, ReciprocalOverflow>;
    unsafe fn recip_finite_unchecked(self) -> Self;
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

                fn recip(self) -> Positive<$t> {
                    let y = (1.0 as $t) / self.get();
                    unsafe { Positive::<$t>::new_unchecked(y) }
                }

                fn try_recip_finite(self) -> Result<Self, ReciprocalOverflow> {
                    let y = (1.0 as $t) / self.get();
                    if y.is_finite() {
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

/// Operations that are valid for the non-strict positive finite carrier family.
pub trait NonStrictCarrierOps: NonStrictBounds + Sized {
    type Scalar;

    fn from_positive_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber>;
    /// SAFETY: The caller must ensure that `x` is a valid non-negative finite number.
    unsafe fn from_positive_scalar_unchecked(x: Self:: Scalar) -> Self;
    fn into_scalar(self) -> Self::Scalar;
    fn zero() -> Self;
    fn from_usize(x: usize) -> Self;
}

macro_rules! impl_non_strict_carrier_ops {
    ($($t:ty),* $(,)?) => {
        $(
            impl NonStrictCarrierOps for NonStrict<$t> {
                type Scalar = $t;

                fn from_positive_scalar(x: Self::Scalar) -> Result<Self, InvalidNumber> {
                    NonStrict::<$t>::new(x).map_err(|e| e.into())
                }

                unsafe fn from_positive_scalar_unchecked(x: Self::Scalar) -> Self {
                    #[cfg(debug_assertions)]
                    {
                        NonStrict::<$t>::new(x).expect("Invalid number passed to from_positive_scalar_unchecked")
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
