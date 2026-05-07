// These names are used by the macro expansions below. They must be imported
// at each invocation site, not here, because `macro_rules!` item paths resolve
// in the module where the macro expands.
//
// use std::{
//     cmp::Ordering,
//     fmt::{Debug, Display, Formatter, Result as FmtResult},
//     hash::{Hash, Hasher},
// };
//
// use crate::{
//     types::float::{
//         FloatScalar, InvalidNumber, NonStrict, NonStrictCarrierOps, Positive,
//         Strict, StrictCarrierOps, TransparentOver,
//     },
//     error::ReciprocalOverflow,
// };

macro_rules! impl_wrapper_common {
    ($name:ident, $carrier:ident, $carrier_ops:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $name<T>(pub $carrier<T>);
        // where
        //     $carrier<T>: $carrier_ops<Scalar = T>;

        impl<T> Debug for $name<T>
        where
            $carrier<T>: Debug,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                f.debug_tuple(stringify!($name)).field(&self.0).finish()
            }
        }

        impl<T> PartialEq for $name<T>
        where
            $carrier<T>: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<T> Eq for $name<T> where $carrier<T>: Eq {}

        impl<T> PartialOrd for $name<T>
        where
            $carrier<T>: PartialOrd,
        {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<T> Ord for $name<T>
        where
            $carrier<T>: Ord,
        {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.cmp(&other.0)
            }
        }

        impl<T> Hash for $name<T>
        where
            $carrier<T>: Hash,
        {
            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher,
            {
                self.0.hash(state);
            }
        }

        unsafe impl<T> TransparentOver for $name<T>
        where
            $carrier<T>: $carrier_ops<Scalar = T>,
        {
            type Inner = $carrier<T>;
        }

        impl<T> From<$carrier<T>> for $name<T>
        where
            $carrier<T>: $carrier_ops<Scalar = T>,
        {
            fn from(x: $carrier<T>) -> Self {
                Self(x)
            }
        }

        impl<T> $name<T>
        where
            $carrier<T>: $carrier_ops<Scalar = T>,
        {
            pub fn new(x: $carrier<T>) -> Self {
                Self(x)
            }

            pub fn into_scalar(self) -> T {
                <$carrier<T> as $carrier_ops>::into_scalar(self.0)
            }
        }

        impl<T> Display for $name<T>
        where
            $carrier<T>: $carrier_ops<Scalar = T>,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                write!(f, "{}", self.0)
            }
        }
    };
}

macro_rules! newtypes_strict {
    ($($name:ident),* $(,)?) => {
        $(
            crate::types::float::macros::impl_wrapper_common!(
                $name,
                Strict,
                StrictCarrierOps
            );

            impl<T> $name<T>
            where
                Strict<T>: StrictCarrierOps<Scalar = T>,
            {
                pub fn from_scalar(x: T) -> Result<Self, InvalidNumber> {
                    <Strict<T> as StrictCarrierOps>::from_positive_scalar(x)
                        .map($name)
                }

                /// # Safety
                /// The caller must ensure that `x` is positive and finite.
                pub unsafe fn from_scalar_unchecked(x: T) -> Self {
                    $name(unsafe {
                        <Strict<T> as StrictCarrierOps>::from_positive_scalar_unchecked(x)
                    })
                }

            }
        )*
    };
}

macro_rules! newtypes_non_strict {
    ($($name:ident),* $(,)?) => {
        $(
            crate::types::float::macros::impl_wrapper_common!(
                $name,
                NonStrict,
                NonStrictCarrierOps
            );

            impl<T> $name<T>
            where
                NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
            {
                pub fn from_scalar(x: T) -> Result<Self, InvalidNumber> {
                    <NonStrict<T> as NonStrictCarrierOps>::from_non_negative_scalar(x)
                        .map($name)
                }

                /// # Safety
                /// The caller must ensure that `x` is non-negative and finite.
                pub unsafe fn from_scalar_unchecked(x: T) -> Self {
                    $name(unsafe {
                        <NonStrict<T> as NonStrictCarrierOps>::from_non_negative_scalar_unchecked(x)
                    })
                }

                pub fn zero() -> Self {
                    $name(
                        <NonStrict<T> as NonStrictCarrierOps>::zero()
                    )
                }
            }

            impl<T> std::iter::Sum for $name<T>
                where
                    T: FloatScalar,
                    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
                {
                    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                        let total: T = iter.map(|x| x.into_scalar()).sum();
                        Self(
                            NonStrict::<T>::from_non_negative_scalar(total)
                            .expect("sum of non-negative finite floats overflowed"),
                        )
                    }
                }

            impl<'a, T> std::iter::Sum<&'a $name<T>> for $name<T>
                where
                    T: FloatScalar,
                    NonStrict<T>: NonStrictCarrierOps<Scalar = T> + Copy,
                {
                    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                        let total: T = iter.map(|x| x.into_scalar()).sum();
                        Self(
                            NonStrict::<T>::from_non_negative_scalar(total)
                            .expect("sum of non-negative finite floats overflowed"),
                        )
                    }
                }
        )*
    };
}

pub(crate) use impl_wrapper_common;
pub(crate) use newtypes_non_strict;
pub(crate) use newtypes_strict;
