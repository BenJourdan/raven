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
//         NonStrict, NonStrictBounds, NonStrictCarrierOps, Positive, Strict, StrictBounds,
//         StrictCarrierOps, TransparentOver, WrapsCarrierFloat,
//     },
//     error::ReciprocalOverflow,
// };

macro_rules! impl_wrapper_common {
    ($name:ident, $carrier:ident, $bounds:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $name<T>(pub $carrier<T>);
        // where
        //     $carrier<T>: $bounds;

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

        impl<T> $name<T>
        where
            $carrier<T>: $bounds,
        {
            pub fn new(x: $carrier<T>) -> Self {
                Self(x)
            }
        }

        impl<T> Display for $name<T>
        where
            $carrier<T>: $bounds,
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
            crate::types::float::macros::impl_wrapper_common!($name, Strict, StrictBounds);

            impl<T> $name<T>
            where
                Strict<T>: StrictCarrierOps<Scalar = T>,
            {
                pub fn recip(self) -> Positive<T> {
                    <Strict<T> as StrictCarrierOps>::recip(self.0)
                }

                pub fn try_recip_finite(
                    self,
                ) -> Result<Self, ReciprocalOverflow> {
                    <Strict<T> as StrictCarrierOps>::try_recip_finite(self.0)
                        .map($name)
                }
                /// # Safety
                /// The caller must ensure that 'self' is a finite positive value.
                pub unsafe fn recip_finite_unchecked(self) -> Self {
                    unsafe {
                        $name(
                            <Strict<T> as StrictCarrierOps>::recip_finite_unchecked(self.0)
                        )
                    }
                }
            }
        )*
    };
}

macro_rules! newtypes_non_strict {
    ($($name:ident),* $(,)?) => {
        $(
            crate::types::float::macros::impl_wrapper_common!($name, NonStrict, NonStrictBounds);

            impl<T> $name<T>
            where
                NonStrict<T>: NonStrictCarrierOps<Scalar = T>,
            {
                pub fn zero() -> Self {
                    $name(
                        <NonStrict<T> as NonStrictCarrierOps>::zero()
                    )
                }
            }
        )*
    };
}

pub(crate) use impl_wrapper_common;
pub(crate) use newtypes_non_strict;
pub(crate) use newtypes_strict;
