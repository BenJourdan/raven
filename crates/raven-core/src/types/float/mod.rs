mod bounds;
mod carrier_ops;
mod infra;
mod macros;
mod quantities;
mod scalar;

pub use quantities::*;
pub use scalar::{FP_EPSILON32, FP_EPSILON64};

pub(crate) use bounds::{NonStrictBounds, StrictBounds};
pub(crate) use carrier_ops::{NonStrictCarrierOps, StrictCarrierOps};
pub(crate) use infra::TransparentOver;
pub use infra::{convert, reinterpret_slice, reinterpret_vec, WrapsCarrierFloat};
pub(crate) use scalar::{FloatScalar, NonStrict, Positive, Signed, Strict};
