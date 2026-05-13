mod carrier_ops;
mod infra;
mod macros;
mod quantities;
mod scalar;

pub use quantities::*;
pub use scalar::{FP_EPSILON32, FP_EPSILON64};

pub use carrier_ops::{InvalidNumber, NonStrictCarrierOps, StrictCarrierOps};
pub(crate) use infra::TransparentOver;
pub use infra::{reinterpret_slice, reinterpret_vec};
pub use scalar::{FloatScalar, NonStrict, Strict};
