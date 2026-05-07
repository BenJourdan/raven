use std::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
};

use crate::types::{
    FloatScalar,
    float::{
        InvalidNumber, NonStrict, NonStrictCarrierOps, Strict, StrictCarrierOps, TransparentOver,
    },
};

use super::macros::{newtypes_non_strict, newtypes_strict};

newtypes_strict!(EdgeWeight, NodeDegree, Volume);

newtypes_non_strict!(
    Contribution,
    SmoothedContribution,
    FDelta,
    SmoothingTermDelta,
    HB,
    HS,
);
