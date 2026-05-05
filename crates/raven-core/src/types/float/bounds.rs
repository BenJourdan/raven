use std::{fmt, hash::Hash};

use super::{NonStrict, Strict};

pub trait StrictBounds: Eq + PartialEq + PartialOrd + Ord + Hash + fmt::Display {}

impl StrictBounds for Strict<f32> {}
impl StrictBounds for Strict<f64> {}

pub trait NonStrictBounds: Eq + PartialEq + PartialOrd + Ord + Hash + fmt::Display {}

impl NonStrictBounds for NonStrict<f32> {}
impl NonStrictBounds for NonStrict<f64> {}
