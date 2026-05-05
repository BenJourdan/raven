use typed_floats::{NonNaNFinite, PositiveFinite, StrictlyPositive, StrictlyPositiveFinite};

pub const FP_EPSILON32: StrictlyPositiveFinite<f32> =
    unsafe { StrictlyPositiveFinite::<f32>::new_unchecked(1e-6) };
pub const FP_EPSILON64: StrictlyPositiveFinite<f64> =
    unsafe { StrictlyPositiveFinite::<f64>::new_unchecked(1e-12) };


pub type Strict<T> = StrictlyPositiveFinite<T>;
pub type NonStrict<T> = PositiveFinite<T>;
pub type Positive<T> = StrictlyPositive<T>;
pub type Signed<T> = NonNaNFinite<T>;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

pub trait FloatScalar: 
    private::Sealed 
    + Sized
    + num_traits::Float
{
    const ZERO: Self;
    const ONE: Self;

    fn from_bool(x: bool) -> Self{
        if x {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

}

impl FloatScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
impl FloatScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
