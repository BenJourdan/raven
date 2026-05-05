pub trait PowerOfTwo {}

pub struct ConstPow2<const N: usize>;

macro_rules! impl_power_of_two_up_to_1024 {
    ($($pow:expr),* $(,)?) => {
        $(
            impl PowerOfTwo for ConstPow2<$pow> {}
        )*
    };
}

impl_power_of_two_up_to_1024! {
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
}
