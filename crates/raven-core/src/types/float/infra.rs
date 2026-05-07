use super::{FloatScalar, NonStrict, Strict};

/// Marks transparent wrappers that can be reinterpreted as their inner carrier.
///
/// This is used by `reinterpret_vec` and `reinterpret_slice` to do zero-copy
/// casts between wrapper types that share the exact same carrier type.
pub unsafe trait TransparentOver {
    type Inner;
}

unsafe impl<T: FloatScalar> TransparentOver for Strict<T> {
    type Inner = Strict<T>;
}

unsafe impl<T: FloatScalar> TransparentOver for NonStrict<T> {
    type Inner = NonStrict<T>;
}

/// Zero-copy reinterpret a vector of `T` as a vector of `U`.
pub fn reinterpret_vec<T, U>(v: Vec<T>) -> Vec<U>
where
    T: TransparentOver,
    U: TransparentOver<Inner = T::Inner>,
{
    use std::mem::{align_of, size_of};

    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());

    let len = v.len();
    let cap = v.capacity();
    let ptr = v.as_ptr();
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(ptr as *mut U, len, cap) }
}

/// Zero-copy reinterpret a slice of `T` as a slice of `U`.
pub fn reinterpret_slice<T, U>(s: &[T]) -> &[U]
where
    T: TransparentOver,
    U: TransparentOver<Inner = T::Inner>,
{
    use std::mem::{align_of, size_of};

    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const U, s.len()) }
}
