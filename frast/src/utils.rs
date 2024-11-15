use tfhe::core_crypto::prelude::{*, polynomial_algorithms::*};

pub type SeededLWE = SeededLweCiphertext<u64>;
pub type LWE = LweCiphertext<Vec<u64>>;
pub type LWEList = LweCiphertextList<Vec<u64>>;

pub fn get_val_and_error<C: Container<Element=u64>>(
    lwe_secret_key: &LweSecretKey<Vec<u64>>,
    lwe_ctxt: &LweCiphertext<C>,
    correct_val: u64,
    delta: u64,
) -> (u64, u32) {
    let decrypted_u64 = decrypt_lwe_ciphertext(&lwe_secret_key, &lwe_ctxt).0;
    let err = {
        let correct_val = correct_val * delta;
        let d0 = decrypted_u64.wrapping_sub(correct_val);
        let d1 = correct_val.wrapping_sub(decrypted_u64);
        std::cmp::min(d0, d1)
    };
    let bit_error = if err != 0 {64 - err.leading_zeros()} else {0};
    let rounding = (decrypted_u64 & (delta >> 1)) << 1;
    let decoded = (decrypted_u64.wrapping_add(rounding)) / delta;

    return (decoded, bit_error);
}

pub fn get_val_and_abs_error<C: Container<Element=u64>>(
    lwe_secret_key: &LweSecretKey<Vec<u64>>,
    lwe_ctxt: &LweCiphertext<C>,
    correct_val: u64,
    delta: u64,
) -> (u64, u64) {
    let decrypted_u64 = decrypt_lwe_ciphertext(&lwe_secret_key, &lwe_ctxt).0;
    let err = {
        let correct_val = correct_val * delta;
        let d0 = decrypted_u64.wrapping_sub(correct_val);
        let d1 = correct_val.wrapping_sub(decrypted_u64);
        std::cmp::min(d0, d1)
    };
    let rounding = (decrypted_u64 & (delta >> 1)) << 1;
    let decoded = (decrypted_u64.wrapping_add(rounding)) / delta;

    return (decoded, err);
}

pub fn glwe_ciphertext_monic_monomial_div_assign<Scalar, ContMut>(
    glwe: &mut GlweCiphertext<ContMut>,
    monomial_degree: MonomialDegree,
) where
    Scalar: UnsignedInteger,
    ContMut: ContainerMut<Element=Scalar>,
{
    for mut poly in glwe.as_mut_polynomial_list().iter_mut() {
        polynomial_wrapping_monic_monomial_div_assign(&mut poly, monomial_degree);
    }
}

pub fn glwe_ciphertext_clone_from<Scalar, OutputCont, InputCont>(
    dst: &mut GlweCiphertext<OutputCont>,
    src: &GlweCiphertext<InputCont>,
) where
    Scalar: UnsignedTorus,
    InputCont: Container<Element=Scalar>,
    OutputCont: ContainerMut<Element=Scalar>,
{
    debug_assert!(dst.glwe_size() == src.glwe_size());
    debug_assert!(dst.polynomial_size() == src.polynomial_size());
    dst.as_mut().clone_from_slice(src.as_ref());
}


pub fn bit_length(input: usize) -> usize {
    if input == 0 {return 1};

    let mut bit_len = 0;
    let mut tmp = input;
    while tmp > 0 {
        tmp >>= 1;
        bit_len += 1;
    }

    bit_len
}

/* -------- Macro -------- */
// https://docs.rs/itertools/0.7.8/src/itertools/lib.rs.html#247-269
#[allow(unused_macros)]
macro_rules! izip {
    (@ __closure @ ($a:expr)) => { |a| (a,) };
    (@ __closure @ ($a:expr, $b:expr)) => { |(a, b)| (a, b) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr)) => { |((a, b), c)| (a, b, c) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr)) => { |(((a, b), c), d)| (a, b, c, d) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr)) => { |((((a, b), c), d), e)| (a, b, c, d, e) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr)) => { |(((((a, b), c), d), e), f)| (a, b, c, d, e, f) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr)) => { |((((((a, b), c), d), e), f), g)| (a, b, c, d, e, f, e) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr)) => { |(((((((a, b), c), d), e), f), g), h)| (a, b, c, d, e, f, g, h) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr)) => { |((((((((a, b), c), d), e), f), g), h), i)| (a, b, c, d, e, f, g, h, i) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr)) => { |(((((((((a, b), c), d), e), f), g), h), i), j)| (a, b, c, d, e, f, g, h, i, j) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr)) => { |((((((((((a, b), c), d), e), f), g), h), i), j), k)| (a, b, c, d, e, f, g, h, i, j, k) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr)) => { |(((((((((((a, b), c), d), e), f), g), h), i), j), k), l)| (a, b, c, d, e, f, g, h, i, j, k, l) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr)) => { |((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m)| (a, b, c, d, e, f, g, h, i, j, k, l, m) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr, $n:expr)) => { |(((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m), n)| (a, b, c, d, e, f, g, h, i, j, k, l, m, n) };
    (@ __closure @ ($a:expr, $b:expr, $c:expr, $d:expr, $e: expr, $f:expr, $g:expr, $h:expr, $i: expr, $j: expr, $k: expr, $l: expr, $m:expr, $n:expr, $o:expr)) => { |((((((((((((((a, b), c), d), e), f), g), h), i), j), k), l), m), n), o)| (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) };

    ( $first:expr $(,)?) => {
        {
            #[allow(unused_imports)]
            use $crate::core_crypto::commons::utils::ZipChecked;
            ::core::iter::IntoIterator::into_iter($first)
        }
    };
    ( $first:expr, $($rest:expr),+ $(,)?) => {
        {
            #[allow(unused_imports)]
            use tfhe::core_crypto::commons::utils::ZipChecked;
            ::core::iter::IntoIterator::into_iter($first)
                $(.zip_checked($rest))*
                .map($crate::utils::izip!(@ __closure @ ($first, $($rest),*)))
        }
    };
}

#[allow(unused_imports)]
pub(crate) use izip;

