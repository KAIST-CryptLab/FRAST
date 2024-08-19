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

pub fn glwe_clone_from<Scalar: UnsignedInteger>(mut dst: GlweCiphertextMutView<Scalar>, src: GlweCiphertextView<Scalar>) {
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
