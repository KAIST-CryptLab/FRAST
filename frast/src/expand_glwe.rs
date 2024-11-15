use std::collections::HashMap;
use aligned_vec::ABox;
use tfhe::core_crypto::{
    prelude::*,
    fft_impl::fft64::c64,
};
use crate::utils::*;
use crate::automorphism::*;

pub fn expand_glwe<Scalar: UnsignedTorus + Sync + Send>(
    glwe_in: GlweCiphertextView<Scalar>,
    ksk_map: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) -> GlweCiphertextListOwned<Scalar> {
    let n = glwe_in.polynomial_size().0;
    let mut buf = GlweCiphertextOwned::new(Scalar::ZERO, glwe_in.glwe_size(), glwe_in.polynomial_size(), glwe_in.ciphertext_modulus());

    let mut glwe_out_list = GlweCiphertextListOwned::new(Scalar::ZERO, glwe_in.glwe_size(), glwe_in.polynomial_size(), GlweCiphertextCount(glwe_in.polynomial_size().0), glwe_in.ciphertext_modulus());
    let mut glwe_out_first = glwe_out_list.get_mut(0);
    for (mut poly_out, poly) in glwe_out_first.as_mut_polynomial_list().iter_mut().zip(glwe_in.as_polynomial_list().iter()) {
        poly_out.as_mut().clone_from_slice(poly.as_ref());
    }

    for i in 1..=log2(n) {
        let k = n / (1 << (i - 1)) + 1;
        let ksk = ksk_map.get(&k).unwrap();
        for b in 0..(1 << (i - 1)) {
            let mut tmp = GlweCiphertext::new(Scalar::ZERO, glwe_in.glwe_size(), glwe_in.polynomial_size(), glwe_in.ciphertext_modulus());
            glwe_ciphertext_clone_from(&mut tmp, &glwe_out_list.get(b));
            ksk.auto(&mut buf, &tmp);

            glwe_ciphertext_add(&mut glwe_out_list.get_mut(b), &tmp, &buf);
            glwe_ciphertext_sub(&mut glwe_out_list.get_mut(b + (1 << (i - 1))), &tmp, &buf);
            glwe_ciphertext_monic_monomial_div_assign(&mut glwe_out_list.get_mut(b + (1 << (i - 1))), MonomialDegree(1 << (i-1)));
        }
    }

    glwe_out_list
}

pub fn encode_bits_into_glwe_ciphertext<Scalar, G>(
    glwe_secret_key: &GlweSecretKeyOwned<Scalar>,
    bit_list: &[Scalar],
    ggsw_bit_decomp_base_log: DecompositionBaseLog,
    ggsw_bit_decomp_level_count: DecompositionLevelCount,
    noise_parameters: impl DispersionParameter,
    generator: &mut EncryptionRandomGenerator<G>,
    ciphertext_modulus: CiphertextModulus<Scalar>,
) -> Vec<GlweCiphertextListOwned<Scalar>>
where
    Scalar: UnsignedTorus,
    G: ByteRandomGenerator,
{
    let glwe_size = glwe_secret_key.glwe_dimension().to_glwe_size();
    let polynomial_size = glwe_secret_key.polynomial_size();
    let num_glwe_list = bit_list.len() / polynomial_size.0;
    let num_glwe_list = if bit_list.len() % polynomial_size.0 == 0 {num_glwe_list} else {num_glwe_list + 1};

    let mut vec_glwe_list = vec![GlweCiphertextList::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(ggsw_bit_decomp_level_count.0),
        ciphertext_modulus,
    ); num_glwe_list];

    for (idx, glwe_list) in vec_glwe_list.iter_mut().enumerate() {
        for (k, mut glwe) in glwe_list.iter_mut().enumerate() {
            let log_scale = Scalar::BITS - ggsw_bit_decomp_base_log.0 * (k + 1) - log2(polynomial_size.0);
            let pt = PlaintextList::from_container(
                (0..polynomial_size.0).map(|i| {
                    let bit_idx = idx * polynomial_size.0 + i;
                    if bit_idx < bit_list.len() {
                        bit_list[bit_idx] << log_scale
                    } else {
                        Scalar::ZERO
                    }
                }).collect::<Vec<Scalar>>()
            );

            encrypt_glwe_ciphertext(&glwe_secret_key, &mut glwe, &pt, noise_parameters, generator);
        }
    }

    vec_glwe_list
}
