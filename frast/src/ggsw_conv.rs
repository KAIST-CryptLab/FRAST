use tfhe::core_crypto::{
    prelude::{*, polynomial_algorithms::*},
    fft_impl::fft64::{
        c64,
        crypto::ggsw::FourierGgswCiphertextListView,
    },
};
use crate::utils::*;
#[cfg(feature = "multithread")]
use rayon::prelude::*;

pub fn generate_fourier_ggsw_key<Scalar, G>(
    glwe_secret_key: &GlweSecretKeyOwned<Scalar>,
    ggsw_key_decomp_base_log: DecompositionBaseLog,
    ggsw_key_decomp_level: DecompositionLevelCount,
    noise_parameters: impl DispersionParameter,
    ciphertext_modulus: CiphertextModulus<Scalar>,
    generator: &mut EncryptionRandomGenerator<G>,
) -> FourierGgswCiphertextList<Vec<c64>>
where
    Scalar: UnsignedTorus,
    G: ByteRandomGenerator,
{
    let glwe_dimension = glwe_secret_key.glwe_dimension();
    let glwe_size = glwe_dimension.to_glwe_size();
    let polynomial_size = glwe_secret_key.polynomial_size();
    let glwe_sk_poly_list = glwe_secret_key.as_polynomial_list();

    let mut ggsw_key = GgswCiphertextList::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        ggsw_key_decomp_base_log,
        ggsw_key_decomp_level,
        GgswCiphertextCount(glwe_dimension.0),
        ciphertext_modulus,
    );
    for mut ggsw in ggsw_key.iter_mut() {
        encrypt_constant_ggsw_ciphertext(
            glwe_secret_key,
            &mut ggsw,
            Plaintext(Scalar::ZERO),
            noise_parameters,
            generator,
        );
    }

    for (i, mut ggsw) in ggsw_key.iter_mut().enumerate() {
        let glwe_sk_poly_i = glwe_sk_poly_list.get(i);
        for (row, mut glwe) in ggsw.as_mut_glwe_list().iter_mut().enumerate() {
            let k = row / glwe_size.0;
            let log_scale = Scalar::BITS - (k + 1) * ggsw_key_decomp_base_log.0;

            let mut buf = Polynomial::new(Scalar::ZERO, polynomial_size);
            for (elem, sk) in buf.iter_mut().zip(glwe_sk_poly_i.iter()) {
                *elem = (*sk).wrapping_neg() << log_scale;
            }

            let col = row % glwe_size.0;
            if col < glwe_dimension.0 {
                let mut mask = glwe.get_mut_mask();
                let mut mask = mask.as_mut_polynomial_list();
                let mut mask = mask.get_mut(col);
                polynomial_wrapping_add_assign(&mut mask, &buf);
            } else {
                let mut body = glwe.get_mut_body();
                let mut body = body.as_mut_polynomial();
                polynomial_wrapping_add_assign(&mut body, &buf);
            }
        }
    }

    let mut fourier_ggsw_key = FourierGgswCiphertextList::new(
        vec![
            c64::default();
            glwe_dimension.0 * polynomial_size.to_fourier_polynomial_size().0
                * glwe_size.0
                * glwe_size.0
                * ggsw_key_decomp_level.0
        ],
        glwe_dimension.0,
        glwe_size,
        polynomial_size,
        ggsw_key_decomp_base_log,
        ggsw_key_decomp_level,
    );

    for (mut fourier_ggsw, ggsw) in fourier_ggsw_key.as_mut_view().into_ggsw_iter().zip(ggsw_key.iter()) {
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw);
    }

    fourier_ggsw_key
}

pub fn glev_to_ggsw<Scalar>(
    fourier_ggsw_key: FourierGgswCiphertextListView,
    vec_glev: &[GlweCiphertextListOwned<Scalar>],
    ggsw_bit_decomp_base_log: DecompositionBaseLog,
    ggsw_bit_decomp_level_count: DecompositionLevelCount,
    ciphertext_modulus: CiphertextModulus<Scalar>,
) -> GgswCiphertextListOwned<Scalar>
where
    Scalar: UnsignedTorus,
{
    let glwe_size = fourier_ggsw_key.glwe_size();
    let glwe_dimension = glwe_size.to_glwe_dimension();
    let polynomial_size = fourier_ggsw_key.polynomial_size();
    let num_bits = vec_glev.len();

    let mut ggsw_bit_list = GgswCiphertextList::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
        GgswCiphertextCount(num_bits),
        ciphertext_modulus,
    );

    #[cfg(not(feature = "multithread"))]
    for (i, mut ggsw_bit) in ggsw_bit_list.iter_mut().enumerate() {
        for (col, mut glwe_list) in ggsw_bit.as_mut_glwe_list().chunks_exact_mut(glwe_size.0).enumerate() {
            let glwe_bit = vec_glev.get(i).unwrap().get(col);
            let (mut glwe_mask_list, mut glwe_body_list) = glwe_list.split_at_mut(glwe_dimension.0);

            for (mut glwe_mask, fourier_ggsw) in glwe_mask_list.iter_mut().zip(fourier_ggsw_key.as_view().into_ggsw_iter()) {
                add_external_product_assign(&mut glwe_mask, &fourier_ggsw, &glwe_bit)
            }
            glwe_ciphertext_clone_from(&mut glwe_body_list.get_mut(0), &glwe_bit);
        }
    }
    #[cfg(feature = "multithread")]
    ggsw_bit_list.par_iter_mut().enumerate().for_each(|(i, mut ggsw_bit)| {
        for (col, mut glwe_list) in ggsw_bit.as_mut_glwe_list().chunks_exact_mut(glwe_size.0).enumerate() {
            let glwe_bit = vec_glev.get(i).unwrap().get(col);
            let (mut glwe_mask_list, mut glwe_body_list) = glwe_list.split_at_mut(glwe_dimension.0);

            for (mut glwe_mask, fourier_ggsw) in glwe_mask_list.iter_mut().zip(fourier_ggsw_key.as_view().into_ggsw_iter()) {
                add_external_product_assign(&mut glwe_mask, &fourier_ggsw, &glwe_bit)
            }
            glwe_ciphertext_clone_from(&mut glwe_body_list.get_mut(0), &glwe_bit.as_view());
        }
    });

    ggsw_bit_list
}
