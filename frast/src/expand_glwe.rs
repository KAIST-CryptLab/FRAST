use std::collections::HashMap;
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::algorithms::polynomial_algorithms::*;
use crate::utils::*;

// The following codes are extension of rlweExpand
// from https://github.com/KULeuven-COSIC/SortingHat
// to glweExpand with arbitrary GLWE dimension.
pub struct GLWEKeyswitchKey<Scalar: UnsignedTorus> {
    ksks: GlweCiphertextListOwned<Scalar>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    subs_k: usize,
}

impl <Scalar: UnsignedTorus + Sync + Send> GLWEKeyswitchKey<Scalar> {
    pub fn allocate(
        decomp_base_log: DecompositionBaseLog,
        decomp_level_count: DecompositionLevelCount,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        subs_k: usize,
    ) -> Self {
        let glwe_size = glwe_dimension.to_glwe_size();
        let ksks = GlweCiphertextListOwned::new(Scalar::ZERO, glwe_size, polynomial_size, GlweCiphertextCount(glwe_dimension.0 * decomp_level_count.0), CiphertextModulus::new_native());
        GLWEKeyswitchKey {
            ksks: ksks,
            decomp_base_log,
            decomp_level_count,
            glwe_dimension,
            polynomial_size,
            subs_k: subs_k,
        }
    }

    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    pub fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.decomp_level_count
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        self.glwe_dimension
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }

    /// Fill this object with the appropriate key switching key
    /// that is used for the substitution (subs) operation
    /// where after_key is {S_i(X)} and before_key is computed as {S_i(X^k)}.
    pub fn fill_with_subs_keyswitch_key<G: ByteRandomGenerator>(
        &mut self,
        before_key: &mut GlweSecretKeyOwned<Scalar>,
        after_key: &GlweSecretKeyOwned<Scalar>,
        k: usize,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator<G>,
    ) {
        debug_assert!(self.glwe_dimension == before_key.glwe_dimension());
        debug_assert!(self.glwe_dimension == after_key.glwe_dimension());
        debug_assert!(self.polynomial_size == before_key.polynomial_size());
        debug_assert!(self.polynomial_size == after_key.polynomial_size());

        let mut before_poly_list = PolynomialList::new(
            Scalar::ZERO,
            self.polynomial_size,
            PolynomialCount(self.glwe_dimension.0),
        );
        for (mut before_poly, after_poly) in before_poly_list.iter_mut()
            .zip(after_key.as_polynomial_list().iter())
        {
            let out = eval_x_k(after_poly.as_view(), k);
            before_poly.as_mut().clone_from_slice(out.as_ref());
        }
        *before_key = GlweSecretKey::from_container(before_poly_list.into_container(), self.polynomial_size);

        self.fill_with_keyswitch_key(before_key, after_key, noise_parameters, generator);
        self.subs_k = k;
    }

    /// Fill this object with the appropriate key switching key
    /// that transforms ciphertexts under before_key to ciphertexts under after_key.
    pub fn fill_with_keyswitch_key<G: ByteRandomGenerator>(
        &mut self,
        before_key: &GlweSecretKeyOwned<Scalar>,
        after_key: &GlweSecretKeyOwned<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator<G>
    ) {
        debug_assert!(self.glwe_dimension == before_key.glwe_dimension());
        debug_assert!(self.glwe_dimension == after_key.glwe_dimension());
        debug_assert!(self.polynomial_size == before_key.polynomial_size());
        debug_assert!(self.polynomial_size == after_key.polynomial_size());

        let decomp_level_count = self.decomp_level_count.0;
        let decomp_base_log = self.decomp_base_log.0;

        for (s_i, mut ksk_i) in before_key.as_polynomial_list().iter()
            .zip(self.ksks.chunks_exact_mut(decomp_level_count))
        {
            for (level, mut ksk_ij) in (1..=decomp_level_count).zip(ksk_i.iter_mut()) {
                let mut tmp = PlaintextList::new(Scalar::ZERO, PlaintextCount(self.polynomial_size.0));
                tmp.as_mut().clone_from_slice(s_i.as_ref());

                let log_scale = Scalar::BITS - decomp_base_log * level;
                for elem in tmp.iter_mut() {
                    *elem.0 = elem.0.wrapping_mul(Scalar::ONE << log_scale);
                }

                encrypt_glwe_ciphertext(after_key, &mut ksk_ij, &tmp, noise_parameters, generator);
            }
        }
    }

    pub fn keyswitch_ciphertext(
        &self,
        mut after: GlweCiphertextMutView<Scalar>,
        before: GlweCiphertextView<Scalar>,
    ) {
        let (mut after_mask, mut after_body) = after.get_mut_mask_and_body();
        after_mask.as_mut().fill(Scalar::ZERO);
        after_body.as_mut().clone_from_slice(before.get_body().as_ref());

        let decomposer = SignedDecomposer::<Scalar>::new(self.decomp_base_log, self.decomp_level_count);

        let glwe_size = self.glwe_dimension.to_glwe_size();
        let polynomial_size = self.polynomial_size;
        let decomp_level_count = self.decomp_level_count.0;

        let mut before_mask_decomp = PolynomialListOwned::new(Scalar::ZERO, polynomial_size, PolynomialCount(decomp_level_count));

        let before_masks = before.get_mask();
        for (a_i, ksk_i) in before_masks.as_polynomial_list().iter()
            .zip(self.ksks.chunks_exact(decomp_level_count))
        {
            for (j, a_ij) in a_i.iter().enumerate() {
                let mut term = decomposer.decompose(*a_ij);
                for level in 1..=self.decomp_level_count.0 {
                    *before_mask_decomp.get_mut(level - 1).as_mut().get_mut(j).unwrap() = term.next().unwrap().value();
                }
            }

            for (decomp_j, ksk_ij) in before_mask_decomp.iter().zip(ksk_i.iter().rev()) {
                let mut buf = GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, before.ciphertext_modulus());
                for (mut buf_poly, ksk_ij_poly) in buf.as_mut_polynomial_list().iter_mut()
                    .zip(ksk_ij.as_polynomial_list().iter())
                {
                    polynomial_wrapping_mul(&mut buf_poly, &decomp_j, &ksk_ij_poly);
                }

                glwe_ciphertext_sub_assign(&mut after, &buf);
            }
        }
    }

    pub fn subs(
        &self,
        after: GlweCiphertextMutView<Scalar>,
        before: GlweCiphertextView<Scalar>,
    ) {
        let mut before_power = GlweCiphertextOwned::new(Scalar::ZERO, before.glwe_size(), before.polynomial_size(), before.ciphertext_modulus());
        for (mut poly_power, poly) in before_power.as_mut_polynomial_list().iter_mut().zip(before.as_polynomial_list().iter()) {
            poly_power.as_mut().clone_from_slice(eval_x_k(poly, self.subs_k).as_ref());
        }

        self.keyswitch_ciphertext(after, before_power.as_view());
    }
}

pub fn gen_all_subs_ksk<Scalar, G>(
    decomp_base_log: DecompositionBaseLog,
    decomp_level: DecompositionLevelCount,
    glwe_secret_key: &GlweSecretKeyOwned<Scalar>,
    noise_parameters: impl DispersionParameter,
    generator: &mut EncryptionRandomGenerator<G>,
) -> HashMap<usize, GLWEKeyswitchKey<Scalar>>
where
    Scalar: UnsignedTorus + Sync + Send,
    G: ByteRandomGenerator,
{
    let glwe_dimension = glwe_secret_key.glwe_dimension();
    let polynomial_size = glwe_secret_key.polynomial_size();

    let mut hm = HashMap::new();
    for i in 1..=log2(polynomial_size.0) {
        let k = polynomial_size.0 / (1 << (i - 1)) + 1;
        let mut glwe_ksk = GLWEKeyswitchKey::allocate(decomp_base_log, decomp_level, glwe_dimension, polynomial_size, i);
        let mut before_key = glwe_secret_key.clone();

        glwe_ksk.fill_with_subs_keyswitch_key(&mut before_key, &glwe_secret_key, k, noise_parameters, generator);
        hm.insert(k, glwe_ksk);
    }

    hm
}

pub fn trace<Scalar: UnsignedTorus + Sync + Send>(
    glwe_in: GlweCiphertextView<Scalar>,
    ksk_map: &HashMap<usize, GLWEKeyswitchKey<Scalar>>,
) -> GlweCiphertextOwned<Scalar> {
    let n = glwe_in.polynomial_size().0;
    let mut buf = GlweCiphertextOwned::new(Scalar::ZERO, glwe_in.glwe_size(), glwe_in.polynomial_size(), glwe_in.ciphertext_modulus());
    let mut out: GlweCiphertext<Vec<Scalar>> = GlweCiphertext::new(Scalar::ZERO, glwe_in.glwe_size(), glwe_in.polynomial_size(), glwe_in.ciphertext_modulus());
    out.as_mut().clone_from_slice(glwe_in.as_ref());

    for i in 1..=log2(n) {
        let k = n / (1 << (i - 1)) + 1;
        let ksk = ksk_map.get(&k).unwrap();
        ksk.subs(buf.as_mut_view(), out.as_view());
        glwe_ciphertext_add_assign(&mut out, &buf);
    }

    out
}

pub fn expand_glwe<Scalar: UnsignedTorus + Sync + Send>(
    glwe_in: GlweCiphertextView<Scalar>,
    ksk_map: &HashMap<usize, GLWEKeyswitchKey<Scalar>>,
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
        debug_assert_eq!(k, ksk.subs_k);
        for b in 0..(1 << (i - 1)) {
            let mut tmp = GlweCiphertext::new(Scalar::ZERO, glwe_in.glwe_size(), glwe_in.polynomial_size(), glwe_in.ciphertext_modulus());
            glwe_clone_from(tmp.as_mut_view(), glwe_out_list.get(b));
            ksk.subs(buf.as_mut_view(), tmp.as_view());

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


#[inline]
pub const fn log2(input: usize) -> usize {
    core::mem::size_of::<usize>() * 8 - (input.leading_zeros() as usize) - 1
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k<Scalar>(poly: PolynomialView<'_, Scalar>, k: usize) -> PolynomialOwned<Scalar>
where
    Scalar: UnsignedTorus,
{
    let mut out = PolynomialOwned::new(Scalar::ZERO, poly.polynomial_size());
    eval_x_k_in_memory(&mut out, poly, k);
    out
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k_in_memory<Scalar>(out: &mut PolynomialOwned<Scalar>, poly: PolynomialView<'_, Scalar>, k: usize)
where
    Scalar: UnsignedTorus,
{
    assert_eq!(k % 2, 1);
    assert!(poly.polynomial_size().0.is_power_of_two());
    *out.as_mut().get_mut(0).unwrap() = *poly.as_ref().get(0).unwrap();
    for i in 1..poly.polynomial_size().0 {
        // i-th term becomes ik-th term, but reduced by n
        let j = i * k % poly.polynomial_size().0;
        let sign = if ((i * k) / poly.polynomial_size().0) % 2 == 0
        { Scalar::ONE } else { Scalar::MAX };
        let c = *poly.as_ref().get(i).unwrap();
        *out.as_mut().get_mut(j).unwrap() = sign.wrapping_mul(c);
    }
}
