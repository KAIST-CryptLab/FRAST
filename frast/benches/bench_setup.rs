use std::collections::HashMap;

use aligned_vec::ABox;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use frast::{expand_glwe::*, gen_all_auto_keys, ggsw_conv::*, keygen::*, utils::*, AutomorphKey, FftType, Frast, FRAST_HE_PARAM1, FRAST_PARAM, MODULUS, MODULUS_BIT};
use rand::Rng;
use tfhe::core_crypto::{prelude::*, fft_impl::fft64::c64};
#[cfg(feature = "multithread")]
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator};

fn criterion_benchmark(c: &mut Criterion) {
    // Generator and buffer setting
    let mut boxed_seeder = new_seeder();
    let seeder = boxed_seeder.as_mut();

    let mut secret_generator = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
    let mut encryption_generator = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

    let mut computation_buffers = ComputationBuffers::new();

    // Set plaintext and encrypt
    let modulus_bit = MODULUS_BIT;
    let modulus_sup: usize = 1 << modulus_bit;

    let symmetric_param = FRAST_PARAM;
    let num_branches = symmetric_param.num_branches;
    let round = symmetric_param.round;

    let frast_he_param = FRAST_HE_PARAM1;
    let param = frast_he_param.he_param;
    let ggsw_bit_decomp_base_log = frast_he_param.ggsw_bit_decomp_base_log;
    let ggsw_bit_decomp_level_count = frast_he_param.ggsw_bit_decomp_level_count;
    let subs_decomp_base_log = frast_he_param.subs_decomp_base_log;
    let subs_decomp_level_count = frast_he_param.subs_decomp_level_count;
    let ggsw_key_decomp_base_log = frast_he_param.ggsw_key_decomp_base_log;
    let ggsw_key_decomp_level_count = frast_he_param.ggsw_key_decomp_level_count;

    let (
        _lwe_secret_key,
        glwe_secret_key,
        _lwe_secret_key_after_ks,
        fourier_bsk,
        _ksk,
    ) = keygen_basic(
        &param,
        &mut secret_generator,
        &mut encryption_generator,
        &mut computation_buffers,
    );
    let _fourier_bsk = fourier_bsk.as_view();

    // Set GGSW(-s)
    let fourier_ggsw_key = generate_fourier_ggsw_key(
        &glwe_secret_key,
        ggsw_key_decomp_base_log,
        ggsw_key_decomp_level_count,
        param.glwe_modular_std_dev,
        param.ciphertext_modulus,
        &mut encryption_generator,
    );
    let fourier_ggsw_key = fourier_ggsw_key.as_view();

    // Set rotation keys
    let all_ksk = gen_all_auto_keys(
        subs_decomp_base_log,
        subs_decomp_level_count,
        FftType::Split(39),
        &glwe_secret_key,
        param.glwe_modular_std_dev,
        &mut encryption_generator,
    );

    let mut rng = rand::thread_rng();
    let mut symmetric_key: Vec::<u64> = Vec::new();
    for _ in 0..(2*num_branches) {
        symmetric_key.push(rng.gen_range(0..modulus_sup) as u64);
    }

    // Get round keys
    let symmetric_key_u8: Vec<u8> = symmetric_key.iter().map(|x| *x as u8).collect();
    let frast = Frast::new(symmetric_key_u8.as_slice());
    let mut round_key = Vec::<Vec<u64>>::with_capacity(round);
    let mut round_key_bit = Vec::<u64>::with_capacity(round * 4 * (num_branches - 2) + 4);
    for r in 0..round {
        let cur_round_key: Vec<u64> = frast.get_round_key(r).iter().map(|x| *x as u64).collect();

        if r == 0 {
            for cur_round_key_elem in cur_round_key.iter().skip(1) {
                for b in 0..4 {
                    round_key_bit.push((cur_round_key_elem >> b) & 0x1);
                }
            }
        } else {
            let cur_round_key1 = cur_round_key[1];
            for cur_round_key_elem in cur_round_key.iter().skip(2) {
                let cur_round_key_diff = (cur_round_key_elem - cur_round_key1) % (MODULUS as u64);
                for b in 0..4 {
                    round_key_bit.push((cur_round_key_diff >> b) & 0x1);
                }
            }
        }

        round_key.push(cur_round_key);
    }

    // vec_glwe_list[i][k] contains GLWE(sum_j b_{iN+j} / NB^{k+1} X^j)
    let vec_glwe_list = encode_bits_into_glwe_ciphertext(
        &glwe_secret_key,
        round_key_bit.as_ref(),
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
        param.glwe_modular_std_dev,
        &mut encryption_generator,
        param.ciphertext_modulus,
    );

    c.bench_function("frast setup", |b| b.iter(|| setup(
        black_box(&vec_glwe_list),
        black_box(&all_ksk),
        black_box(&fourier_ggsw_key),
        black_box(round_key_bit.len()),
        black_box(ggsw_bit_decomp_base_log),
        black_box(ggsw_bit_decomp_level_count),
        black_box(param.polynomial_size),
        black_box(param.glwe_dimension.to_glwe_size()),
        black_box(param.ciphertext_modulus),
    )));
}

fn setup(
    vec_glwe_list: &Vec<GlweCiphertextList<Vec<u64>>>,
    all_ksk: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
    fourier_ggsw_key: &FourierGgswCiphertextList<&[c64]>,
    num_round_key_bits: usize,
    ggsw_bit_decomp_base_log: DecompositionBaseLog,
    ggsw_bit_decomp_level_count: DecompositionLevelCount,
    polynomial_size: PolynomialSize,
    glwe_size: GlweSize,
    ciphertext_modulus: CiphertextModulus<u64>,
) {
    // expand_glwe(vec_glwe_list[i][k])[j] contains GLWE(b_{iN+j} / B^{k+1})
    let num_glwe_list = vec_glwe_list.len();
    let mut vec_expanded_glwe_list = vec![GlweCiphertextListOwned::new(
        0u64,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(polynomial_size.0),
        ciphertext_modulus,
    ); num_glwe_list * ggsw_bit_decomp_level_count.0];
    #[cfg(not(feature = "multithread"))]
    vec_expanded_glwe_list.iter_mut().enumerate().for_each(|(i, expanded_glwe)| {
        let glwe_list_idx = i / ggsw_bit_decomp_level_count.0;
        let k = i % ggsw_bit_decomp_level_count.0;
        *expanded_glwe = expand_glwe(vec_glwe_list[glwe_list_idx].get(k), all_ksk);
    });
    #[cfg(feature = "multithread")]
    vec_expanded_glwe_list.par_iter_mut().enumerate().for_each(|(i, expanded_glwe)| {
        let glwe_list_idx = i / ggsw_bit_decomp_level_count.0;
        let k = i % ggsw_bit_decomp_level_count.0;
        *expanded_glwe = expand_glwe(vec_glwe_list[glwe_list_idx].get(k), &all_ksk);
    });

    let mut vec_glev = vec![GlweCiphertextList::new(
        0u64,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(ggsw_bit_decomp_level_count.0),
        ciphertext_modulus,
    ); num_round_key_bits];
    for (i, glev) in vec_glev.iter_mut().enumerate() {
        for k in 0..ggsw_bit_decomp_level_count.0 {
            let expanded_glwe_list_idx = (i / polynomial_size.0) * ggsw_bit_decomp_level_count.0 + k;
            let coeff_idx = i % polynomial_size.0;
            glwe_ciphertext_clone_from(
                &mut glev.get_mut(k),
                &vec_expanded_glwe_list[expanded_glwe_list_idx].get(coeff_idx),
            );
        }
    }

    let ggsw_bit_list = glev_to_ggsw(
        fourier_ggsw_key.as_view(),
        vec_glev.as_ref(),
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
        ciphertext_modulus,
    );

    let mut fourier_ggsw_bit_list = FourierGgswCiphertextList::new(
        vec![c64::default();
            num_round_key_bits
                * polynomial_size.to_fourier_polynomial_size().0
                * glwe_size.0
                * glwe_size.0
                * ggsw_bit_decomp_level_count.0
        ],
        num_round_key_bits,
        glwe_size,
        polynomial_size,
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
    );
    for (mut fourier_ggsw_bit, ggsw) in fourier_ggsw_bit_list.as_mut_view().into_ggsw_iter().zip(ggsw_bit_list.iter()) {
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw_bit);
    }

    let _vec_fourier_ggsw_bit = fourier_ggsw_bit_list.as_view().into_ggsw_iter().collect::<Vec<FourierGgswCiphertext<&[c64]>>>();
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(100);
    targets = criterion_benchmark
}
criterion_main!(benches);
