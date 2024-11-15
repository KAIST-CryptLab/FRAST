use aligned_vec::CACHELINE_ALIGN;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dyn_stack::ReborrowMut;
use frast::{expand_glwe::*, gen_all_auto_keys, ggsw_conv::*, keygen::*, pbs::*, utils::*, FftType, Frast, FRAST_HE_PARAM1, FRAST_PARAM, MODULUS, MODULUS_BIT, PARAM_ONLINE};
use rand::Rng;
#[cfg(feature = "multithread")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tfhe::core_crypto::{prelude::{*, polynomial_algorithms::*}, fft_impl::fft64::{c64, crypto::{ggsw::cmux, bootstrap::FourierLweBootstrapKeyView}}};

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
    let delta_log = 64 - modulus_bit;
    let delta = 1_u64 << delta_log;

    let symmetric_param = FRAST_PARAM;
    let num_branches = symmetric_param.num_branches;
    let round = symmetric_param.round;

    let frast_he_param = FRAST_HE_PARAM1;
    let param = frast_he_param.he_param;

    let param_online = PARAM_ONLINE;
    let ks_level_to_online = DecompositionLevelCount(5);
    let ks_base_log_to_online = DecompositionBaseLog(3);
    let ks_level_from_online = DecompositionLevelCount(5);
    let ks_base_log_from_online = DecompositionBaseLog(3);

    let ggsw_bit_decomp_base_log = frast_he_param.ggsw_bit_decomp_base_log;
    let ggsw_bit_decomp_level_count = frast_he_param.ggsw_bit_decomp_level_count;
    let subs_decomp_base_log = frast_he_param.subs_decomp_base_log;
    let subs_decomp_level_count = frast_he_param.subs_decomp_level_count;
    let ggsw_key_decomp_base_log = frast_he_param.ggsw_key_decomp_base_log;
    let ggsw_key_decomp_level_count = frast_he_param.ggsw_key_decomp_level_count;

    let (
        lwe_secret_key,
        glwe_secret_key,
        _lwe_secret_key_after_ks,
        fourier_bsk,
        ksk,
    ) = keygen_basic(
        &param,
        &mut secret_generator,
        &mut encryption_generator,
        &mut computation_buffers,
    );
    let fourier_bsk = fourier_bsk.as_view();

    let (
        lwe_secret_key_online,
        _glwe_secret_key_online,
        lwe_secret_key_after_ks_online,
        fourier_bsk_online,
        ksk_online,
    ) = keygen_basic(
        &param_online,
        &mut secret_generator,
        &mut encryption_generator,
        &mut computation_buffers,
    );
    let fourier_bsk_online = fourier_bsk_online.as_view();

    let ksk_to_online = allocate_and_generate_new_lwe_keyswitch_key(
        &lwe_secret_key,
        &lwe_secret_key_after_ks_online,
        ks_base_log_to_online,
        ks_level_to_online,
        param_online.lwe_modular_std_dev,
        param_online.ciphertext_modulus,
        &mut encryption_generator,
    );
    let _ksk_from_online = allocate_and_generate_new_lwe_keyswitch_key(
        &lwe_secret_key_online,
        &lwe_secret_key,
        ks_base_log_from_online,
        ks_level_from_online,
        param.glwe_modular_std_dev,
        param.ciphertext_modulus,
        &mut encryption_generator,
    );

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


    // Set FRAST
    let symmetric_key_u8: Vec<u8> = symmetric_key.iter().map(|x| *x as u8).collect();
    let mut frast = Frast::new(symmetric_key_u8.as_slice());
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

    // Set random sboxes
    let nonce = 0x0123456789abcdef as u64;
    frast.set_nonce(&nonce);
    let keystream = frast.generate_keystream_block();

    let mut all_sboxes = Vec::<u64>::with_capacity(round);
    for r in 0..round {
        let sbox = frast.get_sbox(r);
        for val in sbox.iter() {
            all_sboxes.push(*val as u64);
        }
    }

    print!("\nSymmetric key:");
    for val in symmetric_key.iter() {
        print!("{val:x}");
    }
    println!("\nNonce: {nonce:x}\n");


    // vec_glwe_list[i][k] contains RLWE(sum_j b_{iN+j} / NB^{k+1} X^j)
    let vec_glwe_list = encode_bits_into_glwe_ciphertext(
        &glwe_secret_key,
        round_key_bit.as_ref(),
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
        param.glwe_modular_std_dev,
        &mut encryption_generator,
        param.ciphertext_modulus,
    );
    let num_glwe_list = vec_glwe_list.len();

    // ======== Setup Phase ========
    // expand_glwe(vec_glwe_list[i][k])[j] contains GLWE(b_{iN+j} / B^{k+1})
    println!("Expanding glwe_list...");
    let mut vec_expanded_glwe_list = vec![GlweCiphertextListOwned::new(
        0u64,
        param.glwe_dimension.to_glwe_size(),
        param.polynomial_size,
        GlweCiphertextCount(param.polynomial_size.0),
        param.ciphertext_modulus,
    ); num_glwe_list * ggsw_bit_decomp_level_count.0];
    #[cfg(not(feature = "multithread"))]
    vec_expanded_glwe_list.iter_mut().enumerate().for_each(|(i, expanded_glwe)| {
        let glwe_list_idx = i / ggsw_bit_decomp_level_count.0;
        let k = i % ggsw_bit_decomp_level_count.0;
        *expanded_glwe = expand_glwe(vec_glwe_list[glwe_list_idx].get(k), &all_ksk);
    });
    #[cfg(feature = "multithread")]
    vec_expanded_glwe_list.par_iter_mut().enumerate().for_each(|(i, expanded_glwe)| {
        let glwe_list_idx = i / ggsw_bit_decomp_level_count.0;
        let k = i % ggsw_bit_decomp_level_count.0;
        *expanded_glwe = expand_glwe(vec_glwe_list[glwe_list_idx].get(k), &all_ksk);
    });

    let mut vec_glev = vec![GlweCiphertextList::new(
        0u64,
        param.glwe_dimension.to_glwe_size(),
        param.polynomial_size,
        GlweCiphertextCount(ggsw_bit_decomp_level_count.0),
        param.ciphertext_modulus,
    ); round_key_bit.len()];
    for (i, glev) in vec_glev.iter_mut().enumerate() {
        for k in 0..ggsw_bit_decomp_level_count.0 {
            let expanded_glwe_list_idx = (i / param.polynomial_size.0) * ggsw_bit_decomp_level_count.0 + k;
            let coeff_idx = i % param.polynomial_size.0;
            glwe_ciphertext_clone_from(
                &mut glev.get_mut(k),
                &vec_expanded_glwe_list[expanded_glwe_list_idx].get(coeff_idx),
            );
        }
    }

    let ggsw_bit_list = glev_to_ggsw(
        fourier_ggsw_key,
        vec_glev.as_ref(),
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
        param.ciphertext_modulus,
    );

    let mut fourier_ggsw_bit_list = FourierGgswCiphertextList::new(
        vec![c64::default();
            round_key_bit.len()
                * param.polynomial_size.to_fourier_polynomial_size().0
                * param.glwe_dimension.to_glwe_size().0
                * param.glwe_dimension.to_glwe_size().0
                * ggsw_bit_decomp_level_count.0
        ],
        round_key_bit.len(),
        param.glwe_dimension.to_glwe_size(),
        param.polynomial_size,
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
    );
    for (mut fourier_ggsw_bit, ggsw) in fourier_ggsw_bit_list.as_mut_view().into_ggsw_iter().zip(ggsw_bit_list.iter()) {
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw_bit);
    }

    let vec_fourier_ggsw_bit = fourier_ggsw_bit_list.as_view().into_ggsw_iter().collect::<Vec<FourierGgswCiphertext<&[c64]>>>();

    // Plain state
    let mut ic = vec![0u64; num_branches];
    let mut state = vec![0u64; num_branches];
    for i in 0..num_branches {
        ic[i] = (i % MODULUS) as u64;
        state[i] = (i % MODULUS) as u64;
    }

    // ======== Plain ========
    let mut plain_state_vec = Vec::<Vec<u64>>::with_capacity(round);
    let mut plain_linear_sum_vec = Vec::<u64>::with_capacity(round);

    for r in 0..round {
        let mut cur_state_vec = Vec::<u64>::with_capacity(num_branches);
        let offset = modulus_sup * r;

        let mut linear_sum = 0;
        let sbox = &all_sboxes[offset..offset+modulus_sup];

        let sbox_erf = if r % 5 < 4 {
            (0..MODULUS).map(|i| {
                if i < MODULUS / 2 {
                    sbox[i]
                } else {
                    (MODULUS as u64 - sbox[i - MODULUS/2]) % MODULUS as u64
                }
            }).collect::<Vec<u64>>()
        } else {
            (0..MODULUS).map(|i| sbox[i]).collect::<Vec<u64>>()
        };
        let sbox_crf = if r % 5 < 4 {
            (0..MODULUS).map(|i| {
                if i < MODULUS / 2 {
                    sbox[i + MODULUS / 2]
                } else {
                    (MODULUS as u64 - sbox[i]) % MODULUS as u64
                }
            }).collect::<Vec<u64>>()
        } else {
            (0..MODULUS).map(|i| sbox[i]).collect::<Vec<u64>>()
        };

        for i in 1..num_branches {
            let sbox_in = (state[0] + round_key[r][i]) % (MODULUS as u64);
            state[i] += sbox_erf[sbox_in as usize];
            state[i] %= MODULUS as u64;

            linear_sum += state[i];
        }
        linear_sum += round_key[r][0];
        linear_sum %= MODULUS as u64;
        plain_linear_sum_vec.push(linear_sum);

        state[0] += sbox_crf[linear_sum as usize];
        state[0] %= MODULUS as u64;

        for val in state.iter() {
            cur_state_vec.push(*val);
        }
        plain_state_vec.push(cur_state_vec);

        if r == round - 1 {
            for (key1, key2) in keystream.iter().zip(state.iter()) {
                assert_eq!(*key1, *key2 as u8);
            }
        }
    }

    // HE state
    let mut he_state = Vec::<LWE>::with_capacity(num_branches);
    for i in 0..num_branches {
        let triv_ct = allocate_and_trivially_encrypt_new_lwe_ciphertext(
            lwe_secret_key.lwe_dimension().to_lwe_size(),
            Plaintext(ic[i] * delta),
            param.ciphertext_modulus,
        );
        he_state.push(triv_ct);
    }

    let mut init_linear_sum = 0;
    for i in 1..num_branches {
        init_linear_sum += ic[i] as u64;
    }
    init_linear_sum %= MODULUS as u64;

    let mut he_round_key = Vec::<Vec<LWE>>::with_capacity(round);
    for cur_round_key in round_key.iter() {
        let mut he_cur_round_key = Vec::<LWE>::with_capacity(num_branches);
        for cur_round_key_elem in cur_round_key.iter() {
            he_cur_round_key.push(allocate_and_encrypt_new_lwe_ciphertext(
                &lwe_secret_key,
                Plaintext(*cur_round_key_elem * delta),
                param.glwe_modular_std_dev,
                param.ciphertext_modulus,
                &mut encryption_generator,
            ));
        }
        he_round_key.push(he_cur_round_key);
    }

    c.bench_function("frast eval", |b| b.iter(|| eval_frast(
        black_box(&mut he_state),
        black_box(&he_round_key),
        black_box(&vec_fourier_ggsw_bit),
        black_box(init_linear_sum),
        black_box(&all_sboxes),
        black_box(round),
        black_box(num_branches),
        black_box(delta),
        black_box(delta_log),
        black_box(&ksk),
        black_box(fourier_bsk),
        black_box(&ksk_online),
        black_box(fourier_bsk_online),
        black_box(&ksk_to_online),
    )));
}

fn eval_frast(
    mut he_state: &mut Vec<LWE>,
    he_round_key: &Vec<Vec<LWE>>,
    vec_fourier_ggsw_bit: &Vec<FourierGgswCiphertext<&[c64]>>,
    init_linear_sum: u64,
    all_sboxes: &Vec<u64>,
    round: usize,
    num_branches: usize,
    delta: u64,
    delta_log: usize,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    ksk_online: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk_online: FourierLweBootstrapKeyView,
    ksk_to_online: &LweKeyswitchKeyOwned<u64>,
) {
    let mut linear_sum = allocate_and_trivially_encrypt_new_lwe_ciphertext(he_state[0].lwe_size(), Plaintext(init_linear_sum * delta), he_state[0].ciphertext_modulus());

    for r in 0..round {
        let sbox = &all_sboxes[(r * MODULUS)..((r+1) * MODULUS)];

        let sbox_erf = if r % 5 < 4 {
            (0..MODULUS).map(|i| {
                if i < MODULUS / 2 {
                    sbox[i]
                } else {
                    (MODULUS as u64 - sbox[i - MODULUS/2]) % MODULUS as u64
                }
            }).collect::<Vec<u64>>()
        } else {
            (0..MODULUS).map(|i| sbox[i]).collect::<Vec<u64>>()
        };
        let sbox_crf = if r % 5 < 4 {
            (0..MODULUS).map(|i| {
                if i < MODULUS / 2 {
                    sbox[i + MODULUS / 2]
                } else {
                    (MODULUS as u64 - sbox[i]) % MODULUS as u64
                }
            }).collect::<Vec<u64>>()
        } else {
            (0..MODULUS).map(|i| sbox[i]).collect::<Vec<u64>>()
        };

        if r == 0 {
            eval_first_round_expanding_part_negacyclic(
                &mut he_state,
                &mut linear_sum,
                &sbox_erf,
                num_branches,
                delta,
                fourier_bsk,
                &vec_fourier_ggsw_bit[0..4*(num_branches-1)],
            );
        } else {
            let offset = 4 + 4 * r * (num_branches-2);
            if r % 5 < 4 {
                eval_round_expanding_part_negacyclic(
                    &mut he_state,
                    &mut linear_sum,
                    &he_round_key[r][1],
                    &sbox_erf,
                    num_branches,
                    delta,
                    &ksk,
                    fourier_bsk,
                    &vec_fourier_ggsw_bit[offset..(offset + 4*(num_branches-2))],
                );
            } else {
                #[cfg(not(feature = "multithread"))]
                eval_round_expanding_part(
                    &mut he_state,
                    &mut linear_sum,
                    &he_round_key[r][1],
                    &sbox_erf,
                    num_branches,
                    delta,
                    &ksk,
                    fourier_bsk,
                    &vec_fourier_ggsw_bit[offset..(offset + 4*(num_branches-2))],
                );
                #[cfg(feature = "multithread")]
                par_eval_round_expanding_part(
                    &mut he_state,
                    &mut linear_sum,
                    &he_round_key[r][1],
                    &sbox_erf,
                    num_branches,
                    delta,
                    &ksk,
                    fourier_bsk,
                    &vec_fourier_ggsw_bit[offset..(offset + 4*(num_branches-2))],
                );
            }
        }

        lwe_ciphertext_add_assign(&mut linear_sum, &he_round_key[r][0]);

        if r % 5 < 4 {
            eval_round_contracting_part_negacyclic(&mut he_state[0], &mut linear_sum, &sbox_crf, delta, &ksk, fourier_bsk, true, true);
        } else {
            if r < round - 1 {
                eval_round_contracting_part(&mut he_state[0], &mut linear_sum, &sbox_crf, delta_log, &ksk, fourier_bsk, true, true);
            } else {
                eval_round_contracting_part(&mut he_state[0], &mut linear_sum, &sbox_crf, delta_log, &ksk, fourier_bsk, false, true);
            }
        }

        if r < round - 1 {
            lwe_ciphertext_sub_assign(&mut linear_sum, &he_round_key[r][0]);
        }
    }

    // let he_keystream_bits = bit_decompose_keystream_to_online(&he_state, fourier_bsk_online, &ksk_to_online, &ksk_online, num_branches, MODULUS);
    bit_decompose_keystream_to_online(&he_state, fourier_bsk_online, &ksk_to_online, &ksk_online, num_branches, MODULUS);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[allow(unused)]
fn eval_first_round_expanding_part(
    he_state: &mut Vec<LWE>,
    linear_sum: &mut LWE,
    sbox: &[u64],
    num_branches: usize,
    delta: u64,
    fourier_bsk: FourierLweBootstrapKeyView,
    fourier_ggsw_bits_list: &[FourierGgswCiphertext<&[c64]>],
) {
    let polynomial_size = fourier_bsk.polynomial_size();
    let ciphertext_modulus = he_state[0].ciphertext_modulus();
    let mut fourier_ggsw_bit_iter = fourier_ggsw_bits_list.into_iter();

    let sbox_odd = (0..MODULUS/2).map(|i| (sbox[i] - sbox[i + MODULUS/2]) % ((2 * MODULUS) as u64)).collect::<Vec<u64>>();
    let sbox_even = (0..MODULUS).map(|i| (sbox[i] + sbox[(i + MODULUS/2) % MODULUS])).collect::<Vec<u64>>();

    let box_size_odd = 2 * polynomial_size.0 / MODULUS;
    let box_size_even = 2 * polynomial_size.0 / (2 * MODULUS);

    // TODO: fix buffer size
    let mut buffers = ComputationBuffers::new();
    let fft = Fft::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();
    buffers.resize(
        2 *
        programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
            fourier_bsk.glwe_size(),
            polynomial_size,
            fft,
        ).unwrap().unaligned_bytes_required()
    );
    let stack = buffers.stack();

    let accumulator_odd = generate_negacyclic_accumulator_from_two_sbox_half(
        MODULUS,
        ciphertext_modulus,
        fourier_bsk,
        delta / 2,
        &sbox_odd,
        1 << (u64::BITS - 2),
        &vec![3u64; MODULUS / 2],
    );
    let (local_accumulator_odd_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_odd.as_ref().iter().copied());

    let accumulator_even = generate_negacyclic_accumulator_from_sbox_half(2 * MODULUS, ciphertext_modulus, fourier_bsk, delta / 2, &sbox_even);
    let stack = stack.rb_mut();
    let (local_accumulator_even_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_even.as_ref().iter().copied());

    for i in 1..num_branches {
        let mut buf_odd = GlweCiphertext::from_container(local_accumulator_odd_data.to_vec(), polynomial_size, ciphertext_modulus);
        let mut buf_even = GlweCiphertext::from_container(local_accumulator_even_data.to_vec(), polynomial_size, ciphertext_modulus);

        for b in 0..4 {
            let fourier_ggsw_bit = fourier_ggsw_bit_iter.next().unwrap();

            let mut buf_odd_shifted = buf_odd.clone();
            for mut poly in buf_odd_shifted.as_mut_polynomial_list().iter_mut() {
                polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size_odd));
            }
            cmux(buf_odd.as_mut_view(), buf_odd_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());

            if b < 3 {
                let mut buf_even_shifted = buf_even.clone();
                for mut poly in buf_even_shifted.as_mut_polynomial_list().iter_mut() {
                    polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size_even));
                }
                cmux(buf_even.as_mut_view(), buf_even_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());
            }
        }

        let mut sbox_out = LweCiphertext::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        let mut sbox_out_odd = LweCiphertext::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        let mut sbox_out_even = LweCiphertext::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        extract_lwe_sample_from_glwe_ciphertext(&buf_odd, &mut sbox_out_odd, MonomialDegree(0));
        extract_lwe_sample_from_glwe_ciphertext(&buf_even, &mut sbox_out_even, MonomialDegree(0));
        lwe_ciphertext_add(&mut sbox_out, &sbox_out_odd, &sbox_out_even);

        lwe_ciphertext_add_assign(&mut he_state[i], &sbox_out);
        lwe_ciphertext_add_assign(linear_sum, &sbox_out);
    }
}

#[allow(unused)]
fn eval_first_round_expanding_part_negacyclic(
    he_state: &mut Vec<LWE>,
    linear_sum: &mut LWE,
    sbox: &[u64],
    num_branches: usize,
    delta: u64,
    fourier_bsk: FourierLweBootstrapKeyView,
    fourier_ggsw_bits_list: &[FourierGgswCiphertext<&[c64]>],
) {
    let polynomial_size = fourier_bsk.polynomial_size();
    let ciphertext_modulus = he_state[0].ciphertext_modulus();
    let box_size = 2 * polynomial_size.0 / MODULUS;

    let mut fourier_ggsw_bit_iter = fourier_ggsw_bits_list.into_iter();

    let mut buffers = ComputationBuffers::new();
    let fft = Fft::new(polynomial_size);
    let fft = fft.as_view();

    // TODO: fix buffer size
    buffers.resize(
        2 * programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
            fourier_bsk.glwe_size(),
            polynomial_size,
            fft,
        ).unwrap().unaligned_bytes_required()
    );
    let stack = buffers.stack();

    let accumulator = generate_negacyclic_accumulator_from_sbox_half(
        MODULUS,
        ciphertext_modulus,
        fourier_bsk,
        delta,
        sbox,
    );
    let (local_accumulator_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());

    for i in 1..num_branches {
        let mut buf = GlweCiphertext::from_container(local_accumulator_data.to_vec(), polynomial_size, ciphertext_modulus);

        for b in 0..4 {
            let fourier_ggsw_bit = fourier_ggsw_bit_iter.next().unwrap();

            let mut buf_shifted = buf.clone();
            for mut poly in buf_shifted.as_mut_polynomial_list().iter_mut() {
                polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size));
            }
            cmux(buf.as_mut_view(), buf_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());
        }

        let mut sbox_out = LWE::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        extract_lwe_sample_from_glwe_ciphertext(&buf, &mut sbox_out, MonomialDegree(0));

        lwe_ciphertext_add_assign(&mut he_state[i], &sbox_out);
        lwe_ciphertext_add_assign(linear_sum, &sbox_out);
    }
}

#[cfg(not(feature = "multithread"))]
fn eval_round_expanding_part(
    he_state: &mut Vec<LWE>,
    linear_sum: &mut LWE,
    second_round_key: &LWE,
    sbox: &[u64],
    num_branches: usize,
    delta: u64,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    fourier_bits_list: &[FourierGgswCiphertext<&[c64]>],
) {
    let polynomial_size = fourier_bsk.polynomial_size();
    let ciphertext_modulus = he_state[0].ciphertext_modulus();

    let mut fourier_ggsw_bit_iter = fourier_bits_list.into_iter();

    let sbox_odd = (0..MODULUS/2).map(|i| (sbox[i] - sbox[i + MODULUS/2]) % ((2 * MODULUS) as u64)).collect::<Vec<u64>>();
    let sbox_even = (0..MODULUS).map(|i| (sbox[i] + sbox[(i + MODULUS/2) % MODULUS])).collect::<Vec<u64>>();

    let box_size_odd = 2 * polynomial_size.0 / MODULUS;
    let box_size_even = 2 * polynomial_size.0 / (2 * MODULUS);

    let mut buffers = ComputationBuffers::new();
    let fft = Fft::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();

    // TODO: fix buffer size
    buffers.resize(
        2 *
        programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
            fourier_bsk.glwe_size(),
            polynomial_size,
            fft,
        ).unwrap().unaligned_bytes_required()
    );
    let stack = buffers.stack();

    let accumulator_odd = generate_negacyclic_accumulator_from_two_sbox_half(
        MODULUS,
        ciphertext_modulus,
        fourier_bsk,
        delta / 2,
        &sbox_odd,
        1 << (u64::BITS - 2),
        &vec![3u64; MODULUS / 2],
    );
    let (mut local_accumulator_odd_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_odd.as_ref().iter().copied());
    let mut local_accumulator_odd = GlweCiphertextMutView::from_container(
        &mut *local_accumulator_odd_data,
        polynomial_size,
        ciphertext_modulus,
    );

    let accumulator_even = generate_negacyclic_accumulator_from_sbox_half(2 * MODULUS, ciphertext_modulus, fourier_bsk, delta / 2, &sbox_even);
    let stack = stack.rb_mut();
    let (mut local_accumulator_even_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_even.as_ref().iter().copied());
    let mut local_accumulator_even = GlweCiphertextMutView::from_container(
        &mut *local_accumulator_even_data,
        polynomial_size,
        ciphertext_modulus,
    );

    let mut ct = he_state[0].clone();
    lwe_ciphertext_add_assign(&mut ct, &second_round_key);
    let mut ct_ks = LWE::new(0u64, ksk.output_lwe_size(), ciphertext_modulus);
    keyswitch_lwe_ciphertext(&ksk, &ct, &mut ct_ks);

    gen_blind_rotate_local_assign(fourier_bsk, local_accumulator_odd.as_mut_view(), ModulusSwitchOffset(0), LutCountLog(1), ct_ks.as_ref(), fft, stack.rb_mut());

    let mut ct_msb = LWE::new(0u64, ct.lwe_size(), ciphertext_modulus);
    extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_odd, &mut ct_msb, MonomialDegree(1));
    lwe_ciphertext_plaintext_add_assign(&mut ct_msb, Plaintext(1 << (u64::BITS - 2)));

    let mut ct_lsbs = ct.clone();
    lwe_ciphertext_sub_assign(&mut ct_lsbs, &ct_msb);
    let mut ct_lsbs_ks = LWE::new(0u64, ksk.output_lwe_size(), ciphertext_modulus);
    keyswitch_lwe_ciphertext(&ksk, &ct_lsbs, &mut ct_lsbs_ks);
    set_scale_by_pbs(&ct_lsbs_ks, &mut ct_lsbs, MODULUS, delta / 2, fourier_bsk);
    keyswitch_lwe_ciphertext(&ksk, &ct_lsbs, &mut ct_lsbs_ks);

    fourier_bsk.blind_rotate_assign(local_accumulator_even.as_mut_view(), ct_lsbs_ks.as_ref(), fft, stack.rb_mut());

    for i in 1..num_branches {
        let mut buf_odd = GlweCiphertext::from_container(local_accumulator_odd_data.to_vec(), polynomial_size, ciphertext_modulus);
        let mut buf_even = GlweCiphertext::from_container(local_accumulator_even_data.to_vec(), polynomial_size, ciphertext_modulus);

        if i > 1 {
            for b in 0..4 {
                let fourier_ggsw_bit = fourier_ggsw_bit_iter.next().unwrap();

                let mut buf_odd_shifted = buf_odd.clone();
                for mut poly in buf_odd_shifted.as_mut_polynomial_list().iter_mut() {
                    polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size_odd));
                }
                cmux(buf_odd.as_mut_view(), buf_odd_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());

                if b < 3 {
                    let mut buf_even_shifted = buf_even.clone();
                    for mut poly in buf_even_shifted.as_mut_polynomial_list().iter_mut() {
                        polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size_even));
                    }
                    cmux(buf_even.as_mut_view(), buf_even_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());
                }
            }
        }

        let mut sbox_out = LWE::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        let mut sbox_out_odd = LWE::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        let mut sbox_out_even = LWE::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        extract_lwe_sample_from_glwe_ciphertext(&buf_odd, &mut sbox_out_odd, MonomialDegree(0));
        extract_lwe_sample_from_glwe_ciphertext(&buf_even, &mut sbox_out_even, MonomialDegree(0));
        lwe_ciphertext_add(&mut sbox_out, &sbox_out_odd, &sbox_out_even);

        lwe_ciphertext_add_assign(&mut he_state[i], &sbox_out);
        lwe_ciphertext_add_assign(linear_sum, &sbox_out);
    }
}

#[cfg(feature = "multithread")]
fn par_eval_round_expanding_part(
    he_state: &mut Vec<LWE>,
    linear_sum: &mut LWE,
    second_round_key: &LWE,
    sbox: &[u64],
    num_branches: usize,
    delta: u64,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    fourier_ggsw_bits_list: &[FourierGgswCiphertext<&[c64]>],
) {
    use rayon::iter::IntoParallelRefIterator;

    let polynomial_size = fourier_bsk.polynomial_size();
    let ciphertext_modulus = he_state[0].ciphertext_modulus();
    let sbox = (0..MODULUS).map(|i| sbox[i] as usize).collect::<Vec<usize>>();

    let mut lwe_in = he_state[0].clone();
    lwe_ciphertext_add_assign(&mut lwe_in, &second_round_key);
    let mut lwe_in_ks = LWE::new(0u64, ksk.output_lwe_size(), ciphertext_modulus);
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    let mut vec_sbox_i_out_list = vec![LweCiphertextList::new(0u64, he_state[0].lwe_size(), LweCiphertextCount(num_branches-1), ciphertext_modulus); MODULUS_BIT];
    let vec_sbox_decomp = decompose_sbox(&sbox, MODULUS, MODULUS_BIT);
    vec_sbox_decomp[0..MODULUS_BIT].par_iter().enumerate()
        .zip(vec_sbox_i_out_list.par_iter_mut())
        .for_each(|((i, sbox_i), sbox_i_out_list)| {
            let box_size = 2 * polynomial_size.0 / (MODULUS >> i);
            let mut buffers = ComputationBuffers::new();
            let fft = Fft::new(fourier_bsk.polynomial_size());
            let fft = fft.as_view();

            // TODO: fix buffer size
            buffers.resize(
                programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
                    fourier_bsk.glwe_size(),
                    polynomial_size,
                    fft,
                ).unwrap().unaligned_bytes_required()
            );
            let stack = buffers.stack();

            let sbox_i_half = (0..(MODULUS >> (i+1))).map(|x| sbox_i[x] as u64).collect::<Vec<u64>>();
            let accumulator_sbox_i = generate_negacyclic_accumulator_from_sbox_half(
                MODULUS >> i,
                ciphertext_modulus,
                fourier_bsk,
                delta >> (i+1),
                &sbox_i_half,
            );

            let (mut local_accumulator_sbox_i_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_sbox_i.as_ref().iter().copied());
            let mut local_accumulator_sbox_i = GlweCiphertextMutView::from_container(
                &mut *local_accumulator_sbox_i_data,
                polynomial_size,
                ciphertext_modulus,
            );

            gen_blind_rotate_local_assign(fourier_bsk, local_accumulator_sbox_i.as_mut_view(), ModulusSwitchOffset(i), LutCountLog(0), lwe_in_ks.as_ref(), fft, stack.rb_mut());

            for (j, mut sbox_i_out) in sbox_i_out_list.iter_mut().enumerate() {
                let branch_idx = j + 1;
                let mut buf = GlweCiphertext::from_container(local_accumulator_sbox_i_data.to_vec(), polynomial_size, ciphertext_modulus);

                if branch_idx > 1 {
                    for b in 0..(MODULUS_BIT - i) {
                        let fourier_ggsw_bit = fourier_ggsw_bits_list[4 * (branch_idx - 2) + b];

                        let mut buf_shifted = buf.clone();
                        for mut poly in buf_shifted.as_mut_polynomial_list().iter_mut() {
                            polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size));
                        }
                        cmux(buf.as_mut_view(), buf_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());
                    }
                }
                extract_lwe_sample_from_glwe_ciphertext(&buf, &mut sbox_i_out, MonomialDegree(0));
            }
        });

    for (_, sbox_i_out_list) in vec_sbox_i_out_list.iter().enumerate() {
        for (j, sbox_i_out) in sbox_i_out_list.iter().enumerate() {
            let branch_idx = j + 1;
            lwe_ciphertext_add_assign(&mut he_state[branch_idx], &sbox_i_out);
            lwe_ciphertext_add_assign(linear_sum, &sbox_i_out);
        }
    }
    for branch_idx in 1..num_branches {
        let pt = Plaintext(vec_sbox_decomp[MODULUS_BIT][0] as u64 * (delta >> MODULUS_BIT));
        lwe_ciphertext_plaintext_add_assign(&mut he_state[branch_idx], pt);
        lwe_ciphertext_plaintext_add_assign(linear_sum, pt);
    }
}

fn eval_round_contracting_part(
    he_first_state: &mut LWE,
    linear_sum: &mut LWE,
    sbox: &[u64],
    delta_log: usize,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    do_refresh: bool,
    is_pbs_many: bool,
) {
    if do_refresh {
        let (sbox_out, refreshed) = sbox_call_and_refresh(&linear_sum, &sbox, delta_log, ksk, fourier_bsk, is_pbs_many);
        lwe_ciphertext_add_assign(he_first_state, &sbox_out);
        *linear_sum = refreshed.clone();
    } else {
        let sbox_out = sbox_call(&linear_sum, &sbox, delta_log, ksk, fourier_bsk, is_pbs_many);
        lwe_ciphertext_add_assign(he_first_state, &sbox_out);
    }
}

fn eval_round_contracting_part_negacyclic(
    he_first_state: &mut LWE,
    linear_sum: &mut LWE,
    sbox: &[u64],
    delta: u64,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    do_refresh: bool,
    is_pbs_many: bool,
) {
    let ciphertext_modulus = he_first_state.ciphertext_modulus();
    if do_refresh {
        let he_in = linear_sum.clone();
        let mut sbox_out = LWE::new(0u64, fourier_bsk.output_lwe_dimension().to_lwe_size(), ciphertext_modulus);

        gen_pbs_with_bts(
            sbox_out.as_mut_view(),
            linear_sum.as_mut_view(),
            he_in.as_view(),
            ksk.as_view(),
            fourier_bsk,
            delta,
            MODULUS,
            sbox,
            is_pbs_many,
        );

        lwe_ciphertext_add_assign(he_first_state, &sbox_out);
    } else {
        let mut linear_sum_ks = LWE::new(0u64, ksk.output_lwe_size(), ciphertext_modulus);
        keyswitch_lwe_ciphertext(ksk, linear_sum, &mut linear_sum_ks);

        let accumulator = generate_negacyclic_accumulator_from_sbox_half(
            MODULUS,
            ciphertext_modulus,
            fourier_bsk,
            delta,
            sbox,
        );

        let mut sbox_out = LWE::new(0u64, fourier_bsk.output_lwe_dimension().to_lwe_size(), ciphertext_modulus);
        programmable_bootstrap_lwe_ciphertext(&linear_sum_ks, &mut sbox_out, &accumulator, &fourier_bsk);

        lwe_ciphertext_add_assign(he_first_state, &sbox_out);
    }
}

fn eval_round_expanding_part_negacyclic(
    he_state: &mut Vec<LWE>,
    linear_sum: &mut LWE,
    second_round_key: &LWE,
    sbox: &[u64],
    num_branches: usize,
    delta: u64,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    fourier_bits_list: &[FourierGgswCiphertext<&[c64]>],
) {
    let polynomial_size = fourier_bsk.polynomial_size();
    let ciphertext_modulus = he_state[0].ciphertext_modulus();
    let box_size = 2 * polynomial_size.0 / MODULUS;

    let mut fourier_ggsw_bit_iter = fourier_bits_list.into_iter();

    let mut buffers = ComputationBuffers::new();
    let fft = Fft::new(polynomial_size);
    let fft = fft.as_view();

    // TODO: fix buffer size
    buffers.resize(
        2 * programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
            fourier_bsk.glwe_size(),
            polynomial_size,
            fft,
        ).unwrap().unaligned_bytes_required()
    );
    let stack = buffers.stack();

    let accumulator = generate_negacyclic_accumulator_from_sbox_half(
        MODULUS,
        ciphertext_modulus,
        fourier_bsk,
        delta,
        sbox,
    );
    let (mut local_accumulator_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
    let mut local_accumulator = GlweCiphertextMutView::from_container(
        &mut *local_accumulator_data,
        polynomial_size,
        ciphertext_modulus,
    );

    let mut he_input = he_state[0].clone();
    lwe_ciphertext_add_assign(&mut he_input, &second_round_key);

    let mut he_input_ks = LWE::new(0u64, ksk.output_lwe_size(), ciphertext_modulus);
    keyswitch_lwe_ciphertext(ksk, &he_input, &mut he_input_ks);

    gen_blind_rotate_local_assign(
        fourier_bsk,
        local_accumulator.as_mut_view(),
        ModulusSwitchOffset(0),
        LutCountLog(0),
        he_input_ks.as_ref(),
        fft,
        stack.rb_mut(),
    );

    for i in 1..num_branches {
        let mut buf = GlweCiphertext::from_container(local_accumulator_data.to_vec(), polynomial_size, ciphertext_modulus);

        if i > 1 {
            for b in 0..4 {
                let fourier_ggsw_bit = fourier_ggsw_bit_iter.next().unwrap();

                let mut buf_shifted = buf.clone();
                for mut poly in buf_shifted.as_mut_polynomial_list().iter_mut() {
                    polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size));
                }
                cmux(buf.as_mut_view(), buf_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());
            }
        }

        let mut sbox_out = LWE::new(0u64, he_state[0].lwe_size(), ciphertext_modulus);
        extract_lwe_sample_from_glwe_ciphertext(&buf, &mut sbox_out, MonomialDegree(0));

        lwe_ciphertext_add_assign(&mut he_state[i], &sbox_out);
        lwe_ciphertext_add_assign(linear_sum, &sbox_out);
    }
}

#[allow(unused)]
fn bit_decompose_keystream(
    he_state: &Vec<LWE>,
    fourier_bsk: FourierLweBootstrapKeyView,
    ksk: &LweKeyswitchKeyOwned<u64>,
    num_branches: usize,
    modulus_sup: usize,
) -> LweCiphertextListOwned<u64> {
    let mut he_keystream_bits = LweCiphertextList::new(0u64, fourier_bsk.output_lwe_dimension().to_lwe_size(), LweCiphertextCount(MODULUS_BIT * num_branches), he_state[0].ciphertext_modulus());

    for (mut he_bits_list, he_in) in he_keystream_bits.chunks_exact_mut(MODULUS_BIT).zip(he_state.iter()) {
        bit_decomposition_into_msb_encoding(
            he_in,
            &mut he_bits_list,
            modulus_sup,
            fourier_bsk,
            ksk,
        );
    }

    he_keystream_bits
}

#[allow(unused)]
fn bit_decompose_keystream_to_online(
    he_state: &Vec<LWE>,
    fourier_bsk_online: FourierLweBootstrapKeyView,
    ksk_to_online: &LweKeyswitchKeyOwned<u64>,
    ksk_online: &LweKeyswitchKeyOwned<u64>,
    num_branches: usize,
    modulus_sup: usize,
) -> LweCiphertextListOwned<u64> {
    let mut he_keystream_bits = LweCiphertextList::new(0u64, ksk_online.output_lwe_size(), LweCiphertextCount(MODULUS_BIT * num_branches), he_state[0].ciphertext_modulus());

    for (mut he_bits_list, he_in) in he_keystream_bits.chunks_exact_mut(MODULUS_BIT).zip(he_state.iter()) {
        let mut buf_list = LweCiphertextList::new(0u64, fourier_bsk_online.output_lwe_dimension().to_lwe_size(), LweCiphertextCount(MODULUS_BIT), he_in.ciphertext_modulus());
        bit_decomposition_into_msb_encoding(
            he_in,
            &mut buf_list,
            modulus_sup,
            fourier_bsk_online,
            ksk_to_online,
        );

        for (mut he_bit, buf) in he_bits_list.iter_mut().zip(buf_list.iter()) {
            keyswitch_lwe_ciphertext(&ksk_online, &buf, &mut he_bit);
        }
    }

    he_keystream_bits
}

#[allow(unused)]
#[cfg(feature = "multithread")]
fn par_bit_decompose_keystream(
    he_state: &Vec<LWE>,
    fourier_bsk: FourierLweBootstrapKeyView,
    ksk: &LweKeyswitchKeyOwned<u64>,
    num_branches: usize,
    modulus_sup: usize,
) -> LweCiphertextListOwned<u64> {
    use rayon::iter::IntoParallelRefIterator;

    let mut he_keystream_bits = LweCiphertextList::new(0u64, he_state[0].lwe_size(), LweCiphertextCount(MODULUS_BIT * num_branches), he_state[0].ciphertext_modulus());

    he_keystream_bits.par_chunks_exact_mut(MODULUS_BIT)
        .zip(he_state.par_iter())
        .for_each(|(mut he_bits_list, he_in)| {
            bit_decomposition_into_msb_encoding(
                he_in,
                &mut he_bits_list,
                modulus_sup,
                fourier_bsk,
                ksk,
            );
        });

    he_keystream_bits
}

fn sbox_call(
    lwe_in: &LWE,
    sbox: &[u64],
    delta_log: usize,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    is_pbs_many: bool,
) -> LWE {
    let ciphertext_modulus = lwe_in.ciphertext_modulus();

    let mut lwe_out = LWE::new(0u64, lwe_in.lwe_size(), ciphertext_modulus);

    let sbox = (0..MODULUS).map(|i| sbox[i] as usize).collect::<Vec<usize>>();
    wop_pbs(
        lwe_out.as_mut_view(),
        lwe_in.as_view(),
        ksk.as_view(),
        fourier_bsk,
        DeltaLog(delta_log),
        MODULUS,
        &sbox,
        is_pbs_many,
    );

    lwe_out
}

fn sbox_call_and_refresh(
    lwe_in: &LWE,
    sbox: &[u64],
    delta_log: usize,
    ksk: &LweKeyswitchKeyOwned<u64>,
    fourier_bsk: FourierLweBootstrapKeyView,
    is_pbs_many: bool,
) -> (LWE, LWE) {
    let ciphertext_modulus = lwe_in.ciphertext_modulus();

    let mut lwe_out = LWE::new(0u64, lwe_in.lwe_size(), ciphertext_modulus);
    let mut lwe_refreshed = LWE::new(0u64, lwe_in.lwe_size(), ciphertext_modulus);

    let sbox = (0..MODULUS).map(|i| sbox[i] as usize).collect::<Vec<usize>>();
    wop_pbs_with_bts(
        lwe_out.as_mut_view(),
        lwe_refreshed.as_mut_view(),
        lwe_in.as_view(),
        ksk.as_view(),
        fourier_bsk,
        DeltaLog(delta_log),
        MODULUS,
        &sbox,
        is_pbs_many,
    );

    (lwe_out, lwe_refreshed)
}
