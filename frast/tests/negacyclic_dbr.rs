use std::time::{Instant, Duration};

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::ReborrowMut;
use rand::Rng;
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::algorithms::polynomial_algorithms::*;
use tfhe::core_crypto::fft_impl::fft64::{c64, crypto::ggsw::cmux};
use frast::{MODULUS, MODULUS_BIT, FRAST_HE_PARAM1};
use frast::{utils::*, keygen::*, expand_glwe::*, ggsw_conv::*, pbs::*};

fn main() {
    // Generator and buffer setting
    let mut boxed_seeder = new_seeder();
    let seeder = boxed_seeder.as_mut();

    let mut secret_generator = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
    let mut encryption_generator = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

    let mut computation_buffers = ComputationBuffers::new();

    let modulus_bit = MODULUS_BIT;
    let modulus_sup = MODULUS;
    let delta_log = 64 - modulus_bit;
    let delta = 1u64 << delta_log;

    let frast_he_param = FRAST_HE_PARAM1;
    let param = frast_he_param.he_param;
    let ggsw_bit_decomp_base_log = frast_he_param.ggsw_bit_decomp_base_log;
    let ggsw_bit_decomp_level_count = frast_he_param.ggsw_bit_decomp_level_count;
    let subs_decomp_base_log = frast_he_param.subs_decomp_base_log;
    let subs_decomp_level_count = frast_he_param.subs_decomp_level_count;
    let ggsw_key_decomp_base_log = frast_he_param.ggsw_key_decomp_base_log;
    let ggsw_key_decomp_level_count = frast_he_param.ggsw_key_decomp_level_count;

    // Set FRAST
    let num_branch = 32;
    let num_round = 32;

    let mut rng = rand::thread_rng();
    let sbox_half = (0..modulus_sup/2).map(|_| rng.gen_range(0..modulus_sup) as u64).collect::<Vec<u64>>();
    let sbox = (0..modulus_sup).map(|i| {
        if i < modulus_sup / 2 {
            sbox_half[i]
        } else {
            (modulus_sup as u64 - sbox_half[i - modulus_sup / 2]) % modulus_sup as u64
        }
    }).collect::<Vec<u64>>();

    let mut vec_round_key = Vec::<Vec<u64>>::with_capacity(num_round);
    let mut vec_round_key_bit = Vec::<u64>::with_capacity(num_round * modulus_bit * num_branch);

    for _ in 0..num_round {
        let cur_round_key = (0..num_branch).map(|_| rng.gen_range(0..modulus_sup) as u64).collect::<Vec<u64>>();

        for rk in cur_round_key.iter() {
            for bit_idx in 0..modulus_bit {
                vec_round_key_bit.push((rk & (1 << bit_idx)) >> bit_idx);
            }
        }
        vec_round_key.push(cur_round_key);
    }

    // Set eval keys
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

    let fourier_ggsw_key = generate_fourier_ggsw_key(
        &glwe_secret_key,
        ggsw_key_decomp_base_log,
        ggsw_key_decomp_level_count,
        param.glwe_modular_std_dev,
        param.ciphertext_modulus,
        &mut encryption_generator,
    );
    let fourier_ggsw_key = fourier_ggsw_key.as_view();

    let all_ksk = gen_all_subs_ksk(
        subs_decomp_base_log,
        subs_decomp_level_count,
        &glwe_secret_key,
        param.glwe_modular_std_dev,
        &mut encryption_generator,
    );

    // Setup Phase
    let now = Instant::now();
    // vec![
    //     [GLWE(b_0 q/NB + b_1 q/NB X + ... + b_{N-1} q/NB X^{N-1}), ..., GLWE(b_0 q/NB^l + b_1 q/NB^l X + ... + b_{N-1} q/NB^l X^{N-1})],
    //     ...
    //     [GLWE(b_{(m-1)N} q/NB + b_{(m-1)N+1} q/NB X + ... + b_{mN-1} q/NB X^{N-1}), ..., GLWE(b_{(m-1)N} q/NB^l + b_{(m-1)N+1} q/NB X + ... + b_{mN-1} q/NB X^{N-1})]
    // ]
    let vec_glwe_list = encode_bits_into_glwe_ciphertext(
        &glwe_secret_key,
        vec_round_key_bit.as_ref(),
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
        param.glwe_modular_std_dev,
        &mut encryption_generator,
        param.ciphertext_modulus,
    );
    let num_glwe_list = vec_glwe_list.len();

    // vec![
    //     [GLWE(b_0 q/B), ..., GLWE(b_{N-1} q/B)],
    //     ...,
    //     [GLWE(b_0 q/B^l, ..., GLWE(b_{N-1} q/B^l))],
    //     ...,
    //     [GLWE(b_{(m-1)N} q/B), ..., GLWE(b_{mN-1} q/B)],
    //     ...,
    //     [GLWE(b_{(m-1)N} q/B^l), ..., GLWE(b_{mN-1} q/B^l)],
    // ]
    let mut vec_expanded_glwe_list = vec![GlweCiphertextList::new(
        0u64,
        param.glwe_dimension.to_glwe_size(),
        param.polynomial_size,
        GlweCiphertextCount(param.polynomial_size.0),
        param.ciphertext_modulus,
    ); num_glwe_list * ggsw_bit_decomp_level_count.0];
    vec_expanded_glwe_list.iter_mut().enumerate().for_each(|(i, expanded_glwe)| {
        let glwe_list_idx = i / ggsw_bit_decomp_level_count.0;
        let k = i % ggsw_bit_decomp_level_count.0;
        *expanded_glwe = expand_glwe(vec_glwe_list[glwe_list_idx].get(k), &all_ksk);
    });

    // vec![
    //     GLEV(b_0) = [GLWE(b_0 q/B), ..., GLWE(b_0 q/B^l)],
    //     ...,
    //     GLEV(b_{mN-1}) = [GLWE(b_{mN-1} q/B), ..., GLWE(b_{mN-1} q/B^l)],
    // ]
    let mut vec_glev = vec![GlweCiphertextList::new(
        0u64,
        param.glwe_dimension.to_glwe_size(),
        param.polynomial_size,
        GlweCiphertextCount(ggsw_bit_decomp_level_count.0),
        param.ciphertext_modulus,
    ); num_round * modulus_bit * num_branch];
    for (i, glev) in vec_glev.iter_mut().enumerate() {
        for k in 0..ggsw_bit_decomp_level_count.0 {
            let expanded_glwe_list_idx = (i / param.polynomial_size.0) * ggsw_bit_decomp_level_count.0 + k;
            let coeff_idx = i % param.polynomial_size.0;
            glwe_clone_from(
                glev.get_mut(k),
                vec_expanded_glwe_list[expanded_glwe_list_idx].get(coeff_idx),
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
            num_round * modulus_bit * num_branch
                * param.polynomial_size.to_fourier_polynomial_size().0
                * param.glwe_dimension.to_glwe_size().0
                * param.glwe_dimension.to_glwe_size().0
                * ggsw_bit_decomp_level_count.0
        ],
        num_round * modulus_bit * num_branch,
        param.glwe_dimension.to_glwe_size(),
        param.polynomial_size,
        ggsw_bit_decomp_base_log,
        ggsw_bit_decomp_level_count,
    );
    for (mut fourier_ggsw, ggsw) in fourier_ggsw_bit_list.as_mut_view().into_ggsw_iter().zip(ggsw_bit_list.iter()) {
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw);
    }
    let mut fourier_ggsw_bit_iter = fourier_ggsw_bit_list.as_view().into_ggsw_iter();

    let time_setup = now.elapsed();
    println!("Setup Time: {} s", time_setup.as_millis() as f64 / 1000f64);

    // Eval
    let mut state = (0..num_branch).map(|i| i as u64).collect::<Vec<u64>>();
    let mut he_state = (0..num_branch).map(|i| {
        allocate_and_trivially_encrypt_new_lwe_ciphertext(
            lwe_secret_key.lwe_dimension().to_lwe_size(),
            Plaintext(state[i] * delta),
            param.ciphertext_modulus,
        )
    }).collect::<Vec<LWE>>();

    let mut vec_he_round_key = Vec::<Vec<LWE>>::with_capacity(num_round);
    for round_key in vec_round_key.iter() {
        let he_round_key = (0..num_branch).map(|i| {
            allocate_and_encrypt_new_lwe_ciphertext(
                &lwe_secret_key,
                Plaintext(delta * round_key[i]),
                param.glwe_modular_std_dev,
                param.ciphertext_modulus,
                &mut encryption_generator,
            )
        }).collect::<Vec<LWE>>();
        vec_he_round_key.push(he_round_key);
    }
    let mut he_crf_sum = allocate_and_trivially_encrypt_new_lwe_ciphertext(
        lwe_secret_key.lwe_dimension().to_lwe_size(),
        Plaintext(0),
        param.ciphertext_modulus,
    );
    let mut he_prev_round_key = he_crf_sum.clone();

    let mut he_input_ks = LWE::new(0u64, ksk.output_lwe_size(), param.ciphertext_modulus);
    let mut he_output = LWE::new(0u64, fourier_bsk.output_lwe_dimension().to_lwe_size(), param.ciphertext_modulus);

    let polynomial_size = param.polynomial_size;
    let box_size = 2 * polynomial_size.0 / modulus_sup;

    let mut time_eval = Duration::ZERO;
    for (r, (round_key, he_round_key)) in vec_round_key.iter().zip(vec_he_round_key.iter()).enumerate() {
        println!("Round {}", r+1);
        // Plain
        let mut crf_sum = round_key[0].clone();
        for i in 1..num_branch {
            state[i] += sbox[(state[0] + round_key[i]) as usize % modulus_sup];
            state[i] %= modulus_sup as u64;

            crf_sum += state[i];
            crf_sum %= modulus_sup as u64;
        }

        state[0] += sbox[crf_sum as usize] % modulus_sup as u64;
        state[0] %= modulus_sup as u64;
        print!("Plain:");
        for i in 0..num_branch {
            print!(" {:>2}", state[i]);
        }
        println!();

        // HE
        let now = Instant::now();
        let mut buffers = ComputationBuffers::new();
        let fft = Fft::new(polynomial_size);
        let fft = fft.as_view();

        buffers.resize(
            2 * programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
                fourier_bsk.glwe_size(),
                polynomial_size,
                fft,
            ).unwrap().unaligned_bytes_required()
        );
        let stack = buffers.stack();

        let accumulator = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            param.ciphertext_modulus,
            fourier_bsk,
            delta,
            &sbox_half,
        );
        let (mut local_accumulator_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            polynomial_size,
            param.ciphertext_modulus,
        );

        keyswitch_lwe_ciphertext(&ksk, &he_state[0], &mut he_input_ks);
        gen_blind_rotate_local_assign(
            fourier_bsk,
            local_accumulator.as_mut_view(),
            ModulusSwitchOffset(0),
            LutCountLog(0),
            he_input_ks.as_ref(),
            fft,
            stack.rb_mut(),
        );

        for _ in 0..modulus_bit {
            let _ = fourier_ggsw_bit_iter.next();
        }

        for i in 1..num_branch {
            let mut buf = GlweCiphertext::from_container(
                local_accumulator_data.to_vec(),
                polynomial_size,
                param.ciphertext_modulus,
            );

            for b in 0..modulus_bit {
                // let fourier_ggsw_bit = cur_fourier_ggsw_bit_iter.next().unwrap();
                let fourier_ggsw_bit = fourier_ggsw_bit_iter.next().unwrap();

                let mut buf_shifted = buf.clone();
                for mut poly in buf_shifted.as_mut_polynomial_list().iter_mut() {
                    polynomial_wrapping_monic_monomial_div_assign(&mut poly, MonomialDegree((1 << b) * box_size));
                }
                cmux(buf.as_mut_view(), buf_shifted.as_mut_view(), fourier_ggsw_bit.as_view(), fft, stack.rb_mut());
            }

            let mut he_sbox_out = LWE::new(0u64, he_state[0].lwe_size(), param.ciphertext_modulus);
            extract_lwe_sample_from_glwe_ciphertext(&buf, &mut he_sbox_out, MonomialDegree(0));

            lwe_ciphertext_add_assign(&mut he_state[i], &he_sbox_out);
            lwe_ciphertext_add_assign(&mut he_crf_sum, &he_sbox_out);
        }

        lwe_ciphertext_add_assign(&mut he_crf_sum, &he_round_key[0]);
        lwe_ciphertext_sub_assign(&mut he_crf_sum, &he_prev_round_key);
        he_prev_round_key = he_round_key[0].clone();

        keyswitch_lwe_ciphertext(&ksk, &he_crf_sum, &mut he_input_ks);

        programmable_bootstrap_lwe_ciphertext(&he_input_ks, &mut he_output, &accumulator, &fourier_bsk);
        lwe_ciphertext_add_assign(&mut he_state[0], &he_output);
        time_eval += now.elapsed();

        let mut vec_val = Vec::<u64>::with_capacity(num_branch);
        let mut vec_err = Vec::<u32>::with_capacity(num_branch);
        for i in 0..num_branch {
            let (val, err) = get_val_and_error(&lwe_secret_key, &he_state[i], state[i], delta);
            vec_val.push(val);
            vec_err.push(err);
        }
        print!("HE   :");
        for i in 0..num_branch {
            print!(" {:>2}", vec_val[i]);
        }
        println!();

        print!("Err  :");
        for i in 0..num_branch {
            print!(" {:>2}", vec_err[i]);
        }
        println!();

        let (_, err) = get_val_and_abs_error(&lwe_secret_key, &he_crf_sum, crf_sum, delta);
        let log_err = (err as f64).log2();
        println!("Crf  : {log_err}\n");
    }
    println!("\nEval time: {} s", time_eval.as_millis() as f64 / 1000f64);
}