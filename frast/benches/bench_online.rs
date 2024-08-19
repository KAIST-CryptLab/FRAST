use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use frast::{keygen::*, pbs::*, MODULUS, FRAST_HE_PARAM1};
use tfhe::shortint::{prelude::*, CiphertextModulus};
use tfhe::core_crypto::{
    prelude::*,
    fft_impl::fft64::crypto::bootstrap::FourierLweBootstrapKeyView,
};

fn criterion_benchmark(c: &mut Criterion) {
    let param_elisabeth: ClassicPBSParameters = ClassicPBSParameters {
        lwe_dimension: LweDimension(784),
        glwe_dimension: GlweDimension(3),
        polynomial_size: PolynomialSize(512),
        lwe_modular_std_dev: StandardDev(0.00000240455),
        glwe_modular_std_dev: StandardDev(0.0000000000025729745),
        pbs_base_log: DecompositionBaseLog(19),
        pbs_level: DecompositionLevelCount(1),
        ks_level: DecompositionLevelCount(2),
        ks_base_log: DecompositionBaseLog(6),
        message_modulus: MessageModulus(4),
        carry_modulus: CarryModulus(2),
        ciphertext_modulus: CiphertextModulus::new_native(),
        encryption_key_choice: EncryptionKeyChoice::Big,
    };
    let param_kreyvium: ClassicPBSParameters = ClassicPBSParameters {
        lwe_dimension: LweDimension(684),
        glwe_dimension: GlweDimension(3),
        polynomial_size: PolynomialSize(512),
        lwe_modular_std_dev: StandardDev(0.0000204378),
        glwe_modular_std_dev: StandardDev(0.00000000000345253),
        pbs_base_log: DecompositionBaseLog(18),
        pbs_level: DecompositionLevelCount(1),
        ks_level: DecompositionLevelCount(3),
        ks_base_log: DecompositionBaseLog(4),
        message_modulus: MessageModulus(2),
        carry_modulus: CarryModulus(2),
        ciphertext_modulus: CiphertextModulus::new_native(),
        encryption_key_choice: EncryptionKeyChoice::Big,
    };

    let param_list = [param_kreyvium, param_elisabeth];
    let bench_id_list = ["kreyvium", "elisabeth"];
    let mut group = c.benchmark_group("online");

    for (id, param_online) in bench_id_list.iter().zip(param_list.iter()) {
        // Generator and buffer setting
        let mut boxed_seeder = new_seeder();
        let seeder = boxed_seeder.as_mut();

        let mut secret_generator = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
        let mut encryption_generator = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

        let mut computation_buffers = ComputationBuffers::new();

        // Set plaintext and encrypt
        let frast_he_param = FRAST_HE_PARAM1;
        let param = frast_he_param.he_param;

        let ks_level_to_online = DecompositionLevelCount(5);
        let ks_base_log_to_online = DecompositionBaseLog(3);
        let ks_level_from_online = DecompositionLevelCount(5);
        let ks_base_log_from_online = DecompositionBaseLog(3);

        let (
            lwe_secret_key,
            _,
            _,
            _,
            _,
        ) = keygen_basic(
            &param,
            &mut secret_generator,
            &mut encryption_generator,
            &mut computation_buffers,
        );

        let (
            lwe_secret_key_online,
            _glwe_secret_key_online,
            lwe_secret_key_after_ks_online,
            fourier_bsk_online,
            _,
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
        let ksk_from_online = allocate_and_generate_new_lwe_keyswitch_key(
            &lwe_secret_key_online,
            &lwe_secret_key,
            ks_base_log_from_online,
            ks_level_from_online,
            param.glwe_modular_std_dev,
            param.ciphertext_modulus,
            &mut encryption_generator,
        );

        let mut he_bits = LweCiphertextList::new(0u64, ksk_to_online.output_lwe_size(), LweCiphertextCount(2), param_online.ciphertext_modulus);
        for mut he_bit in he_bits.iter_mut() {
            encrypt_lwe_ciphertext(&lwe_secret_key_after_ks_online, &mut he_bit, Plaintext(0), param_online.lwe_modular_std_dev, &mut encryption_generator);
        }

        group.bench_with_input(BenchmarkId::new(*id, *id), &param_online, |b, &_| b.iter(|| online(
            black_box(&he_bits),
            black_box(fourier_bsk_online),
            black_box(&ksk_from_online),
        )));
    }
}

fn online(
    he_bits: &LweCiphertextListOwned<u64>,
    fourier_bsk_online: FourierLweBootstrapKeyView,
    ksk_from_online: &LweKeyswitchKeyOwned<u64>,
) {
    let ciphertext_modulus = he_bits.ciphertext_modulus();

    let mut he_out = LweCiphertext::new(0u64, ksk_from_online.output_lwe_size(), ciphertext_modulus);
    let mut he_buf = LweCiphertext::new(0u64, fourier_bsk_online.output_lwe_dimension().to_lwe_size(), ciphertext_modulus);

    for (i, he_bit) in he_bits.iter().enumerate() {
        let delta_log = 64 - (5 - i);
        let accumulator = generate_negacyclic_accumulator_from_sbox_half(
            MODULUS,
            ciphertext_modulus,
            fourier_bsk_online,
            1u64 << (delta_log - 1),
            &[1u64.wrapping_neg(); MODULUS / 2],
        );

        let mut he_pbs = LweCiphertext::new(0u64, fourier_bsk_online.output_lwe_dimension().to_lwe_size(), ciphertext_modulus);
        programmable_bootstrap_lwe_ciphertext(&he_bit, &mut he_pbs, &accumulator, &fourier_bsk_online);

        lwe_ciphertext_plaintext_add_assign(&mut he_pbs, Plaintext(1u64 << (delta_log - 1)));
        lwe_ciphertext_add_assign(&mut he_buf, &he_pbs);
    }

    keyswitch_lwe_ciphertext(&ksk_from_online, &he_buf, &mut he_out);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
