use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use frast::{keygen_basic, generate_negacyclic_accumulator_from_sbox_half};
use rand::Rng;
use tfhe::{shortint::{prelude::*, CiphertextModulus}, core_crypto::{prelude::*, fft_impl::fft64::c64}};

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
    let param_frast = PARAM_MESSAGE_2_CARRY_2;

    let param_list = [param_elisabeth, param_kreyvium, param_frast];
    let bench_id_list = ["elisabeth", "kreyvium", "frast"];
    let mut group = c.benchmark_group("gen_pbs");

    for (id, param) in bench_id_list.iter().zip(param_list.iter()) {
        // Set random generators and buffers
        let mut boxed_seeder = new_seeder();
        let seeder = boxed_seeder.as_mut();

        let mut secret_generator = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
        let mut encryption_generator = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

        let mut computation_buffers = ComputationBuffers::new();

        // Set keys
        let (
            lwe_secret_key,
            _,
            lwe_secret_key_after_ks,
            fourier_bsk,
            _,
        ) = keygen_basic(
            &param,
            &mut secret_generator,
            &mut encryption_generator,
            &mut computation_buffers
        );
        let fourier_bsk = fourier_bsk.as_view();

        // Set plaintext and encrypt
        let modulus_bit = 4;
        let modulus_sup: usize = 1 << modulus_bit;
        let delta = 1_u64 << (64 - modulus_bit);

        let mut rng = rand::thread_rng();
        let input = rng.gen_range(0..modulus_sup);
        let pt = Plaintext(input as u64 * delta);
        let ct_in = allocate_and_encrypt_new_lwe_ciphertext(&lwe_secret_key_after_ks, pt, param.lwe_modular_std_dev, param.ciphertext_modulus, &mut encryption_generator);

        let sbox_half = (0..modulus_sup/2).map(|_| rng.gen_range(0..modulus_sup) as u64).collect::<Vec<u64>>();
        let accumulator = generate_negacyclic_accumulator_from_sbox_half(modulus_sup, param.ciphertext_modulus, fourier_bsk, delta, &sbox_half);
        let mut ct_out = LweCiphertext::new(0u64, lwe_secret_key.lwe_dimension().to_lwe_size(), param.ciphertext_modulus);

        // c.bench_function("gen_pbs_elisabeth", |b| b.iter(|| gen_pbs(black_box(&ct_in), black_box(&mut ct_out), black_box(&accumulator), black_box(&fourier_bsk))));
        group.bench_with_input(BenchmarkId::new(*id, *id), &param, |b, &_| b.iter(|| gen_pbs(black_box(&ct_in), black_box(&mut ct_out), black_box(&accumulator), black_box(&fourier_bsk))));
    }

}

fn gen_pbs<C: Container<Element = c64>>(
    input: &LweCiphertextOwned<u64>,
    output: &mut LweCiphertextOwned<u64>,
    accumulator: &GlweCiphertextOwned<u64>,
    fourier_bsk: &FourierLweBootstrapKey<C>,
) {
    programmable_bootstrap_lwe_ciphertext(input, output, accumulator, fourier_bsk);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);