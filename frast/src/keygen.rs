use tfhe::{
    shortint::parameters::ClassicPBSParameters,
    core_crypto::{
        prelude::*,
        commons::math::random::{RandomGenerable, UniformBinary},
    }
};

pub fn keygen_basic<Scalar, G>(
    param: &ClassicPBSParameters,
    secret_generator: &mut SecretRandomGenerator<G>,
    encryption_generator: &mut EncryptionRandomGenerator<G>,
    buffers: &mut ComputationBuffers,
) -> (
    LweSecretKey<Vec<Scalar>>,
    GlweSecretKey<Vec<Scalar>>,
    LweSecretKey<Vec<Scalar>>,
    FourierLweBootstrapKeyOwned,
    LweKeyswitchKey<Vec<Scalar>>,
)
where
    Scalar: UnsignedTorus + Sync + Send + RandomGenerable<UniformBinary>,
    G: ParallelByteRandomGenerator,
{
    let small_lwe_secret_key: LweSecretKey<Vec<Scalar>> = LweSecretKey::generate_new_binary(param.lwe_dimension, secret_generator);
    let glwe_secret_key: GlweSecretKey<Vec<Scalar>> = GlweSecretKey::generate_new_binary(param.glwe_dimension, param.polynomial_size, secret_generator);
    let large_lwe_secret_key: LweSecretKey<Vec<Scalar>> = glwe_secret_key.clone().into_lwe_secret_key();

    let lwe_secret_key = large_lwe_secret_key;
    let lwe_secret_key_after_ks = small_lwe_secret_key;

    let bootstrap_key = par_allocate_and_generate_new_lwe_bootstrap_key(
        &lwe_secret_key_after_ks,
        &glwe_secret_key,
        param.pbs_base_log,
        param.pbs_level,
        param.glwe_modular_std_dev,
        CiphertextModulus::new_native(),
        encryption_generator,
    );

    let mut fourier_bsk = FourierLweBootstrapKey::new(
        bootstrap_key.input_lwe_dimension(),
        bootstrap_key.glwe_size(),
        bootstrap_key.polynomial_size(),
        bootstrap_key.decomposition_base_log(),
        bootstrap_key.decomposition_level_count(),
    );

    let fft = Fft::new(bootstrap_key.polynomial_size());
    let fft = fft.as_view();
    buffers.resize(
        convert_standard_lwe_bootstrap_key_to_fourier_mem_optimized_requirement(fft)
        .unwrap()
        .unaligned_bytes_required(),
    );
    let stack = buffers.stack();

    convert_standard_lwe_bootstrap_key_to_fourier_mem_optimized(
        &bootstrap_key,
        &mut fourier_bsk,
        fft,
        stack,
    );
    drop(bootstrap_key);

    let ksk = allocate_and_generate_new_lwe_keyswitch_key(
        &lwe_secret_key,
        &lwe_secret_key_after_ks,
        param.ks_base_log,
        param.ks_level,
        param.lwe_modular_std_dev,
        CiphertextModulus::new_native(),
        encryption_generator,
    );

    (lwe_secret_key, glwe_secret_key, lwe_secret_key_after_ks, fourier_bsk, ksk)
}
