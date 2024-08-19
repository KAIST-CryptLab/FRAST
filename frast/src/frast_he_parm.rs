use serde::{Serialize, Deserialize};
use tfhe::shortint::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct FrastHEParameters {
    pub he_param: ClassicPBSParameters,
    pub ggsw_bit_decomp_base_log: DecompositionBaseLog,
    pub ggsw_bit_decomp_level_count: DecompositionLevelCount,
    pub subs_decomp_base_log: DecompositionBaseLog,
    pub subs_decomp_level_count: DecompositionLevelCount,
    pub ggsw_key_decomp_base_log: DecompositionBaseLog,
    pub ggsw_key_decomp_level_count: DecompositionLevelCount,
}

pub const FRAST_HE_PARAM1: FrastHEParameters = FrastHEParameters {
    he_param: _PARAM_M2_C2,
    ggsw_bit_decomp_base_log: DecompositionBaseLog(7),
    ggsw_bit_decomp_level_count: DecompositionLevelCount(3),
    subs_decomp_base_log: DecompositionBaseLog(9),
    subs_decomp_level_count: DecompositionLevelCount(5),
    ggsw_key_decomp_base_log: DecompositionBaseLog(9),
    ggsw_key_decomp_level_count: DecompositionLevelCount(5),
};

const _PARAM_M2_C2: ClassicPBSParameters = ClassicPBSParameters { // PARAM_MESSAGE_2_CARRY_2
    lwe_dimension: LweDimension(742),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_modular_std_dev: StandardDev(0.000007069849454709433),
    glwe_modular_std_dev: StandardDev(0.00000000000000029403601535432533),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(3),
    message_modulus: MessageModulus(4),
    carry_modulus: CarryModulus(4),
    ciphertext_modulus: CiphertextModulus::new_native(),
    encryption_key_choice: EncryptionKeyChoice::Big,
};

pub const PARAM_ONLINE: ClassicPBSParameters = ClassicPBSParameters { // kreyvium parameter
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
