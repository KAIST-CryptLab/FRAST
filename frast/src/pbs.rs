use aligned_vec::CACHELINE_ALIGN;
use tfhe::core_crypto::{
    prelude::{*, polynomial_algorithms::*},
    fft_impl::{
        fft64::{
            c64,
            crypto::{bootstrap::FourierLweBootstrapKeyView, ggsw::cmux},
            math::fft::FftView,
        },
        common::pbs_modulus_switch,
    },
};
use dyn_stack::{PodStack, ReborrowMut};
use crate::fourier_poly_mult::*;

// https://docs.rs/itertools/0.7.8/src/itertools/lib.rs.html#247-269
// From tfhe::src::core_crypto::commons::utils
#[allow(unused_macros)]
macro_rules! izip {
    // eg. __izip_closure!(((a, b), c) => (a, b, c) , dd , ee )
    (@ __closure @ $p:pat => $tup:expr) => {
        |$p| $tup
    };

    // The "b" identifier is a different identifier on each recursion level thanks to hygiene.
    (@ __closure @ $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )*) => {
        izip!(@ __closure @ ($p, b) => ( $($tup)*, b ) $( , $tail )*)
    };

    ( $first:expr $(,)?) => {
        {
            #[allow(unused_imports)]
            use tfhe::core_crypto::commons::utils::ZipChecked;
            ::core::iter::IntoIterator::into_iter($first)
        }
    };
    ( $first:expr, $($rest:expr),+ $(,)?) => {
        {
            #[allow(unused_imports)]
            use tfhe::core_crypto::commons::utils::ZipChecked;
            ::core::iter::IntoIterator::into_iter($first)
                $(.zip_checked($rest))*
                .map(izip!(@ __closure @ a => (a) $( , $rest )*))
        }
    };
}


// ======== Generating accumulators for GenPBS ========
pub fn generate_accumulator<Scalar, F>(
    modulus_sup: usize,
    ciphertext_modulus: CiphertextModulus<Scalar>,
    bsk: FourierLweBootstrapKeyView,
    delta: Scalar,
    f: F
) -> GlweCiphertext<Vec<Scalar>>
where
    Scalar: UnsignedInteger + UnsignedTorus + CastFrom<usize>,
    F: Fn(Scalar) -> Scalar,
{
    let polynomial_size = bsk.polynomial_size().0;
    let box_size = polynomial_size / modulus_sup;

    let mut accumulator = vec![Scalar::ZERO; polynomial_size];

    for i in 0..modulus_sup {
        let index = i * box_size;
        accumulator[index..index + box_size]
            .iter_mut()
            .for_each(|a| {
                *a = f(i.cast_into()) * delta;
            });
    }

    let half_box_size = box_size / 2;

    for a_i in accumulator[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }

    accumulator.rotate_left(half_box_size);

    let accumulator_plaintext = PlaintextListOwned::from_container(accumulator);

    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        bsk.glwe_size(),
        &accumulator_plaintext,
        ciphertext_modulus,
    )
}

pub fn generate_negacyclic_accumulator_from_sbox_half<Scalar>(
    modulus_sup: usize,
    ciphertext_modulus: CiphertextModulus<Scalar>,
    bsk: FourierLweBootstrapKeyView,
    delta: Scalar,
    sbox_half: &[Scalar],
) -> GlweCiphertext<Vec<Scalar>>
where
    Scalar: UnsignedInteger + UnsignedTorus + CastFrom<usize>
{
    let poly_size = bsk.polynomial_size().0;
    let box_size = 2 * poly_size / modulus_sup;
    let mut accumulator = vec![Scalar::ZERO; poly_size];

    for i in 0..(modulus_sup/2) {
        let index = i * box_size;
        accumulator[index..index + box_size]
            .iter_mut()
            .for_each(|a| *a = sbox_half[i as usize] * delta)
    }

    let half_box_size = box_size / 2;
    for a_i in accumulator[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }
    accumulator.rotate_left(half_box_size);

    let accumulator_plaintext = PlaintextList::from_container(accumulator);
    let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
        bsk.glwe_size(),
        &accumulator_plaintext,
        ciphertext_modulus,
    );

    accumulator
}

// ======== ComBo ========
pub fn bts_by_combo<Scalar, ContMut>(
    mut lwe_out: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta_log: DeltaLog,
    modulus_sup: usize,
)
where
    Scalar: UnsignedInteger + UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    ContMut: ContainerMut<Element=Scalar>,
{
    debug_assert!(lwe_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        lwe_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );

    // Set tables for the negacyclic functions
    let sbox_odd_half = (0..modulus_sup/2).map(|i| {
        let val = 2 * i + 1;
        let val = val % (2 * modulus_sup);
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let id_neg_half = (0..modulus_sup/2).map(|i| {
        let val = 2 * i + 1;
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    // Set accumulators
    let accumulator_sbox_odd = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &sbox_odd_half,
    );

    let accumulator_id_neg = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &id_neg_half,
    );

    // Keyswitch input ciphertext
    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    // Odd function evaluation
    let mut lwe_odd_input = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_odd_input, &accumulator_id_neg, &fourier_bsk);
    let lwe_odd_input_body = lwe_odd_input.get_mut_body().data;
    *lwe_odd_input_body = lwe_odd_input_body.wrapping_sub(Scalar::ONE << (delta_log.0 - 1));

    let mut lwe_odd_input_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_odd_input, &mut lwe_odd_input_ks);

    programmable_bootstrap_lwe_ciphertext(&lwe_odd_input_ks, &mut lwe_out, &accumulator_sbox_odd, &fourier_bsk);

    // Even function evaluation
    let lwe_out_body = lwe_out.get_mut_body().data;
    *lwe_out_body = lwe_out_body.wrapping_sub(Scalar::ONE << (delta_log.0 - 1));
}

pub fn wop_pbs_by_combo<Scalar, ContMut>(
    mut lwe_out: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta_log: DeltaLog,
    modulus_sup: usize,
    sbox: &[usize],
)
where
    Scalar: UnsignedInteger + UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    ContMut: ContainerMut<Element=Scalar>,
{
    debug_assert!(lwe_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        lwe_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );

    // Set tables for the negacyclic functions
    let sbox_odd_half = (0..modulus_sup/2).map(|i| {
        let val = sbox[i] - sbox[(modulus_sup - i - 1) % modulus_sup];
        let val = val % (2 * modulus_sup);
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let sbox_even_half = (0..modulus_sup/2).map(|i| {
        let val = sbox[i] + sbox[(modulus_sup - i - 1) % modulus_sup];
        let val = val % (2 * modulus_sup);
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let id_neg_half = (0..modulus_sup/2).map(|i| {
        let val = 2 * i + 1;
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let abs_neg_half = (0..modulus_sup/2).map(|i| {
        let val = 4 * i + modulus_sup + 2;
        (val << (delta_log.0 - 2)).cast_into()
    }).collect::<Vec<Scalar>>();

    // Set accumulators
    let accumulator_sbox_odd = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &sbox_odd_half,
    );

    let accumulator_sbox_even = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &sbox_even_half,
    );

    let accumulator_id_neg = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &id_neg_half,
    );

    let accumulator_abs_neg = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &abs_neg_half,
    );

    // Keyswitch input ciphertext
    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    // Odd function evaluation
    let mut lwe_odd_input = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_odd_input, &accumulator_id_neg, &fourier_bsk);
    let lwe_odd_input_body = lwe_odd_input.get_mut_body().data;
    *lwe_odd_input_body = lwe_odd_input_body.wrapping_sub(Scalar::ONE << (delta_log.0 - 1));

    let mut lwe_odd_input_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_odd_input, &mut lwe_odd_input_ks);

    programmable_bootstrap_lwe_ciphertext(&lwe_odd_input_ks, &mut lwe_out, &accumulator_sbox_odd, &fourier_bsk);

    // Even function evaluation
    let mut lwe_even_input = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_even_input, &accumulator_abs_neg, &fourier_bsk);
    let lwe_even_input_body = lwe_even_input.get_mut_body().data;
    *lwe_even_input_body = lwe_even_input_body.wrapping_sub(((modulus_sup + 2) << (delta_log.0 - 2)).cast_into());

    let mut lwe_even_input_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_even_input, &mut lwe_even_input_ks);

    let mut lwe_out_even = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_even_input_ks, &mut lwe_out_even, &accumulator_sbox_even, &fourier_bsk);

    lwe_ciphertext_add_assign(&mut lwe_out, &lwe_out_even);
}

pub fn wop_pbs_with_bts_by_combo<Scalar, ContMut>(
    mut pbs_out: LweCiphertext<ContMut>,
    mut bts_out: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta_log: DeltaLog,
    modulus_sup: usize,
    sbox: &[usize],
)
where
    Scalar: UnsignedInteger + UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    ContMut: ContainerMut<Element=Scalar>,
{
    debug_assert!(pbs_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(bts_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        pbs_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "pbs_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        pbs_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        bts_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "bts_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        bts_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );

    // Set tables for the negacyclic functions
    let sbox_odd_half = (0..modulus_sup/2).map(|i| {
        let val = sbox[i] - sbox[(modulus_sup - i - 1) % modulus_sup];
        let val = val % (2 * modulus_sup);
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let sbox_even_half = (0..modulus_sup/2).map(|i| {
        let val = sbox[i] + sbox[(modulus_sup - i - 1) % modulus_sup];
        let val = val % (2 * modulus_sup);
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let id_neg_half = (0..modulus_sup/2).map(|i| {
        let val = 2 * i + 1;
        (val << (delta_log.0 - 1)).cast_into()
    }).collect::<Vec<Scalar>>();

    let abs_neg_half = (0..modulus_sup/2).map(|i| {
        let val = 4 * i + modulus_sup + 2;
        (val << (delta_log.0 - 2)).cast_into()
    }).collect::<Vec<Scalar>>();

    // Set accumulators
    let accumulator_sbox_odd = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &sbox_odd_half,
    );

    let accumulator_sbox_even = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &sbox_even_half,
    );

    let accumulator_id_neg = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &id_neg_half,
    );

    let accumulator_abs_neg = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        Scalar::ONE,
        &abs_neg_half,
    );

    // Keyswitch input ciphertext
    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    // Odd function evaluation
    let mut lwe_odd_input = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_odd_input, &accumulator_id_neg, &fourier_bsk);
    let lwe_odd_input_body = lwe_odd_input.get_mut_body().data;
    *lwe_odd_input_body = lwe_odd_input_body.wrapping_sub(Scalar::ONE << (delta_log.0 - 1));

    let mut lwe_odd_input_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_odd_input, &mut lwe_odd_input_ks);

    programmable_bootstrap_lwe_ciphertext(&lwe_odd_input_ks, &mut pbs_out, &accumulator_sbox_odd, &fourier_bsk);

    programmable_bootstrap_lwe_ciphertext(&lwe_odd_input_ks, &mut bts_out, &accumulator_id_neg, &fourier_bsk);
    let bts_out_body = bts_out.get_mut_body().data;
    *bts_out_body = bts_out_body.wrapping_sub(Scalar::ONE << (delta_log.0 - 1));

    // Even function evaluation
    let mut lwe_even_input = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_even_input, &accumulator_abs_neg, &fourier_bsk);
    let lwe_even_input_body = lwe_even_input.get_mut_body().data;
    *lwe_even_input_body = lwe_even_input_body.wrapping_sub(((modulus_sup + 2) << (delta_log.0 - 2)).cast_into());

    let mut lwe_even_input_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_even_input, &mut lwe_even_input_ks);

    let mut lwe_out_even = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    programmable_bootstrap_lwe_ciphertext(&lwe_even_input_ks, &mut lwe_out_even, &accumulator_sbox_even, &fourier_bsk);

    lwe_ciphertext_add_assign(&mut pbs_out, &lwe_out_even);
}

// ======== New WoP-PBS ========
pub fn generate_negacyclic_accumulator_from_two_sbox_half<Scalar>(
    modulus_sup: usize,
    ciphertext_modulus: CiphertextModulus<Scalar>,
    bsk: FourierLweBootstrapKeyView,
    delta1: Scalar,
    sbox_half1: &[Scalar],
    delta2: Scalar,
    sbox_half2: &[Scalar],
) -> GlweCiphertext<Vec<Scalar>>
where
    Scalar: UnsignedTorus + CastFrom<usize>
{
    let poly_size = bsk.polynomial_size().0;
    let box_size = 2 * poly_size / modulus_sup;
    let mut accumulator = vec![Scalar::ZERO; poly_size];

    for i in 0..(modulus_sup/2) {
        let index = i * box_size;
        for offset in 0..box_size {
            if offset % 2 == 0 {
                accumulator[index + offset] = sbox_half1[i] * delta1;
            } else {
                accumulator[index + offset] = sbox_half2[i] * delta2;
            }
        }
    }

    let half_box_size = box_size / 2;
    for a_i in accumulator[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }
    accumulator.rotate_left(half_box_size);

    let accumulator_plaintext = PlaintextList::from_container(accumulator);
    let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
        bsk.glwe_size(),
        &accumulator_plaintext,
        ciphertext_modulus,
    );

    accumulator
}

pub fn gen_blind_rotate_assign<Scalar, InputCont, OutputCont, AccCont, KeyCont>(
    input: &LweCiphertext<InputCont>,
    output: &mut LweCiphertext<OutputCont>,
    accumulator: &GlweCiphertext<AccCont>,
    mod_switch_offset: ModulusSwitchOffset,
    log_lut_count: LutCountLog,
    fourier_bsk: &FourierLweBootstrapKey<KeyCont>,
) where
    Scalar: UnsignedTorus + CastInto<usize>,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
    AccCont: Container<Element = Scalar>,
    KeyCont: Container<Element = c64>,
{
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());
    assert_eq!(
        output.ciphertext_modulus(),
        accumulator.ciphertext_modulus()
    );

    let mut buffers = ComputationBuffers::new();

    let fft = Fft::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();

    buffers.resize(
        programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<Scalar>(
            fourier_bsk.glwe_size(),
            fourier_bsk.polynomial_size(),
            fft,
        )
        .unwrap()
        .unaligned_bytes_required(),
    );

    let stack = buffers.stack();

    let accumulator = accumulator.as_view();
    let (mut local_accumulator_data, stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
    let mut local_accumulator = GlweCiphertextMutView::from_container(
        &mut *local_accumulator_data,
        accumulator.polynomial_size(),
        accumulator.ciphertext_modulus(),
    );

    gen_blind_rotate_local_assign(
        fourier_bsk.as_view(),
        local_accumulator.as_mut_view(),
        mod_switch_offset,
        log_lut_count,
        input.as_ref(),
        fft,
        stack,
    );

    extract_lwe_sample_from_glwe_ciphertext(&local_accumulator, output, MonomialDegree(0));
}

pub fn gen_blind_rotate_local_assign<Scalar: UnsignedTorus + CastInto<usize>>(
    bsk: FourierLweBootstrapKeyView<'_>,
    mut lut: GlweCiphertextMutView<'_, Scalar>,
    mod_switch_offset: ModulusSwitchOffset,
    log_lut_count: LutCountLog,
    lwe: &[Scalar],
    fft: FftView<'_>,
    mut stack: PodStack<'_>,
) {
    let (lwe_body, lwe_mask) = lwe.split_last().unwrap();

    let lut_poly_size = lut.polynomial_size();
    let ciphertext_modulus = lut.ciphertext_modulus();
    let monomial_degree = pbs_modulus_switch(
        *lwe_body,
        lut_poly_size,
        mod_switch_offset,
        log_lut_count,
    );
    debug_assert!(ciphertext_modulus.is_native_modulus());

    lut.as_mut_polynomial_list()
        .iter_mut()
        .for_each(|mut poly| {
            polynomial_wrapping_monic_monomial_div_assign(
                &mut poly,
                MonomialDegree(monomial_degree),
            )
        });

    // We initialize the ct_0 used for the successive cmuxes
    let mut ct0 = lut;

    for (lwe_mask_element, bootstrap_key_ggsw) in izip!(lwe_mask.iter(), bsk.into_ggsw_iter())
    {
        if *lwe_mask_element != Scalar::ZERO {
            let stack = stack.rb_mut();
            // We copy ct_0 to ct_1
            let (mut ct1, stack) =
                stack.collect_aligned(CACHELINE_ALIGN, ct0.as_ref().iter().copied());
            let mut ct1 = GlweCiphertextMutView::from_container(
                &mut *ct1,
                lut_poly_size,
                ciphertext_modulus,
            );

            // We rotate ct_1 by performing ct_1 <- ct_1 * X^{a_hat}
            for mut poly in ct1.as_mut_polynomial_list().iter_mut() {
                polynomial_wrapping_monic_monomial_mul_assign(
                    &mut poly,
                    MonomialDegree(pbs_modulus_switch(
                        *lwe_mask_element,
                        lut_poly_size,
                        mod_switch_offset,
                        log_lut_count,
                    )),
                );
            }

            // ct1 is re-created each loop it can be moved, ct0 is already a view, but
            // as_mut_view is required to keep borrow rules consistent
            cmux(ct0.as_mut_view(), ct1, bootstrap_key_ggsw, fft, stack);
        }
    }

    // if !ciphertext_modulus.is_native_modulus() {
    //     // When we convert back from the fourier domain, integer values will contain up to 53
    //     // MSBs with information. In our representation of power of 2 moduli < native modulus we
    //     // fill the MSBs and leave the LSBs empty, this usage of the signed decomposer allows to
    //     // round while keeping the data in the MSBs
    //     let signed_decomposer = SignedDecomposer::new(
    //         DecompositionBaseLog(ciphertext_modulus.get().ilog2() as usize),
    //         DecompositionLevelCount(1),
    //     );
    //     ct0.as_mut()
    //         .iter_mut()
    //         .for_each(|x| *x = signed_decomposer.closest_representable(*x));
    // }
}

pub fn set_scale_by_pbs<Scalar>(
    lwe_in: &LweCiphertextOwned<Scalar>,
    lwe_out: &mut LweCiphertextOwned<Scalar>,
    modulus_sup: usize,
    delta_out: Scalar,
    fourier_bsk: FourierLweBootstrapKeyView,
) where
    Scalar: UnsignedInteger + UnsignedTorus + CastFrom<usize> + CastInto<usize>,
{
    let accumulator_id = generate_accumulator(
        modulus_sup / 2,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        delta_out,
        |i| i,
    );

    programmable_bootstrap_lwe_ciphertext(
        lwe_in,
        lwe_out,
        &accumulator_id,
        &fourier_bsk,
    );
}

pub fn wop_pbs<Scalar, ContMut>(
    mut lwe_out: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta_log: DeltaLog,
    modulus_sup: usize,
    sbox: &[usize],
    is_pbs_many: bool,
) where
    Scalar: UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    ContMut: ContainerMut<Element=Scalar>,
{
    debug_assert!(lwe_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        lwe_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );

    lwe_out.as_mut().fill(Scalar::ZERO);

    let sbox_odd = (0..modulus_sup/2).map(|i| ((sbox[i] - sbox[i + modulus_sup/2]) % (2 * modulus_sup)).cast_into()).collect::<Vec<Scalar>>();
    let sbox_even = (0..modulus_sup/2).map(|i| ((sbox[i] + sbox[i + modulus_sup/2])).cast_into()).collect::<Vec<Scalar>>();

    let fft = Fft::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();

    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    let mut lwe_out_odd = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    let mut lwe_out_even = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());

    let mut lwe_msb = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());

    let mut lwe_lsbs = LweCiphertext::from_container(lwe_in.as_ref().to_vec(), lwe_in.ciphertext_modulus());
    let mut lwe_lsbs_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());

    if is_pbs_many {
        let mut buffers = ComputationBuffers::new();
        buffers.resize(
            2 * programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<Scalar>(
                fourier_bsk.glwe_size(),
                fourier_bsk.polynomial_size(),
                fft,
            ).unwrap().unaligned_bytes_required()
        );
        let stack = buffers.stack();

        let accumulator_odd = generate_negacyclic_accumulator_from_two_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_odd,
            Scalar::ONE << (Scalar::BITS - 2),
            &vec![(3 as usize).cast_into(); modulus_sup / 2],
        );
        let (mut local_accumulator_odd_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_odd.as_ref().iter().copied());
        let mut local_accumulator_odd = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_odd_data,
            fourier_bsk.polynomial_size(),
            lwe_in.ciphertext_modulus(),
        );

        gen_blind_rotate_local_assign(fourier_bsk, local_accumulator_odd.as_mut_view(), ModulusSwitchOffset(0), LutCountLog(1), lwe_in_ks.as_ref(), fft, stack.rb_mut());
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_odd, &mut lwe_out_odd, MonomialDegree(0));
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_odd, &mut lwe_msb, MonomialDegree(1));
        lwe_ciphertext_plaintext_add_assign(&mut lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

        lwe_ciphertext_sub_assign(&mut lwe_lsbs, &lwe_msb);
        keyswitch_lwe_ciphertext(&ksk, &lwe_lsbs, &mut lwe_lsbs_ks);

        let accumulator_even = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_even,
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_lsbs_ks, &mut lwe_out_even, &accumulator_even, &fourier_bsk);
    } else {
        let accumulator_odd = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_odd,
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_out_odd, &accumulator_odd, &fourier_bsk);

        let accumulator_msb = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (Scalar::BITS - 2),
            &vec![(3 as usize).cast_into(); modulus_sup / 2],
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_msb, &accumulator_msb, &fourier_bsk);
        lwe_ciphertext_plaintext_add_assign(&mut lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

        lwe_ciphertext_sub_assign(&mut lwe_lsbs, &lwe_msb);
        keyswitch_lwe_ciphertext(&ksk, &lwe_lsbs, &mut lwe_lsbs_ks);

        let accumulator_even = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_even,
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_lsbs_ks, &mut lwe_out_even, &accumulator_even, &fourier_bsk);
    }

    lwe_ciphertext_add_assign(&mut lwe_out, &lwe_out_odd);
    lwe_ciphertext_add_assign(&mut lwe_out, &lwe_out_even);
}

pub fn wop_pbs_with_bts<Scalar, ContMut>(
    mut lwe_out: LweCiphertext<ContMut>,
    mut lwe_refreshed: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta_log: DeltaLog,
    modulus_sup: usize,
    sbox: &[usize],
    is_pbs_many: bool,
) where
    Scalar: UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    ContMut: ContainerMut<Element=Scalar>,
{
    debug_assert!(lwe_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(lwe_refreshed.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        lwe_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_refreshed.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );

    lwe_out.as_mut().fill(Scalar::ZERO);
    lwe_refreshed.as_mut().fill(Scalar::ZERO);

    let sbox_odd = (0..modulus_sup/2).map(|i| ((sbox[i] - sbox[i + modulus_sup/2]) % (2 * modulus_sup)).cast_into()).collect::<Vec<Scalar>>();
    let sbox_even = (0..modulus_sup/2).map(|i| ((sbox[i] + sbox[i + modulus_sup/2])).cast_into()).collect::<Vec<Scalar>>();

    let fft = Fft::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();

    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    let mut lwe_out_odd = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    let mut lwe_out_even = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    let mut lwe_msb = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());

    let mut lwe_lsbs = LweCiphertext::from_container(lwe_in.as_ref().to_vec(), lwe_in.ciphertext_modulus());
    let mut lwe_lsbs_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());

    if is_pbs_many {
        let mut buffers = ComputationBuffers::new();
        buffers.resize(
            2 * programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<Scalar>(
                fourier_bsk.glwe_size(),
                fourier_bsk.polynomial_size(),
                fft,
            ).unwrap().unaligned_bytes_required()
        );
        let stack = buffers.stack();

        let accumulator_odd = generate_negacyclic_accumulator_from_two_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_odd,
            Scalar::ONE << (Scalar::BITS - 2),
            &vec![(3 as usize).cast_into(); modulus_sup / 2],
        );
        let (mut local_accumulator_odd_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_odd.as_ref().iter().copied());
        let mut local_accumulator_odd = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_odd_data,
            fourier_bsk.polynomial_size(),
            lwe_in.ciphertext_modulus(),
        );

        gen_blind_rotate_local_assign(fourier_bsk, local_accumulator_odd.as_mut_view(), ModulusSwitchOffset(0), LutCountLog(1), lwe_in_ks.as_ref(), fft, stack.rb_mut());
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_odd, &mut lwe_out_odd, MonomialDegree(0));
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_odd, &mut lwe_msb, MonomialDegree(1));
        lwe_ciphertext_plaintext_add_assign(&mut lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

        lwe_ciphertext_sub_assign(&mut lwe_lsbs, &lwe_msb);
        keyswitch_lwe_ciphertext(&ksk, &lwe_lsbs, &mut lwe_lsbs_ks);

        let accumulator_even = generate_negacyclic_accumulator_from_two_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_even,
            Scalar::ONE << delta_log.0,
            &(0..modulus_sup/2).map(|i| i.cast_into()).collect::<Vec<Scalar>>(),
        );
        let (mut local_accumulator_even_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator_even.as_ref().iter().copied());
        let mut local_accumulator_even = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_even_data,
            fourier_bsk.polynomial_size(),
            lwe_in.ciphertext_modulus(),
        );

        gen_blind_rotate_local_assign(fourier_bsk, local_accumulator_even.as_mut_view(), ModulusSwitchOffset(0), LutCountLog(1), lwe_lsbs_ks.as_ref(), fft, stack.rb_mut());
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_even, &mut lwe_out_even, MonomialDegree(0));
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator_even, &mut lwe_lsbs, MonomialDegree(1));
    } else {
        let accumulator_odd = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_odd,
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_out_odd, &accumulator_odd, &fourier_bsk);

        let accumulator_msb = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (Scalar::BITS - 2),
            &vec![(3 as usize).cast_into(); modulus_sup / 2],
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_msb, &accumulator_msb, &fourier_bsk);
        lwe_ciphertext_plaintext_add_assign(&mut lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

        lwe_ciphertext_sub_assign(&mut lwe_lsbs, &lwe_msb);
        keyswitch_lwe_ciphertext(&ksk, &lwe_lsbs, &mut lwe_lsbs_ks);

        let accumulator_even = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0 - 1),
            &sbox_even,
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_lsbs_ks, &mut lwe_out_even, &accumulator_even, &fourier_bsk);

        let accumulator_id = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (delta_log.0),
            &(0..modulus_sup/2).map(|i| i.cast_into()).collect::<Vec<Scalar>>(),
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_lsbs_ks, &mut lwe_lsbs, &accumulator_id, &fourier_bsk);
    }

    lwe_ciphertext_add_assign(&mut lwe_out, &lwe_out_odd);
    lwe_ciphertext_add_assign(&mut lwe_out, &lwe_out_even);
    lwe_ciphertext_add_assign(&mut lwe_refreshed, &lwe_msb);
    lwe_ciphertext_add_assign(&mut lwe_refreshed, &lwe_lsbs);
}

pub fn gen_pbs_with_bts<Scalar, ContMut>(
    mut lwe_out: LweCiphertext<ContMut>,
    mut lwe_refreshed: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta: Scalar,
    modulus_sup: usize,
    sbox_half: &[Scalar],
    is_pbs_many: bool,
) where
    Scalar: UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    ContMut: ContainerMut<Element=Scalar>,
{
    debug_assert!(lwe_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(lwe_refreshed.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        lwe_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_refreshed.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );

    lwe_out.as_mut().fill(Scalar::ZERO);
    lwe_refreshed.as_mut().fill(Scalar::ZERO);

    let fft = Fft::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();

    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    let mut lwe_msb = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    let mut lwe_lsbs = LweCiphertext::new(Scalar::ZERO, lwe_in.lwe_size(), lwe_in.ciphertext_modulus());
    let mut lwe_lsbs_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());

    if is_pbs_many {
        let mut buffers = ComputationBuffers::new();
        buffers.resize(
            2 * programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<Scalar>(
                fourier_bsk.glwe_size(),
                fourier_bsk.polynomial_size(),
                fft,
            ).unwrap().unaligned_bytes_required()
        );
        let stack = buffers.stack();

        let accumulator = generate_negacyclic_accumulator_from_two_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            delta,
            &sbox_half,
            Scalar::ONE << (Scalar::BITS - 2),
            &vec![(3 as usize).cast_into(); modulus_sup / 2],
        );
        let (mut local_accumulator_data, mut stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            fourier_bsk.polynomial_size(),
            lwe_in.ciphertext_modulus(),
        );

        gen_blind_rotate_local_assign(fourier_bsk, local_accumulator.as_mut_view(), ModulusSwitchOffset(0), LutCountLog(1), lwe_in_ks.as_ref(), fft, stack.rb_mut());
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator, &mut lwe_out, MonomialDegree(0));
        extract_lwe_sample_from_glwe_ciphertext(&local_accumulator, &mut lwe_msb, MonomialDegree(1));
        lwe_ciphertext_plaintext_add_assign(&mut lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

        lwe_ciphertext_sub(&mut lwe_lsbs, &lwe_in, &lwe_msb);
        keyswitch_lwe_ciphertext(&ksk, &lwe_lsbs, &mut lwe_lsbs_ks);
    } else {
        let accumulator = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            delta,
            &sbox_half,
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_out, &accumulator, &fourier_bsk);

        let accumulator_msb = generate_negacyclic_accumulator_from_sbox_half(
            modulus_sup,
            lwe_in.ciphertext_modulus(),
            fourier_bsk,
            Scalar::ONE << (Scalar::BITS - 2),
            &vec![(3 as usize).cast_into(); modulus_sup / 2],
        );
        programmable_bootstrap_lwe_ciphertext(&lwe_in_ks, &mut lwe_msb, &accumulator_msb, &fourier_bsk);
        lwe_ciphertext_plaintext_add_assign(&mut lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

        lwe_ciphertext_sub(&mut lwe_lsbs, &lwe_in, &lwe_msb);
        keyswitch_lwe_ciphertext(&ksk, &lwe_lsbs, &mut lwe_lsbs_ks);
    }

    let accumulator_id = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        lwe_in.ciphertext_modulus(),
        fourier_bsk,
        delta,
        &(0..modulus_sup/2).map(|i| i.cast_into()).collect::<Vec<Scalar>>(),
    );
    programmable_bootstrap_lwe_ciphertext(&lwe_lsbs_ks, &mut lwe_lsbs, &accumulator_id, &fourier_bsk);

    lwe_ciphertext_add_assign(&mut lwe_refreshed, &lwe_msb);
    lwe_ciphertext_add_assign(&mut lwe_refreshed, &lwe_lsbs);
}

#[cfg(feature = "multithread")]
pub fn par_wop_pbs<Scalar, ContMut>(
    mut lwe_out: LweCiphertext<ContMut>,
    lwe_in: LweCiphertext<&'_ [Scalar]>,
    ksk: LweKeyswitchKey<&'_ [Scalar]>,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    delta_log: DeltaLog,
    modulus_sup: usize,
    modulus_bit: usize,
    sbox: &[usize],
) where
    Scalar: UnsignedTorus + CastFrom<usize> + CastInto<usize> + Sync + Send,
    ContMut: ContainerMut<Element = Scalar> + Sync + Send,
{
    use rayon::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};

    debug_assert!(lwe_out.ciphertext_modulus() == lwe_in.ciphertext_modulus());
    debug_assert!(
        ksk.ciphertext_modulus().is_native_modulus(),
        "This operation only supports native moduli"
    );
    debug_assert!(
        lwe_out.lwe_size().to_lwe_dimension() == ksk.output_key_lwe_dimension(),
        "lwe_out needs to have an lwe_size of {}, got {}",
        ksk.output_key_lwe_dimension().0,
        lwe_out.lwe_size().to_lwe_dimension().0,
    );
    debug_assert!(
        lwe_in.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size(),
        "lwe_in needs to have an LWE dimension of {}, got {}",
        fourier_bsk.output_lwe_dimension().to_lwe_size().0,
        lwe_in.lwe_size().0,
    );
    debug_assert!(
        ksk.output_key_lwe_dimension() == fourier_bsk.input_lwe_dimension(),
        "ksk needs to have an output LWE dimension of {}, got {}",
        fourier_bsk.input_lwe_dimension().0,
        ksk.output_key_lwe_dimension().0,
    );
    debug_assert!(modulus_sup == (1 << modulus_bit));

    lwe_out.as_mut().fill(Scalar::ZERO);
    let vec_sbox_decomp = decompose_sbox(sbox, modulus_sup, modulus_bit);
    let mut buf_list = LweCiphertextList::new(Scalar::ZERO, lwe_out.lwe_size(), LweCiphertextCount(modulus_bit), lwe_out.ciphertext_modulus());

    let mut lwe_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_out.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_ks);

    vec_sbox_decomp[0..modulus_bit].par_iter().enumerate()
        .zip(buf_list.par_iter_mut())
        .for_each(|((i, sbox_i), mut buf)| {
            let sbox_i_half = (0..(modulus_sup >> (i+1))).map(|x| sbox_i[x].cast_into()).collect::<Vec<Scalar>>();
            let accumulator_sbox_i = generate_negacyclic_accumulator_from_sbox_half(
                modulus_sup >> i,
                lwe_out.ciphertext_modulus(),
                fourier_bsk,
                Scalar::ONE << (delta_log.0 - (i + 1)),
                &sbox_i_half,
            );
            gen_blind_rotate_assign(&lwe_ks, &mut buf, &accumulator_sbox_i, ModulusSwitchOffset(i), LutCountLog(0), &fourier_bsk);
        });
    for buf in buf_list.iter() {
        lwe_ciphertext_add_assign(&mut lwe_out, &buf);
    }
    lwe_ciphertext_plaintext_add_assign(&mut lwe_out, Plaintext((vec_sbox_decomp[modulus_bit][0] << (delta_log.0 - modulus_bit)).cast_into()));
}

pub fn decompose_sbox(sbox: &[usize], modulus_sup: usize, modulus_bit: usize) -> Vec<Vec<usize>> {
    debug_assert!(modulus_sup == (1 << modulus_bit));
    debug_assert!(sbox.len() == modulus_sup);

    let mut vec_sbox_decomp = Vec::<Vec<usize>>::with_capacity(modulus_bit + 1);
    for i in 0..modulus_bit {
        let sbox_i = (0..(modulus_sup >> i)).map(|m| {
            let m_msb_flipped = (m + (modulus_sup >> (i + 1))) % (modulus_sup >> i);
            let mut val = 0;
            for j in 0..(1 << i) {
                val += sbox[(j * (modulus_sup >> i) + m) % modulus_sup];
                val -= sbox[(j * (modulus_sup >> i) + m_msb_flipped) % modulus_sup];
            }
            val % (modulus_sup << (i + 1))
        }).collect::<Vec<usize>>();

        vec_sbox_decomp.push(sbox_i);
    }

    let mut val = 0;
    for i in 0..modulus_sup {
        val += sbox[i];
    }
    vec_sbox_decomp.push(vec![val % (modulus_sup << modulus_bit); 1]);

    vec_sbox_decomp
}

pub fn bit_decomposition_into_msb_encoding<Scalar, InCont, OutCont, KskCont>(
    lwe_in: &LweCiphertext<InCont>,
    lwe_out_list: &mut LweCiphertextList<OutCont>,
    modulus_sup: usize,
    fourier_bsk: FourierLweBootstrapKeyView<'_>,
    ksk: &LweKeyswitchKey<KskCont>,
) where
    Scalar: UnsignedTorus + CastFrom<usize> + CastInto<usize>,
    InCont: Container<Element=Scalar>,
    OutCont: ContainerMut<Element=Scalar>,
    KskCont: Container<Element=Scalar>,
{
    debug_assert!(lwe_in.lwe_size() == ksk.input_key_lwe_dimension().to_lwe_size());
    debug_assert!(lwe_out_list.lwe_size() == fourier_bsk.output_lwe_dimension().to_lwe_size());
    debug_assert!(ksk.output_lwe_size() == fourier_bsk.input_lwe_dimension().to_lwe_size());
    debug_assert!(lwe_in.ciphertext_modulus() == ksk.ciphertext_modulus());

    let ciphertext_modulus = lwe_in.ciphertext_modulus();
    let polynomial_size = fourier_bsk.polynomial_size();

    let num_bits = lwe_out_list.lwe_ciphertext_count().0;
    debug_assert!(modulus_sup == 1 << num_bits);

    let mut lwe_in_ks = LweCiphertext::new(Scalar::ZERO, ksk.output_lwe_size(), lwe_in.ciphertext_modulus());
    keyswitch_lwe_ciphertext(&ksk, &lwe_in, &mut lwe_in_ks);

    let lwe_msb = &mut lwe_out_list.get_mut(num_bits - 1);
    let sbox_msb = vec![Scalar::ONE; modulus_sup / 2];
    let accumulator = generate_negacyclic_accumulator_from_sbox_half(
        modulus_sup,
        ciphertext_modulus,
        fourier_bsk,
        Scalar::ONE << (Scalar::BITS - 2),
        &sbox_msb,
    );
    let accumulator = accumulator.as_view();

    let fft = Fft::new(polynomial_size);
    let fft = fft.as_view();

    let mut buffers = ComputationBuffers::new();
    buffers.resize(
        programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<Scalar>(
            fourier_bsk.glwe_size(),
            polynomial_size,
            fft,
        )
        .unwrap()
        .unaligned_bytes_required(),
    );
    let stack = buffers.stack();

    let (mut local_accumulator_data, stack) = stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
    let mut local_accumulator = GlweCiphertextMutView::from_container(
        &mut *local_accumulator_data,
        accumulator.polynomial_size(),
        accumulator.ciphertext_modulus(),
    );

    gen_blind_rotate_local_assign(
        fourier_bsk,
        local_accumulator.as_mut_view(),
        ModulusSwitchOffset(0),
        LutCountLog(0),
        lwe_in_ks.as_ref(),
        fft,
        stack,
    );
    extract_lwe_sample_from_glwe_ciphertext(&local_accumulator, lwe_msb, MonomialDegree(0));
    lwe_ciphertext_plaintext_sub_assign(lwe_msb, Plaintext(Scalar::ONE << (Scalar::BITS - 2)));

    let box_size = 2 * polynomial_size.0 / modulus_sup;
    for (bit_idx, mut lwe_out) in lwe_out_list.iter_mut().take(num_bits - 1).enumerate() {
        let multiplier = Polynomial::from_container((0..polynomial_size.0).map(|i| {
            if i == 0 {
                (((modulus_sup/2 - 1) & (1 << bit_idx)) >> bit_idx).cast_into()
            } else if i % box_size == 0 {
                let cur = i / box_size;
                let cur_bit = (cur & (1 << bit_idx)) >> bit_idx;
                let prev_bit = ((cur - 1) & (1 << bit_idx)) >> bit_idx;
                ((cur_bit - prev_bit) % 4).cast_into()
            } else {
                Scalar::ZERO
            }
        }).collect::<Vec<Scalar>>());

        let mut glwe_out = GlweCiphertext::new(Scalar::ZERO, fourier_bsk.glwe_size(), fourier_bsk.polynomial_size(), ciphertext_modulus);
        fourier_glwe_polynomial_mult(&mut glwe_out, &local_accumulator, &multiplier, modulus_sup);
        extract_lwe_sample_from_glwe_ciphertext(&glwe_out, &mut lwe_out, MonomialDegree(0));
    }
}
