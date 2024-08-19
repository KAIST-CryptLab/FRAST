use tfhe::core_crypto::prelude::*;

pub fn fourier_polynomial_torus_integer_mult<Scalar, ContOut, ContLhs, ContRhs>(
    out: &mut Polynomial<ContOut>,
    lhs_torus: &Polynomial<ContLhs>,
    rhs_int: &Polynomial<ContRhs>,
    modulus_sup: usize,
) where
    Scalar: UnsignedTorus,
    ContOut: ContainerMut<Element=Scalar>,
    ContLhs: Container<Element=Scalar>,
    ContRhs: Container<Element=Scalar>,
{
    debug_assert!(out.polynomial_size() == lhs_torus.polynomial_size());
    debug_assert!(out.polynomial_size() == rhs_int.polynomial_size());

    let polynomial_size = out.polynomial_size();
    let wrapper_glwe_size = GlweSize(1);
    let wrapper_decomp_base_log = DecompositionBaseLog(modulus_sup);
    let wrapper_decomp_level_count = DecompositionLevelCount(1);
    let ciphertext_modulus = CiphertextModulus::new_native();

    out.as_mut().fill(Scalar::ZERO);
    let mut glwe_out = GlweCiphertextMutView::from_container(
        out.as_mut(),
        polynomial_size,
        ciphertext_modulus,
    );

    let wrapper_ggsw_lhs = GgswCiphertext::from_container(
        lhs_torus.as_ref(),
        wrapper_glwe_size,
        polynomial_size,
        wrapper_decomp_base_log,
        ciphertext_modulus,
    );
    let mut wrapper_fourier_ggsw_lhs = FourierGgswCiphertext::new(
        wrapper_glwe_size,
        polynomial_size,
        wrapper_decomp_base_log,
        wrapper_decomp_level_count,
    );
    convert_standard_ggsw_ciphertext_to_fourier(&wrapper_ggsw_lhs, &mut wrapper_fourier_ggsw_lhs);

    let wrapper_glwe_rhs = GlweCiphertext::from_container(
        (0..polynomial_size.0).map(|i| {
            let int_val = *rhs_int.as_ref().get(i).unwrap();
            int_val << (Scalar::BITS - wrapper_decomp_base_log.0)
        }).collect::<Vec<Scalar>>(),
        polynomial_size,
        ciphertext_modulus,
    );

    let fft = Fft::new(polynomial_size);
    let fft = fft.as_view();

    let mut buffers = ComputationBuffers::new();
    buffers.resize(
        add_external_product_assign_mem_optimized_requirement::<Scalar>(
            wrapper_glwe_size,
            polynomial_size,
            fft,
        )
        .unwrap()
        .unaligned_bytes_required(),
    );
    let stack = buffers.stack();

    add_external_product_assign_mem_optimized(
        &mut glwe_out,
        &wrapper_fourier_ggsw_lhs,
        &wrapper_glwe_rhs,
        fft,
        stack,
    );
}

pub fn fourier_glwe_polynomial_mult<Scalar, ContOut, ContLhs, ContRhs>(
    out: &mut GlweCiphertext<ContOut>,
    lhs: &GlweCiphertext<ContLhs>,
    rhs: &Polynomial<ContRhs>,
    modulus_sup: usize,
) where 
    Scalar: UnsignedTorus,
    ContOut: ContainerMut<Element=Scalar>,
    ContLhs: Container<Element=Scalar>,
    ContRhs: Container<Element=Scalar>,
{
    debug_assert!(out.polynomial_size() == lhs.polynomial_size());
    debug_assert!(out.glwe_size() == lhs.glwe_size());
    debug_assert!(lhs.polynomial_size() == rhs.polynomial_size());
    debug_assert!(out.ciphertext_modulus() == lhs.ciphertext_modulus());

    out.as_mut().fill(Scalar::ZERO);
    for (mut out_poly, lhs_poly) in out.as_mut_polynomial_list().iter_mut().zip(lhs.as_polynomial_list().iter()) {
        fourier_polynomial_torus_integer_mult(&mut out_poly, &lhs_poly, rhs, modulus_sup);
    }
}
