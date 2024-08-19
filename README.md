# FRAST
This is an implementation of the FRAST cipher (proposed in [FRAST: TFHE-friendly Cipher Based on Random S-boxes](https://eprint.iacr.org/2024/745)) using TFHE-rs library.

## Tests
There are two tests: 'negacyclic_dbr' and 'frast'.
* 'negacyclic_dbr': test for evaluating double blind rotation
* 'frast': test for evaluating the frast cipher

To run each test, run the following command in the [frast](frast) directory:
```bash
cargo test --release --test `test_name`
```

## Benchmarks
There are foure benchmarks: 'bench_gen_pbs', 'bench_setup', 'bench_frast' and 'bench_online'.
* 'bench_gen_pbs': benchmark for generalized PBS
* 'bench_setup': benchmark for the setup phase
* 'bench_frast': benchmark for the frast cipher in the offline phase
* 'bench_online': benchmark for the online phase

To run each benchmark, run the following command in the [frast](frast) directory:
```bash
cargo bench --bench `bench_name`
```

## Error Analysis
The error anlaysis of the frast evaluation is given in [err.sage](./err.sage).
The failure probability of the FRAST evaluation is given by 2^-80.32.
