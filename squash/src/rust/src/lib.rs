mod data;
mod hyperparameters;
mod mvnorm;
mod registration;
mod state;

use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::sample_multivariate_normal_repeatedly;
use crate::state::{State, StateFixedComponents};
use dahl_randompartition::cpp::CppParameters;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::epa::EpaParameters;
use dahl_randompartition::fixed::FixedPartitionParameters;
use dahl_randompartition::frp::FrpParameters;
use dahl_randompartition::jlp::JlpParameters;
use dahl_randompartition::lsp::LspParameters;
use dahl_randompartition::oldsp::OldSpParameters;
use dahl_randompartition::sp::SpParameters;
use dahl_randompartition::up::UpParameters;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use roxido::*;

#[roxido]
fn state_r2rust(state: Rval) -> Rval {
    Rval::external_pointer_encode(State::from_r(state, &mut pc), Rval::new("state", &mut pc))
}

#[roxido]
fn state_rust2r_as_reference(state: Rval) -> Rval {
    Rval::external_pointer_decode_as_reference::<State>(state).to_r(&mut pc)
}

#[roxido]
fn hyperparameters_r2rust(hyperparameters: Rval) -> Rval {
    Rval::external_pointer_encode(
        Hyperparameters::from_r(hyperparameters, &mut pc),
        Rval::new("hyperparameters", &mut pc),
    )
}

#[roxido]
fn data_r2rust(data: Rval) -> Rval {
    Rval::external_pointer_encode(Data::from_r(data, &mut pc), Rval::new("data", &mut pc))
}

#[roxido]
fn rust_free(x: Rval) -> Rval {
    match x.external_pointer_tag().as_str() {
        "data" => {
            let _ = x.external_pointer_decode::<Data>();
        }
        "state" => {
            let _ = x.external_pointer_decode::<State>();
        }
        "hyperparameters" => {
            let _ = x.external_pointer_decode::<Hyperparameters>();
        }
        str => {
            panic!("Unrecognized type ID: {}.", str)
        }
    };
    Rval::nil()
}

#[roxido]
fn fit(
    n_updates: Rval,
    data: Rval,
    state: Rval,
    fixed: Rval,
    hyperparameters: Rval,
    partition_distribution: Rval,
) -> Rval {
    let n_updates = n_updates.as_usize();
    let data = Rval::external_pointer_decode_as_reference::<Data>(data);
    let state_tag = state.external_pointer_tag();
    let mut state = Rval::external_pointer_decode::<State>(state);
    let fixed = StateFixedComponents::from_r(fixed, &mut pc);
    let hyperparameters =
        Rval::external_pointer_decode_as_reference::<Hyperparameters>(hyperparameters);
    if data.n_global_covariates() != state.n_global_covariates()
        || hyperparameters.n_global_covariates() != state.n_global_covariates()
    {
        panic!("Inconsistent number of global covariates.")
    }
    if data.n_clustered_covariates() != state.n_clustered_covariates()
        || hyperparameters.n_clustered_covariates() != state.n_clustered_covariates()
    {
        panic!("Inconsistent number of clustered covariates.")
    }
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rng2 = Pcg64Mcg::from_seed(seed);
    macro_rules! distr_macro {
        ($tipe:ty) => {{
            let partition_distribution =
                partition_distribution.external_pointer_decode_as_reference::<$tipe>();
            for _ in 0..n_updates {
                state = state.mcmc_iteration(
                    &fixed,
                    &data,
                    &hyperparameters,
                    partition_distribution,
                    &mut rng,
                    &mut rng2,
                );
            }
        }};
    }
    let prior_name = partition_distribution.external_pointer_tag().as_str();
    match prior_name {
        "fixed" => distr_macro!(FixedPartitionParameters),
        "up" => distr_macro!(UpParameters),
        "jlp" => distr_macro!(JlpParameters),
        "crp" => distr_macro!(CrpParameters),
        "epa" => distr_macro!(EpaParameters),
        "lsp" => distr_macro!(LspParameters),
        "cpp-up" => distr_macro!(CppParameters<UpParameters>),
        "cpp-jlp" => distr_macro!(CppParameters<JlpParameters>),
        "cpp-crp" => distr_macro!(CppParameters<CrpParameters>),
        "frp" => distr_macro!(FrpParameters),
        "oldsp-up" => distr_macro!(OldSpParameters<UpParameters>),
        "oldsp-jlp" => distr_macro!(OldSpParameters<JlpParameters>),
        "oldsp-crp" => distr_macro!(OldSpParameters<CrpParameters>),
        "sp-up" => distr_macro!(SpParameters<UpParameters>),
        "sp-jlp" => distr_macro!(SpParameters<JlpParameters>),
        "sp-crp" => distr_macro!(SpParameters<CrpParameters>),
        _ => panic!("Unsupported distribution: {}", prior_name),
    }
    state = state.canonicalize();
    Rval::external_pointer_encode(state, state_tag)
}

#[roxido]
fn sample_multivariate_normal(n_samples: Rval, mean: Rval, precision: Rval) -> Rval {
    let n_samples = n_samples.as_usize();
    let n = mean.len();
    let (_, mean) = mean.coerce_double(&mut pc).unwrap();
    let mean = DVector::from_iterator(n, mean.iter().cloned());
    let (_, precision) = precision.coerce_double(&mut pc).unwrap();
    let precision = DMatrix::from_iterator(n, n, precision.iter().cloned());
    let (rval, slice) = Rval::new_matrix_double(n, n_samples, &mut pc);
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let x = sample_multivariate_normal_repeatedly(n_samples, mean, precision, &mut rng).unwrap();
    slice.clone_from_slice(x.as_slice());
    rval.transpose(&mut pc)
}

#[roxido]
fn ready() -> Rval {
    Rval::new(true, &mut pc)
}

#[roxido]
fn myrnorm(n: Rval, mean: Rval, sd: Rval) -> Rval {
    unsafe {
        use rbindings::*;
        use std::convert::TryFrom;
        let (mean, sd) = (Rf_asReal(mean.0), Rf_asReal(sd.0));
        let length = isize::try_from(Rf_asInteger(n.0)).unwrap();
        let vec = Rf_protect(Rf_allocVector(REALSXP, length));
        let slice = Rval(vec).slice_double().unwrap();
        GetRNGstate();
        for x in slice {
            *x = Rf_rnorm(mean, sd);
        }
        PutRNGstate();
        Rf_unprotect(1);
        Rval(vec)
    }
}

#[roxido]
fn convolve2(a: Rval, b: Rval) -> Rval {
    let (a, xa) = a.coerce_double(&mut pc).unwrap();
    let (b, xb) = b.coerce_double(&mut pc).unwrap();
    let (ab, xab) = Rval::new_vector_double(a.len() + b.len() - 1, &mut pc);
    for xabi in xab.iter_mut() {
        *xabi = 0.0
    }
    for (i, xai) in xa.iter().enumerate() {
        for (j, xbj) in xb.iter().enumerate() {
            xab[i + j] += xai * xbj;
        }
    }
    ab
}

#[roxido]
fn zero(f: Rval, guesses: Rval, stol: Rval, rho: Rval) -> Rval {
    let slice = guesses.slice_double().unwrap();
    let (mut x0, mut x1, tol) = (slice[0], slice[1], stol.as_f64());
    if tol <= 0.0 {
        panic!("non-positive tol value");
    }
    let symbol = Rval::new_symbol("x", &mut pc);
    let feval = |x: f64| {
        let mut pc = Pc::new();
        symbol.assign(Rval::new(x, &mut pc), rho);
        f.eval(rho, &mut pc).unwrap().as_f64()
    };
    let mut f0 = feval(x0);
    if f0 == 0.0 {
        return Rval::new(x0, &mut pc);
    }
    let f1 = feval(x1);
    if f1 == 0.0 {
        return Rval::new(x1, &mut pc);
    }
    if f0 * f1 > 0.0 {
        panic!("x[0] and x[1] have the same sign");
    }
    loop {
        let xc = 0.5 * (x0 + x1);
        if (x0 - x1).abs() < tol {
            return Rval::new(xc, &mut pc);
        }
        let fc = feval(xc);
        if fc == 0.0 {
            return Rval::new(xc, &mut pc);
        }
        if f0 * fc > 0.0 {
            x0 = xc;
            f0 = fc;
        } else {
            x1 = xc;
        }
    }
}
