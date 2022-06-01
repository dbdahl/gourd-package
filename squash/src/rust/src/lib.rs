mod data;
mod hyperparameters;
mod mvnorm;
mod registration;
mod state;

use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::sample_multivariate_normal_repeatedly;
use crate::state::State;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use roxido::*;

#[roxido]
fn state_r2rust(state: Rval) -> Rval {
    Rval::external_pointer_encode(State::from_r(state, &mut pc))
}

#[roxido]
fn state_rust2r(state: Rval) -> Rval {
    Rval::external_pointer_decode::<State>(state).to_r(&mut pc)
}

#[roxido]
fn mk_data(response: Rval, global_covariates: Rval, clustered_covariates: Rval) -> Rval {
    let (_, response_slice) = response.coerce_double(&mut pc).unwrap();
    let n_items = response_slice.len();
    let response = DVector::from_column_slice(response_slice);
    let (_, global_covariates_slice) = global_covariates.coerce_double(&mut pc).unwrap();
    let n_global_covariates = global_covariates.ncol();
    let global_covariates =
        DMatrix::from_column_slice(n_items, n_global_covariates, global_covariates_slice);
    let (_, clustered_covariates_slice) = clustered_covariates.coerce_double(&mut pc).unwrap();
    let n_clustered_covariates = clustered_covariates.ncol();
    let clustered_covariates =
        DMatrix::from_column_slice(n_items, n_clustered_covariates, clustered_covariates_slice);
    let data = Data::new(response, global_covariates, clustered_covariates).unwrap();
    Rval::external_pointer_encode(data)
}

#[roxido]
fn mk_hyperparameters(
    precision_response_shape: Rval,
    precision_response_rate: Rval,
    global_coefficients_mean: Rval,
    global_coefficients_precision: Rval,
    clustered_coefficients_mean: Rval,
    clustered_coefficients_precision: Rval,
) -> Rval {
    let precision_response_shape = precision_response_shape.as_f64();
    let precision_response_rate = precision_response_rate.as_f64();
    let global_coefficients_mean =
        DVector::from_column_slice(global_coefficients_mean.coerce_double(&mut pc).unwrap().1);
    let global_coefficients_precision = DMatrix::from_column_slice(
        global_coefficients_mean.len(),
        global_coefficients_mean.len(),
        global_coefficients_precision
            .coerce_double(&mut pc)
            .unwrap()
            .1,
    );
    let clustered_coefficients_mean = DVector::from_column_slice(
        clustered_coefficients_mean
            .coerce_double(&mut pc)
            .unwrap()
            .1,
    );
    let clustered_coefficients_precision = DMatrix::from_column_slice(
        clustered_coefficients_mean.len(),
        clustered_coefficients_mean.len(),
        clustered_coefficients_precision
            .coerce_double(&mut pc)
            .unwrap()
            .1,
    );
    let hyperparameters = Hyperparameters::new(
        precision_response_shape,
        precision_response_rate,
        global_coefficients_mean,
        global_coefficients_precision,
        clustered_coefficients_mean,
        clustered_coefficients_precision,
    )
    .unwrap();
    Rval::external_pointer_encode(hyperparameters)
}

#[roxido]
fn fit(n_updates: Rval, state: Rval, data: Rval, hyperparameters: Rval) -> Rval {
    let n_updates = n_updates.as_usize();
    let mut state = Rval::external_pointer_decode::<State>(state);
    let data = Rval::external_pointer_decode::<Data>(data);
    let hyperparameters = Rval::external_pointer_decode::<Hyperparameters>(hyperparameters);
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    for _ in 0..n_updates {
        state = state.mcmc_iteration(&data, &hyperparameters, &mut rng);
    }
    Rval::external_pointer_encode(state)
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
