use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;

pub fn sample_multivariate_normal_repeatedly<R: Rng>(
    n_samples: usize,
    mean: DVector<f64>,
    precision: DMatrix<f64>,
    rng: &mut R,
) -> Option<DMatrix<f64>> {
    let lit = match prepare(precision) {
        None => return None,
        Some(lit) => lit,
    };
    let u = DMatrix::from_fn(lit.nrows(), n_samples, |_, _| StandardNormal.sample(rng));
    let mut result = lit * u;
    for mut col in result.column_iter_mut() {
        col += &mean;
    }
    Some(result)
}

#[allow(dead_code)]
pub fn sample_multivariate_normal<R: Rng>(
    mean: DVector<f64>,
    precision: DMatrix<f64>,
    rng: &mut R,
) -> Option<DVector<f64>> {
    let lit = match prepare(precision) {
        None => return None,
        Some(lit) => lit,
    };
    let u = DVector::from_fn(lit.nrows(), |_, _| StandardNormal.sample(rng));
    let result = mean + lit * u;
    Some(result)
}

#[allow(dead_code)]
pub fn sample_multivariate_normal_v2<R: Rng>(
    mut precision_times_mean: DVector<f64>,
    precision: DMatrix<f64>,
    rng: &mut R,
) -> Option<DVector<f64>> {
    let n = precision.nrows();
    if !precision.is_square() || precision.nrows() != n {
        return None;
    }
    let chol = match precision.cholesky() {
        None => return None,
        Some(chol) => chol,
    };
    chol.solve_mut(&mut precision_times_mean);
    let mut l_inverse = DMatrix::identity(n, n);
    chol.l_dirty()
        .solve_lower_triangular_unchecked_mut(&mut l_inverse);
    let lit = l_inverse.transpose();
    let u = DVector::from_fn(lit.nrows(), |_, _| StandardNormal.sample(rng));
    let result = precision_times_mean + lit * u;
    Some(result)
}

fn prepare(precision: DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = precision.nrows();
    if !precision.is_square() || precision.nrows() != n {
        return None;
    }
    let chol = match precision.cholesky() {
        None => return None,
        Some(chol) => chol,
    };
    let mut l_inverse = DMatrix::identity(n, n);
    chol.l_dirty()
        .solve_lower_triangular_unchecked_mut(&mut l_inverse);
    Some(l_inverse.transpose())
}
