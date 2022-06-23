use crate::state::State;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;
use roxido::*;

#[derive(Debug)]
pub struct Data {
    response: DVector<f64>,
    global_covariates: DMatrix<f64>,
    global_covariates_transpose: DMatrix<f64>,
    global_covariates_transpose_times_self: DMatrix<f64>,
    clustered_covariates: DMatrix<f64>,
    missing: Vec<(usize, f64)>,
}

impl Data {
    pub fn new(
        response: DVector<f64>,
        global_covariates: DMatrix<f64>,
        clustered_covariates: DMatrix<f64>,
    ) -> Option<Self> {
        let n_items = response.nrows();
        if global_covariates.nrows() != n_items || clustered_covariates.nrows() != n_items {
            return None;
        }
        let global_covariates_transpose = global_covariates.transpose();
        let global_covariates_transpose_times_self =
            global_covariates_transpose.clone() * &global_covariates;
        Some(Self {
            response,
            global_covariates,
            global_covariates_transpose,
            global_covariates_transpose_times_self,
            clustered_covariates,
            missing: Vec::new(),
        })
    }

    pub fn from_r(data: Rval, pc: &mut Pc) -> Self {
        let (_, response_slice) = data.get_list_element(0).coerce_double(pc).unwrap();
        let n_items = response_slice.len();
        let response = DVector::from_column_slice(response_slice);
        let (global_covariates_rval, global_covariates_slice) =
            data.get_list_element(1).coerce_double(pc).unwrap();
        let n_global_covariates = global_covariates_rval.ncol();
        let global_covariates =
            DMatrix::from_column_slice(n_items, n_global_covariates, global_covariates_slice);
        let (clustered_covariates_rval, clustered_covariates_slice) =
            data.get_list_element(2).coerce_double(pc).unwrap();
        let n_clustered_covariates = clustered_covariates_rval.ncol();
        let clustered_covariates =
            DMatrix::from_column_slice(n_items, n_clustered_covariates, clustered_covariates_slice);
        Data::new(response, global_covariates, clustered_covariates).unwrap()
    }

    pub fn impute<T: Rng>(&mut self, state: &State, rng: &mut T) {
        if !self.missing.is_empty() {
            let stdev = 1.0 / state.precision_response().sqrt();
            let normal = Normal::new(0.0, stdev).unwrap();
            for &(item, _) in &self.missing {
                let label = state.clustering().allocation()[item];
                let parameter = &state.clustered_coefficients()[label];
                let mean = (self.global_covariates().row(item) * state.global_coefficients())
                    .index((0, 0))
                    + (self.clustered_covariates().row(item) * parameter).index((0, 0));
                self.response[item] = mean + normal.sample(rng)
            }
        }
    }

    pub fn response(&self) -> &DVector<f64> {
        &self.response
    }

    pub fn global_covariates(&self) -> &DMatrix<f64> {
        &self.global_covariates
    }

    pub fn global_covariates_transpose(&self) -> &DMatrix<f64> {
        &self.global_covariates_transpose
    }

    pub fn global_covariates_transpose_times_self(&self) -> &DMatrix<f64> {
        &self.global_covariates_transpose_times_self
    }

    pub fn clustered_covariates(&self) -> &DMatrix<f64> {
        &self.clustered_covariates
    }

    pub fn declare_missing(&mut self, items: Vec<usize>) {
        for &(index, value) in &self.missing {
            self.response[index] = value
        }
        self.missing = items
            .iter()
            .enumerate()
            .map(|(index, &item)| (index, self.response[item]))
            .collect();
    }

    pub fn n_items(&self) -> usize {
        self.global_covariates.nrows()
    }

    pub fn n_global_covariates(&self) -> usize {
        self.global_covariates.ncols()
    }

    pub fn n_clustered_covariates(&self) -> usize {
        self.clustered_covariates.ncols()
    }

    pub fn missing(&self) -> &[(usize, f64)] {
        &self.missing
    }
}
