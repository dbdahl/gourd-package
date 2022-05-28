use nalgebra::{DMatrix, DVector};

pub struct Data {
    response: DVector<f64>,
    global_covariates: DMatrix<f64>,
    global_covariates_transpose: DMatrix<f64>,
    global_covariates_transpose_times_self: DMatrix<f64>,
    clustered_covariates: DMatrix<f64>,
    missing_items: Option<Vec<usize>>,
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
            missing_items: None,
        })
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
        self.missing_items = Some(items);
    }
    pub fn n_items(&self) -> usize {
        self.global_covariates.nrows()
    }
    pub fn n_global_coefficients(&self) -> usize {
        self.global_covariates.ncols()
    }
    pub fn n_clustered_coefficients(&self) -> usize {
        self.clustered_covariates.ncols()
    }
}
