use nalgebra::{DMatrix, DVector};
use roxido::*;

#[allow(dead_code)]
pub struct Data {
    response: DVector<f64>,
    global_covariates: DMatrix<f64>,
    global_covariates_transpose: DMatrix<f64>,
    global_covariates_transpose_times_self: DMatrix<f64>,
    clustered_covariates: DMatrix<f64>,
    missing_items: Option<Vec<usize>>,
}

impl Data {
    #[allow(dead_code)]
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

    pub fn missing_items(&self) -> &Option<Vec<usize>> {
        &self.missing_items
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

    #[allow(dead_code)]
    pub fn declare_missing(&mut self, items: Vec<usize>) {
        self.missing_items = if items.is_empty() {
            None
        } else {
            let max = *items.iter().max().unwrap();
            if max >= self.n_items() {
                panic!("Missing indices are out of bounds.")
            }
            Some(items)
        }
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
}
