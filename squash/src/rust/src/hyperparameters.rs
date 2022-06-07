use nalgebra::{DMatrix, DVector};
use roxido::*;

#[allow(dead_code)]
pub struct Hyperparameters {
    precision_response_shape: f64,
    precision_response_rate: f64,
    global_coefficients_mean: DVector<f64>,
    global_coefficients_precision: DMatrix<f64>,
    global_coefficients_precision_times_mean: DVector<f64>,
    clustered_coefficients_mean: DVector<f64>,
    clustered_coefficients_precision: DMatrix<f64>,
    clustered_coefficients_precision_times_mean: DVector<f64>,
    clustered_coefficients_precision_l_inv_transpose: DMatrix<f64>,
}

impl Hyperparameters {
    #[allow(dead_code)]
    pub fn new(
        precision_response_shape: f64,
        precision_response_rate: f64,
        global_coefficients_mean: DVector<f64>,
        global_coefficients_precision: DMatrix<f64>,
        clustered_coefficients_mean: DVector<f64>,
        clustered_coefficients_precision: DMatrix<f64>,
    ) -> Option<Self> {
        if precision_response_shape <= 0.0 {
            return None;
        }
        if precision_response_rate <= 0.0 {
            return None;
        }
        if global_coefficients_mean.len() != global_coefficients_mean.nrows() {
            return None;
        }
        if !global_coefficients_precision.is_square() {
            return None;
        }
        // Check of positive semi-definiteness?
        if clustered_coefficients_mean.len() != clustered_coefficients_mean.nrows() {
            return None;
        }
        if !clustered_coefficients_precision.is_square() {
            return None;
        }
        // Check of positive semi-definiteness?
        let global_coefficients_precision_times_mean =
            global_coefficients_precision.clone() * &global_coefficients_mean;
        let clustered_coefficients_precision_times_mean =
            clustered_coefficients_precision.clone() * &clustered_coefficients_mean;
        let clustered_coefficients_precision_l_inv_transpose =
            match crate::mvnorm::prepare(clustered_coefficients_precision.clone()) {
                Some(lit) => lit,
                None => return None,
            };
        Some(Self {
            precision_response_shape,
            precision_response_rate,
            global_coefficients_mean,
            global_coefficients_precision,
            global_coefficients_precision_times_mean,
            clustered_coefficients_mean,
            clustered_coefficients_precision,
            clustered_coefficients_precision_times_mean,
            clustered_coefficients_precision_l_inv_transpose,
        })
    }

    pub fn from_r(hyperparameters: Rval, pc: &mut Pc) -> Self {
        let precision_response_shape = hyperparameters.get_list_element(0).as_f64();
        let precision_response_rate = hyperparameters.get_list_element(1).as_f64();
        let (_, global_coefficients_mean_slice) = hyperparameters
            .get_list_element(2)
            .coerce_double(pc)
            .unwrap();
        let global_coefficients_mean = DVector::from_column_slice(global_coefficients_mean_slice);
        let (global_coefficients_precision_rval, global_coefficients_precision_slice) =
            hyperparameters
                .get_list_element(3)
                .coerce_double(pc)
                .unwrap();
        let global_coefficients_precision = DMatrix::from_column_slice(
            global_coefficients_precision_rval.nrow(),
            global_coefficients_precision_rval.ncol(),
            global_coefficients_precision_slice,
        );
        let (_, clustered_coefficients_mean_slice) = hyperparameters
            .get_list_element(4)
            .coerce_double(pc)
            .unwrap();
        let clustered_coefficients_mean =
            DVector::from_column_slice(clustered_coefficients_mean_slice);
        let (clustered_coefficients_precision_rval, clustered_coefficients_precision_slice) =
            hyperparameters
                .get_list_element(5)
                .coerce_double(pc)
                .unwrap();
        let clustered_coefficients_precision = DMatrix::from_column_slice(
            clustered_coefficients_precision_rval.nrow(),
            clustered_coefficients_precision_rval.ncol(),
            clustered_coefficients_precision_slice,
        );
        Self::new(
            precision_response_shape,
            precision_response_rate,
            global_coefficients_mean,
            global_coefficients_precision,
            clustered_coefficients_mean,
            clustered_coefficients_precision,
        )
        .unwrap()
    }

    pub fn n_global_covariates(&self) -> usize {
        self.global_coefficients_mean.len()
    }

    pub fn n_clustered_covariates(&self) -> usize {
        self.clustered_coefficients_mean.len()
    }

    pub fn precision_response_shape(&self) -> f64 {
        self.precision_response_shape
    }

    pub fn precision_response_rate(&self) -> f64 {
        self.precision_response_rate
    }

    pub fn global_coefficients_precision(&self) -> &DMatrix<f64> {
        &self.global_coefficients_precision
    }

    pub fn global_coefficients_precision_times_mean(&self) -> &DVector<f64> {
        &self.global_coefficients_precision_times_mean
    }

    pub fn clustered_coefficients_mean(&self) -> &DVector<f64> {
        &self.clustered_coefficients_mean
    }

    pub fn clustered_coefficients_precision(&self) -> &DMatrix<f64> {
        &self.clustered_coefficients_precision
    }

    pub fn clustered_coefficients_precision_times_mean(&self) -> &DVector<f64> {
        &self.clustered_coefficients_precision_times_mean
    }

    pub fn clustered_coefficients_precision_l_inv_transpose(&self) -> &DMatrix<f64> {
        &self.clustered_coefficients_precision_l_inv_transpose
    }
}
