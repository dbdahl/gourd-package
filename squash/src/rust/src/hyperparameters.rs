use na::{DMatrix, DVector};
use nalgebra as na;

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
        Some(Self {
            precision_response_shape,
            precision_response_rate,
            global_coefficients_mean,
            global_coefficients_precision,
            global_coefficients_precision_times_mean,
            clustered_coefficients_mean,
            clustered_coefficients_precision,
            clustered_coefficients_precision_times_mean,
        })
    }

    #[allow(dead_code)]
    pub fn precision_response_shape(&self) -> f64 {
        self.precision_response_shape
    }

    #[allow(dead_code)]
    pub fn precision_response_rate(&self) -> f64 {
        self.precision_response_rate
    }

    #[allow(dead_code)]
    pub fn global_coefficients_precision(&self) -> &DMatrix<f64> {
        &self.global_coefficients_precision
    }

    #[allow(dead_code)]
    pub fn global_coefficients_precision_times_mean(&self) -> &DVector<f64> {
        &self.global_coefficients_precision_times_mean
    }

    #[allow(dead_code)]
    pub fn clustered_coefficients_mean(&self) -> &DVector<f64> {
        &self.clustered_coefficients_mean
    }

    #[allow(dead_code)]
    pub fn clustered_coefficients_precision(&self) -> &DMatrix<f64> {
        &self.clustered_coefficients_precision
    }

    #[allow(dead_code)]
    pub fn clustered_coefficients_precision_times_mean(&self) -> &DVector<f64> {
        &self.clustered_coefficients_precision_times_mean
    }
}
