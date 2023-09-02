use nalgebra::{DMatrix, DVector};
use roxido::*;

#[derive(Debug)]
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
    pub shrinkage_reference: Option<usize>,
    pub shrinkage_shape: Option<f64>,
    pub shrinkage_rate: Option<f64>,
}

impl Hyperparameters {
    pub fn new(
        precision_response_shape: f64,
        precision_response_rate: f64,
        global_coefficients_mean: DVector<f64>,
        global_coefficients_precision: DMatrix<f64>,
        clustered_coefficients_mean: DVector<f64>,
        clustered_coefficients_precision: DMatrix<f64>,
        shrinkage_reference: Option<usize>,
        shrinkage_shape: Option<f64>,
        shrinkage_rate: Option<f64>,
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
            shrinkage_reference,
            shrinkage_shape,
            shrinkage_rate,
        })
    }

    pub fn from_r(hyperparameters: RObject, pc: &mut Pc) -> Self {
        let hyperparameters = hyperparameters.as_list().stop();
        let precision_response_shape = hyperparameters.get(0).unwrap().as_f64().stop();
        let precision_response_rate = hyperparameters.get(1).unwrap().as_f64().stop();
        let global_coefficients_mean = hyperparameters
            .get(2)
            .unwrap()
            .as_vector()
            .stop()
            .to_mode_double(pc);
        let global_coefficients_mean_slice = global_coefficients_mean.slice();
        let global_coefficients_mean = DVector::from_column_slice(global_coefficients_mean_slice);
        let global_coefficients_precision_rval = hyperparameters
            .get(3)
            .stop()
            .as_matrix()
            .stop()
            .to_mode_double(pc);

        let global_coefficients_precision_slice = global_coefficients_precision_rval.slice();
        let global_coefficients_precision = DMatrix::from_column_slice(
            global_coefficients_precision_rval.nrow(),
            global_coefficients_precision_rval.ncol(),
            global_coefficients_precision_slice,
        );
        let clustered_coefficients_mean_slice = hyperparameters
            .get(4)
            .stop()
            .as_vector()
            .stop()
            .to_mode_double(pc)
            .slice();
        let clustered_coefficients_mean =
            DVector::from_column_slice(clustered_coefficients_mean_slice);
        let clustered_coefficients_precision_rval = hyperparameters
            .get(5)
            .stop()
            .as_matrix()
            .stop()
            .to_mode_double(pc);
        let clustered_coefficients_precision_slice = clustered_coefficients_precision_rval.slice();
        let clustered_coefficients_precision = DMatrix::from_column_slice(
            clustered_coefficients_precision_rval.nrow(),
            clustered_coefficients_precision_rval.ncol(),
            clustered_coefficients_precision_slice,
        );
        let shrinkage_reference = hyperparameters.get(6).stop().as_usize().stop() - 1;
        let shrinkage_shape = hyperparameters.get(7).unwrap().as_f64();
        let shrinkage_rate = hyperparameters.get(8).unwrap().as_f64();
        fn wrap(x: f64) -> Option<f64> {
            if x.is_nan() || x.is_infinite() || x <= 0.0 {
                None
            } else {
                Some(x)
            }
        }
        Self::new(
            precision_response_shape,
            precision_response_rate,
            global_coefficients_mean,
            global_coefficients_precision,
            clustered_coefficients_mean,
            clustered_coefficients_precision,
            Some(shrinkage_reference),
            wrap(shrinkage_shape.stop()),
            wrap(shrinkage_rate.stop()),
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
