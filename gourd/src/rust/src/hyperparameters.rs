use crate::validate_list;
use dahl_randompartition::prelude::*;
use nalgebra::{DMatrix, DVector};
use roxido::*;

#[derive(Debug)]
pub struct Hyperparameters {
    precision_response_shape: Shape,
    precision_response_rate: Rate,
    global_coefficients_mean: DVector<f64>,
    global_coefficients_precision: DMatrix<f64>,
    global_coefficients_precision_times_mean: DVector<f64>,
    clustered_coefficients_mean: DVector<f64>,
    clustered_coefficients_precision: DMatrix<f64>,
    clustered_coefficients_precision_times_mean: DVector<f64>,
    clustered_coefficients_precision_l_inv_transpose: DMatrix<f64>,
    pub shrinkage_option: Option<ShrinkageHyperparameters>,
    pub grit_option: Option<GritHyperparameters>,
}

#[derive(Debug)]
pub struct ShrinkageHyperparameters {
    pub reference: Option<usize>,
    pub shape: Shape,
    pub rate: Rate,
}

impl ShrinkageHyperparameters {
    pub fn from_r(shrinkage: RObject) -> ShrinkageHyperparameters {
        let list = validate_list(shrinkage, &["reference", "shape", "rate"], "shrinkage");
        let reference_rval = list.get(0).stop();
        let reference = if reference_rval.is_null() {
            None
        } else {
            Some(
                reference_rval
                    .as_usize()
                    .stop_str("Shrinkage reference should be an integer")
                    - 1,
            )
        };
        let shape = Shape::new(
            list.get(1)
                .unwrap()
                .as_f64()
                .stop_str("Shrinkage shape should be a numeric value"),
        )
        .unwrap_or_else(|| stop!("Shape of shrinkage is not valid"));
        let rate = Rate::new(
            list.get(2)
                .unwrap()
                .as_f64()
                .stop_str("Shrinkage rate should be a numeric value"),
        )
        .unwrap_or_else(|| stop!("Rate of shrinkage is not valid"));
        ShrinkageHyperparameters {
            reference,
            shape,
            rate,
        }
    }
}

#[derive(Debug)]
pub struct GritHyperparameters {
    pub shape1: Shape,
    pub shape2: Shape,
}

impl GritHyperparameters {
    pub fn from_r(grit: RObject) -> GritHyperparameters {
        let list = validate_list(grit, &["shape1", "shape2"], "grit");
        let shape1 = Shape::new(
            list.get(0)
                .unwrap()
                .as_f64()
                .stop_str("Grit shape should be a numeric value"),
        )
        .unwrap_or_else(|| stop!("Shape 1 of grit is not valid"));
        let shape2 = Shape::new(
            list.get(1)
                .unwrap()
                .as_f64()
                .stop_str("Grit shape should be a numeric value"),
        )
        .unwrap_or_else(|| stop!("Shape 2 of grit is not valid"));
        GritHyperparameters { shape1, shape2 }
    }
}

impl Hyperparameters {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        precision_response_shape: Shape,
        precision_response_rate: Rate,
        global_coefficients_mean: DVector<f64>,
        global_coefficients_precision: DMatrix<f64>,
        clustered_coefficients_mean: DVector<f64>,
        clustered_coefficients_precision: DMatrix<f64>,
        shrinkage_option: Option<ShrinkageHyperparameters>,
        grit_option: Option<GritHyperparameters>,
    ) -> Option<Self> {
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
            shrinkage_option,
            grit_option,
        })
    }

    pub fn from_r(hyperparameters: RObject, pc: &mut Pc) -> Self {
        let hyperparameters = validate_list(
            hyperparameters,
            &[
                "precision_response_shape",
                "precision_response_rate",
                "global_coefficients_mean",
                "global_coefficients_precision",
                "clustered_coefficients_mean",
                "clustered_coefficients_precision",
                "shrinkage",
                "grit",
            ],
            "hyperparameters",
        );
        let precision_response_shape = Shape::new(
            hyperparameters
                .get(0)
                .unwrap()
                .as_f64()
                .stop_str("Invalid rate parameter for precision response"),
        )
        .unwrap_or_else(|| stop!("Invalid shape parameter for precision response"));
        let precision_response_rate = Rate::new(
            hyperparameters
                .get(1)
                .unwrap()
                .as_f64()
                .stop_str("Invalid rate parameter for precision response"),
        )
        .unwrap_or_else(|| stop!("Invalid rate parameter for precision response"));
        let global_coefficients_mean = DVector::from_column_slice(
            hyperparameters
                .get(2)
                .unwrap()
                .as_vector()
                .stop_str("Invalid global coefficients mean")
                .to_mode_double(pc)
                .slice(),
        );
        let global_coefficients_precision_rval = hyperparameters
            .get(3)
            .stop()
            .as_matrix()
            .stop_str("Precision matrix of the global coefficients should be a matrix")
            .to_mode_double(pc);
        let nrows = global_coefficients_precision_rval.nrow();
        let ncols = global_coefficients_precision_rval.ncol();
        if nrows != ncols {
            stop!("Precision matrix of the global coefficients should be a square matrix");
        }
        if global_coefficients_mean.len() != nrows {
            stop!("The dimensions of the precision matrix of the global coefficients does not match it's mean");
        }
        let global_coefficients_precision =
            DMatrix::from_column_slice(nrows, ncols, global_coefficients_precision_rval.slice());
        let clustered_coefficients_mean = DVector::from_column_slice(
            hyperparameters
                .get(4)
                .stop()
                .as_vector()
                .stop_str("Invalid clustered coefficients mean")
                .to_mode_double(pc)
                .slice(),
        );
        let clustered_coefficients_precision_rval = hyperparameters
            .get(5)
            .stop()
            .as_matrix()
            .stop_str("Precision matrix of the clustered coefficients should be a matrix")
            .to_mode_double(pc);
        let nrows = clustered_coefficients_precision_rval.nrow();
        let ncols = clustered_coefficients_precision_rval.ncol();
        if nrows != ncols {
            stop!("Precision matrix of the clustered coefficients should be a square matrix");
        }
        if clustered_coefficients_mean.len() != nrows {
            stop!("The dimensions of the precision matrix of the clustered coefficients does not match it's mean");
        }
        let clustered_coefficients_precision_slice = clustered_coefficients_precision_rval.slice();
        let clustered_coefficients_precision =
            DMatrix::from_column_slice(nrows, ncols, clustered_coefficients_precision_slice);
        let shrinkage_rval = hyperparameters.get(6).stop();
        let shrinkage_option = if shrinkage_rval.is_null() {
            None
        } else {
            Some(ShrinkageHyperparameters::from_r(shrinkage_rval))
        };
        let grit_rval = hyperparameters.get(7).stop();
        let grit_option = if grit_rval.is_null() {
            None
        } else {
            Some(GritHyperparameters::from_r(grit_rval))
        };
        Self::new(
            precision_response_shape,
            precision_response_rate,
            global_coefficients_mean,
            global_coefficients_precision,
            clustered_coefficients_mean,
            clustered_coefficients_precision,
            shrinkage_option,
            grit_option,
        )
        .unwrap()
    }

    pub fn n_global_covariates(&self) -> usize {
        self.global_coefficients_mean.len()
    }

    pub fn n_clustered_covariates(&self) -> usize {
        self.clustered_coefficients_mean.len()
    }

    pub fn precision_response_shape(&self) -> Shape {
        self.precision_response_shape
    }

    pub fn precision_response_rate(&self) -> Rate {
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
