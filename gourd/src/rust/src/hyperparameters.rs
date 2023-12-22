use dahl_randompartition::prelude::*;
use nalgebra::{DMatrix, DVector};
use roxido::*;

#[derive(Debug)]
pub struct Hyperparameters {
    precision_response_shape: Shape,
    precision_response_rate: Rate,
    global_coefficients_mean: DVector<f64>,
    global_coefficients_precision: DMatrix<f64>,
    clustered_coefficients_mean: DVector<f64>,
    clustered_coefficients_precision: DMatrix<f64>,
    global_coefficients_precision_times_mean: DVector<f64>,
    clustered_coefficients_precision_times_mean: DVector<f64>,
    clustered_coefficients_precision_l_inv_transpose: DMatrix<f64>,
    pub shrinkage: ShrinkageHyperparameters,
    pub grit: GritHyperparameters,
}

#[derive(Debug)]
pub struct ShrinkageHyperparameters {
    pub reference: Option<usize>,
    pub shape: Shape,
    pub rate: Rate,
}

impl FromR for ShrinkageHyperparameters {
    fn from_r(x: RObject, _pc: &mut Pc) -> Result<Self, String> {
        let x = x.as_list()?;
        let mut map = x.make_map();
        let result = Self {
            reference: map.get("reference")?.as_usize().ok(),
            shape: Shape::new(map.get("shape")?.as_f64()?).ok_or("Invalid shape")?,
            rate: Rate::new(map.get("rate")?.as_f64()?).ok_or("Invalid rate")?,
        };
        map.exhaustive()?;
        Ok(result)
    }
}

#[derive(Debug)]
pub struct GritHyperparameters {
    pub shape1: Shape,
    pub shape2: Shape,
}

impl FromR for GritHyperparameters {
    fn from_r(x: RObject, _pc: &mut Pc) -> Result<Self, String> {
        let x = x.as_list()?;
        let mut map = x.make_map();
        let result = Self {
            shape1: Shape::new(map.get("shape1")?.as_f64()?).ok_or("Invalid shape1")?,
            shape2: Shape::new(map.get("shape2")?.as_f64()?).ok_or("Invalid shape2")?,
        };
        map.exhaustive()?;
        Ok(result)
    }
}

fn helper_mean_precision(
    map: &mut roxido::r::RListMap,
    vector_name: &str,
    matrix_name: &str,
    pc: &mut Pc,
) -> Result<(DVector<f64>, DMatrix<f64>), String> {
    let r1 = DVector::from_column_slice(
        map.get(vector_name)?
            .as_vector()?
            .to_mode_double(pc)
            .slice(),
    );
    let n = r1.len();
    let x = map.get(matrix_name)?.as_matrix()?.to_mode_double(pc);
    if x.nrow() != n || x.ncol() != n {
        return Err(format!(
            "To match '{}', '{}' is expected to be a {}-by-{} square matrix",
            vector_name, matrix_name, n, n
        ));
    }
    let r2 = DMatrix::from_column_slice(n, n, x.slice());
    Ok((r1, r2))
}

impl FromR for Hyperparameters {
    fn from_r(x: RObject, pc: &mut Pc) -> Result<Self, String> {
        let x = x.as_list()?;
        let mut map = x.make_map();
        let (global_coefficients_mean, global_coefficients_precision) = helper_mean_precision(
            &mut map,
            "global_coefficients_mean",
            "global_coefficients_precision",
            pc,
        )?;
        let (clustered_coefficients_mean, clustered_coefficients_precision) =
            helper_mean_precision(
                &mut map,
                "clustered_coefficients_mean",
                "clustered_coefficients_precision",
                pc,
            )?;
        let global_coefficients_precision_times_mean =
            global_coefficients_precision.clone() * &global_coefficients_mean;
        let clustered_coefficients_precision_times_mean =
            clustered_coefficients_precision.clone() * &clustered_coefficients_mean;
        let clustered_coefficients_precision_l_inv_transpose =
            match crate::mvnorm::prepare(clustered_coefficients_precision.clone()) {
                Some(lit) => lit,
                None => return Err("Cannot decompose clustered coefficients precision".to_owned()),
            };
        let result = Self {
            precision_response_shape: Shape::new(map.get("precision_response_shape")?.as_f64()?)
                .ok_or("Invalid shape for response precision")?,
            precision_response_rate: Rate::new(map.get("precision_response_rate")?.as_f64()?)
                .ok_or("Invalid rate for response precision")?,
            global_coefficients_mean,
            global_coefficients_precision,
            clustered_coefficients_mean,
            clustered_coefficients_precision,
            global_coefficients_precision_times_mean,
            clustered_coefficients_precision_times_mean,
            clustered_coefficients_precision_l_inv_transpose,
            shrinkage: ShrinkageHyperparameters::from_r(map.get("shrinkage")?, pc)?,
            grit: GritHyperparameters::from_r(map.get("grit")?, pc)?,
        };
        map.exhaustive()?;
        Ok(result)
    }
}

impl Hyperparameters {
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
