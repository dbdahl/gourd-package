use crate::membership::MembershipGenerator;
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
    membership_generator: MembershipGenerator,
    missing: Vec<(usize, DVector<f64>)>,
}

impl Data {
    pub fn new(
        response: DVector<f64>,
        global_covariates: DMatrix<f64>,
        clustered_covariates: DMatrix<f64>,
        item_sizes: Vec<usize>,
    ) -> Option<Self> {
        let n_observations = response.nrows();
        if global_covariates.nrows() != n_observations
            || clustered_covariates.nrows() != n_observations
        {
            return None;
        }
        let membership_generator = MembershipGenerator::new(item_sizes);
        if membership_generator.n_observations() != n_observations {
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
            membership_generator,
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
        let (_, item_sizes_slice) = data.get_list_element(3).coerce_integer(pc).unwrap();
        let item_sizes: Vec<_> = item_sizes_slice
            .iter()
            .map(|x| usize::try_from(*x).unwrap())
            .collect();
        Data::new(
            response,
            global_covariates,
            clustered_covariates,
            item_sizes,
        )
        .unwrap()
    }

    pub fn impute<T: Rng>(&mut self, state: &State, rng: &mut T) {
        if !self.missing.is_empty() {
            let stdev = 1.0 / state.precision_response().sqrt();
            let normal = Normal::new(0.0, stdev).unwrap();
            for &(item, _) in &self.missing {
                let label = state.clustering().allocation()[item];
                let parameter = &state.clustered_coefficients()[label];
                let rows = self.membership_generator.indices_of_item(item);
                let mean = (self.global_covariates().select_rows(rows)
                    * state.global_coefficients())
                    + (self.clustered_covariates().select_rows(rows) * parameter);
                for (&row, &m) in rows.iter().zip(mean.iter()) {
                    self.response[row] = m + normal.sample(rng);
                }
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

    pub fn membership_generator(&self) -> &MembershipGenerator {
        &self.membership_generator
    }

    pub fn declare_missing(&mut self, items: Vec<usize>) {
        for (item, value) in &self.missing {
            let rows = self.membership_generator.indices_of_item(*item);
            for (&row, &v) in rows.iter().zip(value.iter()) {
                self.response[row] = v;
            }
        }
        self.missing = items
            .iter()
            .map(|&item| {
                let rows = self.membership_generator.indices_of_item(item);
                (item, self.response.select_rows(&rows[..]))
            })
            .collect();
    }

    pub fn n_items(&self) -> usize {
        self.membership_generator().n_items()
    }

    pub fn n_observations(&self) -> usize {
        self.global_covariates.nrows()
    }

    pub fn n_global_covariates(&self) -> usize {
        self.global_covariates.ncols()
    }

    pub fn n_clustered_covariates(&self) -> usize {
        self.clustered_covariates.ncols()
    }

    pub fn missing(&self) -> &[(usize, DVector<f64>)] {
        &self.missing
    }
}
