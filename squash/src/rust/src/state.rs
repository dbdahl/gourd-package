use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::{sample_multivariate_normal_v2, sample_multivariate_normal_v3};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::distr::FullConditional;
use dahl_randompartition::mcmc::update_neal_algorithm8;
use dahl_randompartition::perm::Permutation;
use nalgebra::DVector;
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use roxido::*;

#[derive(Debug)]
pub struct State {
    precision_response: f64,
    global_coefficients: DVector<f64>,
    clustering: Clustering,
    clustered_coefficients: Vec<DVector<f64>>,
    permutation: Permutation,
}

impl State {
    pub fn new(
        precision_response: f64,
        global_coefficients: DVector<f64>,
        clustering: Clustering,
        clustered_coefficients: Vec<DVector<f64>>,
        permutation: Permutation,
    ) -> Option<Self> {
        if precision_response <= 0.0 {
            return None;
        }
        if clustered_coefficients.len() != clustering.max_label() + 1 {
            return None;
        }
        let ncol = clustered_coefficients[0].len();
        for coef in &clustered_coefficients {
            if coef.len() != ncol {
                return None;
            }
        }
        if permutation.n_items() != clustering.n_items() {
            return None;
        }
        Some(Self {
            precision_response,
            global_coefficients,
            clustering,
            clustered_coefficients,
            permutation,
        })
    }

    pub fn from_r(state: Rval, pc: &mut Pc) -> Self {
        let precision_response = state.get_list_element(0).as_f64();
        let (_, global_coefficients_slice) = state.get_list_element(1).coerce_double(pc).unwrap();
        let global_coefficients = DVector::from_column_slice(global_coefficients_slice);
        let (_, clustering_slice) = state.get_list_element(2).coerce_integer(pc).unwrap();
        let clustering = Clustering::from_slice(clustering_slice);
        let clustered_coefficients_rval = state.get_list_element(3);
        let mut clustered_coefficients = Vec::with_capacity(clustered_coefficients_rval.len());
        for i in 0..clustered_coefficients.capacity() {
            let element = clustered_coefficients_rval.get_list_element(i);
            let (_, slice) = element.coerce_double(pc).unwrap();
            clustered_coefficients.push(DVector::from_column_slice(slice));
        }
        let permutation =
            Permutation::from_slice(state.get_list_element(4).coerce_integer(pc).unwrap().1)
                .unwrap();
        Self::new(
            precision_response,
            global_coefficients,
            clustering,
            clustered_coefficients,
            permutation,
        )
        .unwrap()
    }

    pub fn to_r(&self, pc: &mut Pc) -> Rval {
        let result = Rval::new_list(5, pc);
        result.set_list_element(0, Rval::new(self.precision_response, pc));
        result.set_list_element(1, Rval::new(self.global_coefficients.as_slice(), pc));
        let x: Vec<_> = self
            .clustering
            .allocation()
            .iter()
            .map(|&label| i32::try_from(label + 1).unwrap())
            .collect();
        result.set_list_element(2, Rval::new(&x[..], pc));
        let rval = Rval::new_list(self.clustered_coefficients.len(), pc);
        for (i, coef) in self.clustered_coefficients.iter().enumerate() {
            rval.set_list_element(i, Rval::new(coef.as_slice(), pc));
        }
        result.set_list_element(3, rval);
        let x: Vec<_> = self
            .permutation
            .as_slice()
            .iter()
            .map(|&label| i32::try_from(label + 1).unwrap())
            .collect();
        result.set_list_element(4, Rval::new(&x[..], pc));
        result
    }

    pub fn precision_response(&self) -> f64 {
        self.precision_response
    }

    pub fn global_coefficients(&self) -> &DVector<f64> {
        &self.global_coefficients
    }

    pub fn clustering(&self) -> &Clustering {
        &self.clustering
    }

    pub fn clustered_coefficients(&self) -> &Vec<DVector<f64>> {
        &self.clustered_coefficients
    }

    pub fn n_global_covariates(&self) -> usize {
        self.global_coefficients.len()
    }

    pub fn n_clustered_covariates(&self) -> usize {
        self.clustered_coefficients[0].len()
    }

    pub fn canonicalize(mut self) -> Self {
        let (clustering, map) = self.clustering.relabel(0, None, true);
        let map = map.unwrap();
        let mut new_clustered_coefficients = Vec::with_capacity(clustering.n_clusters());
        for i in map {
            new_clustered_coefficients.push(self.clustered_coefficients[i].clone());
        }
        self.clustering = clustering;
        self.clustered_coefficients = new_clustered_coefficients;
        self
    }

    const NEGATIVE_LN_SQRT_2PI: f64 = -0.91893853320467274178032973640561763986139747363778;

    #[inline]
    fn log_likelihood_kernel(&self, data: &Data, item: usize, response: f64, label: usize) -> f64 {
        let parameter = &self.clustered_coefficients[label];
        (response
            - (data.global_covariates().row(item) * &self.global_coefficients).index((0, 0))
            - (data.clustered_covariates().row(item) * parameter).index((0, 0)))
        .powi(2)
    }

    pub fn log_likelihood_contributions(&self, data: &Data) -> Vec<f64> {
        let log_normalizing_constant =
            Self::NEGATIVE_LN_SQRT_2PI + 0.5 * self.precision_response.ln();
        let negative_half_precision = -0.5 * self.precision_response;
        let mut result = Vec::with_capacity(data.n_items());
        for ((item, &label), &response) in self
            .clustering
            .allocation()
            .iter()
            .enumerate()
            .zip(data.response())
        {
            let x = log_normalizing_constant
                + negative_half_precision * self.log_likelihood_kernel(data, item, response, label);
            result.push(x);
        }
        result
    }

    pub fn log_likelihood_contributions_of_missing(&self, data: &Data) -> Vec<f64> {
        if let Some(missing_items) = data.missing_items() {
            let response = data.original_response().as_ref().unwrap();
            let log_normalizing_constant =
                Self::NEGATIVE_LN_SQRT_2PI + 0.5 * self.precision_response.ln();
            let negative_half_precision = -0.5 * self.precision_response;
            let mut result = Vec::with_capacity(missing_items.len());
            for &item in missing_items {
                let label = self.clustering().allocation()[item];
                let y = response[item];
                let x = log_normalizing_constant
                    + negative_half_precision * self.log_likelihood_kernel(data, item, y, label);
                result.push(x);
            }
            result
        } else {
            Vec::new()
        }
    }

    pub fn log_likelihood_of<'a, T: Iterator<Item = &'a usize>>(
        &'a self,
        data: &Data,
        items: T,
    ) -> f64 {
        let log_normalizing_constant =
            Self::NEGATIVE_LN_SQRT_2PI + 0.5 * self.precision_response.ln();
        let negative_half_precision = -0.5 * self.precision_response;
        let mut sum = 0.0;
        let mut len: u32 = 0;
        for &item in items {
            let label = self.clustering.allocation()[item];
            let response = data.response()[item];
            sum += self.log_likelihood_kernel(data, item, response, label);
            len += 1;
        }
        (len as f64) * log_normalizing_constant + negative_half_precision + sum
    }

    pub fn log_likelihood(&self, data: &Data) -> f64 {
        let log_normalizing_constant =
            Self::NEGATIVE_LN_SQRT_2PI + 0.5 * self.precision_response.ln();
        let negative_half_precision = -0.5 * self.precision_response;
        let mut sum = 0.0;
        for ((item, &label), &response) in self
            .clustering
            .allocation()
            .iter()
            .enumerate()
            .zip(data.response())
        {
            sum += self.log_likelihood_kernel(data, item, response, label);
        }
        (data.n_items() as f64) * log_normalizing_constant + negative_half_precision + sum
    }

    pub fn mcmc_iteration<S: FullConditional, T: Rng>(
        mut self,
        fixed: &StateFixedComponents,
        data: &mut Data,
        hyperparameters: &Hyperparameters,
        partition_distribution: &S,
        rng: &mut T,
        rng2: &mut T,
    ) -> Self {
        data.impute(&self, rng);
        if !fixed.precision_response {
            self.precision_response = Self::update_precision_response(
                &self.global_coefficients,
                &self.clustering,
                &self.clustered_coefficients,
                data,
                hyperparameters,
                rng,
            );
        }
        if !fixed.global_coefficients {
            self.global_coefficients = Self::update_global_coefficients(
                self.precision_response,
                &self.clustering,
                &self.clustered_coefficients,
                data,
                hyperparameters,
                rng,
            );
        }
        if !fixed.clustering {
            self.clustering = Self::update_clustering(
                self.clustering,
                &mut self.clustered_coefficients,
                self.precision_response,
                &self.global_coefficients,
                &self.permutation,
                data,
                hyperparameters,
                partition_distribution,
                rng,
                rng2,
            );
        }
        if !fixed.clustered_coefficients {
            self.clustered_coefficients = Self::update_clustered_coefficients(
                self.clustered_coefficients,
                &self.clustering,
                self.precision_response,
                &self.global_coefficients,
                data,
                hyperparameters,
                rng,
            );
        }
        if !fixed.permutation {
            panic!("Random permutation is not yet implemented.")
        }
        self
    }

    fn update_precision_response<T: Rng>(
        global_coefficients: &DVector<f64>,
        clustering: &Clustering,
        clustered_coefficients: &[DVector<f64>],
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) -> f64 {
        let shape = hyperparameters.precision_response_shape() + 0.5 * data.n_items() as f64;
        let residuals = data.response()
            - data.global_covariates() * global_coefficients
            - Self::dot_products(clustering, clustered_coefficients, data);
        let sum_of_squared_residuals: f64 = residuals.fold(0.0, |acc, x| acc + x * x);
        let rate = hyperparameters.precision_response_rate() + 0.5 * sum_of_squared_residuals;
        let scale = 1.0 / rate;
        Gamma::new(shape, scale).unwrap().sample(rng)
    }

    fn update_global_coefficients<T: Rng>(
        precision_response: f64,
        clustering: &Clustering,
        clustered_coefficients: &[DVector<f64>],
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) -> DVector<f64> {
        let precision = hyperparameters.global_coefficients_precision()
            + precision_response * data.global_covariates_transpose_times_self();
        let partial_residuals =
            data.response() - Self::dot_products(clustering, clustered_coefficients, data);
        let precision_times_mean = hyperparameters.global_coefficients_precision_times_mean()
            + precision_response * data.global_covariates_transpose() * partial_residuals;
        sample_multivariate_normal_v2(precision_times_mean, precision, rng).unwrap()
    }

    fn update_clustering<S: FullConditional, T: Rng>(
        clustering: Clustering,
        clustered_coefficients: &mut Vec<DVector<f64>>,
        precision_response: f64,
        global_coefficients: &DVector<f64>,
        permutation: &Permutation,
        data: &Data,
        hyperparameters: &Hyperparameters,
        partition_distribution: &S,
        rng: &mut T,
        rng2: &mut T,
    ) -> Clustering {
        let negative_half_precision = -0.5 * precision_response;
        let mut log_likelihood_contribution_fn = |item: usize, label: usize, is_new: bool| {
            if is_new {
                let parameter = sample_multivariate_normal_v3(
                    hyperparameters.clustered_coefficients_mean(),
                    hyperparameters.clustered_coefficients_precision_l_inv_transpose(),
                    rng2,
                );
                if label >= clustered_coefficients.len() {
                    clustered_coefficients.resize(label + 1, parameter);
                } else {
                    clustered_coefficients[label] = parameter;
                }
            };
            let parameter = &clustered_coefficients[label];
            negative_half_precision
                * (data.response()[item]
                    - (data.global_covariates().row(item) * global_coefficients).index((0, 0))
                    - (data.clustered_covariates().row(item) * parameter).index((0, 0)))
                .powi(2)
        };
        update_neal_algorithm8(
            1,
            clustering,
            permutation,
            partition_distribution,
            &mut log_likelihood_contribution_fn,
            rng,
        )
    }

    fn update_clustered_coefficients<T: Rng>(
        mut clustered_coefficients: Vec<DVector<f64>>,
        clustering: &Clustering,
        precision_response: f64,
        global_coefficients: &DVector<f64>,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) -> Vec<DVector<f64>> {
        for (label, clustered_coefficient) in clustered_coefficients.iter_mut().enumerate() {
            let indices = clustering.items_of(label);
            if indices.is_empty() {
                continue;
            }
            let w = data.clustered_covariates().select_rows(&indices);
            let wt = w.transpose();
            let precision = hyperparameters.clustered_coefficients_precision()
                + precision_response * wt.clone() * &w;
            let partial_residuals = data.response().select_rows(&indices)
                - data.global_covariates().select_rows(&indices) * global_coefficients;
            let mean = hyperparameters.clustered_coefficients_precision_times_mean()
                + precision_response * wt * partial_residuals;
            *clustered_coefficient = sample_multivariate_normal_v2(mean, precision, rng).unwrap()
        }
        clustered_coefficients
    }

    fn dot_products(
        clustering: &Clustering,
        clustered_coefficients: &[DVector<f64>],
        data: &Data,
    ) -> DVector<f64> {
        DVector::from_iterator(
            data.n_items(),
            data.clustered_covariates()
                .row_iter()
                .zip(clustering.allocation())
                .map(|(w, &label)| (w * &clustered_coefficients[label])[(0, 0)]),
        )
    }
}

pub struct StateFixedComponents {
    precision_response: bool,
    global_coefficients: bool,
    clustering: bool,
    clustered_coefficients: bool,
    permutation: bool,
}

impl StateFixedComponents {
    pub fn new(
        precision_response: bool,
        global_coefficients: bool,
        clustering: bool,
        clustered_coefficients: bool,
        permutation: bool,
    ) -> Self {
        Self {
            precision_response,
            global_coefficients,
            clustering,
            clustered_coefficients,
            permutation,
        }
    }
    pub fn from_r(x: Rval, pc: &mut Pc) -> Self {
        let (_, x) = x.coerce_logical(pc).unwrap();
        Self::new(x[0] != 0, x[1] != 0, x[2] != 0, x[3] != 0, x[4] != 0)
    }
}
