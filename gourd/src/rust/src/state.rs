use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::{sample_multivariate_normal_v2, sample_multivariate_normal_v3};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::distr::FullConditional;
use dahl_randompartition::distr::ProbabilityMassFunction;
use dahl_randompartition::mcmc::update_neal_algorithm8;
use dahl_randompartition::perm::Permutation;
use dahl_randompartition::sp::SpParameters;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use roxido::*;

#[derive(Debug)]
pub struct State {
    pub precision_response: f64,
    pub global_coefficients: DVector<f64>,
    pub clustering: Clustering,
    pub clustered_coefficients: Vec<DVector<f64>>,
}

impl State {
    pub fn new(
        precision_response: f64,
        global_coefficients: DVector<f64>,
        clustering: Clustering,
        clustered_coefficients: Vec<DVector<f64>>,
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
        Some(Self {
            precision_response,
            global_coefficients,
            clustering,
            clustered_coefficients,
        })
    }

    pub fn from_r(state: Rval, pc: &mut Pc) -> Self {
        let precision_response = state.get_list_element(0).as_f64();
        let (_, global_coefficients_slice) = state.get_list_element(1).coerce_double(pc).unwrap();
        let global_coefficients = DVector::from_column_slice(global_coefficients_slice);
        let clustering = {
            let (_, clustering_slice) = state.get_list_element(2).coerce_integer(pc).unwrap();
            let clust: Vec<_> = clustering_slice.iter().map(|&x| (x as usize) - 1).collect();
            Clustering::from_vector(clust)
        };
        let clustered_coefficients_rval = state.get_list_element(3);
        let mut clustered_coefficients = Vec::with_capacity(clustered_coefficients_rval.len());
        for i in 0..clustered_coefficients.capacity() {
            let element = clustered_coefficients_rval.get_list_element(i);
            let (_, slice) = element.coerce_double(pc).unwrap();
            clustered_coefficients.push(DVector::from_column_slice(slice));
        }
        Self::new(
            precision_response,
            global_coefficients,
            clustering,
            clustered_coefficients,
        )
        .unwrap()
    }

    pub fn to_r(&self, pc: &mut Pc) -> Rval {
        let result = Rval::new_list(4, pc);
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
    fn mk_constants(precision_response: f64) -> (f64, f64) {
        let log_normalizing_constant = Self::NEGATIVE_LN_SQRT_2PI + 0.5 * precision_response.ln();
        let negative_half_precision = -0.5 * precision_response;
        (log_normalizing_constant, negative_half_precision)
    }

    #[inline]
    fn log_likelihood_kernel(
        &self,
        data: &Data,
        rows: &[usize],
        response: &DVector<f64>,
        label: usize,
    ) -> f64 {
        let parameter = &self.clustered_coefficients[label];
        (response
            - (data.global_covariates().select_rows(rows) * &self.global_coefficients)
            - (data.clustered_covariates().select_rows(rows) * parameter))
            .map(|x| x.powi(2))
            .sum()
    }

    pub fn log_likelihood_of<T: IntoIterator<Item = usize>>(&self, data: &Data, items: T) -> f64 {
        let (log_normalizing_constant, negative_half_precision) =
            Self::mk_constants(self.precision_response);
        let mut sum = 0.0;
        let mut len = 0;
        for item in items {
            let label = self.clustering.allocation()[item];
            let rows = data.membership_generator().indices_of_item(item);
            let response = data.response().select_rows(rows);
            sum += self.log_likelihood_kernel(data, rows, &response, label);
            len += rows.len();
        }
        (len as f64) * log_normalizing_constant + negative_half_precision * sum
    }

    pub fn log_likelihood(&self, data: &Data) -> f64 {
        self.log_likelihood_of(data, 0..data.n_items())
    }

    pub fn log_likelihood_contributions(&self, data: &Data) -> Vec<f64> {
        let (log_normalizing_constant, negative_half_precision) =
            Self::mk_constants(self.precision_response);
        let mut result = Vec::with_capacity(data.n_items());
        for item in 0..data.n_items() {
            let label = self.clustering.allocation()[item];
            let rows = data.membership_generator().indices_of_item(item);
            let response = data.response().select_rows(rows);
            let value = (rows.len() as f64) * log_normalizing_constant
                + negative_half_precision
                    * self.log_likelihood_kernel(data, rows, &response, label);
            result.push(value);
        }
        result
    }

    pub fn log_likelihood_contributions_of_missing(&self, data: &Data) -> Vec<f64> {
        if data.missing().is_empty() {
            return Vec::new();
        }
        let (log_normalizing_constant, negative_half_precision) =
            Self::mk_constants(self.precision_response);
        let mut result = Vec::with_capacity(data.missing().len());
        for (item, response) in data.missing() {
            let label = self.clustering().allocation()[*item];
            let rows = data.membership_generator().indices_of_item(*item);
            let x = (response.len() as f64) * log_normalizing_constant
                + negative_half_precision * self.log_likelihood_kernel(data, rows, response, label);
            result.push(x);
        }
        result
    }

    pub fn mcmc_iteration<S: FullConditional, T: Rng>(
        &mut self,
        mcmc_tuning: &McmcTuning,
        data: &mut Data,
        hyperparameters: &Hyperparameters,
        partition_distribution: &S,
        rng: &mut T,
        rng2: &mut T,
    ) {
        data.impute(self, rng);
        if mcmc_tuning.update_precision_response {
            Self::update_precision_response(
                &mut self.precision_response,
                &self.global_coefficients,
                &self.clustering,
                &self.clustered_coefficients,
                data,
                hyperparameters,
                rng,
            );
        }
        if mcmc_tuning.update_global_coefficients {
            Self::update_global_coefficients(
                &mut self.global_coefficients,
                self.precision_response,
                &self.clustering,
                &self.clustered_coefficients,
                data,
                hyperparameters,
                rng,
            );
        }
        if mcmc_tuning.update_clustering {
            let permutation = Permutation::natural_and_fixed(self.clustering.n_items());
            Self::update_clustering(
                &mut self.clustering,
                &mut self.clustered_coefficients,
                self.precision_response,
                &self.global_coefficients,
                &permutation,
                data,
                hyperparameters,
                partition_distribution,
                rng,
                rng2,
            );
        }
        if mcmc_tuning.update_clustered_coefficients {
            Self::update_clustered_coefficients(
                &mut self.clustered_coefficients,
                &self.clustering,
                self.precision_response,
                &self.global_coefficients,
                data,
                hyperparameters,
                rng,
            );
        }
    }

    pub fn update_precision_response<T: Rng>(
        precision_response: &mut f64,
        global_coefficients: &DVector<f64>,
        clustering: &Clustering,
        clustered_coefficients: &[DVector<f64>],
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) {
        let shape = hyperparameters.precision_response_shape() + 0.5 * data.n_observations() as f64;
        let residuals = data.response()
            - data.global_covariates() * global_coefficients
            - Self::clustered_covariates_times_parameters(clustering, clustered_coefficients, data);
        let sum_of_squared_residuals: f64 = residuals.fold(0.0, |acc, x| acc + x * x);
        let rate = hyperparameters.precision_response_rate() + 0.5 * sum_of_squared_residuals;
        let scale = 1.0 / rate;
        *precision_response = Gamma::new(shape, scale).unwrap().sample(rng);
    }

    pub fn update_global_coefficients<T: Rng>(
        global_coefficients: &mut DVector<f64>,
        precision_response: f64,
        clustering: &Clustering,
        clustered_coefficients: &[DVector<f64>],
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) {
        let precision = hyperparameters.global_coefficients_precision()
            + precision_response * data.global_covariates_transpose_times_self();
        let partial_residuals = data.response()
            - Self::clustered_covariates_times_parameters(clustering, clustered_coefficients, data);
        let precision_times_mean = hyperparameters.global_coefficients_precision_times_mean()
            + precision_response * data.global_covariates_transpose() * partial_residuals;
        *global_coefficients =
            sample_multivariate_normal_v2(precision_times_mean, precision, rng).unwrap();
    }

    fn clustered_covariates_times_parameters(
        clustering: &Clustering,
        clustered_coefficients: &[DVector<f64>],
        data: &Data,
    ) -> DVector<f64> {
        let mut result = DVector::from_element(data.n_observations(), 0.0);
        for item in 0..data.n_items() {
            let rows = data.membership_generator().indices_of_item(item);
            let label = &clustering.allocation()[item];
            for (&row, &value) in rows.iter().zip(
                (data.clustered_covariates().select_rows(rows) * &clustered_coefficients[*label])
                    .iter(),
            ) {
                result[row] = value;
            }
        }
        result
    }

    fn update_clustering<S: FullConditional, T: Rng>(
        clustering: &mut Clustering,
        clustered_coefficients: &mut Vec<DVector<f64>>,
        precision_response: f64,
        global_coefficients: &DVector<f64>,
        permutation: &Permutation,
        data: &Data,
        hyperparameters: &Hyperparameters,
        partition_distribution: &S,
        rng: &mut T,
        rng2: &mut T,
    ) {
        struct CommonItemCache {
            difference: DVector<f64>,
            clustered_covariates: DMatrix<f64>,
        }
        let cacher = |item: usize| {
            let rows = data.membership_generator().indices_of_item(item);
            let response = data.response().select_rows(rows);
            let difference =
                response - (data.global_covariates().select_rows(rows) * global_coefficients);
            let clustered_covariates = data.clustered_covariates().select_rows(rows);
            CommonItemCache {
                difference,
                clustered_covariates,
            }
        };
        let negative_half_precision = -0.5 * precision_response;
        let mut log_likelihood_contribution_fn =
            |_item: usize, cache: &CommonItemCache, label: usize, is_new: bool| {
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
                    * (&cache.difference - (&cache.clustered_covariates * parameter))
                        .map(|x| x.powi(2))
                        .sum()
            };
        update_neal_algorithm8(
            1,
            clustering,
            permutation,
            partition_distribution,
            &mut log_likelihood_contribution_fn,
            cacher,
            rng,
        )
    }

    pub fn update_clustering_temporal<S: FullConditional, T: Rng, U: Fn(&Clustering) -> f64>(
        clustering: &mut Clustering,
        clustered_coefficients: &mut Vec<DVector<f64>>,
        precision_response: f64,
        global_coefficients: &DVector<f64>,
        permutation: &Permutation,
        data: &Data,
        hyperparameters: &Hyperparameters,
        partition_distribution: &S,
        partition_distribution_next: Option<(Clustering, SpParameters<CrpParameters>)>,
        rng: &mut T,
        rng2: &mut T,
    ) {
        struct CommonItemCache {
            pd: Option<(Clustering, SpParameters<CrpParameters>)>,
            difference: DVector<f64>,
            clustered_covariates: DMatrix<f64>,
        }
        let cacher = |item: usize| {
            let rows = data.membership_generator().indices_of_item(item);
            let response = data.response().select_rows(rows);
            let difference =
                response - (data.global_covariates().select_rows(rows) * global_coefficients);
            let clustered_covariates = data.clustered_covariates().select_rows(rows);
            CommonItemCache {
                pd: partition_distribution_next.clone(),
                difference,
                clustered_covariates,
            }
        };
        let negative_half_precision = -0.5 * precision_response;
        let mut log_likelihood_contribution_fn =
            |item: usize, cache: &CommonItemCache, label: usize, is_new: bool| {
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
                let likelihood1 = negative_half_precision
                    * (&cache.difference - (&cache.clustered_covariates * parameter))
                        .map(|x| x.powi(2))
                        .sum();
                let likelihood2 = match cache.pd {
                    Some((p, pd)) => {
                        pd.anchor.allocate(item, label);
                        pd.log_pmf(&p)
                    }
                    None => 0.0,
                };
                likelihood1 + likelihood2
            };
        update_neal_algorithm8(
            1,
            clustering,
            permutation,
            partition_distribution,
            &mut log_likelihood_contribution_fn,
            cacher,
            rng,
        )
    }

    pub fn update_clustered_coefficients<T: Rng>(
        clustered_coefficients: &mut [DVector<f64>],
        clustering: &Clustering,
        precision_response: f64,
        global_coefficients: &DVector<f64>,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) {
        for (label, clustered_coefficient) in clustered_coefficients.iter_mut().enumerate() {
            let items = clustering.items_of(label);
            if items.is_empty() {
                continue;
            }
            let rows = data.membership_generator().indices_of_items(items.iter());
            let w = data.clustered_covariates().select_rows(rows.clone());
            let wt = w.transpose();
            let precision = hyperparameters.clustered_coefficients_precision()
                + precision_response * wt.clone() * &w;
            let partial_residuals = data.response().select_rows(rows.clone())
                - data.global_covariates().select_rows(rows) * global_coefficients;
            let mean = hyperparameters.clustered_coefficients_precision_times_mean()
                + precision_response * wt * partial_residuals;
            *clustered_coefficient = sample_multivariate_normal_v2(mean, precision, rng).unwrap()
        }
    }
}

pub struct McmcTuning {
    pub update_precision_response: bool,
    pub update_global_coefficients: bool,
    pub update_clustering: bool,
    pub update_clustered_coefficients: bool,
    pub n_items_per_permutation_update: Option<u32>,
    pub shrinkage_slice_step_size: Option<f64>,
}

impl McmcTuning {
    pub fn new(
        update_precision_response: bool,
        update_global_coefficients: bool,
        update_clustering: bool,
        update_clustered_coefficients: bool,
        n_items_per_permutation_update: Option<u32>,
        shrinkage_slice_step_size: Option<f64>,
    ) -> Result<Self, &'static str> {
        if let Some(s) = shrinkage_slice_step_size {
            if s.is_nan() || s.is_infinite() || s < 0.0 {
                return Err("shrinkage_slice_step_size must be a finite nonpositive number.");
            }
        }
        Ok(Self {
            update_precision_response,
            update_global_coefficients,
            update_clustering,
            update_clustered_coefficients,
            n_items_per_permutation_update,
            shrinkage_slice_step_size,
        })
    }
    pub fn from_r(x: Rval, _pc: &mut Pc) -> Self {
        Self::new(
            x.get_list_element(0).as_bool(),
            x.get_list_element(1).as_bool(),
            x.get_list_element(2).as_bool(),
            x.get_list_element(3).as_bool(),
            Some(x.get_list_element(4).as_i32().try_into().unwrap()),
            Some(x.get_list_element(5).as_f64()),
        )
        .unwrap()
    }
}
