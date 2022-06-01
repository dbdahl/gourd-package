use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::{sample_multivariate_normal, sample_multivariate_normal_v2};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::mcmc::update_neal_algorithm8_v2;
use dahl_randompartition::perm::Permutation;
use dahl_randompartition::prelude::Mass;
use na::DVector;
use nalgebra as na;
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use roxido::*;

#[allow(dead_code)]
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
        State::new(
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
            .map(|&label| i32::try_from(label).unwrap())
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
            .map(|&label| i32::try_from(label).unwrap())
            .collect();
        result.set_list_element(4, Rval::new(&x[..], pc));
        result
    }

    #[allow(dead_code)]
    fn precision_response(&self) -> f64 {
        self.precision_response
    }

    #[allow(dead_code)]
    fn global_coefficients(&self) -> &DVector<f64> {
        &self.global_coefficients
    }

    #[allow(dead_code)]
    fn permutation(&self) -> &Permutation {
        &self.permutation
    }

    pub fn mcmc_iteration<T: Rng>(
        mut self,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) -> Self {
        self.precision_response = Self::update_precision_response(
            &self.global_coefficients,
            &self.clustering,
            &self.clustered_coefficients,
            data,
            hyperparameters,
            rng,
        );
        self.global_coefficients = Self::update_global_coefficients(
            self.precision_response,
            &self.clustering,
            &self.clustered_coefficients,
            data,
            hyperparameters,
            rng,
        );
        self.clustering = Self::update_clustering(
            self.clustering,
            &mut self.clustered_coefficients,
            self.precision_response,
            &self.global_coefficients,
            &self.permutation,
            data,
            hyperparameters,
            rng,
        );
        self.clustered_coefficients = Self::update_clustered_coefficients(
            self.clustered_coefficients,
            &self.clustering,
            self.precision_response,
            &self.global_coefficients,
            data,
            hyperparameters,
            rng,
        );
        self
    }

    #[allow(dead_code)]
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
        Gamma::new(shape, rate).unwrap().sample(rng)
    }

    #[allow(dead_code)]
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

    fn update_clustering<T: Rng>(
        clustering: Clustering,
        clustered_coefficients: &mut Vec<DVector<f64>>,
        precision_response: f64,
        global_coefficients: &DVector<f64>,
        permutation: &Permutation,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) -> Clustering {
        use rand::SeedableRng;
        use rand_pcg::Pcg64Mcg;
        let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
        rng.fill(&mut seed);
        let mut rng2 = Pcg64Mcg::from_seed(seed);
        let prior = CrpParameters::new_with_mass(data.n_items(), Mass::new(1.0));
        let mut log_likelihood_contribution_fn = |item: usize, label: usize, is_new: bool| {
            if is_new {
                let parameter = sample_multivariate_normal(
                    hyperparameters.clustered_coefficients_mean().to_owned(),
                    hyperparameters
                        .clustered_coefficients_precision()
                        .to_owned(),
                    &mut rng2,
                )
                .unwrap();
                if label >= clustered_coefficients.len() {
                    clustered_coefficients.resize(label + 1, parameter);
                } else {
                    clustered_coefficients[label] = parameter;
                }
            };
            let parameter = &clustered_coefficients[label];
            -precision_response / 0.5
                * (*data.response().index(item)
                    - (data.global_covariates().row(item) * global_coefficients).index((0, 0))
                    - (data.clustered_covariates().row(item) * parameter).index((0, 0)))
                .powi(2)
        };
        update_neal_algorithm8_v2(
            1,
            clustering,
            &permutation,
            &prior,
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
            if indices.len() == 0 {
                continue;
            }
            let w = data.clustered_covariates().select_rows(&indices);
            let wt = w.transpose();
            let precision = hyperparameters.clustered_coefficients_precision()
                + precision_response * wt.clone() * &w;
            let partial_residuals = data.response().select_rows(&indices)
                - data.global_covariates().select_rows(&indices) * global_coefficients;
            let mean = hyperparameters.global_coefficients_precision_times_mean()
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
