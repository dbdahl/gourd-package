use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::sample_multivariate_normal;
use dahl_randompartition::clust::Clustering;
use na::DVector;
use nalgebra as na;
use rand::Rng;
use rand_distr::{Distribution, Gamma};

#[allow(dead_code)]
pub struct State {
    precision_response: f64,
    global_coefficients: DVector<f64>,
    clustering: Clustering,
    clustered_coefficients: Vec<DVector<f64>>,
}

impl State {
    #[allow(dead_code)]
    fn precision_response(&self) -> f64 {
        self.precision_response
    }

    #[allow(dead_code)]
    fn global_coefficients(&self) -> &DVector<f64> {
        &self.global_coefficients
    }

    #[allow(dead_code)]
    fn mcmc_iteration<T: Rng>(
        &mut self,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) {
        self.update_precision_response(data, hyperparameters, rng);
        self.update_global_coefficients(data, hyperparameters, rng);
    }

    #[allow(dead_code)]
    fn update_precision_response<T: Rng>(
        &mut self,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) {
        let shape = hyperparameters.precision_response_shape() + 0.5 * data.n_items() as f64;
        let residuals = data.response()
            - data.global_covariates() * self.global_coefficients()
            - self.dot_products(data);
        let sum_of_squared_residuals: f64 = residuals.fold(0.0, |acc, x| acc + x * x);
        let rate = hyperparameters.precision_response_rate() + 0.5 * sum_of_squared_residuals;
        self.precision_response = Gamma::new(shape, rate).unwrap().sample(rng);
    }

    #[allow(dead_code)]
    fn update_global_coefficients<T: Rng>(
        &mut self,
        data: &Data,
        hyperparameters: &Hyperparameters,
        rng: &mut T,
    ) {
        let precision = hyperparameters.global_coefficients_precision()
            + self.precision_response() * data.global_covariates_transpose_times_self();
        let partial_residuals = data.response() - self.dot_products(data);
        let mean = hyperparameters.global_coefficients_precision_times_mean()
            + self.precision_response() * data.global_covariates_transpose() * partial_residuals;
        self.global_coefficients = sample_multivariate_normal(mean, precision, rng).unwrap();
    }

    fn dot_products(&self, data: &Data) -> DVector<f64> {
        DVector::from_iterator(
            data.n_items(),
            data.clustered_covariates()
                .row_iter()
                .zip(self.clustering.allocation())
                .map(|(w, &label)| (w * &self.clustered_coefficients[label])[(0, 0)]),
        )
    }
}

/*
impl State {
    fn update_clustering(&mut self) {}
    fn sample_clustered_coefficient_prior<T: Rng>(&self, data: &Data, rng: &mut T) -> DVector<f64> {
        let mean = 0.0;
        let sd = (1.0 / self.precision_coefficients).sqrt();
        let distr = Normal::new(mean, sd).unwrap();
        let mut x = Vec::with_capacity(data.n_clustered_coefficients());
        for _ in 0..x.capacity() {
            x.push(distr.sample(rng));
        }
        na::DVector::from(x)
    }
    fn update_clustered_coefficient<T: Rng>(&self, data: &Data, subset: &[usize], rng: &mut T) {
        let y = data.response().select_rows(subset);
        let W = data.clustered_covariates().select_rows(subset);
        let Wt = W.transpose();
        let WtW = Wt.clone() * W;
        let Wty = Wt
            * (y - data.global_covariates().select_rows(subset) * self.global_coefficients.clone());
    }
}
*/
