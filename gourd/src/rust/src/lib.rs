mod registration {
    include!(concat!(env!("OUT_DIR"), "/registration.rs"));
}

mod data;
mod hyperparameters;
mod membership;
mod monitor;
mod mvnorm;
mod state;

use crate::data::Data;
use crate::hyperparameters::{GritHyperparameters, Hyperparameters, ShrinkageHyperparameters};
use crate::monitor::Monitor;
use crate::mvnorm::sample_multivariate_normal_repeatedly;
use crate::state::{McmcTuning, State};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::cpp::CppParameters;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::distr::{ProbabilityMassFunction, ProbabilityMassFunctionPartial};
use dahl_randompartition::epa::EpaParameters;
use dahl_randompartition::fixed::FixedPartitionParameters;
use dahl_randompartition::jlp::JlpParameters;
use dahl_randompartition::lsp::LspParameters;
use dahl_randompartition::mcmc::update_partition_gibbs;
use dahl_randompartition::perm::Permutation;
use dahl_randompartition::prelude::*;
use dahl_randompartition::shrink::Shrinkage;
use dahl_randompartition::sp::SpParameters;
use dahl_randompartition::up::UpParameters;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use roxido::*;
use slice_sampler::univariate::stepping_out;
use statrs::distribution::{Continuous, Gamma};
use walltime::TicToc;

#[roxido]
fn rngs_new() -> RObject {
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let rng2 = Pcg64Mcg::from_seed(seed);
    let result = R::new_list(2, pc);
    result
        .set(0, &R::encode(rng, &"rng".to_r(pc), true, pc))
        .stop();
    result
        .set(1, &R::encode(rng2, &"rng".to_r(pc), true, pc))
        .stop();
    result
}

#[roxido]
fn state_r2rust(state: RObject) -> RObject {
    let state = State::from_r(state, pc);
    R::encode(state, &"state".to_r(pc), true, pc)
}

#[roxido]
fn state_rust2r(state: RObject) -> RObject {
    state
        .as_external_ptr()
        .stop()
        .decode_as_ref::<State>()
        .to_r(pc)
}

#[roxido]
fn monitor_new() -> RObject {
    R::encode(Monitor::<u32>::new(), &"monitor".to_r(pc), true, pc)
}

#[roxido]
fn monitor_rate(monitor: RObject) -> RObject {
    monitor
        .as_external_ptr()
        .stop()
        .decode_as_ref::<Monitor<u32>>()
        .rate()
}

#[roxido]
fn monitor_reset(monitor: RObject) -> RObject {
    let monitor = monitor
        .as_external_ptr()
        .stop()
        .decode_as_mut::<Monitor<u32>>();
    monitor.reset();
}

#[roxido]
fn hyperparameters_r2rust(hyperparameters: RObject) -> RObject {
    R::encode(
        Hyperparameters::from_r(hyperparameters, pc),
        &"hyperparameters".to_r(pc),
        true,
        pc,
    )
}

#[roxido]
fn data_r2rust(data: RObject, missing_items: RObject) -> RObject {
    let mut data = Data::from_r(data, pc);
    let missing_items = missing_items.as_vector().stop().to_mode_integer(pc).slice();
    let missing_items: Vec<_> = missing_items
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    data.declare_missing(missing_items);
    R::encode(data, &"data".to_r(pc), true, pc)
}

fn permutation_to_r(permutation: &Permutation, rval: &RObject) {
    for (x, y) in permutation
        .as_slice()
        .iter()
        .zip(rval.as_vector().stop().as_mode_integer().stop().slice())
    {
        *y = i32::try_from(*x).unwrap() + 1;
    }
}

fn scalar_shrinkage_to_r(shrinkage: &ScalarShrinkage, rval: &RObject) {
    rval.as_vector().stop().as_mode_double().stop().slice()[0] = shrinkage.get();
}

fn shrinkage_to_r(shrinkage: &Shrinkage, rval: &RObject) {
    for (x, y) in rval
        .as_vector()
        .stop()
        .as_mode_double()
        .stop()
        .slice()
        .iter_mut()
        .zip(shrinkage.as_slice())
    {
        *x = y.get()
    }
}

fn grit_to_r(grit: &Grit, rval: &RObject) {
    rval.as_vector().stop().as_mode_double().stop().slice()[0] = grit.get();
}

struct Group {
    data: Data,
    state: State,
    hyperparameters: Hyperparameters,
}

struct All {
    units: Vec<Group>,
    n_items: usize,
}

#[roxido]
fn all(all: RObject) -> RObject {
    let all = all.as_list().stop();
    let mut vec = Vec::with_capacity(all.len());
    let mut n_items = None;
    for i in 0..all.len() {
        let list = all.get(i).stop().as_list().stop();
        let (data, state, hyperparameters) =
            (list.get(0).stop(), list.get(1).stop(), list.get(2).stop());
        let data = Data::from_r(data, pc);
        let state = State::from_r(state, pc);
        let hyperparameters = Hyperparameters::from_r(hyperparameters, pc);
        if n_items.is_none() {
            n_items = Some(data.n_items());
        }
        assert_eq!(data.n_items(), n_items.unwrap());
        assert_eq!(state.clustering().n_items(), n_items.unwrap());
        assert_eq!(
            hyperparameters.n_global_covariates(),
            data.n_global_covariates()
        );
        assert_eq!(
            hyperparameters.n_clustered_covariates(),
            data.n_clustered_covariates()
        );
        vec.push(Group {
            data,
            state,
            hyperparameters,
        });
    }
    assert!(!all.is_empty());
    let all = All {
        units: vec,
        n_items: n_items.unwrap(),
    };
    R::encode(all, &"all".to_r(pc), true, pc)
}

struct GlobalMcmcTuning {
    n_loops: usize,
    n_loops_burnin: usize,
    n_scans_per_loop: usize,
    n_saves: usize,
    update_anchor: bool,
    n_permutation_updates_per_scan: usize,
    n_items_per_permutation_update: Option<usize>,
    shrinkage_slice_step_size: Option<f64>,
    validation_data: Option<Vec<Data>>,
}

impl GlobalMcmcTuning {
    fn from_r(x: RObject, pc: &mut Pc) -> Self {
        let list = validate_list(
            x,
            &[
                "n_loops",
                "n_loops_burnin",
                "n_scans_per_loop",
                "n_saves",
                "update_anchor",
                "n_permutations_updates_per_scan",
                "n_items_per_permutation_update",
                "shrinkage_slice_step_size",
                "grit_slice_step_size",
                "validation_data",
            ],
            "global_mcmc_tuning",
        );
        let n_scans_per_loop = list
            .get(2)
            .stop()
            .as_usize()
            .stop_str("Number of scans per loop should be an integer");
        let n_loops = list
            .get(0)
            .stop()
            .as_usize()
            .stop_str("Number of loops should be an integer")
            / n_scans_per_loop;
        let n_loops_burnin = list
            .get(1)
            .stop()
            .as_usize()
            .stop_str("Number of loops for burnin should be an integer")
            / n_scans_per_loop;
        let n_saves = n_loops - n_loops_burnin;
        let update_anchor = list
            .get(3)
            .unwrap()
            .as_bool()
            .stop_str("Update anchor should be an logical");
        let n_permutation_updates_per_scan = list
            .get(4)
            .unwrap()
            .as_usize()
            .stop_str("Number of permutation updates per scan should be an integer");
        let y = list.get(5).unwrap();
        let n_items_per_permutation_update = if y.is_null() || y.is_na() || y.is_nan() {
            None
        } else {
            Some(
                y.as_usize()
                    .stop_str("Number of items per permutation update should be an integer"),
            )
        };
        let y = list.get(6).unwrap();
        let shrinkage_slice_step_size = if y.is_null() || y.is_na() || y.is_nan() {
            None
        } else {
            Some(
                y.as_f64()
                    .stop_str("Slice size for shrinkage should be a numeric"),
            )
        };
        let y = list
            .get(7)
            .stop()
            .as_list()
            .stop_str("Validation data should be a list");
        let validation_data = if y.is_null() || y.is_na() || y.is_nan() {
            None
        } else {
            let n_units = y.len();
            let mut vd = Vec::with_capacity(n_units);
            for k in 0..n_units {
                vd.push(Data::from_r(y.get(k).unwrap(), pc));
            }
            Some(vd)
        };
        Self {
            n_loops,
            n_loops_burnin,
            n_scans_per_loop,
            n_saves,
            update_anchor,
            n_permutation_updates_per_scan,
            n_items_per_permutation_update,
            shrinkage_slice_step_size,
            validation_data,
        }
    }
}

struct GlobalHyperparametersTemporal {
    grit: Grit,
    baseline_concentration: Concentration,
    shrinkage: ShrinkageHyperparameters,
}

impl GlobalHyperparametersTemporal {
    fn from_r(x: RObject, _pc: &mut Pc) -> Self {
        let list = validate_list(
            x,
            &["grit", "baselinse_concentration", "shrinkage"],
            "global_hyperparameters_temporal",
        );
        Self {
            grit: Grit::new(
                list.get(0)
                    .stop()
                    .as_f64()
                    .stop_str("Grit parameter should be a numeric"),
            )
            .unwrap_or_else(|| stop!("Invalid grit parameter")),
            baseline_concentration: Concentration::new(
                list.get(1)
                    .stop()
                    .as_f64()
                    .stop_str("Grit parameter should be a numeric"),
            )
            .unwrap_or_else(|| stop!("Invalid concentration parameter")),
            shrinkage: ShrinkageHyperparameters::from_r(list.get(2).stop()),
        }
    }
}

struct ResultsTemporal<'a> {
    rval: RObject<roxido::r::Vector, roxido::r::List>,
    counter: usize,
    n_items: usize,
    unit_partitions: Vec<&'a mut [i32]>,
    permutations: &'a mut [i32],
    shrinkages: &'a mut [f64],
    log_likelihoods: &'a mut [f64],
}

impl<'a> ResultsTemporal<'a> {
    pub fn new(tuning: &GlobalMcmcTuning, n_items: usize, n_units: usize, pc: &mut Pc) -> Self {
        let unit_partitions_rval = R::new_list(n_units, pc);
        let mut unit_partitions = Vec::with_capacity(n_units);
        for t in 0..n_units {
            let rval = R::new_matrix_integer(n_items, tuning.n_saves, pc);
            unit_partitions.push(rval.slice());
            unit_partitions_rval.set(t, &rval).stop();
        }
        let permutations_rval = R::new_matrix_integer(n_items, tuning.n_saves, pc);
        let shrinkages_rval = R::new_vector_double(tuning.n_saves, pc);
        let log_likelihoods_rval = R::new_vector_double(tuning.n_saves, pc);
        let rval = R::new_list(7, pc); // Extra 3 items for rates after looping.
        rval.set_names(
            &[
                "unit_partitions",
                "permutation",
                "shrinkage",
                "log_likelihood",
                "permutation_acceptance_rate",
                "shrinkage_slice_n_evaluations_rate",
                "wall_times",
            ]
            .to_r(pc),
        )
        .stop();
        rval.set(0, &unit_partitions_rval).stop();
        rval.set(1, &permutations_rval).stop();
        rval.set(2, &shrinkages_rval).stop();
        rval.set(3, &log_likelihoods_rval).stop();
        Self {
            rval,
            counter: 0,
            n_items,
            unit_partitions,
            permutations: permutations_rval.slice(),
            shrinkages: shrinkages_rval.slice(),
            log_likelihoods: log_likelihoods_rval.slice(),
        }
    }

    pub fn push<'b, I: Iterator<Item = &'b Clustering>>(
        &mut self,
        unit_partitions: I,
        permutation: &Permutation,
        shrinkage: f64,
        log_likelihoods: f64,
    ) {
        let range = (self.counter * self.n_items)..((self.counter + 1) * self.n_items);
        for (clustering, full_slice) in unit_partitions.zip(self.unit_partitions.iter_mut()) {
            let slice = &mut full_slice[range.clone()];
            clustering.relabel_into_slice(1, slice);
        }
        let slice = &mut self.permutations[range];
        for (x, y) in slice.iter_mut().zip(permutation.as_slice()) {
            *x = *y as i32 + 1;
        }
        self.shrinkages[self.counter] = shrinkage;
        self.log_likelihoods[self.counter] = log_likelihoods;
        self.counter += 1;
    }
}

#[roxido]
fn fit_temporal_model(
    all_ptr: RObject,
    unit_mcmc_tuning: RObject,
    global_hyperparameters: RObject,
    global_mcmc_tuning: RObject,
) -> RObject {
    let all: &mut All = all_ptr.as_external_ptr().stop().decode_as_mut();
    let unit_mcmc_tuning = McmcTuning::from_r(unit_mcmc_tuning, pc);
    let global_hyperparameters = GlobalHyperparametersTemporal::from_r(global_hyperparameters, pc);
    let global_mcmc_tuning = GlobalMcmcTuning::from_r(global_mcmc_tuning, pc);
    let n_units = all.units.len();
    let mut results = ResultsTemporal::new(&global_mcmc_tuning, all.n_items, n_units, pc);
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rng2 = Pcg64Mcg::from_seed(seed);
    let fastrand = fastrand::Rng::with_seed(rng.gen());
    let fastrand_option = &mut Some(fastrand);
    let anchor = all.units[rng.gen_range(0..all.units.len())]
        .state
        .clustering()
        .clone();
    let mut shrinkage_value = ScalarShrinkage::new(1.0).unwrap();
    let shrinkage = Shrinkage::constant(shrinkage_value, all.n_items);
    let permutation = Permutation::random(all.n_items, &mut rng);
    let baseline_distribution =
        CrpParameters::new(all.n_items, global_hyperparameters.baseline_concentration);
    let mut partition_distribution = SpParameters::new(
        anchor,
        shrinkage,
        permutation,
        global_hyperparameters.grit,
        baseline_distribution,
    )
    .unwrap();
    struct Timers {
        units: TicToc,
        anchor: TicToc,
        permutation: TicToc,
        shrinkage: TicToc,
    }
    let mut timers = Timers {
        units: TicToc::new(),
        anchor: TicToc::new(),
        permutation: TicToc::new(),
        shrinkage: TicToc::new(),
    };
    let mut permutation_n_acceptances: u64 = 0;
    let mut shrinkage_slice_n_evaluations: u64 = 0;
    for loop_counter in 0..global_mcmc_tuning.n_loops {
        for _ in 0..global_mcmc_tuning.n_scans_per_loop {
            // Update each unit
            timers.units.tic();
            for time in 0..all.units.len() {
                let (left, not_left) = all.units.split_at_mut(time);
                if time == 0 {
                    shrinkage_value = partition_distribution.shrinkage[0];
                    partition_distribution
                        .shrinkage
                        .set_constant(ScalarShrinkage::zero());
                } else {
                    if time == 1 {
                        partition_distribution
                            .shrinkage
                            .set_constant(shrinkage_value);
                    }
                    partition_distribution.anchor = left.last().unwrap().state.clustering.clone();
                };
                let (middle, right) = not_left.split_at_mut(1);
                let unit = middle.first_mut().unwrap();
                let clustering_next = right.get(0).map(|x| &x.state.clustering);
                unit.data.impute(&unit.state, &mut rng);
                if unit_mcmc_tuning.update_precision_response {
                    State::update_precision_response(
                        &mut unit.state.precision_response,
                        &unit.state.global_coefficients,
                        &unit.state.clustering,
                        &unit.state.clustered_coefficients,
                        &unit.data,
                        &unit.hyperparameters,
                        &mut rng,
                    );
                }
                if unit_mcmc_tuning.update_global_coefficients {
                    State::update_global_coefficients(
                        &mut unit.state.global_coefficients,
                        unit.state.precision_response,
                        &unit.state.clustering,
                        &unit.state.clustered_coefficients,
                        &unit.data,
                        &unit.hyperparameters,
                        &mut rng,
                    );
                }
                if unit_mcmc_tuning.update_clustering {
                    let permutation =
                        Permutation::natural_and_fixed(unit.state.clustering.n_items());
                    State::update_clustering_temporal(
                        &mut unit.state.clustering,
                        &mut unit.state.clustered_coefficients,
                        unit.state.precision_response,
                        &unit.state.global_coefficients,
                        &permutation,
                        &unit.data,
                        &unit.hyperparameters,
                        &partition_distribution,
                        clustering_next,
                        &mut rng,
                        &mut rng2,
                    );
                }
                if unit_mcmc_tuning.update_clustered_coefficients {
                    State::update_clustered_coefficients(
                        &mut unit.state.clustered_coefficients,
                        &unit.state.clustering,
                        unit.state.precision_response,
                        &unit.state.global_coefficients,
                        &unit.data,
                        &unit.hyperparameters,
                        &mut rng,
                    );
                }
            }
            timers.units.toc();
            // Helper
            let compute_log_likelihood = |pd: &SpParameters<CrpParameters>| {
                all.units
                    .par_iter()
                    .fold_with(0.0, |acc, x| acc + pd.log_pmf(&x.state.clustering))
                    .sum::<f64>()
            };
            // Update permutation
            timers.permutation.tic();
            if let Some(k) = global_mcmc_tuning.n_items_per_permutation_update {
                let mut log_target_current: f64 = compute_log_likelihood(&partition_distribution);
                for _ in 0..global_mcmc_tuning.n_permutation_updates_per_scan {
                    partition_distribution
                        .permutation
                        .partial_shuffle(k, &mut rng);
                    let log_target_proposal = compute_log_likelihood(&partition_distribution);
                    let log_hastings_ratio = log_target_proposal - log_target_current;
                    if 0.0 <= log_hastings_ratio
                        || rng.gen_range(0.0..1.0_f64).ln() < log_hastings_ratio
                    {
                        log_target_current = log_target_proposal;
                        if loop_counter >= global_mcmc_tuning.n_loops_burnin {
                            permutation_n_acceptances += 1;
                        }
                    } else {
                        partition_distribution.permutation.partial_shuffle_undo(k);
                    }
                }
            }
            timers.permutation.toc();
            // Update shrinkage
            timers.shrinkage.tic();
            if let Some(w) = global_mcmc_tuning.shrinkage_slice_step_size {
                if let Some(reference) = global_hyperparameters.shrinkage.reference {
                    let shrinkage_prior_distribution = Gamma::new(
                        global_hyperparameters.shrinkage.shape.get(),
                        global_hyperparameters.shrinkage.rate.get(),
                    )
                    .unwrap();
                    let tuning_parameters = stepping_out::TuningParameters::new().width(w);
                    let (_s_new, n_evaluations) =
                        stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                            partition_distribution.shrinkage[reference].get(),
                            |s| match ScalarShrinkage::new(s) {
                                None => f64::NEG_INFINITY,
                                Some(shrinkage) => {
                                    partition_distribution
                                        .shrinkage
                                        .rescale_by_reference(reference, shrinkage);
                                    shrinkage_prior_distribution.ln_pdf(s)
                                        + compute_log_likelihood(&partition_distribution)
                                }
                            },
                            true,
                            &tuning_parameters,
                            fastrand_option,
                        );
                    // // Not necessary... see implementation of slice_sampler function.
                    // partition_distribution.shrinkage.rescale_by_reference(
                    //     global_hyperparameters.shrinkage_reference,
                    //     ScalarShrinkage::new(_s_new).unwrap(),
                    // );
                    if loop_counter >= global_mcmc_tuning.n_loops_burnin {
                        shrinkage_slice_n_evaluations += u64::from(n_evaluations);
                    }
                }
            }
            timers.shrinkage.toc();
        }
        // Report
        if loop_counter >= global_mcmc_tuning.n_loops_burnin {
            let log_likelihood = match &global_mcmc_tuning.validation_data {
                Some(validation_data) => all
                    .units
                    .par_iter()
                    .zip(validation_data.par_iter())
                    .fold(
                        || 0.0,
                        |acc, (group, data)| acc + group.state.log_likelihood(data),
                    )
                    .sum(),
                None => all
                    .units
                    .par_iter()
                    .fold(
                        || 0.0,
                        |acc, group| acc + group.state.log_likelihood(&group.data),
                    )
                    .sum(),
            };
            results.push(
                all.units.iter().map(|x| &x.state.clustering),
                &partition_distribution.permutation,
                if let Some(reference) = global_hyperparameters.shrinkage.reference {
                    partition_distribution.shrinkage[reference].get()
                } else {
                    f64::NAN
                },
                log_likelihood,
            );
        }
    }
    let denominator = (global_mcmc_tuning.n_saves * global_mcmc_tuning.n_scans_per_loop) as f64;
    results
        .rval
        .set(
            4,
            &(permutation_n_acceptances as f64
                / (global_mcmc_tuning.n_permutation_updates_per_scan as f64 * denominator))
                .to_r(pc),
        )
        .stop();
    results
        .rval
        .set(
            5,
            &(shrinkage_slice_n_evaluations as f64 / denominator).to_r(pc),
        )
        .stop();
    results
        .rval
        .set(
            6,
            &[
                timers.units.as_secs_f64(),
                timers.anchor.as_secs_f64(),
                timers.permutation.as_secs_f64(),
                timers.shrinkage.as_secs_f64(),
            ]
            .to_r(pc),
        )
        .stop();
    results.rval
}

struct GlobalHyperparametersHierarchical {
    grit: Grit,
    baseline_concentration: Concentration,
    anchor_concentration: Concentration,
    shrinkage_value: ScalarShrinkage,
    shrinkage_hyperparameter: ShrinkageHyperparameters,
}

impl GlobalHyperparametersHierarchical {
    fn from_r(x: RObject, _pc: &mut Pc) -> Self {
        let list = validate_list(
            x,
            &[
                "grit",
                "baseline_concentration",
                "anchor_concentration",
                "shrinkage_value",
                "shrinkage_hyperparameters",
            ],
            "global_hyperparameters_hierarchical",
        );
        Self {
            grit: Grit::new(
                list.get(0)
                    .stop()
                    .as_f64()
                    .stop_str("Grit should be a numeric"),
            )
            .unwrap_or_else(|| stop!("Invalid grit parameter")),
            baseline_concentration: Concentration::new(
                list.get(1)
                    .stop()
                    .as_f64()
                    .stop_str("Baseline concentration should be a numeric"),
            )
            .unwrap_or_else(|| stop!("Invalid concentration parameter")),
            anchor_concentration: Concentration::new(
                list.get(2)
                    .stop()
                    .as_f64()
                    .stop_str("Anchor concentration should be a numeric"),
            )
            .unwrap_or_else(|| stop!("Invalid concentration parameter")),
            shrinkage_value: ScalarShrinkage::new(
                list.get(3).stop().as_f64().stop_str(" should be a numeric"),
            )
            .unwrap_or_else(|| stop!("Invalid scalar shrinkage parameter")),
            shrinkage_hyperparameter: ShrinkageHyperparameters::from_r(list.get(4).stop()),
        }
    }
}

struct ResultsHierarchical<'a> {
    rval: RObject<roxido::r::Vector, roxido::r::List>,
    counter: usize,
    n_items: usize,
    unit_partitions: Vec<&'a mut [i32]>,
    anchors: &'a mut [i32],
    permutations: &'a mut [i32],
    shrinkages: &'a mut [f64],
    log_likelihoods: &'a mut [f64],
}

impl<'a> ResultsHierarchical<'a> {
    pub fn new(tuning: &GlobalMcmcTuning, n_items: usize, n_units: usize, pc: &mut Pc) -> Self {
        let unit_partitions_rval = R::new_list(n_units, pc);
        let mut unit_partitions = Vec::with_capacity(n_units);
        for t in 0..n_units {
            let rval = R::new_matrix_integer(n_items, tuning.n_saves, pc);
            unit_partitions.push(rval.slice());
            unit_partitions_rval.set(t, &rval).stop();
        }
        let anchors_rval = R::new_matrix_integer(n_items, tuning.n_saves, pc);
        let permutations_rval = R::new_matrix_integer(n_items, tuning.n_saves, pc);
        let shrinkages_rval = R::new_vector_double(tuning.n_saves, pc);
        let log_likelihoods_rval = R::new_vector_double(tuning.n_saves, pc);
        let rval = R::new_list(8, pc); // Extra 3 items for rates after looping.
        rval.set_names(
            &[
                "unit_partitions",
                "anchor",
                "permutation",
                "shrinkage",
                "log_likelihood",
                "permutation_acceptance_rate",
                "shrinkage_slice_n_evaluations_rate",
                "wall_times",
            ]
            .to_r(pc),
        )
        .stop();
        rval.set(0, &unit_partitions_rval).stop();
        rval.set(1, &anchors_rval).stop();
        rval.set(2, &permutations_rval).stop();
        rval.set(3, &shrinkages_rval).stop();
        rval.set(4, &log_likelihoods_rval).stop();
        Self {
            rval,
            counter: 0,
            n_items,
            unit_partitions,
            anchors: anchors_rval.slice(),
            permutations: permutations_rval.slice(),
            shrinkages: shrinkages_rval.slice(),
            log_likelihoods: log_likelihoods_rval.slice(),
        }
    }

    pub fn push<'b, I: Iterator<Item = &'b Clustering>>(
        &mut self,
        unit_partitions: I,
        anchor: &Clustering,
        permutation: &Permutation,
        shrinkage: f64,
        log_likelihoods: f64,
    ) {
        let range = (self.counter * self.n_items)..((self.counter + 1) * self.n_items);
        for (clustering, full_slice) in unit_partitions.zip(self.unit_partitions.iter_mut()) {
            let slice = &mut full_slice[range.clone()];
            clustering.relabel_into_slice(1, slice);
        }
        let slice = &mut self.anchors[range.clone()];
        anchor.relabel_into_slice(1, slice);
        let slice = &mut self.permutations[range];
        for (x, y) in slice.iter_mut().zip(permutation.as_slice()) {
            *x = *y as i32 + 1;
        }
        self.shrinkages[self.counter] = shrinkage;
        self.log_likelihoods[self.counter] = log_likelihoods;
        self.counter += 1;
    }
}

#[roxido]
fn fit_hierarchical_model(
    all_ptr: RObject,
    unit_mcmc_tuning: RObject,
    global_hyperparameters: RObject,
    global_mcmc_tuning: RObject,
) -> RObject {
    let all: &mut All = all_ptr.as_external_ptr().stop().decode_as_mut();
    let unit_mcmc_tuning = McmcTuning::from_r(unit_mcmc_tuning, pc);
    let global_hyperparameters =
        GlobalHyperparametersHierarchical::from_r(global_hyperparameters, pc);
    let global_mcmc_tuning = GlobalMcmcTuning::from_r(global_mcmc_tuning, pc);
    let n_units = all.units.len();
    let mut results = ResultsHierarchical::new(&global_mcmc_tuning, all.n_items, n_units, pc);
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rngs: Vec<_> = std::iter::repeat_with(|| {
        let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
        rng.fill(&mut seed);
        let r1 = Pcg64Mcg::from_seed(seed);
        rng.fill(&mut seed);
        let r2 = Pcg64Mcg::from_seed(seed);
        (r1, r2)
    })
    .take(all.units.len())
    .collect();
    let fastrand = fastrand::Rng::with_seed(rng.gen());
    let fastrand_option = &mut Some(fastrand);
    let anchor = all.units[rng.gen_range(0..all.units.len())]
        .state
        .clustering()
        .clone();
    let shrinkage = Shrinkage::constant(global_hyperparameters.shrinkage_value, all.n_items);
    let permutation = Permutation::random(all.n_items, &mut rng);
    let grit = global_hyperparameters.grit;
    let baseline_distribution =
        CrpParameters::new(all.n_items, global_hyperparameters.baseline_concentration);
    let anchor_distribution =
        CrpParameters::new(all.n_items, global_hyperparameters.anchor_concentration);
    let anchor_update_permutation = Permutation::natural_and_fixed(all.n_items);
    let mut partition_distribution =
        SpParameters::new(anchor, shrinkage, permutation, grit, baseline_distribution).unwrap();
    struct Timers {
        units: TicToc,
        anchor: TicToc,
        permutation: TicToc,
        shrinkage: TicToc,
    }
    let mut timers = Timers {
        units: TicToc::new(),
        anchor: TicToc::new(),
        permutation: TicToc::new(),
        shrinkage: TicToc::new(),
    };
    let mut permutation_n_acceptances = 0;
    let mut shrinkage_slice_n_evaluations = 0;
    for loop_counter in 0..global_mcmc_tuning.n_loops {
        for _ in 0..global_mcmc_tuning.n_scans_per_loop {
            // Update each unit
            timers.units.tic();
            all.units
                .par_iter_mut()
                .zip(rngs.par_iter_mut())
                .for_each(|(unit, (r1, r2))| {
                    unit.state.mcmc_iteration(
                        &unit_mcmc_tuning,
                        &mut unit.data,
                        &unit.hyperparameters,
                        &partition_distribution,
                        r1,
                        r2,
                    )
                });
            timers.units.toc();
            // Update anchor
            timers.anchor.tic();
            let mut pd = partition_distribution.clone();
            let mut compute_log_likelihood = |item: usize, anchor: &Clustering| {
                pd.anchor = anchor.clone();
                all.units
                    .par_iter()
                    .fold_with(0.0, |acc, x| {
                        acc + pd.log_pmf_partial(item, &x.state.clustering)
                    })
                    .sum::<f64>()
            };
            if global_mcmc_tuning.update_anchor {
                update_partition_gibbs(
                    1,
                    &mut partition_distribution.anchor,
                    &anchor_update_permutation,
                    &anchor_distribution,
                    &mut compute_log_likelihood,
                    &mut rng,
                )
            }
            timers.anchor.toc();
            // Helper
            let compute_log_likelihood = |pd: &SpParameters<CrpParameters>| {
                all.units
                    .par_iter()
                    .fold_with(0.0, |acc, x| acc + pd.log_pmf(&x.state.clustering))
                    .sum::<f64>()
            };
            // Update permutation
            timers.permutation.tic();
            if let Some(k) = global_mcmc_tuning.n_items_per_permutation_update {
                let mut log_target_current: f64 = compute_log_likelihood(&partition_distribution);
                for _ in 0..global_mcmc_tuning.n_permutation_updates_per_scan {
                    partition_distribution
                        .permutation
                        .partial_shuffle(k, &mut rng);
                    let log_target_proposal = compute_log_likelihood(&partition_distribution);
                    let log_hastings_ratio = log_target_proposal - log_target_current;
                    if 0.0 <= log_hastings_ratio
                        || rng.gen_range(0.0..1.0_f64).ln() < log_hastings_ratio
                    {
                        log_target_current = log_target_proposal;
                        if loop_counter >= global_mcmc_tuning.n_loops_burnin {
                            permutation_n_acceptances += 1;
                        }
                    } else {
                        partition_distribution.permutation.partial_shuffle_undo(k);
                    }
                }
            }
            timers.permutation.toc();
            // Update shrinkage
            timers.shrinkage.tic();
            if let Some(w) = global_mcmc_tuning.shrinkage_slice_step_size {
                if let Some(reference) = global_hyperparameters.shrinkage_hyperparameter.reference {
                    let shrinkage_prior_distribution = Gamma::new(
                        global_hyperparameters.shrinkage_hyperparameter.shape.get(),
                        global_hyperparameters.shrinkage_hyperparameter.rate.get(),
                    )
                    .unwrap();
                    let tuning_parameters = stepping_out::TuningParameters::new().width(w);
                    let (_s_new, n_evaluations) =
                        stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                            partition_distribution.shrinkage[reference].get(),
                            |s| match ScalarShrinkage::new(s) {
                                None => f64::NEG_INFINITY,
                                Some(shrinkage) => {
                                    partition_distribution
                                        .shrinkage
                                        .rescale_by_reference(reference, shrinkage);
                                    shrinkage_prior_distribution.ln_pdf(s)
                                        + compute_log_likelihood(&partition_distribution)
                                }
                            },
                            true,
                            &tuning_parameters,
                            fastrand_option,
                        );
                    // // Not necessary... see implementation of slice_sampler function.
                    // partition_distribution.shrinkage.rescale_by_reference(
                    //     global_hyperparameters.shrinkage_reference,
                    //     ScalarShrinkage::new(_s_new).unwrap(),
                    // );
                    if loop_counter >= global_mcmc_tuning.n_loops_burnin {
                        shrinkage_slice_n_evaluations += n_evaluations;
                    }
                }
            }
            timers.shrinkage.toc();
        }
        // Report
        if loop_counter >= global_mcmc_tuning.n_loops_burnin {
            let log_likelihood = match &global_mcmc_tuning.validation_data {
                Some(validation_data) => all
                    .units
                    .par_iter()
                    .zip(validation_data.par_iter())
                    .fold(
                        || 0.0,
                        |acc, (group, data)| acc + group.state.log_likelihood(data),
                    )
                    .sum(),
                None => all
                    .units
                    .par_iter()
                    .fold(
                        || 0.0,
                        |acc, group| acc + group.state.log_likelihood(&group.data),
                    )
                    .sum(),
            };
            results.push(
                all.units.iter().map(|x| &x.state.clustering),
                &partition_distribution.anchor,
                &partition_distribution.permutation,
                if let Some(reference) = global_hyperparameters.shrinkage_hyperparameter.reference {
                    partition_distribution.shrinkage[reference].get()
                } else {
                    f64::NAN
                },
                log_likelihood,
            );
        }
    }
    let denominator = (global_mcmc_tuning.n_saves * global_mcmc_tuning.n_scans_per_loop) as f64;
    results
        .rval
        .set(
            5,
            &(permutation_n_acceptances as f64
                / (global_mcmc_tuning.n_permutation_updates_per_scan as f64 * denominator))
                .to_r(pc),
        )
        .stop();
    results
        .rval
        .set(
            6,
            &(shrinkage_slice_n_evaluations as f64 / denominator).to_r(pc),
        )
        .stop();
    results
        .rval
        .set(
            7,
            &[
                timers.units.as_secs_f64(),
                timers.anchor.as_secs_f64(),
                timers.permutation.as_secs_f64(),
                timers.shrinkage.as_secs_f64(),
            ]
            .to_r(pc),
        )
        .stop();
    results.rval
}

#[roxido]
fn fit(
    n_updates: RObject,
    data: RObject,
    state: RObject,
    hyperparameters: RObject,
    monitor: RObject,
    partition_distribution: RObject,
    mcmc_tuning: RObject,
    permutation: RObject,
    shrinkage: RObject,
    grit: RObject,
    rngs: RObject,
) -> RObject {
    let n_updates: u32 = n_updates.as_i32().stop().try_into().unwrap();
    let data: &mut Data = data.as_external_ptr().stop().decode_as_mut();
    let state: &mut State = state.as_external_ptr().stop().decode_as_mut();
    let hyperparameters: &Hyperparameters =
        hyperparameters.as_external_ptr().stop().decode_as_ref();
    let monitor = monitor
        .as_external_ptr()
        .stop()
        .decode_as_mut::<Monitor<u32>>();
    let mcmc_tuning = McmcTuning::from_r(mcmc_tuning, pc);
    if data.n_global_covariates() != state.n_global_covariates()
        || hyperparameters.n_global_covariates() != state.n_global_covariates()
    {
        stop!("Inconsistent number of global covariates.")
    }
    if data.n_clustered_covariates() != state.n_clustered_covariates()
        || hyperparameters.n_clustered_covariates() != state.n_clustered_covariates()
    {
        stop!("Inconsistent number of clustered covariates.")
    }
    if data.n_items() != state.clustering.n_items() {
        stop!("Inconsistent number of items.")
    }
    let rngs = rngs.as_list().stop();
    let getrng = |i: usize| {
        rngs.get(i)
            .stop()
            .as_external_ptr()
            .stop()
            .decode_as_mut::<Pcg64Mcg>()
    };
    let rng = getrng(0);
    let rng2 = getrng(1);
    #[rustfmt::skip]
    macro_rules! mcmc_update { // (_, HAS_PERMUTATION, HAS_SCALAR_SHRINKAGE, HAS_VECTOR_SHRINKAGE, HAS_GRIT)
        ($tipe:ty, false, false, false, false) => {{
            let partition_distribution = partition_distribution.as_external_ptr().stop().decode_as_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
            }
        }};
        ($tipe:ty, true, false, false, false) => {{
            let partition_distribution = partition_distribution.as_external_ptr().stop().decode_as_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, &permutation);
            }
        }};
        ($tipe:ty, true, true, false, false) => {{
            let partition_distribution = partition_distribution.as_external_ptr().stop().decode_as_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, &permutation);
                if let Some(w) = mcmc_tuning.shrinkage_slice_step_size {
                    if let Some(ShrinkageHyperparameters{shape, rate, ..}) = &hyperparameters.shrinkage_option {
                        dahl_randompartition::mcmc::update_scalar_shrinkage(1, partition_distribution, w, *shape, *rate, &state.clustering, rng);
                        scalar_shrinkage_to_r(&partition_distribution.shrinkage, &shrinkage);
                    }
                }
            }
        }};
        ($tipe:ty, true, false, true, true) => {{
            let partition_distribution = partition_distribution.as_external_ptr().stop().decode_as_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, &permutation);
                if let Some(w) = mcmc_tuning.shrinkage_slice_step_size {
                    if let Some(ShrinkageHyperparameters{reference: Some(reference), shape, rate}) = &hyperparameters.shrinkage_option {
                        dahl_randompartition::mcmc::update_shrinkage(1, partition_distribution, *reference, w, *shape, *rate, &state.clustering, rng);
                        shrinkage_to_r(&partition_distribution.shrinkage, &shrinkage);
                    }
                }
                if let Some(w) = mcmc_tuning.grit_slice_step_size {
                    if let Some(GritHyperparameters{shape1, shape2}) = &hyperparameters.grit_option {
                        dahl_randompartition::mcmc::update_grit(1, partition_distribution, w, *shape1, *shape2, &state.clustering, rng);
                        grit_to_r(&partition_distribution.grit, &grit);
                    }
                }
            }
        }};
    }
    let tag = partition_distribution.as_external_ptr().stop().tag();
    let prior_name = tag.as_str().stop();
    match prior_name {
        "fixed" => mcmc_update!(FixedPartitionParameters, false, false, false, false),
        "up" => mcmc_update!(UpParameters, false, false, false, false),
        "jlp" => mcmc_update!(JlpParameters, true, false, false, false),
        "crp" => mcmc_update!(CrpParameters, false, false, false, false),
        "epa" => mcmc_update!(EpaParameters, true, false, false, false),
        "lsp" => mcmc_update!(LspParameters, true, true, false, false),
        "cpp-up" => mcmc_update!(CppParameters<UpParameters>, false, false, false, false),
        "cpp-jlp" => mcmc_update!(CppParameters<JlpParameters>, false, false, false, false),
        "cpp-crp" => mcmc_update!(CppParameters<CrpParameters>, false, false, false, false),
        "sp-up" => mcmc_update!(SpParameters<UpParameters>, true, false, true, true),
        "sp-jlp" => mcmc_update!(SpParameters<JlpParameters>, true, false, true, true),
        "sp-crp" => mcmc_update!(SpParameters<CrpParameters>, true, false, true, true),
        _ => stop!("Unsupported distribution: {}", prior_name),
    }
    state.canonicalize();
}

#[roxido]
fn log_likelihood_contributions(state: RObject, data: RObject) -> RObject {
    let state: &State = state.as_external_ptr().stop().decode_as_ref();
    let data = data.as_external_ptr().stop().decode_as_ref();
    state.log_likelihood_contributions(data).iter().to_r(pc)
}

#[roxido]
fn log_likelihood_contributions_of_missing(state: RObject, data: RObject) -> RObject {
    let state: &State = state.as_external_ptr().stop().decode_as_ref();
    let data = data.as_external_ptr().stop().decode_as_ref();
    state
        .log_likelihood_contributions_of_missing(data)
        .iter()
        .to_r(pc)
}

#[roxido]
fn log_likelihood_of(state: RObject, data: RObject, items: RObject) -> RObject {
    let state: &State = state.as_external_ptr().stop().decode_as_ref();
    let data = data.as_external_ptr().stop().decode_as_ref();
    let items = items.as_vector().stop().to_mode_integer(pc).slice();
    let items: Vec<_> = items
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    state.log_likelihood_of(data, items)
}

#[roxido]
fn log_likelihood(state: RObject, data: RObject) -> RObject {
    let state: &State = state.as_external_ptr().stop().decode_as_ref();
    let data = data.as_external_ptr().stop().decode_as_ref();
    state.log_likelihood(data)
}

#[roxido]
fn sample_multivariate_normal(n_samples: RObject, mean: RObject, precision: RObject) -> RObject {
    let n_samples = n_samples.as_usize().stop();
    let mean = mean.as_vector().stop().to_mode_double(pc).slice();
    let n = mean.len();
    let mean = DVector::from_iterator(n, mean.iter().cloned());
    let precision = precision.as_matrix().stop().to_mode_double(pc).slice();
    let precision = DMatrix::from_iterator(n, n, precision.iter().cloned());
    let rval = R::new_matrix_double(n, n_samples, pc);
    let slice = rval.slice();
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let x = sample_multivariate_normal_repeatedly(n_samples, mean, precision, &mut rng).unwrap();
    slice.clone_from_slice(x.as_slice());
    rval.transpose(pc)
}

fn validate_list(
    x: RObject,
    expected_names: &[&str],
    arg_name: &str,
) -> RObject<roxido::r::Vector, roxido::r::List> {
    let list = x
        .as_list()
        .stop_closure(|| format!("'{}' should be a list or NULL.", arg_name));
    if list.len() != expected_names.len() {
        stop!("'{}' must be of length {}", arg_name, expected_names.len());
    }
    let names = list.get_names();
    if names.len() != expected_names.len() {
        stop!(
            "'{}' must be an named list with names:\n    {}",
            arg_name,
            expected_names.join("\n    ")
        );
    }
    for (i, name) in expected_names.iter().enumerate() {
        if *name != names.get(i).unwrap() {
            stop!("Element {} of '{}' should be named '{}'", i, arg_name, name);
        }
    }
    list
}
