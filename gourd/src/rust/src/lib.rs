mod data;
mod hyperparameters;
mod membership;
mod monitor;
mod mvnorm;
mod registration;
mod state;

use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::monitor::Monitor;
use crate::mvnorm::sample_multivariate_normal_repeatedly;
use crate::state::{McmcTuning, State};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::cpp::CppParameters;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::distr::ProbabilityMassFunction;
use dahl_randompartition::epa::EpaParameters;
use dahl_randompartition::fixed::FixedPartitionParameters;
use dahl_randompartition::frp::FrpParameters;
use dahl_randompartition::jlp::JlpParameters;
use dahl_randompartition::lsp::LspParameters;
use dahl_randompartition::mcmc::{update_neal_algorithm_full, update_partition_gibbs};
use dahl_randompartition::old2sp::Old2SpParameters;
use dahl_randompartition::oldsp::OldSpParameters;
use dahl_randompartition::perm::Permutation;
use dahl_randompartition::prelude::Mass;
use dahl_randompartition::shrink::{Shrinkage, ShrinkageProbabilities};
use dahl_randompartition::sp::SpParameters;
use dahl_randompartition::sp_mixture::SpMixtureParameters;
use dahl_randompartition::up::UpParameters;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use roxido::*;
use slice_sampler::univariate::stepping_out;
use statrs::distribution::{Continuous, Gamma};

#[roxido]
fn rngs_new() -> Rval {
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let rng2 = Pcg64Mcg::from_seed(seed);
    let result = Rval::new_list(2, pc);
    result.set_list_element(0, Rval::external_pointer_encode(rng, rval!("rng")));
    result.set_list_element(1, Rval::external_pointer_encode(rng2, rval!("rng")));
    result
}

#[roxido]
fn state_r2rust(state: Rval) -> Rval {
    let state = State::from_r(state, pc);
    Rval::external_pointer_encode(state, rval!("state"))
}

#[roxido]
fn state_rust2r_as_reference(state: Rval) -> Rval {
    let state = Rval::external_pointer_decode_as_ref::<State>(state);
    state.to_r(pc)
}

#[roxido]
fn monitor_new() -> Rval {
    Rval::external_pointer_encode(Monitor::<u32>::new(), rval!("monitor"))
}

#[roxido]
fn monitor_rate(monitor: Rval) -> Rval {
    rval!(Rval::external_pointer_decode_as_ref::<Monitor<u32>>(monitor).rate())
}

#[roxido]
fn monitor_reset(monitor: Rval) -> Rval {
    let monitor = Rval::external_pointer_decode_as_mut_ref::<Monitor<u32>>(monitor);
    monitor.reset();
    Rval::nil()
}

#[roxido]
fn hyperparameters_r2rust(hyperparameters: Rval) -> Rval {
    Rval::external_pointer_encode(
        Hyperparameters::from_r(hyperparameters, pc),
        rval!("hyperparameters"),
    )
}

#[roxido]
fn data_r2rust(data: Rval, missing_items: Rval) -> Rval {
    let mut data = Data::from_r(data, pc);
    let (_, missing_items) = missing_items.coerce_integer(pc).unwrap();
    let missing_items: Vec<_> = missing_items
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    data.declare_missing(missing_items);
    Rval::external_pointer_encode(data, rval!("data"))
}

#[roxido]
fn rust_free(x: Rval) -> Rval {
    match x.external_pointer_tag().as_str() {
        "data" => {
            let _ = x.external_pointer_decode::<Data>();
        }
        "state" => {
            let _ = x.external_pointer_decode::<State>();
        }
        "hyperparameters" => {
            let _ = x.external_pointer_decode::<Hyperparameters>();
        }
        "monitor" => {
            let _ = x.external_pointer_decode::<Monitor<u32>>();
        }
        "rng" => {
            let _ = x.external_pointer_decode::<Pcg64Mcg>();
        }
        str => {
            panic!("Unrecognized type ID: {}.", str)
        }
    };
    Rval::nil()
}

fn permutation_to_r(permutation: &Permutation, rval: Rval) {
    for (x, y) in permutation
        .as_slice()
        .iter()
        .zip(rval.slice_mut_integer().unwrap())
    {
        *y = i32::try_from(*x).unwrap() + 1;
    }
}

fn rate_to_r(rate: f64, rval: Rval) {
    rval.slice_mut_double().unwrap()[0] = rate;
}

fn shrinkage_to_r(shrinkage: &Shrinkage, rval: Rval) {
    rval.slice_mut_double()
        .unwrap()
        .copy_from_slice(shrinkage.as_slice());
}

fn shrinkage_probabilities_to_r(shrinkage_probabilities: &ShrinkageProbabilities, rval: Rval) {
    rval.slice_mut_double()
        .unwrap()
        .copy_from_slice(shrinkage_probabilities.as_slice());
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
fn all(all: Rval) -> Rval {
    let mut vec = Vec::with_capacity(all.len());
    let mut n_items = None;
    for i in 0..all.len() {
        let list = all.get_list_element(i);
        let (data, state, hyperparameters) = (
            list.get_list_element(0),
            list.get_list_element(1),
            list.get_list_element(2),
        );
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
    Rval::external_pointer_encode(all, rval!("all"))
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
}

impl GlobalMcmcTuning {
    fn from_r(x: Rval, _pc: &mut Pc) -> Self {
        let n_scans_per_loop = x.get_list_element(2).as_usize();
        let n_loops = x.get_list_element(0).as_usize() / n_scans_per_loop;
        let n_loops_burnin = x.get_list_element(1).as_usize() / n_scans_per_loop;
        let n_saves = n_loops - n_loops_burnin;
        let update_anchor = x.get_list_element(3).as_bool();
        let n_permutation_updates_per_scan = x.get_list_element(4).as_usize();
        let x1 = x.get_list_element(5);
        let n_items_per_permutation_update = if x1.is_nil() {
            None
        } else {
            Some(x1.as_usize())
        };
        let x2 = x.get_list_element(6);
        let shrinkage_slice_step_size = if x2.is_nil() { None } else { Some(x1.as_f64()) };
        Self {
            n_loops,
            n_loops_burnin,
            n_scans_per_loop,
            n_saves,
            update_anchor,
            n_permutation_updates_per_scan,
            n_items_per_permutation_update,
            shrinkage_slice_step_size,
        }
    }
}

struct GlobalHyperparameters {
    baseline_mass: f64,
    anchor_mass: f64,
    shrinkage_reference: usize,
    shrinkage_shape: f64,
    shrinkage_rate: f64,
}

impl GlobalHyperparameters {
    fn from_r(x: Rval, _pc: &mut Pc) -> Self {
        Self {
            baseline_mass: x.get_list_element(0).as_f64(),
            anchor_mass: x.get_list_element(1).as_f64(),
            shrinkage_reference: x.get_list_element(2).as_usize() - 1,
            shrinkage_shape: x.get_list_element(3).as_f64(),
            shrinkage_rate: x.get_list_element(4).as_f64(),
        }
    }
}

struct Results<'a> {
    rval: Rval,
    counter: usize,
    n_items: usize,
    unit_partitions: Vec<&'a mut [i32]>,
    anchors: &'a mut [i32],
    permutations: &'a mut [i32],
    shrinkages: &'a mut [f64],
    log_likelihoods: &'a mut [f64],
}

impl<'a> Results<'a> {
    pub fn new(tuning: &GlobalMcmcTuning, n_items: usize, n_units: usize, pc: &mut Pc) -> Self {
        let unit_partitions_rval = Rval::new_list(n_units, pc);
        let mut unit_partitions = Vec::with_capacity(n_units);
        for t in 0..n_units {
            let (rval, slice) = Rval::new_matrix_integer(n_items, tuning.n_saves, pc);
            unit_partitions.push(slice);
            unit_partitions_rval.set_list_element(t, rval);
        }
        let (anchors_rval, anchors) = Rval::new_matrix_integer(n_items, tuning.n_saves, pc);
        let (permutations_rval, permutations) =
            Rval::new_matrix_integer(n_items, tuning.n_saves, pc);
        let (shrinkages_rval, shrinkages) = Rval::new_vector_double(tuning.n_saves, pc);
        let (log_likelihoods_rval, log_likelihoods) = Rval::new_vector_double(tuning.n_saves, pc);
        let rval = Rval::new_list(7, pc); // Extra 2 items for rates after looping.
        rval.names_gets(Rval::new(
            [
                "unit_partitions",
                "anchor",
                "permutation",
                "shrinkage",
                "log_likelihood",
                "permutation_acceptance_rate",
                "shrinkage_slice_n_evaluations_rate",
            ],
            pc,
        ));
        rval.set_list_element(0, unit_partitions_rval);
        rval.set_list_element(1, anchors_rval);
        rval.set_list_element(2, permutations_rval);
        rval.set_list_element(3, shrinkages_rval);
        rval.set_list_element(4, log_likelihoods_rval);
        Self {
            rval,
            counter: 0,
            n_items,
            unit_partitions,
            anchors,
            permutations,
            shrinkages,
            log_likelihoods,
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
    all_ptr: Rval,
    unit_mcmc_tuning: Rval,
    global_hyperparameters: Rval,
    global_mcmc_tuning: Rval,
    validation_data: Rval,
) -> Rval {
    let mut all: All = all_ptr.external_pointer_decode();
    let unit_mcmc_tuning = McmcTuning::from_r(unit_mcmc_tuning, pc);
    let global_hyperparameters = GlobalHyperparameters::from_r(global_hyperparameters, pc);
    let global_mcmc_tuning = GlobalMcmcTuning::from_r(global_mcmc_tuning, pc);
    let n_units = validation_data.len();
    let validation_data = {
        let mut vd = Vec::with_capacity(n_units);
        for k in 0..n_units {
            vd.push(Data::from_r(validation_data.get_list_element(k), pc));
        }
        vd
    };
    let mut results = Results::new(&global_mcmc_tuning, all.n_items, n_units, pc);
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
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
    let fastrand_option = Some(&fastrand);
    let anchor = all.units[rng.gen_range(0..all.units.len())]
        .state
        .clustering()
        .clone();
    let shrinkage = Shrinkage::constant(1.0, all.n_items).unwrap();
    let permutation = Permutation::random(all.n_items, &mut rng);
    let baseline_mass = Mass::new(global_hyperparameters.baseline_mass);
    let anchor_mass = Mass::new(global_hyperparameters.anchor_mass);
    let baseline_distribution = CrpParameters::new_with_mass(all.n_items, baseline_mass);
    let anchor_distribution = CrpParameters::new_with_mass(all.n_items, anchor_mass);
    let anchor_update_permutation = Permutation::natural_and_fixed(all.n_items);
    let mut partition_distribution =
        SpParameters::new(anchor, shrinkage, permutation, baseline_distribution).unwrap();
    let mut permutation_n_acceptances = 0;
    let mut shrinkage_slice_n_evaluations = 0;
    for loop_counter in 0..global_mcmc_tuning.n_loops {
        for _ in 0..global_mcmc_tuning.n_scans_per_loop {
            // Update each unit
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
            // Update anchor
            let mut pd = partition_distribution.clone();
            let mut compute_log_likelihood = |anchor: &Clustering| {
                pd.anchor = anchor.clone();
                all.units
                    .par_iter()
                    .fold_with(0.0, |acc, x| acc + pd.log_pmf(&x.state.clustering))
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
            // Helper
            let compute_log_likelihood = |pd: &SpParameters<CrpParameters>| {
                all.units
                    .par_iter()
                    .fold_with(0.0, |acc, x| acc + pd.log_pmf(&x.state.clustering))
                    .sum::<f64>()
            };
            // Update permutation
            if let Some(k) = global_mcmc_tuning.n_items_per_permutation_update {
                for _ in 0..global_mcmc_tuning.n_permutation_updates_per_scan {
                    let log_target_current: f64 = compute_log_likelihood(&partition_distribution);
                    partition_distribution
                        .permutation
                        .partial_shuffle(k, &mut rng);
                    let log_target_proposal: f64 = compute_log_likelihood(&partition_distribution);
                    let log_hastings_ratio = log_target_proposal - log_target_current;
                    if 0.0 <= log_hastings_ratio
                        || rng.gen_range(0.0..1.0_f64).ln() < log_hastings_ratio
                    {
                        if loop_counter >= global_mcmc_tuning.n_loops_burnin {
                            permutation_n_acceptances += 1;
                        }
                    } else {
                        partition_distribution.permutation.partial_shuffle_undo(k);
                    }
                }
            }
            // Update shrinkage
            if global_mcmc_tuning.shrinkage_slice_step_size.is_some() {
                let shrinkage_prior_distribution = Gamma::new(
                    global_hyperparameters.shrinkage_shape,
                    global_hyperparameters.shrinkage_rate,
                )
                .unwrap();
                let (s_new, n_evaluations) =
                    stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                        partition_distribution.shrinkage
                            [global_hyperparameters.shrinkage_reference],
                        |s| {
                            if s < 0.0 {
                                return std::f64::NEG_INFINITY;
                            }
                            partition_distribution.shrinkage.rescale_by_reference(
                                global_hyperparameters.shrinkage_reference,
                                s,
                            );
                            shrinkage_prior_distribution.ln_pdf(s)
                                + compute_log_likelihood(&partition_distribution)
                        },
                        true,
                        &stepping_out::TuningParameters::new()
                            .width(global_mcmc_tuning.shrinkage_slice_step_size.unwrap()),
                        fastrand_option,
                    );
                partition_distribution
                    .shrinkage
                    .rescale_by_reference(global_hyperparameters.shrinkage_reference, s_new);
                if loop_counter >= global_mcmc_tuning.n_loops_burnin {
                    shrinkage_slice_n_evaluations += n_evaluations;
                }
            }
        }
        // Report
        if loop_counter >= global_mcmc_tuning.n_loops_burnin {
            let log_likelihood_of_validation = all
                .units
                .iter()
                .zip(validation_data.iter())
                .fold(0.0, |acc, (group, data)| {
                    acc + group.state.log_likelihood(data)
                });
            results.push(
                all.units.iter().map(|x| &x.state.clustering),
                &partition_distribution.anchor,
                &partition_distribution.permutation,
                partition_distribution.shrinkage[global_hyperparameters.shrinkage_reference],
                log_likelihood_of_validation,
            );
        }
    }
    let denominator = (global_mcmc_tuning.n_saves * global_mcmc_tuning.n_scans_per_loop) as f64;
    results.rval.set_list_element(
        5,
        rval!(
            permutation_n_acceptances as f64
                / (global_mcmc_tuning.n_permutation_updates_per_scan as f64 * denominator)
        ),
    );
    results
        .rval
        .set_list_element(6, rval!(shrinkage_slice_n_evaluations as f64 / denominator));
    results.rval
}

#[roxido]
fn fit_all(all_ptr: Rval, shrinkage: Rval, n_updates: Rval, do_baseline_partition: Rval) -> Rval {
    let mut all: All = all_ptr.external_pointer_decode();
    let n_items = all.n_items;
    let fixed = McmcTuning::new(false, false, false, false, Some(2), Some(1.0)).unwrap();
    let shrinkage = Shrinkage::constant(shrinkage.as_f64(), n_items).unwrap();
    let n_updates = n_updates.as_usize();
    let do_baseline_partition = do_baseline_partition.as_bool();
    let (result_rval, result_slice) = if do_baseline_partition {
        Rval::new_matrix_integer(n_items, n_updates, pc)
    } else {
        (Rval::nil(), &mut [] as &mut [i32])
    };
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rng2 = Pcg64Mcg::from_seed(seed);
    let permutation = Permutation::natural_and_fixed(n_items);
    let hyperpartition_prior_distribution = CrpParameters::new_with_mass(n_items, Mass::new(1.0));
    let anchor_initial = all.units[rng.gen_range(0..all.units.len())]
        .state
        .clustering()
        .clone();
    let mut grand_sum = 0.0;
    let upper = if do_baseline_partition { 1 } else { n_items };
    for missing_item in 0..upper {
        let mut sum = 0.0;
        if !do_baseline_partition {
            println!("Missing: {}", missing_item);
            for group in all.units.iter_mut() {
                group.data.declare_missing(vec![missing_item]);
            }
        }
        let mut partition_distribution = SpParameters::new(
            anchor_initial.clone(),
            shrinkage.clone(),
            Permutation::natural_and_fixed(n_items),
            CrpParameters::new_with_mass(n_items, Mass::new(1.0)),
        )
        .unwrap();
        for update_index in 0..n_updates {
            for group in all.units.iter_mut() {
                group.state.mcmc_iteration(
                    &fixed,
                    &mut group.data,
                    &group.hyperparameters,
                    &partition_distribution,
                    &mut rng,
                    &mut rng2,
                );
            }
            let mut anchor_tmp = partition_distribution.anchor.clone();
            let mut log_likelihood_contribution_fn = |proposed_anchor: &Clustering| {
                partition_distribution.anchor = proposed_anchor.clone();
                let mut sum = 0.0;
                for group in all.units.iter() {
                    sum += partition_distribution.log_pmf(group.state.clustering());
                }
                sum
            };
            update_neal_algorithm_full(
                1,
                &mut anchor_tmp,
                &permutation,
                &hyperpartition_prior_distribution,
                &mut log_likelihood_contribution_fn,
                &mut rng,
            );
            if do_baseline_partition {
                partition_distribution.anchor = anchor_tmp.standardize();
                partition_distribution.anchor.relabel_into_slice(
                    1_i32,
                    &mut result_slice[(update_index * n_items)..((update_index + 1) * n_items)],
                );
            } else {
                for group in all.units.iter_mut() {
                    sum += group
                        .state
                        .log_likelihood_contributions_of_missing(&group.data)
                        .iter()
                        .sum::<f64>();
                }
            }
        }
        grand_sum += sum / (n_updates as f64);
    }
    if do_baseline_partition {
        result_rval
    } else {
        rval!(grand_sum)
    }
}

#[roxido]
fn fit(
    n_updates: Rval,
    data: Rval,
    state: Rval,
    hyperparameters: Rval,
    monitor: Rval,
    partition_distribution: Rval,
    mcmc_tuning: Rval,
    permutation: Rval,
    shrinkage: Rval,
    rngs: Rval,
) -> Rval {
    let n_updates: u32 = n_updates.as_i32().try_into().unwrap();
    let data = Rval::external_pointer_decode_as_mut_ref::<Data>(data);
    let state_tag = state.external_pointer_tag();
    let mut state = Rval::external_pointer_decode::<State>(state);
    let hyperparameters = Rval::external_pointer_decode_as_ref::<Hyperparameters>(hyperparameters);
    let monitor_tag = monitor.external_pointer_tag();
    let mut monitor = Rval::external_pointer_decode::<Monitor<u32>>(monitor);
    let mcmc_tuning = McmcTuning::from_r(mcmc_tuning, pc);
    if data.n_global_covariates() != state.n_global_covariates()
        || hyperparameters.n_global_covariates() != state.n_global_covariates()
    {
        panic!("Inconsistent number of global covariates.")
    }
    if data.n_clustered_covariates() != state.n_clustered_covariates()
        || hyperparameters.n_clustered_covariates() != state.n_clustered_covariates()
    {
        panic!("Inconsistent number of clustered covariates.")
    }
    if data.n_items() != state.clustering.n_items() {
        panic!("Inconsistent number of items.")
    }
    let rng = rngs
        .get_list_element(0)
        .external_pointer_decode_as_mut_ref::<Pcg64Mcg>();
    let rng2 = rngs
        .get_list_element(1)
        .external_pointer_decode_as_mut_ref::<Pcg64Mcg>();
    #[rustfmt::skip]
    macro_rules! mcmc_update { // (_, HAS_PERMUTATION, HAS_SCALAR_SHRINKAGE, HAS_VECTOR_SHRINKAGE, HAS_VECTOR_SHRINKAGE_PROBABILITY)
        ($tipe:ty, false, false, false, false) => {{
            let partition_distribution = partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
            }
        }};
        ($tipe:ty, true, false, false, false) => {{
            let partition_distribution = partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation);
            }
        }};
        ($tipe:ty, true, true, false, false) => {{
            let partition_distribution = partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation);
                dahl_randompartition::mcmc::update_scalar_shrinkage(1, partition_distribution, mcmc_tuning.shrinkage_slice_step_size.unwrap(), hyperparameters.shrinkage_shape.unwrap(), hyperparameters.shrinkage_rate.unwrap(), &state.clustering, rng);
                rate_to_r(partition_distribution.rate, shrinkage);
            }
        }};
        ($tipe:ty, true, false, true, false) => {{
            let partition_distribution = partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation);
                dahl_randompartition::mcmc::update_vector_shrinkage(1, partition_distribution, hyperparameters.shrinkage_reference.unwrap(), mcmc_tuning.shrinkage_slice_step_size.unwrap(), hyperparameters.shrinkage_shape.unwrap(), hyperparameters.shrinkage_rate.unwrap(), &state.clustering, rng);
                shrinkage_to_r(&partition_distribution.shrinkage, shrinkage);
            }
        }};
        ($tipe:ty, true, false, false, true) => {{
            let partition_distribution = partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(1, |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, mcmc_tuning.n_items_per_permutation_update.unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation);
                dahl_randompartition::mcmc::update_vector_shrinkage_probabilities(1, partition_distribution, hyperparameters.shrinkage_reference.unwrap(), mcmc_tuning.shrinkage_slice_step_size.unwrap(), hyperparameters.shrinkage_shape.unwrap(), hyperparameters.shrinkage_rate.unwrap(), &state.clustering, rng);
                shrinkage_probabilities_to_r(&partition_distribution.shrinkage_probabilities, shrinkage);
            }
        }};
    }
    let prior_name = partition_distribution.external_pointer_tag().as_str();
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
        "frp" => mcmc_update!(FrpParameters, true, false, true, false),
        "oldsp-up" => mcmc_update!(OldSpParameters<UpParameters>, true, false, true, false),
        "oldsp-jlp" => mcmc_update!(OldSpParameters<JlpParameters>, true, false, true, false),
        "oldsp-crp" => mcmc_update!(OldSpParameters<CrpParameters>, true, false, true, false),
        "old2sp-up" => mcmc_update!(Old2SpParameters<UpParameters>, true, false, true, false),
        "old2sp-jlp" => mcmc_update!(Old2SpParameters<JlpParameters>, true, false, true, false),
        "old2sp-crp" => mcmc_update!(Old2SpParameters<CrpParameters>, true, false, true, false),
        "sp-up" => mcmc_update!(SpParameters<UpParameters>, true, false, true, false),
        "sp-jlp" => mcmc_update!(SpParameters<JlpParameters>, true, false, true, false),
        "sp-crp" => mcmc_update!(SpParameters<CrpParameters>, true, false, true, false),
        "sp-mixture-up" => {
            mcmc_update!(SpMixtureParameters<UpParameters>, true, false, false, true)
        }
        "sp-mixture-jlp" => {
            mcmc_update!(SpMixtureParameters<JlpParameters>, true, false, false, true)
        }
        "sp-mixture-crp" => {
            mcmc_update!(SpMixtureParameters<CrpParameters>, true, false, false, true)
        }
        _ => panic!("Unsupported distribution: {}", prior_name),
    }
    state = state.canonicalize();
    let result = Rval::new_list(2, pc);
    result.set_list_element(0, Rval::external_pointer_encode(state, state_tag));
    result.set_list_element(1, Rval::external_pointer_encode(monitor, monitor_tag));
    result
}

#[roxido]
fn log_likelihood_contributions(state: Rval, data: Rval) -> Rval {
    let state = Rval::external_pointer_decode_as_ref::<State>(state);
    let data = Rval::external_pointer_decode_as_ref::<Data>(data);
    let x = state.log_likelihood_contributions(data);
    rval!(&x[..])
}

#[roxido]
fn log_likelihood_contributions_of_missing(state: Rval, data: Rval) -> Rval {
    let state = Rval::external_pointer_decode_as_ref::<State>(state);
    let data = Rval::external_pointer_decode_as_ref::<Data>(data);
    let x = state.log_likelihood_contributions_of_missing(data);
    rval!(&x[..])
}

#[roxido]
fn log_likelihood_of(state: Rval, data: Rval, items: Rval) -> Rval {
    let state = Rval::external_pointer_decode_as_ref::<State>(state);
    let data = Rval::external_pointer_decode_as_ref::<Data>(data);
    let (_, items) = items.coerce_integer(pc).unwrap();
    let items: Vec<_> = items
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    rval!(state.log_likelihood_of(data, items))
}

#[roxido]
fn log_likelihood(state: Rval, data: Rval) -> Rval {
    let state = Rval::external_pointer_decode_as_ref::<State>(state);
    let data = Rval::external_pointer_decode_as_ref::<Data>(data);
    rval!(state.log_likelihood(data))
}

#[roxido]
fn sample_multivariate_normal(n_samples: Rval, mean: Rval, precision: Rval) -> Rval {
    let n_samples = n_samples.as_usize();
    let n = mean.len();
    let (_, mean) = mean.coerce_double(pc).unwrap();
    let mean = DVector::from_iterator(n, mean.iter().cloned());
    let (_, precision) = precision.coerce_double(pc).unwrap();
    let precision = DMatrix::from_iterator(n, n, precision.iter().cloned());
    let (rval, slice) = Rval::new_matrix_double(n, n_samples, pc);
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let x = sample_multivariate_normal_repeatedly(n_samples, mean, precision, &mut rng).unwrap();
    slice.clone_from_slice(x.as_slice());
    rval.transpose(pc)
}

// Copied from pumpkin/src/rust/src/lib.rs

use dahl_randompartition::epa::SquareMatrixBorrower;
use dahl_randompartition::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::os::raw::c_void;
use std::ptr::NonNull;

#[roxido]
fn new_FixedPartitionParameters(anchor: Rval) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let p = FixedPartitionParameters::new(anchor);
    new_distr_r_ptr("fixed", p, pc)
}

#[roxido]
fn new_UpParameters(n_items: Rval) -> Rval {
    let p = UpParameters::new(n_items.as_usize());
    new_distr_r_ptr("up", p, pc)
}

#[roxido]
fn new_JlpParameters(mass: Rval, permutation: Rval) -> Rval {
    let permutation = mk_permutation(permutation, pc);
    let p = JlpParameters::new(permutation.n_items(), Mass::new(mass.into()), permutation).unwrap();
    new_distr_r_ptr("jlp", p, pc)
}

#[roxido]
fn new_CrpParameters(n_items: Rval, mass: Rval, discount: Rval) -> Rval {
    let n_items = n_items.as_usize();
    let mass = mass.into();
    let discount = discount.into();
    let p = CrpParameters::new_with_mass_and_discount(
        n_items,
        Mass::new_with_variable_constraint(mass, discount),
        Discount::new(discount),
    );
    new_distr_r_ptr("crp", p, pc)
}

#[roxido]
fn new_EpaParameters(similarity: Rval, permutation: Rval, mass: Rval, discount: Rval) -> Rval {
    let ni = similarity.nrow();
    let similarity = SquareMatrixBorrower::from_slice(similarity.try_into().unwrap(), ni);
    let permutation = mk_permutation(permutation, pc);
    let mass = mass.into();
    let discount = discount.into();
    let p = EpaParameters::new(
        similarity,
        permutation,
        Mass::new_with_variable_constraint(mass, discount),
        Discount::new(discount),
    );
    new_distr_r_ptr("epa", p, pc)
}

#[roxido]
fn new_LspParameters(anchor: Rval, rate: Rval, mass: Rval, permutation: Rval) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let rate = rate.into();
    let mass = mass.into();
    let permutation = mk_permutation(permutation, pc);
    let p = LspParameters::new_with_rate(anchor, rate, Mass::new(mass), permutation).unwrap();
    new_distr_r_ptr("lsp", p, pc)
}

#[roxido]
fn new_CppParameters(anchor: Rval, rate: Rval, baseline: Rval, use_vi: Rval, a: Rval) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let rate = rate.into();
    let use_vi = use_vi.as_bool();
    let a = a.into();
    let (baseline_name, baseline_ptr) = unwrap_distr_r_ptr(baseline);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_ptr as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                CppParameters::new(anchor, rate, baseline, use_vi, a).unwrap(),
                pc,
            )
        }};
    }
    match baseline_name {
        "up" => distr_macro!(UpParameters, "cpp-up"),
        "jlp" => distr_macro!(JlpParameters, "cpp-jlp"),
        "crp" => distr_macro!(CrpParameters, "cpp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_name),
    }
}

#[roxido]
fn new_FrpParameters(
    anchor: Rval,
    shrinkage: Rval,
    permutation: Rval,
    mass: Rval,
    discount: Rval,
    power: Rval,
) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let mass = mass.into();
    let discount = discount.into();
    let power = power.into();
    let p = FrpParameters::new(
        anchor,
        shrinkage,
        permutation,
        Mass::new_with_variable_constraint(mass, discount),
        Discount::new(discount),
        Power::new(power),
    )
    .unwrap();
    new_distr_r_ptr("frp", p, pc)
}

#[roxido]
fn new_OldSpParameters(
    anchor: Rval,
    shrinkage: Rval,
    permutation: Rval,
    baseline: Rval,
    use_vi: Rval,
    a: Rval,
    scaling_exponent: Rval,
) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let use_vi = use_vi.as_bool();
    let a = a.into();
    let scaling_exponent = scaling_exponent.into();
    let (baseline_name, baseline_ptr) = unwrap_distr_r_ptr(baseline);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_ptr as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                OldSpParameters::new(
                    anchor,
                    shrinkage,
                    permutation,
                    baseline,
                    use_vi,
                    a,
                    scaling_exponent,
                )
                .unwrap(),
                pc,
            )
        }};
    }
    match baseline_name {
        "up" => distr_macro!(UpParameters, "oldsp-up"),
        "jlp" => distr_macro!(JlpParameters, "oldsp-jlp"),
        "crp" => distr_macro!(CrpParameters, "oldsp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_name),
    }
}

#[roxido]
fn new_Old2SpParameters(anchor: Rval, shrinkage: Rval, permutation: Rval, baseline: Rval) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let (baseline_name, baseline_ptr) = unwrap_distr_r_ptr(baseline);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_ptr as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                Old2SpParameters::new(anchor, shrinkage, permutation, baseline).unwrap(),
                pc,
            )
        }};
    }
    match baseline_name {
        "up" => distr_macro!(UpParameters, "old2sp-up"),
        "jlp" => distr_macro!(JlpParameters, "old2sp-jlp"),
        "crp" => distr_macro!(CrpParameters, "old2sp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_name),
    }
}

#[roxido]
fn new_SpMixtureParameters(
    anchor: Rval,
    shrinkage: Rval,
    permutation: Rval,
    baseline: Rval,
) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let shrinkage = mk_shrinkage_probabilities(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let (baseline_name, baseline_ptr) = unwrap_distr_r_ptr(baseline);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_ptr as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                SpMixtureParameters::new(anchor, shrinkage, permutation, baseline).unwrap(),
                pc,
            )
        }};
    }
    match baseline_name {
        "up" => distr_macro!(UpParameters, "sp-mixture-up"),
        "jlp" => distr_macro!(JlpParameters, "sp-mixture-jlp"),
        "crp" => distr_macro!(CrpParameters, "sp-mixture-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_name),
    }
}

#[roxido]
fn new_SpParameters(anchor: Rval, shrinkage: Rval, permutation: Rval, baseline: Rval) -> Rval {
    let anchor = mk_clustering(anchor, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let (baseline_name, baseline_ptr) = unwrap_distr_r_ptr(baseline);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_ptr as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                SpParameters::new(anchor, shrinkage, permutation, baseline).unwrap(),
                pc,
            )
        }};
    }
    match baseline_name {
        "up" => distr_macro!(UpParameters, "sp-up"),
        "jlp" => distr_macro!(JlpParameters, "sp-jlp"),
        "crp" => distr_macro!(CrpParameters, "sp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_name),
    }
}

fn mk_clustering(partition: Rval, pc: &mut Pc) -> Clustering {
    Clustering::from_slice(partition.coerce_integer(pc).unwrap().1)
}

fn mk_shrinkage(shrinkage: Rval, pc: &mut Pc) -> Shrinkage {
    Shrinkage::from(shrinkage.coerce_double(pc).unwrap().1).unwrap()
}

fn mk_shrinkage_probabilities(shrinkage: Rval, pc: &mut Pc) -> ShrinkageProbabilities {
    ShrinkageProbabilities::from(shrinkage.coerce_double(pc).unwrap().1).unwrap()
}

fn mk_permutation(permutation: Rval, pc: &mut Pc) -> Permutation {
    let vector = permutation
        .coerce_integer(pc)
        .unwrap()
        .1
        .iter()
        .map(|x| *x as usize)
        .collect();
    Permutation::from_vector(vector).unwrap()
}

fn new_distr_r_ptr<T>(name: &str, value: T, pc: &mut Pc) -> Rval {
    use rbindings::*;
    unsafe {
        // Move to Box<_> and then forget about it.
        let ptr = Box::into_raw(Box::new(value)) as *mut c_void;
        let tag = Rval::new_character(name, pc).0;
        let sexp = pc.protect(R_MakeExternalPtr(ptr, tag, R_NilValue));
        R_RegisterCFinalizerEx(sexp, Some(free_distr_r_ptr), 0);
        Rval(sexp)
    }
}

#[no_mangle]
extern "C" fn free_distr_r_ptr(sexp: rbindings::SEXP) {
    fn free_r_ptr_helper<T>(sexp: rbindings::SEXP) {
        unsafe {
            let ptr = rbindings::R_ExternalPtrAddr(sexp) as *mut T;
            if !ptr.is_null() {
                // Convert the raw pointer back to a Box<_> and drop it.
                let _ = Box::from_raw(ptr);
            }
        }
    }
    unsafe {
        match Rval(rbindings::R_ExternalPtrTag(sexp)).try_into().unwrap() {
            "fixed" => free_r_ptr_helper::<FixedPartitionParameters>(sexp),
            "up" => free_r_ptr_helper::<UpParameters>(sexp),
            "jlp" => free_r_ptr_helper::<JlpParameters>(sexp),
            "crp" => free_r_ptr_helper::<CrpParameters>(sexp),
            "epa" => free_r_ptr_helper::<EpaParameters>(sexp),
            "lsp" => free_r_ptr_helper::<LspParameters>(sexp),
            "cpp-up" => free_r_ptr_helper::<CppParameters<UpParameters>>(sexp),
            "cpp-jlp" => free_r_ptr_helper::<CppParameters<JlpParameters>>(sexp),
            "cpp-crp" => free_r_ptr_helper::<CppParameters<CrpParameters>>(sexp),
            "frp" => free_r_ptr_helper::<FrpParameters>(sexp),
            "oldsp-up" => free_r_ptr_helper::<OldSpParameters<UpParameters>>(sexp),
            "oldsp-jlp" => free_r_ptr_helper::<OldSpParameters<JlpParameters>>(sexp),
            "oldsp-crp" => free_r_ptr_helper::<OldSpParameters<CrpParameters>>(sexp),
            "old2sp-up" => free_r_ptr_helper::<Old2SpParameters<UpParameters>>(sexp),
            "old2sp-jlp" => free_r_ptr_helper::<Old2SpParameters<JlpParameters>>(sexp),
            "old2sp-crp" => free_r_ptr_helper::<Old2SpParameters<CrpParameters>>(sexp),
            "sp-up" => free_r_ptr_helper::<SpParameters<UpParameters>>(sexp),
            "sp-jlp" => free_r_ptr_helper::<SpParameters<JlpParameters>>(sexp),
            "sp-crp" => free_r_ptr_helper::<SpParameters<CrpParameters>>(sexp),
            "sp-mixture-up" => free_r_ptr_helper::<SpMixtureParameters<UpParameters>>(sexp),
            "sp-mixture-jlp" => free_r_ptr_helper::<SpMixtureParameters<JlpParameters>>(sexp),
            "sp-mixture-crp" => free_r_ptr_helper::<SpMixtureParameters<CrpParameters>>(sexp),
            name => panic!("Unsupported distribution: {}", name),
        }
    }
}

fn unwrap_distr_r_ptr(x: Rval) -> (&'static str, *mut c_void) {
    unsafe {
        (
            Rval(rbindings::R_ExternalPtrTag(x.0)).try_into().unwrap(),
            rbindings::R_ExternalPtrAddr(x.0),
        )
    }
}
