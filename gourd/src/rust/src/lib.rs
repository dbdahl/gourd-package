mod data;
mod hyperparameters;
mod mvnorm;
mod registration;
mod state;

use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::mvnorm::sample_multivariate_normal_repeatedly;
use crate::state::{State, StateFixedComponents};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::cpp::CppParameters;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::distr::ProbabilityMassFunction;
use dahl_randompartition::epa::EpaParameters;
use dahl_randompartition::fixed::FixedPartitionParameters;
use dahl_randompartition::frp::FrpParameters;
use dahl_randompartition::jlp::JlpParameters;
use dahl_randompartition::lsp::LspParameters;
use dahl_randompartition::mcmc::update_neal_algorithm_full;
use dahl_randompartition::oldsp::OldSpParameters;
use dahl_randompartition::perm::Permutation;
use dahl_randompartition::prelude::Mass;
use dahl_randompartition::shrink::Shrinkage;
use dahl_randompartition::sp::SpParameters;
use dahl_randompartition::up::UpParameters;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use roxido::*;

#[roxido]
fn state_r2rust(state: Rval) -> Rval {
    Rval::external_pointer_encode(State::from_r(state, pc), rval!("state"))
}

#[roxido]
fn state_rust2r_as_reference(state: Rval) -> Rval {
    Rval::external_pointer_decode_as_ref::<State>(state).to_r(pc)
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
        str => {
            panic!("Unrecognized type ID: {}.", str)
        }
    };
    Rval::nil()
}

struct Group {
    data: Data,
    state: State,
    hyperparameters: Hyperparameters,
}

struct All(Vec<Group>);

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
    let all = All(vec);
    Rval::external_pointer_encode(all, rval!("all"))
}

#[roxido]
fn fit_all(all_ptr: Rval, shrinkage: Rval, n_updates: Rval, do_baseline_partition: Rval) -> Rval {
    let mut all: All = all_ptr.external_pointer_decode();
    let n_items = all.0[0].data.n_items();
    let fixed = StateFixedComponents::new(false, false, false, false, true);
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
    let baseline_partition_initial = all.0[rng.gen_range(0..all.0.len())]
        .state
        .clustering()
        .clone();
    let mut grand_sum = 0.0;
    let upper = if do_baseline_partition { 1 } else { n_items };
    for missing_item in 0..upper {
        let mut sum = 0.0;
        if !do_baseline_partition {
            println!("Missing: {}", missing_item);
            for group in all.0.iter_mut() {
                group.data.declare_missing(vec![missing_item]);
            }
        }
        let mut partition_distribution = SpParameters::new(
            baseline_partition_initial.clone(),
            shrinkage.clone(),
            Permutation::natural_and_fixed(n_items),
            CrpParameters::new_with_mass(n_items, Mass::new(1.0)),
        )
        .unwrap();
        for update_index in 0..n_updates {
            for group in all.0.iter_mut() {
                group.state.mcmc_iteration(
                    &fixed,
                    &mut group.data,
                    &group.hyperparameters,
                    &partition_distribution,
                    &mut rng,
                    &mut rng2,
                );
            }
            let mut baseline_partition_tmp = partition_distribution.baseline_partition.clone();
            let mut log_likelihood_contribution_fn = |proposed_baseline_partition: &Clustering| {
                partition_distribution.baseline_partition = proposed_baseline_partition.clone();
                let mut sum = 0.0;
                for group in all.0.iter() {
                    sum += partition_distribution.log_pmf(group.state.clustering());
                }
                sum
            };
            update_neal_algorithm_full(
                1,
                &mut baseline_partition_tmp,
                &permutation,
                &hyperpartition_prior_distribution,
                &mut log_likelihood_contribution_fn,
                &mut rng,
            );
            if do_baseline_partition {
                partition_distribution.baseline_partition = baseline_partition_tmp.standardize();
                partition_distribution
                    .baseline_partition
                    .relabel_into_slice(
                        1_i32,
                        &mut result_slice[(update_index * n_items)..((update_index + 1) * n_items)],
                    );
            } else {
                for group in all.0.iter_mut() {
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
    fixed: Rval,
    hyperparameters: Rval,
    partition_distribution: Rval,
) -> Rval {
    let n_updates = n_updates.as_usize();
    let data = Rval::external_pointer_decode_as_mut_ref::<Data>(data);
    let state_tag = state.external_pointer_tag();
    let mut state = Rval::external_pointer_decode::<State>(state);
    let fixed = StateFixedComponents::from_r(fixed, pc);
    let hyperparameters = Rval::external_pointer_decode_as_ref::<Hyperparameters>(hyperparameters);
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
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rng2 = Pcg64Mcg::from_seed(seed);
    macro_rules! distr_macro {
        ($tipe:ty) => {{
            let partition_distribution =
                partition_distribution.external_pointer_decode_as_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(
                    &fixed,
                    data,
                    hyperparameters,
                    partition_distribution,
                    &mut rng,
                    &mut rng2,
                );
            }
        }};
    }
    let prior_name = partition_distribution.external_pointer_tag().as_str();
    match prior_name {
        "fixed" => distr_macro!(FixedPartitionParameters),
        "up" => distr_macro!(UpParameters),
        "jlp" => distr_macro!(JlpParameters),
        "crp" => distr_macro!(CrpParameters),
        "epa" => distr_macro!(EpaParameters),
        "lsp" => distr_macro!(LspParameters),
        "cpp-up" => distr_macro!(CppParameters<UpParameters>),
        "cpp-jlp" => distr_macro!(CppParameters<JlpParameters>),
        "cpp-crp" => distr_macro!(CppParameters<CrpParameters>),
        "frp" => distr_macro!(FrpParameters),
        "oldsp-up" => distr_macro!(OldSpParameters<UpParameters>),
        "oldsp-jlp" => distr_macro!(OldSpParameters<JlpParameters>),
        "oldsp-crp" => distr_macro!(OldSpParameters<CrpParameters>),
        "sp-up" => distr_macro!(SpParameters<UpParameters>),
        "sp-jlp" => distr_macro!(SpParameters<JlpParameters>),
        "sp-crp" => distr_macro!(SpParameters<CrpParameters>),
        _ => panic!("Unsupported distribution: {}", prior_name),
    }
    state = state.canonicalize();
    Rval::external_pointer_encode(state, state_tag)
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
    rval!(state.log_likelihood_of(data, items.iter()))
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

