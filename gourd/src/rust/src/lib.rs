mod data;
mod hyperparameters;
mod mvnorm;
mod registration;
mod state;

use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
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
    let fixed = McmcTuning::new(false, false, false, false, 2);
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
    hyperparameters: Rval,
    partition_distribution: Rval,
    mcmc_tuning: Rval,
    _missing_items: Rval,
) -> Rval {
    let n_updates = n_updates.as_usize();
    let data = Rval::external_pointer_decode_as_mut_ref::<Data>(data);
    let state_tag = state.external_pointer_tag();
    let mut state = Rval::external_pointer_decode::<State>(state);
    let hyperparameters = Rval::external_pointer_decode_as_ref::<Hyperparameters>(hyperparameters);
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
    let mut rng = Pcg64Mcg::from_seed(r::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rng2 = Pcg64Mcg::from_seed(seed);
    macro_rules! mcmc_update {
        ($tipe:ty, false) => {{
            let partition_distribution =
                partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(
                    &mcmc_tuning,
                    data,
                    hyperparameters,
                    partition_distribution,
                    &mut rng,
                    &mut rng2,
                );
            }
        }};
        ($tipe:ty, true) => {{
            let partition_distribution =
                partition_distribution.external_pointer_decode_as_mut_ref::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(
                    &mcmc_tuning,
                    data,
                    hyperparameters,
                    partition_distribution,
                    &mut rng,
                    &mut rng2,
                );
                dahl_randompartition::mcmc::update_permutation(
                    1,
                    partition_distribution,
                    mcmc_tuning.n_items_per_permutation_update,
                    &state.clustering,
                    &mut rng,
                );
                state.permutation = partition_distribution.permutation.clone();
            }
        }};
    }
    let prior_name = partition_distribution.external_pointer_tag().as_str();
    match prior_name {
        "fixed" => mcmc_update!(FixedPartitionParameters, false),
        "up" => mcmc_update!(UpParameters, false),
        "jlp" => mcmc_update!(JlpParameters, true),
        "crp" => mcmc_update!(CrpParameters, false),
        "epa" => mcmc_update!(EpaParameters, true),
        "lsp" => mcmc_update!(LspParameters, true),
        "cpp-up" => mcmc_update!(CppParameters<UpParameters>, false),
        "cpp-jlp" => mcmc_update!(CppParameters<JlpParameters>, false),
        "cpp-crp" => mcmc_update!(CppParameters<CrpParameters>, false),
        "frp" => mcmc_update!(FrpParameters, true),
        "oldsp-up" => mcmc_update!(OldSpParameters<UpParameters>, true),
        "oldsp-jlp" => mcmc_update!(OldSpParameters<JlpParameters>, true),
        "oldsp-crp" => mcmc_update!(OldSpParameters<CrpParameters>, true),
        "sp-up" => mcmc_update!(SpParameters<UpParameters>, true),
        "sp-jlp" => mcmc_update!(SpParameters<JlpParameters>, true),
        "sp-crp" => mcmc_update!(SpParameters<CrpParameters>, true),
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

// Copied from pumpkin/src/rust/src/lib.rs

use dahl_randompartition::epa::SquareMatrixBorrower;
use dahl_randompartition::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::os::raw::c_void;
use std::ptr::NonNull;

#[roxido]
fn new_FixedPartitionParameters(baseline_partition: Rval) -> Rval {
    let baseline_partition = mk_clustering(baseline_partition, pc);
    let p = FixedPartitionParameters::new(baseline_partition);
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
fn new_LspParameters(baseline_partition: Rval, rate: Rval, mass: Rval, permutation: Rval) -> Rval {
    let baseline_partition = mk_clustering(baseline_partition, pc);
    let rate = rate.into();
    let mass = mass.into();
    let permutation = mk_permutation(permutation, pc);
    let p = LspParameters::new_with_rate(
        baseline_partition,
        Rate::new(rate),
        Mass::new(mass),
        permutation,
    )
    .unwrap();
    new_distr_r_ptr("lsp", p, pc)
}

#[roxido]
fn new_CppParameters(
    baseline_partition: Rval,
    rate: Rval,
    baseline_distribution: Rval,
    use_vi: Rval,
    a: Rval,
) -> Rval {
    let baseline_partition = mk_clustering(baseline_partition, pc);
    let rate = rate.into();
    let use_vi = use_vi.as_bool();
    let a = a.into();
    let (baseline_distribution_name, baseline_distribution_ptr) =
        unwrap_distr_r_ptr(baseline_distribution);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_distribution_ptr as *mut $tipe).unwrap();
            let baseline_distribution = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                CppParameters::new(
                    baseline_partition,
                    Rate::new(rate),
                    baseline_distribution,
                    use_vi,
                    a,
                )
                .unwrap(),
                pc,
            )
        }};
    }
    match baseline_distribution_name {
        "up" => distr_macro!(UpParameters, "cpp-up"),
        "jlp" => distr_macro!(JlpParameters, "cpp-jlp"),
        "crp" => distr_macro!(CrpParameters, "cpp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_distribution_name),
    }
}

#[roxido]
fn new_FrpParameters(
    baseline_partition: Rval,
    shrinkage: Rval,
    permutation: Rval,
    mass: Rval,
    discount: Rval,
    power: Rval,
) -> Rval {
    let baseline_partition = mk_clustering(baseline_partition, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let mass = mass.into();
    let discount = discount.into();
    let power = power.into();
    let p = FrpParameters::new(
        baseline_partition,
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
    baseline_partition: Rval,
    shrinkage: Rval,
    permutation: Rval,
    baseline_distribution: Rval,
    use_vi: Rval,
    a: Rval,
    scaling_exponent: Rval,
) -> Rval {
    let baseline_partition = mk_clustering(baseline_partition, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let use_vi = use_vi.as_bool();
    let a = a.into();
    let scaling_exponent = scaling_exponent.into();
    let (baseline_distribution_name, baseline_distribution_ptr) =
        unwrap_distr_r_ptr(baseline_distribution);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_distribution_ptr as *mut $tipe).unwrap();
            let baseline_distribution = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                OldSpParameters::new(
                    baseline_partition,
                    shrinkage,
                    permutation,
                    baseline_distribution,
                    use_vi,
                    a,
                    scaling_exponent,
                )
                .unwrap(),
                pc,
            )
        }};
    }
    match baseline_distribution_name {
        "up" => distr_macro!(UpParameters, "oldsp-up"),
        "jlp" => distr_macro!(JlpParameters, "oldsp-jlp"),
        "crp" => distr_macro!(CrpParameters, "oldsp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_distribution_name),
    }
}

#[roxido]
fn new_SpParameters(
    baseline_partition: Rval,
    shrinkage: Rval,
    permutation: Rval,
    baseline_distribution: Rval,
) -> Rval {
    let baseline_partition = mk_clustering(baseline_partition, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let (baseline_distribution_name, baseline_distribution_ptr) =
        unwrap_distr_r_ptr(baseline_distribution);
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline_distribution_ptr as *mut $tipe).unwrap();
            let baseline_distribution = unsafe { p.as_ref().clone() };
            new_distr_r_ptr(
                $label,
                SpParameters::new(
                    baseline_partition,
                    shrinkage,
                    permutation,
                    baseline_distribution,
                )
                .unwrap(),
                pc,
            )
        }};
    }
    match baseline_distribution_name {
        "up" => distr_macro!(UpParameters, "sp-up"),
        "jlp" => distr_macro!(JlpParameters, "sp-jlp"),
        "crp" => distr_macro!(CrpParameters, "sp-crp"),
        _ => panic!("Unsupported distribution: {}", baseline_distribution_name),
    }
}

fn mk_clustering(partition: Rval, pc: &mut Pc) -> Clustering {
    Clustering::from_slice(partition.coerce_integer(pc).unwrap().1)
}

fn mk_shrinkage(shrinkage: Rval, pc: &mut Pc) -> Shrinkage {
    Shrinkage::from(shrinkage.coerce_double(pc).unwrap().1).unwrap()
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
                Box::from_raw(ptr);
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
            "sp-up" => free_r_ptr_helper::<SpParameters<UpParameters>>(sexp),
            "sp-jlp" => free_r_ptr_helper::<SpParameters<JlpParameters>>(sexp),
            "sp-crp" => free_r_ptr_helper::<SpParameters<CrpParameters>>(sexp),
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
