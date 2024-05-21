roxido_registration!();
use roxido::*;

mod data;
mod hyperparameters;
mod membership;
mod monitor;
mod mvnorm;
mod state;

use crate::data::Data;
use crate::hyperparameters::Hyperparameters;
use crate::monitor::Monitor;
use crate::mvnorm::sample_multivariate_normal_repeatedly;
use crate::state::{McmcTuning, State};
use dahl_randompartition::clust::Clustering;
use dahl_randompartition::cpp::CppParameters;
use dahl_randompartition::crp::CrpParameters;
use dahl_randompartition::distr::PredictiveProbabilityFunction;
use dahl_randompartition::distr::{
    PartitionSampler, ProbabilityMassFunction, ProbabilityMassFunctionPartial,
};
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
use rand_distr::{Beta as BetaRNG, Distribution, Gamma as GammaRNG};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use slice_sampler::univariate::stepping_out;
use statrs::distribution::{Beta as BetaPDF, Continuous, Gamma as GammaPDF};
use std::fmt::Debug;
use std::ptr::NonNull;
use walltime::TicToc;

#[roxido]
fn new_UpParameters(n_items: usize) {
    let p = UpParameters::new(n_items);
    RExternalPtr::encode(p, "up", pc)
}

#[roxido]
fn new_JlpParameters(concentration: f64, permutation: &RVector) {
    let permutation = mk_permutation(permutation, pc);
    let concentration = Concentration::new(concentration).stop_str("Invalid concentration value");
    let p = JlpParameters::new(permutation.n_items(), concentration, permutation)
        .stop_str("Invalid Jensen Liu parametrization");
    RExternalPtr::encode(p, "jlp", pc)
}

#[roxido]
fn new_CrpParameters(n_items: usize, concentration: f64, discount: f64) {
    let discount = Discount::new(discount).stop_str("Invalid discount value");
    let concentration = Concentration::new_with_discount(concentration, discount)
        .stop_str("Invalid concentration value");
    let p = CrpParameters::new_with_discount(n_items, concentration, discount)
        .stop_str("Invalid CRP parametrization");
    RExternalPtr::encode(p, "crp", pc)
}

#[roxido]
fn new_SpParameters(
    anchor: &RVector,
    shrinkage: &RVector,
    permutation: &RVector,
    grit: f64,
    baseline: &RExternalPtr,
) {
    let anchor = Clustering::from_slice(anchor.to_i32(pc).slice());
    let shrinkage = Shrinkage::from(shrinkage.to_f64(pc).slice()).stop_str("Invalid shrinkage");
    let permutation = mk_permutation(permutation, pc);
    let grit = match Grit::new(grit) {
        Some(grit) => grit,
        None => stop!("Grit value out of range"),
    };
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline.address() as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            RExternalPtr::encode(
                SpParameters::new(anchor, shrinkage, permutation, grit, baseline)
                    .stop_str("Invalid shrinkage partition parametrization"),
                $label,
                pc,
            )
        }};
    }
    let tag = baseline.tag().as_scalar().stop();
    let name = tag.str(pc);
    match name {
        "up" => distr_macro!(UpParameters, "sp-up"),
        "jlp" => distr_macro!(JlpParameters, "sp-jlp"),
        "crp" => distr_macro!(CrpParameters, "sp-crp"),
        _ => stop!("Unsupported distribution: {}", name),
    }
}

fn mk_permutation(permutation: &RVector, pc: &Pc) -> Permutation {
    let vector = permutation
        .to_i32(pc)
        .slice()
        .iter()
        .map(|x| *x as usize)
        .collect();
    Permutation::from_vector(vector).stop_str("Invalid permutation")
}

#[derive(PartialEq, Copy, Clone)]
enum RandomizeShrinkage {
    Fixed,
    Common,
    Cluster,
    Idiosyncratic,
}

#[roxido]
fn prPartition(partition: &RMatrix, prior: &RExternalPtr) {
    let partition = partition.to_i32(pc);
    let matrix = partition.slice();
    let np = partition.nrow();
    let ni = partition.ncol();
    let log_probs_rval = RVector::new(np, pc);
    let log_probs_slice = log_probs_rval.slice_mut();
    macro_rules! distr_macro {
        ($tipe:ty) => {{
            let p = prior.address() as *mut $tipe;
            let distr = unsafe { NonNull::new(p).unwrap().as_mut() };
            for i in 0..np {
                let mut target_labels = Vec::with_capacity(ni);
                for j in 0..ni {
                    target_labels.push((matrix[np * j + i]).try_into().unwrap());
                }
                let target = Clustering::from_vector(target_labels);
                log_probs_slice[i] = distr.log_pmf(&target);
            }
        }};
    }
    let tag = prior.tag().as_scalar().stop();
    let name = tag.str(pc);
    match name {
        "crp" => distr_macro!(CrpParameters),
        "up" => distr_macro!(UpParameters),
        "jlp" => distr_macro!(JlpParameters),
        "sp-up" => distr_macro!(SpParameters<UpParameters>),
        "sp-jlp" => distr_macro!(SpParameters<JlpParameters>),
        "sp-crp" => distr_macro!(SpParameters<CrpParameters>),
        _ => stop!("Unsupported distribution: {}", name),
    }
    log_probs_rval
}

#[roxido]
fn summarize_prior_on_shrinkage_and_grit(
    anchor: &[i32],
    domain_specification: &RList,
    shrinkage_shape: f64,
    shrinkage_rate: f64,
    shrinkage_n: usize,
    grit_shape1: f64,
    grit_shape2: f64,
    grit_n: usize,
    concentration: f64,
    n_mc_samples: i32,
) {
    if !(shrinkage_shape > 0.0) {
        stop!("'shrinkage_shape' should be strictly positive.");
    };
    if !(shrinkage_rate > 0.0) {
        stop!("'shrinkage_rate' should be strictly positive.");
    };
    if shrinkage_n < 2 {
        stop!("'shrinkage_n' must be at least two.");
    }
    if !(grit_shape1 > 0.0) {
        stop!("'grit_shape1' should be strictly positive.");
    };
    if !(grit_shape2 > 0.0) {
        stop!("'grit_shape2' should be strictly positive.");
    };
    if grit_n < 2 {
        stop!("'grit_n' must be at least two.");
    }
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let ((shrinkage_min, shrinkage_max), (grit_min, grit_max)) = match domain_specification
        .get_by_key("n")
    {
        Ok(n) => {
            let Ok(n) = n.as_scalar() else {
                stop!("Element 'n' in 'domain_specification' should be a scalar.");
            };
            let Ok(n) = n.usize() else {
                stop!("Element 'n' in 'domain_specification' cannot be interpreted as positive integer.");
            };
            if n <= 1 {
                stop!("Element 'n' is 'domain_specification' must be at least 2.");
            }
            let Ok(percentile) = domain_specification.get_by_key("percentile") else {
                stop!("Element 'percentile' must be provided when 'n' is provided in 'domain_specification'.");
            };
            let Ok(percentile) = percentile.as_scalar() else {
                stop!("Element 'percentile' in 'domain_specification' should be a scalar.");
            };
            let percentile = percentile.f64();
            if percentile < 0.0 || percentile > 1.0 {
                stop!("Element 'percentile' is 'domain_specification' must be in [0, 1].")
            }
            let Ok(gamma_dist) = GammaRNG::new(shrinkage_shape, 1.0 / shrinkage_rate) else {
                stop!("Cannot construct gamma distribution for shrinkage.");
            };
            let Ok(beta_dist) = BetaRNG::new(grit_shape1, grit_shape2) else {
                stop!("Cannot construct beta distribution for grit.");
            };
            let mut sample = vec![0.0; n];
            sample.fill_with(|| gamma_dist.sample(&mut rng));
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let shrinkage_max = sample[(percentile * (n as f64)).floor() as usize];
            sample.fill_with(|| beta_dist.sample(&mut rng));
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let beta_max = sample[(percentile * (n as f64)).floor() as usize];
            ((0.0, shrinkage_max), (0.0, beta_max))
        }
        Err(_) => {
            let checker = |x| {
                let Ok(y) = domain_specification.get_by_key(x) else {
                    stop!("Element '{x}' is not found in 'domain_specification'.");
                };
                let Ok(y) = y.as_vector() else {
                    stop!("Element '{x}' in 'domain_specification' should be a vector.");
                };
                if y.len() != 2 {
                    stop!("Element '{x}' in 'domain_specification' should be of length 2.");
                };
                let y = y.to_f64(pc);
                let slice = y.slice();
                if slice[0] >= slice[1] {
                    stop!("The first element of '{x}' in 'domain_specification' should be less than the second element.");
                }
                (slice[0], slice[1])
            };
            (checker("shrinkage_lim"), checker("grit_lim"))
        }
    };
    let shrinkage_width = (shrinkage_max - shrinkage_min) / ((shrinkage_n - 1) as f64);
    let grit_width = (grit_max - grit_min) / ((grit_n - 1) as f64);
    let result = RList::with_names(
        &[
            "shrinkage",
            "grit",
            "log_density",
            "expected_rand_index",
            "expected_entropy",
        ],
        pc,
    );
    let shrinkage_rval = RVector::<f64>::new(shrinkage_n, pc);
    let _ = result.set(0, shrinkage_rval);
    let grit_rval = RVector::<f64>::new(grit_n, pc);
    let _ = result.set(1, grit_rval);
    let log_density_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(2, log_density_rval);
    let expected_rand_index_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(3, expected_rand_index_rval);
    let expected_entropy_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(4, expected_entropy_rval);
    let shrinkage_slice = shrinkage_rval.slice_mut();
    let grit_slice = grit_rval.slice_mut();
    let log_density_slice = log_density_rval.slice_mut();
    let expected_rand_index_slice = expected_rand_index_rval.slice_mut();
    let expected_entropy_slice = expected_entropy_rval.slice_mut();
    let Ok(gamma_dist) = GammaPDF::new(shrinkage_shape, shrinkage_rate) else {
        stop!("Cannot construct gamma distribution for shrinkage.");
    };
    let Ok(beta_dist) = BetaPDF::new(grit_shape1, grit_shape2) else {
        stop!("Cannot construct beta distribution for grit.");
    };
    let anchor = Clustering::from_slice(anchor);
    let n_items = anchor.n_items();
    let concentration = Concentration::new(concentration).unwrap();
    let baseline_ppf = CrpParameters::new(n_items, concentration);
    let permutation = Permutation::natural(n_items);
    let mut partition_distribution = SpParameters::new(
        anchor,
        Shrinkage::zero(n_items),
        permutation,
        Grit::one(),
        baseline_ppf,
    )
    .unwrap();
    let n_mc_samples_f64 = n_mc_samples as f64;
    let counts_truth = marginal_counter(partition_distribution.anchor.allocation());
    let summarize_truth = summarize_counts_from_rand_index(&counts_truth);
    for (j, grit) in grit_slice.iter_mut().enumerate() {
        *grit = grit_min + (j as f64) * grit_width;
        let beta_density = beta_dist.ln_pdf(*grit);
        let mut index = shrinkage_n * j;
        for (i, shrinkage) in shrinkage_slice.iter_mut().enumerate() {
            *shrinkage = shrinkage_min + (i as f64) * shrinkage_width;
            let shrinkage_density = gamma_dist.ln_pdf(*shrinkage);
            log_density_slice[index] = shrinkage_density + beta_density;
            partition_distribution
                .shrinkage
                .set_constant(ScalarShrinkage::new(*shrinkage).unwrap());
            partition_distribution.grit.set(*grit);
            let mut sum_rand_index = 0.0;
            let mut sum_entropy = 0.0;
            (0..n_mc_samples).for_each(|_| {
                partition_distribution.permutation.shuffle(&mut rng);
                let sample = partition_distribution.sample(&mut rng);
                let allocation = sample.allocation();
                let result = rand_index_core(
                    partition_distribution.anchor.allocation(),
                    allocation,
                    1.0,
                    true,
                    &counts_truth,
                    summarize_truth,
                );
                sum_rand_index += result.0;
                sum_entropy += result.1.unwrap();
            });
            expected_rand_index_slice[index] = sum_rand_index / n_mc_samples_f64;
            expected_entropy_slice[index] = sum_entropy / n_mc_samples_f64;
            index += 1;
        }
    }
    result
}

#[roxido]
fn rand_index_for_r(labels_truth: &RVector, labels_estimate: &RVector, a: f64) {
    rand_index(
        labels_truth.to_i32(pc).slice(),
        labels_estimate.to_i32(pc).slice(),
        Some(a),
    )
    .stop()
}

fn rand_index<A: AsUsize + Debug>(
    labels_truth: &[A],
    labels_estimate: &[A],
    a: Option<f64>,
) -> Result<f64, &'static str> {
    if labels_truth.len() != labels_estimate.len() {
        return Err("Inconsistent lengths.");
    }
    let a = match a {
        Some(a) => {
            if a < 0.0 || a > 2.0 {
                return Err("Parameter 'a' must be in [0.0, 2.0].");
            }
            a
        }
        None => 1.0,
    };
    Ok(rand_index_engine(labels_truth, labels_estimate, a, false).0)
}

trait AsUsize {
    fn as_usize(&self) -> usize;
}

impl AsUsize for i32 {
    fn as_usize(&self) -> usize {
        (*self).try_into().unwrap()
    }
}

impl AsUsize for usize {
    fn as_usize(&self) -> usize {
        *self
    }
}

fn marginal_counter<B: AsUsize>(labels: &[B]) -> Vec<u32> {
    let mut counts = Vec::new();
    for label in labels {
        let label = label.as_usize();
        if label >= counts.len() {
            counts.resize(label + 1, 0);
        }
        counts[label] += 1;
    }
    counts
}

fn summarize_counts_from_rand_index(counts: &[u32]) -> f64 {
    counts
        .into_iter()
        .filter(|&&zz| zz > 0)
        .map(|&zz| zz as f64)
        .map(|x| x * x)
        .sum::<f64>()
}

fn rand_index_engine<A: AsUsize + Debug>(
    labels_truth: &[A],
    labels_estimate: &[A],
    a: f64,
    with_entropy: bool,
) -> (f64, Option<f64>) {
    let counts_truth = marginal_counter(labels_truth);
    let summarize_truth = summarize_counts_from_rand_index(&counts_truth);
    rand_index_core(
        labels_truth,
        labels_estimate,
        a,
        with_entropy,
        &counts_truth,
        summarize_truth,
    )
}

fn rand_index_core<A: AsUsize + Debug>(
    labels_truth: &[A],
    labels_estimate: &[A],
    a: f64,
    with_entropy: bool,
    counts_truth: &[u32],
    summarize_truth: f64,
) -> (f64, Option<f64>) {
    let counts_estimate = marginal_counter(labels_estimate);
    let n_clusters_truth = counts_truth.len();
    let n_clusters_estimate = counts_estimate.len();
    let mut counts_joint = vec![0; n_clusters_truth * n_clusters_estimate];
    for (label_truth, label_estimate) in labels_truth.iter().zip(labels_estimate.iter()) {
        let label_truth = label_truth.as_usize();
        let label_estimate = label_estimate.as_usize();
        counts_joint[n_clusters_truth * label_estimate + label_truth] += 1;
    }
    fn entropy(counts: &[u32], n: f64) -> f64 {
        -counts
            .into_iter()
            .filter(|&&zz| zz > 0)
            .map(|&zz| zz as f64)
            .map(|x| x / n)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }
    let numerator = a * summarize_truth
        + (2.0 - a) * summarize_counts_from_rand_index(&counts_estimate)
        - 2.0 * summarize_counts_from_rand_index(&counts_joint);
    let n = labels_estimate.len() as f64;
    let rand_index = 1.0 - numerator / (n * (n - 1.0));
    let entropy_option = if with_entropy {
        Some(entropy(&counts_estimate, n))
    } else {
        None
    };
    (rand_index, entropy_option)
}

#[roxido]
fn samplePartition(
    n_samples: usize,
    n_items: usize,
    prior: &RExternalPtr,
    randomize_permutation: bool,
    randomize_shrinkage: &str,
    max: f64,
    shape1: f64,
    shape2: f64,
    n_cores: usize,
) {
    let n_cores = {
        if n_cores == 0 {
            num_cpus::get()
        } else {
            n_cores
        }
    };
    let np = n_samples.max(1);
    let np_per_core = np / n_cores;
    let np_extra = np % n_cores;
    let ni = n_items.max(1);
    let chunk_size = np_per_core * ni;
    let matrix_rval = RMatrix::<i32>::new(ni, np, pc);
    let mut stick = matrix_rval.slice_mut();
    let randomize_shrinkage = match randomize_shrinkage {
        "fixed" => RandomizeShrinkage::Fixed,
        "common" => RandomizeShrinkage::Common,
        "cluster" => RandomizeShrinkage::Cluster,
        "idiosyncratic" => RandomizeShrinkage::Idiosyncratic,
        _ => stop!("Unrecognized 'randomize_shrinkage' value"),
    };
    let max = ScalarShrinkage::new(max).unwrap_or_else(|| stop!("'max' should be non-negative"));
    let shape1 = Shape::new(shape1).unwrap_or_else(|| stop!("'shape1' should be greater than 0"));
    let shape2 = Shape::new(shape2).unwrap_or_else(|| stop!("'shape2' should be greater than 0"));
    macro_rules! distr_macro {
        ($tipe: ty, $callback: tt) => {{
            let _ = crossbeam::scope(|s| {
                let mut nonnull = NonNull::new(prior.address() as *mut $tipe).unwrap();
                let distr = unsafe { nonnull.as_mut() };
                let mut plan = Vec::with_capacity(n_cores);
                let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
                for k in 0..n_cores - 1 {
                    let (left, right) =
                        stick.split_at_mut(chunk_size + if k < np_extra { ni } else { 0 });
                    plan.push((left, distr.clone(), rng.gen::<u128>()));
                    stick = right;
                }
                plan.push((stick, distr.clone(), rng.gen()));
                plan.into_iter().for_each(|mut p| {
                    s.spawn(move |_| {
                        let mut rng = Pcg64Mcg::new(p.2);
                        for j in 0..p.0.len() / ni {
                            #[allow(clippy::redundant_closure_call)]
                            ($callback)(&mut p.1, &mut rng);
                            let clust = p.1.sample(&mut rng).standardize();
                            let labels = clust.allocation();
                            for i in 0..ni {
                                p.0[ni * j + i] = (labels[i] + 1).try_into().unwrap();
                            }
                        }
                    });
                });
            });
        }};
    }
    fn mk_lambda_sp<D: PredictiveProbabilityFunction + Clone>(
        randomize_permutation: bool,
        randomize_shrinkage: RandomizeShrinkage,
        max: ScalarShrinkage,
        shape1: Shape,
        shape2: Shape,
    ) -> impl Fn(&mut SpParameters<D>, &mut Pcg64Mcg) {
        move |distr: &mut SpParameters<D>, rng: &mut Pcg64Mcg| {
            if randomize_permutation {
                distr.permutation.shuffle(rng);
            }
            match randomize_shrinkage {
                RandomizeShrinkage::Fixed => {}
                RandomizeShrinkage::Common => {
                    distr.shrinkage.randomize_common(max, shape1, shape2, rng)
                }
                RandomizeShrinkage::Cluster => distr.shrinkage.randomize_common_cluster(
                    max,
                    shape1,
                    shape2,
                    &distr.anchor,
                    rng,
                ),
                RandomizeShrinkage::Idiosyncratic => distr
                    .shrinkage
                    .randomize_idiosyncratic(max, shape1, shape2, rng),
            };
        }
    }
    let tag = prior.tag().as_scalar().stop();
    let name = tag.str(pc);
    match name {
        "up" => distr_macro!(
            UpParameters,
            (|_distr: &mut UpParameters, _rng: &mut Pcg64Mcg| {})
        ),
        "jlp" => distr_macro!(
            JlpParameters,
            (|_distr: &mut JlpParameters, _rng: &mut Pcg64Mcg| {})
        ),
        "crp" => distr_macro!(
            CrpParameters,
            (|_distr: &mut CrpParameters, _rng: &mut Pcg64Mcg| {})
        ),
        "sp-up" => {
            distr_macro!(
                SpParameters<UpParameters>,
                (mk_lambda_sp::<UpParameters>(
                    randomize_permutation,
                    randomize_shrinkage,
                    max,
                    shape1,
                    shape2
                ))
            );
        }
        "sp-jlp" => {
            distr_macro!(
                SpParameters<JlpParameters>,
                (mk_lambda_sp::<JlpParameters>(
                    randomize_permutation,
                    randomize_shrinkage,
                    max,
                    shape1,
                    shape2
                ))
            );
        }
        "sp-crp" => {
            distr_macro!(
                SpParameters<CrpParameters>,
                (mk_lambda_sp::<CrpParameters>(
                    randomize_permutation,
                    randomize_shrinkage,
                    max,
                    shape1,
                    shape2
                ))
            );
        }
        _ => stop!("Unsupported distribution: {}", name),
    }
    matrix_rval.transpose(pc)
}

#[roxido]
fn rngs_new() {
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let rng2 = Pcg64Mcg::from_seed(seed);
    let result = RList::new(2, pc);
    result.set(0, RExternalPtr::encode(rng, "rng", pc)).stop();
    result.set(1, RExternalPtr::encode(rng2, "rng", pc)).stop();
    result
}

fn permutation_to_r(permutation: &Permutation, rval: &mut RVector<i32>) {
    for (x, y) in permutation.as_slice().iter().zip(rval.slice_mut()) {
        *y = i32::try_from(*x).unwrap() + 1;
    }
}

fn scalar_shrinkage_to_r(shrinkage: &ScalarShrinkage, rval: &mut RVector<f64>) {
    rval.slice_mut()[0] = shrinkage.get();
}

fn shrinkage_to_r(shrinkage: &Shrinkage, rval: &mut RVector<f64>) {
    for (x, y) in rval.slice_mut().iter_mut().zip(shrinkage.as_slice()) {
        *x = y.get()
    }
}

fn grit_to_r(grit: &Grit, rval: &mut RVector<f64>) {
    rval.slice_mut()[0] = grit.get();
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
fn all(all: &RList) {
    let mut vec = Vec::with_capacity(all.len());
    let mut n_items = None;
    for i in 0..all.len() {
        let list = all.get(i).stop().as_list().stop();
        let (data, state, hyperparameters) = (
            list.get(0).stop().as_list().stop(),
            list.get(1).stop().as_list().stop(),
            list.get(2).stop().as_list().stop(),
        );
        let data = Data::from_r(data, pc).stop();
        let state = State::from_r(state, pc).stop();
        let hyperparameters = Hyperparameters::from_r(hyperparameters, pc).stop();
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
    RExternalPtr::encode(all, "all", pc)
}

#[derive(Debug)]
struct GlobalMcmcTuning {
    n_iterations: usize,
    burnin: usize,
    thinning: usize,
    update_anchor: bool,
}

impl GlobalMcmcTuning {
    fn n_saves(&self) -> usize {
        (self.n_iterations - self.burnin) / self.thinning
    }
}

impl FromR<RList, String> for GlobalMcmcTuning {
    fn from_r(x: &RList, _pc: &Pc) -> Result<Self, String> {
        let mut map = x.make_map();
        let result = Self {
            n_iterations: map.get("n_iterations")?.as_scalar()?.usize()?,
            burnin: map.get("burnin")?.as_scalar()?.usize()?,
            thinning: map.get("thinning")?.as_scalar()?.usize()?,
            update_anchor: map.get("update_anchor")?.as_scalar()?.bool()?,
        };
        map.exhaustive()?;
        Ok(result)
    }
}

fn validation_data_from_r(
    validation_data: Option<&RList>,
    pc: &Pc,
) -> Result<Option<Vec<Data>>, String> {
    match validation_data {
        Some(x) => {
            let n_units = x.len();
            let mut vd = Vec::with_capacity(n_units);
            for k in 0..n_units {
                vd.push(Data::from_r(x.get(k).stop().as_list().stop(), pc)?);
            }
            Ok(Some(vd))
        }
        None => Ok(None),
    }
}

struct Timers {
    units: TicToc,
    anchor: TicToc,
    permutation: TicToc,
    shrinkage: TicToc,
    grit: TicToc,
}

struct Results<'a> {
    rval: &'a mut RList,
    counter: usize,
    n_items: usize,
    unit_partitions: Vec<&'a mut [i32]>,
    anchors: &'a mut [i32],
    permutations: &'a mut [i32],
    shrinkages: &'a mut [f64],
    grits: &'a mut [f64],
    log_likelihoods: &'a mut [f64],
    timers: Timers,
}

impl<'a> Results<'a> {
    pub fn new(tuning: &GlobalMcmcTuning, n_items: usize, n_units: usize, pc: &'a Pc) -> Self {
        let n_saves = tuning.n_saves();
        let unit_partitions_rval = RList::new(n_units, pc);
        let mut unit_partitions = Vec::with_capacity(n_units);
        for t in 0..n_units {
            let rval = RMatrix::<i32>::new(n_items, n_saves, pc);
            unit_partitions_rval.set(t, rval).stop();
            unit_partitions.push(rval.slice_mut());
        }
        let anchors_rval = RMatrix::<i32>::new(n_items, n_saves, pc);
        let permutations_rval = RMatrix::<i32>::new(n_items, n_saves, pc);
        let shrinkages_rval = RVector::<f64>::new(n_saves, pc);
        let grits_rval = RVector::<f64>::new(n_saves, pc);
        let log_likelihoods_rval = RVector::<f64>::new(n_saves, pc);
        let names = &[
            "unit_partitions",
            "anchor",
            "permutation",
            "shrinkage",
            "grit",
            "log_likelihood",
            "permutation_acceptance_rate",
            "shrinkage_slice_n_evaluations_rate",
            "grit_slice_n_evaluations_rate",
            "wall_times",
        ];
        let rval = RList::with_names(names, pc); // Extra 4 items for rates after looping.
        rval.set(0, unit_partitions_rval).stop();
        rval.set(1, anchors_rval).stop();
        rval.set(2, permutations_rval).stop();
        rval.set(3, shrinkages_rval).stop();
        rval.set(4, grits_rval).stop();
        rval.set(5, log_likelihoods_rval).stop();
        Self {
            rval,
            counter: 0,
            n_items,
            unit_partitions,
            anchors: anchors_rval.slice_mut(),
            permutations: permutations_rval.slice_mut(),
            shrinkages: shrinkages_rval.slice_mut(),
            grits: grits_rval.slice_mut(),
            log_likelihoods: log_likelihoods_rval.slice_mut(),
            timers: Timers {
                units: TicToc::new(),
                anchor: TicToc::new(),
                permutation: TicToc::new(),
                shrinkage: TicToc::new(),
                grit: TicToc::new(),
            },
        }
    }

    pub fn push<'b, I: Iterator<Item = &'b Clustering>>(
        &mut self,
        unit_partitions: I,
        anchor: Option<&Clustering>,
        permutation: &Permutation,
        shrinkage: f64,
        grit: f64,
        log_likelihoods: f64,
    ) {
        let range = (self.counter * self.n_items)..((self.counter + 1) * self.n_items);
        for (clustering, full_slice) in unit_partitions.zip(self.unit_partitions.iter_mut()) {
            let slice = &mut full_slice[range.clone()];
            clustering.relabel_into_slice(1, slice);
        }
        if let Some(anchor) = anchor {
            let slice = &mut self.anchors[range.clone()];
            anchor.relabel_into_slice(1, slice);
        };
        let slice = &mut self.permutations[range];
        for (x, y) in slice.iter_mut().zip(permutation.as_slice()) {
            *x = *y as i32 + 1;
        }
        self.shrinkages[self.counter] = shrinkage;
        self.grits[self.counter] = grit;
        self.log_likelihoods[self.counter] = log_likelihoods;
        self.counter += 1;
    }

    fn finalize(
        &mut self,
        permutation_rate: f64,
        shrinkage_slice_evaluations_rate: f64,
        grit_slice_evaluations_rate: f64,
        pc: &Pc,
    ) {
        self.rval.set(6, permutation_rate.to_r(pc)).stop();
        self.rval
            .set(7, shrinkage_slice_evaluations_rate.to_r(pc))
            .stop();
        self.rval
            .set(8, grit_slice_evaluations_rate.to_r(pc))
            .stop();
        let wall_times = [
            self.timers.units.as_secs_f64(),
            self.timers.anchor.as_secs_f64(),
            self.timers.permutation.as_secs_f64(),
            self.timers.shrinkage.as_secs_f64(),
            self.timers.grit.as_secs_f64(),
        ]
        .to_r(pc);
        let names = ["units", "anchor", "permutation", "shrinkage", "grit"].to_r(pc);
        wall_times.set_names(names).stop();
        self.rval.set(9, wall_times).stop();
    }
}

#[roxido]
fn fit_dependent(
    model_id: &str,
    all: &mut RExternalPtr,
    anchor_concentration: f64,
    baseline_concentration: f64,
    hyperparameters: &RList,
    unit_mcmc_tuning: &RList,
    global_mcmc_tuning: &RList,
    validation_data: &RObject,
) {
    let do_hierarchical_model = match model_id {
        "hierarchical" => true,
        "temporal" => false,
        model_id => stop!("Unsupported model {}", model_id),
    };
    let all: &mut All = all.decode_mut();
    let validation_data = validation_data_from_r(
        validation_data.as_option().map(|x| {
            x.as_list()
                .stop_str("'validation_data' should be NULL or a list.")
        }),
        pc,
    )
    .stop();
    let unit_mcmc_tuning = McmcTuning::from_r(unit_mcmc_tuning, pc).stop();
    let hyperparameters = Hyperparameters::from_r(hyperparameters, pc).stop();
    let global_mcmc_tuning = GlobalMcmcTuning::from_r(global_mcmc_tuning, pc).stop();
    let n_units = all.units.len();
    let mut results = Results::new(&global_mcmc_tuning, all.n_items, n_units, pc);
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let mut rng2 = Pcg64Mcg::from_seed(seed);
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
    let fastrand_ = fastrand::Rng::with_seed(rng.gen());
    let fastrand = &mut Some(fastrand_);
    let anchor = all.units[rng.gen_range(0..all.units.len())]
        .state
        .clustering()
        .clone();
    let mut shrinkage_value = ScalarShrinkage::new(
        hyperparameters.shrinkage.shape / hyperparameters.shrinkage.rate.get(),
    )
    .stop();
    let grit = Grit::new(
        hyperparameters.grit.shape1
            / (hyperparameters.grit.shape1 + hyperparameters.grit.shape2.get()),
    )
    .stop();
    let permutation = Permutation::random(all.n_items, &mut rng);
    let baseline_distribution = CrpParameters::new(
        all.n_items,
        Concentration::new(baseline_concentration).stop(),
    );
    let anchor_distribution =
        CrpParameters::new(all.n_items, Concentration::new(anchor_concentration).stop());
    let anchor_update_permutation = Permutation::natural_and_fixed(all.n_items);
    let mut partition_distribution = SpParameters::new(
        anchor,
        Shrinkage::constant(shrinkage_value, all.n_items),
        permutation,
        grit,
        baseline_distribution,
    )
    .unwrap();
    let mut permutation_n_acceptances: u64 = 0;
    let mut shrinkage_slice_n_evaluations: u64 = 0;
    let mut grit_slice_n_evaluations: u64 = 0;
    for iteration_counter in
        (0..global_mcmc_tuning.n_iterations).step_by(global_mcmc_tuning.thinning)
    {
        for _ in 0..global_mcmc_tuning.thinning {
            // Update each unit
            results.timers.units.tic();
            if do_hierarchical_model {
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
            } else {
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
                        partition_distribution.anchor =
                            left.last().unwrap().state.clustering.clone();
                    };
                    let (middle, right) = not_left.split_at_mut(1);
                    let unit = middle.first_mut().unwrap();
                    let clustering_next = right.first().map(|x| &x.state.clustering);
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
            }
            results.timers.units.toc();
            // Update anchor
            if do_hierarchical_model {
                results.timers.anchor.tic();
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
                results.timers.anchor.toc();
            }
            // Helper
            let compute_log_likelihood = |pd: &SpParameters<CrpParameters>| {
                if do_hierarchical_model {
                    all.units
                        .par_iter()
                        .fold_with(0.0, |acc, x| acc + pd.log_pmf(&x.state.clustering))
                        .sum::<f64>()
                } else {
                    let mut partition_distribution = pd.clone();
                    let mut shrinkage_value = partition_distribution.shrinkage[0];
                    let mut sum = 0.0;
                    for time in 0..all.units.len() {
                        let (left, not_left) = all.units.split_at(time);
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
                            partition_distribution.anchor =
                                left.last().unwrap().state.clustering.clone();
                        };
                        let (middle, _) = not_left.split_at(1);
                        let unit = middle.first().unwrap();
                        sum += partition_distribution.log_pmf(&unit.state.clustering)
                    }
                    sum
                }
            };
            // Update permutation
            results.timers.permutation.tic();
            if unit_mcmc_tuning.n_permutation_updates_per_scan > 0 {
                let k = unit_mcmc_tuning.n_items_per_permutation_update;
                let mut log_target_current: f64 = compute_log_likelihood(&partition_distribution);
                for _ in 0..unit_mcmc_tuning.n_permutation_updates_per_scan {
                    partition_distribution
                        .permutation
                        .partial_shuffle(k, &mut rng);
                    let log_target_proposal = compute_log_likelihood(&partition_distribution);
                    let log_hastings_ratio = log_target_proposal - log_target_current;
                    if 0.0 <= log_hastings_ratio
                        || rng.gen_range(0.0..1.0_f64).ln() < log_hastings_ratio
                    {
                        log_target_current = log_target_proposal;
                        if iteration_counter >= global_mcmc_tuning.burnin {
                            permutation_n_acceptances += 1;
                        }
                    } else {
                        partition_distribution.permutation.partial_shuffle_undo(k);
                    }
                }
            }
            results.timers.permutation.toc();
            // Update shrinkage
            results.timers.shrinkage.tic();
            if let Some(w) = unit_mcmc_tuning.shrinkage_slice_step_size {
                if let Some(reference) = hyperparameters.shrinkage.reference {
                    let shrinkage_prior_distribution = GammaPDF::new(
                        hyperparameters.shrinkage.shape.get(),
                        hyperparameters.shrinkage.rate.get(),
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
                            fastrand,
                        );
                    // // Not necessary... see implementation of slice_sampler function.
                    // partition_distribution.shrinkage.rescale_by_reference(
                    //     global_hyperparameters.shrinkage_reference,
                    //     ScalarShrinkage::new(_s_new).unwrap(),
                    // );
                    if iteration_counter >= global_mcmc_tuning.burnin {
                        shrinkage_slice_n_evaluations += u64::from(n_evaluations);
                    }
                }
            }
            results.timers.shrinkage.toc();
            // Update grit
            results.timers.grit.tic();
            if let Some(w) = unit_mcmc_tuning.grit_slice_step_size {
                let grit_prior_distribution = BetaPDF::new(
                    hyperparameters.grit.shape1.get(),
                    hyperparameters.grit.shape2.get(),
                )
                .unwrap();
                let tuning_parameters = stepping_out::TuningParameters::new().width(w);
                let (_g_new, n_evaluations) =
                    stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                        partition_distribution.grit.get(),
                        |g| match Grit::new(g) {
                            None => f64::NEG_INFINITY,
                            Some(grit) => {
                                partition_distribution.grit = grit;
                                grit_prior_distribution.ln_pdf(g)
                                    + compute_log_likelihood(&partition_distribution)
                            }
                        },
                        true,
                        &tuning_parameters,
                        fastrand,
                    );
                // // Not necessary... see implementation of slice_sampler function.
                // partition_distribution.shrinkage.rescale_by_reference(
                //     global_hyperparameters.shrinkage_reference,
                //     ScalarShrinkage::new(_s_new).unwrap(),
                // );
                if iteration_counter >= global_mcmc_tuning.burnin {
                    grit_slice_n_evaluations += u64::from(n_evaluations);
                }
            }
            results.timers.grit.toc();
        }
        // Report
        if iteration_counter >= global_mcmc_tuning.burnin {
            let log_likelihood = match &validation_data {
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
                if do_hierarchical_model {
                    Some(&partition_distribution.anchor)
                } else {
                    None
                },
                &partition_distribution.permutation,
                if let Some(reference) = hyperparameters.shrinkage.reference {
                    partition_distribution.shrinkage[reference].get()
                } else {
                    f64::NAN
                },
                partition_distribution.grit.get(),
                log_likelihood,
            );
        }
    }
    let denominator = (global_mcmc_tuning.n_saves() * global_mcmc_tuning.thinning) as f64;
    results.finalize(
        permutation_n_acceptances as f64
            / (unit_mcmc_tuning.n_permutation_updates_per_scan as f64 * denominator),
        shrinkage_slice_n_evaluations as f64 / denominator,
        grit_slice_n_evaluations as f64 / denominator,
        pc,
    );
    results.rval
}

#[roxido]
fn fit(
    n_updates: usize,
    data: &mut RExternalPtr,
    state: &mut RExternalPtr,
    hyperparameters: &RExternalPtr,
    monitor: &mut RExternalPtr,
    partition_distribution: &mut RExternalPtr,
    mcmc_tuning: &RList,
    permutation_bucket: &mut RVector<i32>, // This is used to communicating temporary results back to R
    shrinkage_bucket: &mut RVector<f64>, // This is used to communicating temporary results back to R
    grit_bucket: &mut RVector<f64>, // This is used to communicating temporary results back to R
    rngs: &mut RList,
) {
    let n_updates: u32 = n_updates.try_into().unwrap();
    let data: &mut Data = data.decode_mut();
    let state: &mut State = state.decode_mut();
    let hyperparameters: &Hyperparameters = hyperparameters.decode_ref();
    let monitor = monitor.decode_mut::<Monitor<u32>>();
    let mcmc_tuning = McmcTuning::from_r(mcmc_tuning, pc).stop();
    if data.n_global_covariates() != state.n_global_covariates()
        || hyperparameters.n_global_covariates() != state.n_global_covariates()
    {
        stop!("Inconsistent number of global covariates...\n    data: {}\n    state: {}\n    hyperparameters: {}", data.n_global_covariates(), state.n_global_covariates(), hyperparameters.n_global_covariates());
    }
    if data.n_clustered_covariates() != state.n_clustered_covariates()
        || hyperparameters.n_clustered_covariates() != state.n_clustered_covariates()
    {
        stop!("Inconsistent number of clustered covariates.")
    }
    if data.n_items() != state.clustering.n_items() {
        stop!("Inconsistent number of items.")
    }
    let rng = rngs.get_mut(0).stop().as_external_ptr_mut().stop();
    let rng = rng.decode_mut::<Pcg64Mcg>();
    let rng2 = rngs.get_mut(1).stop().as_external_ptr_mut().stop();
    let rng2 = rng2.decode_mut::<Pcg64Mcg>();
    let tag = partition_distribution.tag().as_scalar().stop();
    let prior_name = tag.str(pc);
    #[rustfmt::skip]
    macro_rules! mcmc_update { // (_, HAS_PERMUTATION, HAS_SCALAR_SHRINKAGE, HAS_VECTOR_SHRINKAGE, HAS_GRIT)
        ($tipe:ty, false, false, false, false) => {{
            let partition_distribution = partition_distribution.decode_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
            }
        }};
        ($tipe:ty, true, false, false, false) => {{
            let partition_distribution = partition_distribution.decode_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(u32::try_from(mcmc_tuning.n_permutation_updates_per_scan).unwrap(), |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, u32::try_from(mcmc_tuning.n_items_per_permutation_update).unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation_bucket);
            }
        }};
        ($tipe:ty, true, true, false, false) => {{
            let partition_distribution = partition_distribution.decode_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(u32::try_from(mcmc_tuning.n_permutation_updates_per_scan).unwrap(), |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, u32::try_from(mcmc_tuning.n_items_per_permutation_update).unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation_bucket);
                if let Some(w) = mcmc_tuning.shrinkage_slice_step_size {
                    dahl_randompartition::mcmc::update_scalar_shrinkage(1, partition_distribution, w, hyperparameters.shrinkage.shape, hyperparameters.shrinkage.rate, &state.clustering, rng);
                    scalar_shrinkage_to_r(&partition_distribution.shrinkage, shrinkage_bucket);
                }
            }
        }};
        ($tipe:ty, true, false, true, true) => {{
            let partition_distribution = partition_distribution.decode_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(u32::try_from(mcmc_tuning.n_permutation_updates_per_scan).unwrap(), |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, u32::try_from(mcmc_tuning.n_items_per_permutation_update).unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, permutation_bucket);
                if let Some(w) = mcmc_tuning.shrinkage_slice_step_size {
                    if let Some(reference) = &hyperparameters.shrinkage.reference {
                        dahl_randompartition::mcmc::update_shrinkage(1, partition_distribution, *reference, w, hyperparameters.shrinkage.shape, hyperparameters.shrinkage.rate, &state.clustering, rng);
                        shrinkage_to_r(&partition_distribution.shrinkage, shrinkage_bucket);
                    }
                }
                if let Some(w) = mcmc_tuning.grit_slice_step_size {
                    dahl_randompartition::mcmc::update_grit(1, partition_distribution, w, hyperparameters.grit.shape1, hyperparameters.grit.shape2, &state.clustering, rng);
                    grit_to_r(&partition_distribution.grit, grit_bucket);
                }
            }
        }};
    }
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
fn log_likelihood_contributions(state: &RExternalPtr, data: &RExternalPtr) {
    let state: &State = state.decode_ref();
    let data = data.decode_ref();
    state.log_likelihood_contributions(data).to_r(pc)
}

#[roxido]
fn log_likelihood_contributions_of_missing(state: &RExternalPtr, data: &RExternalPtr) {
    let state: &State = state.decode_ref();
    let data = data.decode_ref();
    state.log_likelihood_contributions_of_missing(data).to_r(pc)
}

#[roxido]
fn log_likelihood_of(state: &RExternalPtr, data: &RExternalPtr, items: &RVector) {
    let state: &State = state.decode_ref();
    let data = data.decode_ref();
    let items: Vec<_> = items
        .to_i32(pc)
        .slice()
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    state.log_likelihood_of(data, items)
}

#[roxido]
fn log_likelihood(state: &RExternalPtr, data: &RExternalPtr) {
    let state: &State = state.decode_ref();
    let data = data.decode_ref();
    state.log_likelihood(data)
}

#[roxido]
fn sample_multivariate_normal(n_samples: usize, mean: &RVector, precision: &RMatrix) {
    let mean = mean.to_f64(pc);
    let mean = mean.slice();
    let n = mean.len();
    let mean = DVector::from_iterator(n, mean.iter().cloned());
    let precision = precision.to_f64(pc);
    let precision = DMatrix::from_iterator(n, n, precision.slice().iter().cloned());
    let rval = RMatrix::<f64>::new(n, n_samples, pc);
    let slice = rval.slice_mut();
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let x = sample_multivariate_normal_repeatedly(n_samples, mean, precision, &mut rng).unwrap();
    slice.clone_from_slice(x.as_slice());
    rval.transpose(pc)
}

fn validate_list<'a>(list: &'a RList, expected_names: &[&str], arg_name: &str) -> &'a RList {
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
