// The 'roxido_registration' macro is called at the start of the 'lib.rs' file.
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
use dahl_randompartition::mcmc::update_neal_algorithm3;
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
fn new_FixedPartitionParameters(anchor: &RVector) {
    let anchor = Clustering::from_slice(anchor.to_i32(pc).slice());
    let p = FixedPartitionParameters::new(anchor);
    RExternalPtr::encode(p, "fixed", pc)
}

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
fn new_LspParameters(anchor: &RVector, shrinkage: f64, concentration: f64, permutation: &RVector) {
    let anchor = Clustering::from_slice(anchor.to_i32(pc).slice());
    let shrinkage =
        ScalarShrinkage::new(shrinkage).unwrap_or_else(|| stop!("Invalid shrinkage value"));
    let concentration =
        Concentration::new(concentration).unwrap_or_else(|| stop!("Invalid concentration value"));
    let permutation = mk_permutation(permutation, pc);
    let p =
        LspParameters::new_with_shrinkage(anchor, shrinkage, concentration, permutation).unwrap();
    RExternalPtr::encode(p, "lsp", pc)
}

#[roxido]
fn new_CppParameters(anchor: &RVector, rate: f64, baseline: &RExternalPtr, use_vi: bool, a: f64) {
    let anchor = Clustering::from_slice(anchor.to_i32(pc).slice());
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline.address() as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            RExternalPtr::encode(
                CppParameters::new(anchor, rate, baseline, use_vi, a).unwrap(),
                &$label,
                pc,
            )
        }};
    }
    let tag = baseline.tag().as_scalar().stop();
    let name = tag.str(pc);
    match name {
        "up" => distr_macro!(UpParameters, "cpp-up"),
        "jlp" => distr_macro!(JlpParameters, "cpp-jlp"),
        "crp" => distr_macro!(CrpParameters, "cpp-crp"),
        _ => stop!("Unsupported distribution: {}", name),
    }
}

#[roxido]
fn new_SpParameters(
    anchor: &RVector,
    shrinkage: &RVector,
    permutation: &RVector,
    grit: f64,
    baseline: &RExternalPtr,
    shortcut: bool,
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
                SpParameters::new(anchor, shrinkage, permutation, grit, baseline, shortcut)
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
fn summarize_prior_on_shrinkage_and_grit(
    anchor: &RVector,
    shrinkage_shape: f64,
    shrinkage_rate: f64,
    grit_shape1: f64,
    grit_shape2: f64,
    use_crp: bool,
    concentration: f64,
    shrinkage_n: usize,
    grit_n: usize,
    n_mc_samples: i32,
    domain_specification: &RList,
    a: f64,
    n_cores: i32,
) {
    let anchor = anchor.to_i32(pc).slice();
    if shrinkage_shape <= 0.0 {
        stop!("'shrinkage_shape' should be strictly positive.");
    };
    if shrinkage_rate <= 0.0 {
        stop!("'shrinkage_rate' should be strictly positive.");
    };
    if shrinkage_n < 2 {
        stop!("'shrinkage_n' must be at least two.");
    }
    if grit_shape1 <= 0.0 {
        stop!("'grit_shape1' should be strictly positive.");
    };
    if grit_shape2 <= 0.0 {
        stop!("'grit_shape2' should be strictly positive.");
    };
    if grit_n < 2 {
        stop!("'grit_n' must be at least two.");
    }
    if !(0.0..=2.0).contains(&a) {
        stop!("Parameter 'a' must be in [0.0, 2.0].");
    }
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let ((shrinkage_min, shrinkage_max), (grit_min, grit_max)) = match domain_specification
        .get_by_key("n_mc_samples")
    {
        Ok(n_mc_samples) => {
            let Ok(n_mc_samples) = n_mc_samples.as_scalar() else {
                stop!("Element 'n_mc_samples' in 'domain_specification' should be a scalar.");
            };
            let Ok(n_mc_samples) = n_mc_samples.usize() else {
                stop!("Element 'n_mc_samples' in 'domain_specification' cannot be interpreted as positive integer.");
            };
            if n_mc_samples <= 1 {
                stop!("Element 'n_mc_samples' is 'domain_specification' must be at least 2.");
            }
            let Ok(percentile) = domain_specification.get_by_key("percentile") else {
                stop!("Element 'percentile' must be provided when 'n_mc_samples' is provided in 'domain_specification'.");
            };
            let Ok(percentile) = percentile.as_scalar() else {
                stop!("Element 'percentile' in 'domain_specification' should be a scalar.");
            };
            let percentile = percentile.f64();
            if !(0.0..=1.0).contains(&percentile) {
                stop!("Element 'percentile' is 'domain_specification' must be in [0, 1].")
            }
            let Ok(gamma_dist) = GammaRNG::new(shrinkage_shape, 1.0 / shrinkage_rate) else {
                stop!("Cannot construct gamma distribution for shrinkage.");
            };
            let Ok(beta_dist) = BetaRNG::new(grit_shape1, grit_shape2) else {
                stop!("Cannot construct beta distribution for grit.");
            };
            let mut sample = vec![0.0; n_mc_samples];
            sample.fill_with(|| gamma_dist.sample(&mut rng));
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let shrinkage_max = sample[(percentile * (n_mc_samples as f64)).floor() as usize];
            sample.fill_with(|| beta_dist.sample(&mut rng));
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let beta_max = sample[(percentile * (n_mc_samples as f64)).floor() as usize];
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
            "expected_vi",
            "expected_binder",
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
    let expected_vi_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(3, expected_vi_rval);
    let expected_binder_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(4, expected_binder_rval);
    let expected_rand_index_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(5, expected_rand_index_rval);
    let expected_entropy_rval = RMatrix::<f64>::new(shrinkage_n, grit_n, pc);
    let _ = result.set(6, expected_entropy_rval);
    let shrinkage_slice = shrinkage_rval.slice_mut();
    let grit_slice = grit_rval.slice_mut();
    let log_density_slice = log_density_rval.slice_mut();
    let expected_vi_slice = expected_vi_rval.slice_mut();
    let expected_binder_slice = expected_binder_rval.slice_mut();
    let expected_rand_index_slice = expected_rand_index_rval.slice_mut();
    let expected_entropy_slice = expected_entropy_rval.slice_mut();
    let Ok(gamma_dist) = GammaPDF::new(shrinkage_shape, shrinkage_rate) else {
        stop!("Cannot construct gamma distribution for shrinkage.");
    };
    let Ok(beta_dist) = BetaPDF::new(grit_shape1, grit_shape2) else {
        stop!("Cannot construct beta distribution for grit.");
    };
    let log_density_shrinkage: Vec<_> = shrinkage_slice
        .iter_mut()
        .enumerate()
        .map(|(i, shrinkage)| {
            *shrinkage = shrinkage_min + (i as f64) * shrinkage_width;
            gamma_dist.ln_pdf(*shrinkage)
        })
        .collect();
    let log_density_grit: Vec<_> = grit_slice
        .iter_mut()
        .enumerate()
        .map(|(j, grit)| {
            *grit = grit_min + (j as f64) * grit_width;
            beta_dist.ln_pdf(*grit)
        })
        .collect();
    fn engine<D: PredictiveProbabilityFunction + Clone + std::marker::Sync>(
        partition_distribution: SpParameters<D>,
        a: f64,
        n_cores: i32,
        n_mc_samples: i32,
        shrinkage_n: usize,
        grit_n: usize,
        mut rng: Pcg64Mcg,
        shrinkage_slice: &mut [f64],
        grit_slice: &mut [f64],
        log_density_slice: &mut [f64],
        expected_vi_slice: &mut [f64],
        expected_binder_slice: &mut [f64],
        expected_rand_index_slice: &mut [f64],
        expected_entropy_slice: &mut [f64],
        log_density_shrinkage: &[f64],
        log_density_grit: &[f64],
    ) {
        let n_mc_samples_f64 = n_mc_samples as f64;
        let n_items = partition_distribution.anchor.n_items() as f64;
        let counts_truth = marginal_counter(partition_distribution.anchor.allocation());
        let summarize_truth_rand_index = summarize_counts_for_rand_index(&counts_truth);
        let summarize_truth_vi = summarize_counts_for_vi(&counts_truth, n_items);
        let summarize_truth_binder = summarize_counts_for_binder(&counts_truth, n_items);
        let indices: Vec<_> = (0..shrinkage_n * grit_n).collect();
        let n_cores = if n_cores == 0 {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        } else {
            n_cores.max(1) as usize
        };
        indices
            .chunks(n_cores)
            .map(|indices| (indices, rng.random::<u128>()))
            .par_bridge()
            .map(|(indices, seed)| {
                let mut partition_distribution = partition_distribution.clone();
                let mut rng = Pcg64Mcg::new(seed);
                let result = indices
                    .iter()
                    .map(|&index| {
                        let i = index % shrinkage_n;
                        let j = index / shrinkage_n;
                        partition_distribution
                            .shrinkage
                            .set_constant(ScalarShrinkage::new(shrinkage_slice[i]).unwrap());
                        partition_distribution.grit.set(grit_slice[j]);
                        let mut sum_vi = 0.0;
                        let mut sum_binder = 0.0;
                        let mut sum_rand_index = 0.0;
                        let mut sum_entropy = 0.0;
                        (0..n_mc_samples).for_each(|_| {
                            partition_distribution.permutation.shuffle(&mut rng);
                            let sample = partition_distribution.sample(&mut rng);
                            let result = partition_summary_core(
                                partition_distribution.anchor.allocation(),
                                sample.allocation(),
                                a,
                                &counts_truth,
                                summarize_truth_rand_index,
                                summarize_truth_vi,
                                summarize_truth_binder,
                            );
                            sum_rand_index += result.0;
                            sum_vi += result.1;
                            sum_binder += result.2;
                            sum_entropy += result.3;
                        });
                        let log_density = log_density_shrinkage[i] + log_density_grit[j];
                        let expected_vi = sum_vi / n_mc_samples_f64;
                        let expected_binder = sum_binder / n_mc_samples_f64;
                        let expected_rand_index = sum_rand_index / n_mc_samples_f64;
                        let expected_entropy = sum_entropy / n_mc_samples_f64;
                        (
                            index,
                            log_density,
                            expected_vi,
                            expected_binder,
                            expected_rand_index,
                            expected_entropy,
                        )
                    })
                    .collect::<Vec<_>>();
                result
            })
            .flatten()
            .collect::<Vec<_>>()
            .iter()
            .for_each(
                |(
                    index,
                    log_density,
                    expected_vi,
                    expected_binder,
                    expected_rand_index,
                    expected_entropy,
                )| {
                    log_density_slice[*index] = *log_density;
                    expected_vi_slice[*index] = *expected_vi;
                    expected_binder_slice[*index] = *expected_binder;
                    expected_rand_index_slice[*index] = *expected_rand_index;
                    expected_entropy_slice[*index] = *expected_entropy;
                },
            );
    }
    let anchor = Clustering::from_slice(anchor);
    let n_items = anchor.n_items();
    let permutation = Permutation::natural(n_items);
    let concentration = Concentration::new(concentration).unwrap();
    if use_crp {
        let baseline_ppf = CrpParameters::new(n_items, concentration);
        let partition_distribution = SpParameters::new(
            anchor,
            Shrinkage::zero(n_items),
            permutation,
            Grit::one(),
            baseline_ppf,
            true,
        )
        .unwrap();
        engine(
            partition_distribution,
            a,
            n_cores,
            n_mc_samples,
            shrinkage_n,
            grit_n,
            rng,
            shrinkage_slice,
            grit_slice,
            log_density_slice,
            expected_vi_slice,
            expected_binder_slice,
            expected_rand_index_slice,
            expected_entropy_slice,
            &log_density_shrinkage,
            &log_density_grit,
        );
    } else {
        let baseline_ppf = JlpParameters::new(n_items, concentration, permutation.clone()).unwrap();
        let partition_distribution = SpParameters::new(
            anchor,
            Shrinkage::zero(n_items),
            permutation,
            Grit::one(),
            baseline_ppf,
            true,
        )
        .unwrap();
        engine(
            partition_distribution,
            a,
            n_cores,
            n_mc_samples,
            shrinkage_n,
            grit_n,
            rng,
            shrinkage_slice,
            grit_slice,
            log_density_slice,
            expected_vi_slice,
            expected_binder_slice,
            expected_rand_index_slice,
            expected_entropy_slice,
            &log_density_shrinkage,
            &log_density_grit,
        );
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
            if !(0.0..=2.0).contains(&a) {
                return Err("Parameter 'a' must be in [0.0, 2.0].");
            }
            a
        }
        None => 1.0,
    };
    let counts_truth = marginal_counter(labels_truth);
    let summarize_truth_rand_index = summarize_counts_for_rand_index(&counts_truth);
    let (counts_estimate, counts_joint, n) =
        tabulate(labels_truth, labels_estimate, counts_truth.len());
    Ok(rand_index_engine(
        summarize_truth_rand_index,
        &counts_estimate,
        &counts_joint,
        n,
        a,
    ))
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

fn summarize_counts_for_rand_index(counts: &[u32]) -> f64 {
    counts
        .iter()
        .filter(|&&zz| zz > 0)
        .map(|&zz| zz as f64)
        .map(|x| x * x)
        .sum::<f64>()
}

fn rand_index_engine(
    summarize_truth: f64,
    counts_estimate: &[u32],
    counts_joint: &[u32],
    n: f64,
    a: f64,
) -> f64 {
    let numerator = a * summarize_truth
        + (2.0 - a) * summarize_counts_for_rand_index(counts_estimate)
        - 2.0 * summarize_counts_for_rand_index(counts_joint);
    1.0 - numerator / (n * (n - 1.0))
}

fn summarize_counts_for_vi(counts: &[u32], n: f64) -> f64 {
    counts
        .iter()
        .filter(|&&zz| zz > 0)
        .map(|&zz| zz as f64)
        .map(|x| x / n)
        .map(|p| p * p.log2())
        .sum::<f64>()
}

fn vi_engine(
    summarize_truth: f64,
    counts_estimate: &[u32],
    counts_joint: &[u32],
    n: f64,
    a: f64,
) -> f64 {
    a * summarize_truth + (2.0 - a) * summarize_counts_for_vi(counts_estimate, n)
        - 2.0 * summarize_counts_for_vi(counts_joint, n)
}

fn summarize_counts_for_binder(counts: &[u32], n: f64) -> f64 {
    counts
        .iter()
        .filter(|&&zz| zz > 0)
        .map(|&zz| zz as f64)
        .map(|x| x / n)
        .map(|p| p * p)
        .sum::<f64>()
}

fn binder_engine(
    summarize_truth: f64,
    counts_estimate: &[u32],
    counts_joint: &[u32],
    n: f64,
    a: f64,
) -> f64 {
    a * summarize_truth + (2.0 - a) * summarize_counts_for_binder(counts_estimate, n)
        - 2.0 * summarize_counts_for_binder(counts_joint, n)
}

fn summarize_counts_for_entropy(counts: &[u32], n: f64) -> f64 {
    -counts
        .iter()
        .filter(|&&zz| zz > 0)
        .map(|&zz| zz as f64)
        .map(|x| x / n)
        .map(|p| p * p.ln())
        .sum::<f64>()
}

fn entropy_engine(counts_estimate: &[u32], n: f64) -> f64 {
    summarize_counts_for_entropy(counts_estimate, n)
}

fn tabulate<A: AsUsize + Debug>(
    labels_truth: &[A],
    labels_estimate: &[A],
    n_clusters_truth: usize,
) -> (Vec<u32>, Vec<u32>, f64) {
    let counts_estimate = marginal_counter(labels_estimate);
    let n_clusters_estimate = counts_estimate.len();
    let mut counts_joint = vec![0; n_clusters_truth * n_clusters_estimate];
    for (label_truth, label_estimate) in labels_truth.iter().zip(labels_estimate.iter()) {
        let label_truth = label_truth.as_usize();
        let label_estimate = label_estimate.as_usize();
        counts_joint[n_clusters_truth * label_estimate + label_truth] += 1;
    }
    let n = labels_estimate.len() as f64;
    (counts_estimate, counts_joint, n)
}

fn partition_summary_core<A: AsUsize + Debug>(
    labels_truth: &[A],
    labels_estimate: &[A],
    a: f64,
    counts_truth: &[u32],
    summarize_truth_rand_index: f64,
    summarize_truth_vi: f64,
    summarize_truth_binder: f64,
) -> (f64, f64, f64, f64) {
    let (counts_estimate, counts_joint, n) =
        tabulate(labels_truth, labels_estimate, counts_truth.len());
    let rand_index = rand_index_engine(
        summarize_truth_rand_index,
        &counts_estimate,
        &counts_joint,
        n,
        a,
    );
    let vi = vi_engine(summarize_truth_vi, &counts_estimate, &counts_joint, n, a);
    let binder = binder_engine(
        summarize_truth_binder,
        &counts_estimate,
        &counts_joint,
        n,
        a,
    );
    let entropy = entropy_engine(&counts_estimate, n);
    (rand_index, vi, binder, entropy)
}

#[roxido]
fn samplePartition(
    n_samples: usize,
    n_items: usize,
    prior: &RExternalPtr,
    randomize_permutation: bool,
    randomize_shrinkage: &str,
    randomize_grit: bool,
    shrinkage_shape: f64,
    shrinkage_rate: f64,
    grit_shape1: f64,
    grit_shape2: f64,
    n_cores: usize,
) {
    let n_cores = if n_cores == 0 {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
    } else {
        n_cores
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
    let shrinkage_shape = Shape::new(shrinkage_shape)
        .unwrap_or_else(|| stop!("'shrinkage_shape' should be greater than 0"));
    let shrinkage_rate = Rate::new(shrinkage_rate)
        .unwrap_or_else(|| stop!("'shrinkage_rate' should be greater than 0"));
    let grit_shape1 =
        Shape::new(grit_shape1).unwrap_or_else(|| stop!("'grit_shape1' should be greater than 0"));
    let grit_shape2 =
        Shape::new(grit_shape2).unwrap_or_else(|| stop!("'grit_shape2' should be greater than 0"));
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
                    plan.push((left, distr.clone(), rng.random::<u128>()));
                    stick = right;
                }
                plan.push((stick, distr.clone(), rng.random()));
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
    macro_rules! cpp_macro {
        ($tipe: ty) => {{
            match randomize_shrinkage {
                RandomizeShrinkage::Common | RandomizeShrinkage::Fixed => {}
                _ => stop!("Unsupported randomize_shrinkage for this distribution"),
            }
            let distr = prior.decode_ref::<$tipe>();
            let mut plan = Vec::with_capacity(n_cores);
            let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
            for k in 0..n_cores - 1 {
                let (left, right) =
                    stick.split_at_mut(chunk_size + if k < np_extra { ni } else { 0 });
                plan.push((left, distr.clone(), rng.random::<u128>()));
                stick = right;
            }
            plan.push((stick, distr.clone(), rng.random()));
            let log_like = |_i: usize, _indices: &[usize]| 0.0;
            let nup = 1;
            let _ = crossbeam::scope(|s| {
                plan.into_iter().for_each(|p| {
                    s.spawn(move |_| {
                        let perm = Permutation::natural_and_fixed(n_items);
                        let mut rng = Pcg64Mcg::new(p.2);
                        let mut current = distr.anchor.clone();
                        for i in 0..p.0.len() / ni {
                            update_neal_algorithm3(
                                nup,
                                &mut current,
                                &perm,
                                distr,
                                &log_like,
                                &mut rng,
                            );
                            current.relabel_into_slice(
                                1,
                                &mut p.0[(i * n_items)..((i + 1) * n_items)],
                            );
                        }
                    });
                });
            });
        }};
    }
    fn mk_lambda_sp<D: PredictiveProbabilityFunction + Clone>(
        randomize_permutation: bool,
        randomize_shrinkage: RandomizeShrinkage,
        randomize_grit: bool,
        shrinkage_shape: Shape,
        shrinkage_rate: Rate,
        grit_shape1: Shape,
        grit_shape2: Shape,
    ) -> impl Fn(&mut SpParameters<D>, &mut Pcg64Mcg) {
        let beta = BetaRNG::new(grit_shape1.get(), grit_shape2.get()).unwrap();
        move |distr: &mut SpParameters<D>, rng: &mut Pcg64Mcg| {
            if randomize_permutation {
                distr.permutation.shuffle(rng);
            }
            match randomize_shrinkage {
                RandomizeShrinkage::Fixed => {}
                RandomizeShrinkage::Common => {
                    distr
                        .shrinkage
                        .randomize_common(shrinkage_shape, shrinkage_rate, rng)
                }
                RandomizeShrinkage::Cluster => distr.shrinkage.randomize_common_cluster(
                    shrinkage_shape,
                    shrinkage_rate,
                    &distr.anchor,
                    rng,
                ),
                RandomizeShrinkage::Idiosyncratic => {
                    distr
                        .shrinkage
                        .randomize_idiosyncratic(shrinkage_shape, shrinkage_rate, rng)
                }
            };
            if randomize_grit {
                distr.grit = Grit::new(beta.sample(rng)).unwrap()
            }
        }
    }
    let tag = prior.tag().as_scalar().stop();
    let name = tag.str(pc);
    match name {
        "fixed" => distr_macro!(
            FixedPartitionParameters,
            (|_distr: &mut FixedPartitionParameters, _rng: &mut Pcg64Mcg| {})
        ),
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
        "lsp" => {
            match randomize_shrinkage {
                RandomizeShrinkage::Common | RandomizeShrinkage::Fixed => {}
                _ => stop!("Unsupported randomize_shrinkage for this distribution"),
            }
            if randomize_permutation && randomize_shrinkage == RandomizeShrinkage::Common {
                distr_macro!(
                    LspParameters,
                    (|distr: &mut LspParameters, rng: &mut Pcg64Mcg| {
                        distr.permutation.shuffle(rng);
                        let gamma =
                            GammaRNG::new(shrinkage_shape.get(), 1.0 / shrinkage_rate.get())
                                .unwrap();
                        distr.shrinkage = ScalarShrinkage::new(gamma.sample(rng)).unwrap();
                    })
                );
            } else if randomize_permutation {
                distr_macro!(
                    LspParameters,
                    (|distr: &mut LspParameters, rng: &mut Pcg64Mcg| {
                        distr.permutation.shuffle(rng);
                    })
                );
            } else if randomize_shrinkage == RandomizeShrinkage::Common {
                distr_macro!(
                    LspParameters,
                    (|distr: &mut LspParameters, rng: &mut Pcg64Mcg| {
                        let gamma =
                            GammaRNG::new(shrinkage_shape.get(), 1.0 / shrinkage_rate.get())
                                .unwrap();
                        distr.shrinkage = ScalarShrinkage::new(gamma.sample(rng)).unwrap();
                    })
                );
            } else {
                distr_macro!(
                    LspParameters,
                    (|_distr: &mut LspParameters, _rng: &mut Pcg64Mcg| {})
                );
            }
        }
        "cpp-up" => {
            cpp_macro!(CppParameters<UpParameters>);
        }
        "cpp-jlp" => {
            cpp_macro!(CppParameters<JlpParameters>);
        }
        "cpp-crp" => {
            cpp_macro!(CppParameters<CrpParameters>);
        }
        "sp-up" => {
            distr_macro!(
                SpParameters<UpParameters>,
                (mk_lambda_sp::<UpParameters>(
                    randomize_permutation,
                    randomize_shrinkage,
                    randomize_grit,
                    shrinkage_shape,
                    shrinkage_rate,
                    grit_shape1,
                    grit_shape2
                ))
            );
        }
        "sp-jlp" => {
            distr_macro!(
                SpParameters<JlpParameters>,
                (mk_lambda_sp::<JlpParameters>(
                    randomize_permutation,
                    randomize_shrinkage,
                    randomize_grit,
                    shrinkage_shape,
                    shrinkage_rate,
                    grit_shape1,
                    grit_shape2
                ))
            );
        }
        "sp-crp" => {
            distr_macro!(
                SpParameters<CrpParameters>,
                (mk_lambda_sp::<CrpParameters>(
                    randomize_permutation,
                    randomize_shrinkage,
                    randomize_grit,
                    shrinkage_shape,
                    shrinkage_rate,
                    grit_shape1,
                    grit_shape2
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
        "fixed" => distr_macro!(FixedPartitionParameters),
        "up" => distr_macro!(UpParameters),
        "jlp" => distr_macro!(JlpParameters),
        "crp" => distr_macro!(CrpParameters),
        "lsp" => distr_macro!(LspParameters),
        "cpp-up" => distr_macro!(CppParameters<UpParameters>),
        "cpp-jlp" => distr_macro!(CppParameters<JlpParameters>),
        "cpp-crp" => distr_macro!(CppParameters<CrpParameters>),
        "sp-up" => distr_macro!(SpParameters<UpParameters>),
        "sp-jlp" => distr_macro!(SpParameters<JlpParameters>),
        "sp-crp" => distr_macro!(SpParameters<CrpParameters>),
        _ => stop!("Unsupported distribution: {}", name),
    }
    log_probs_rval
}

#[derive(Debug)]
struct GlobalMcmcTuning {
    n_iterations: usize,
    burnin: usize,
    thinning: usize,
    update_anchor: bool,
    idiosyncratic_permutation: bool,
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
            idiosyncratic_permutation: map.get("idiosyncratic_permutation")?.as_scalar()?.bool()?,
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
        let anchors_rval = RMatrix::<i32>::from_value(-1, n_items, n_saves, pc);
        let permutations_rval = RMatrix::<i32>::from_value(-1, n_items, n_saves, pc);
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
        permutation: Option<&Permutation>,
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
        if let Some(permutation) = permutation {
            let slice = &mut self.permutations[range];
            for (x, y) in slice.iter_mut().zip(permutation.as_slice()) {
                *x = *y as i32 + 1;
            }
        };
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
    anchor_anchor: &RVector,
    anchor_shrinkage: &RVector,
    anchor_shrinkage_reference: usize,
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
    let fastrand_ = fastrand::Rng::with_seed(rng.random());
    let fastrand = &mut Some(fastrand_);
    let anchor = all.units[rng.random_range(0..all.units.len())]
        .state
        .clustering()
        .clone();
    let shrinkage_prior_distribution = GammaPDF::new(
        hyperparameters.shrinkage.shape.get(),
        hyperparameters.shrinkage.rate.get(),
    )
    .unwrap();
    let shrinkage_rng = GammaRNG::new(
        hyperparameters.shrinkage.shape.get(),
        1.0 / hyperparameters.shrinkage.rate.get(),
    )
    .stop();
    let shrinkage = Shrinkage::constant(
        ScalarShrinkage::new(shrinkage_rng.sample(&mut rng)).stop(),
        all.n_items,
    );
    let grit_prior_distribution = BetaPDF::new(
        hyperparameters.grit.shape1.get(),
        hyperparameters.grit.shape2.get(),
    )
    .unwrap();
    let grit_rng = BetaRNG::new(
        hyperparameters.grit.shape1.get(),
        hyperparameters.grit.shape2.get(),
    )
    .stop();
    let grit = Grit::new(grit_rng.sample(&mut rng)).stop();
    let baseline_distribution = CrpParameters::new(
        all.n_items,
        Concentration::new(baseline_concentration).stop(),
    );
    let anchor_anchor = anchor_anchor.to_i32(pc);
    let anchor_shrinkage = Shrinkage::from(anchor_shrinkage.to_f64(pc).slice()).stop();
    let anchor_grit = Grit::new(grit_rng.sample(&mut rng)).stop();
    let mut anchor_distribution = SpParameters::new(
        Clustering::from_slice(anchor_anchor.slice()),
        anchor_shrinkage,
        Permutation::random(all.n_items, &mut rng),
        anchor_grit,
        CrpParameters::new(all.n_items, Concentration::new(anchor_concentration).stop()),
        true,
    )
    .unwrap();
    let anchor_update_permutation = Permutation::natural_and_fixed(all.n_items);
    let permutation = Permutation::random(all.n_items, &mut rng);
    let mut dists = std::iter::repeat_with(|| {
        SpParameters::new(
            anchor.clone(),
            shrinkage.clone(),
            permutation.clone(),
            grit,
            baseline_distribution.clone(),
            true,
        )
        .unwrap()
    })
    .take(n_units)
    .collect::<Vec<_>>();
    if !do_hierarchical_model {
        dists[0].anchor = anchor_distribution.anchor.clone();
        for time in 0..n_units - 1 {
            dists[time + 1].anchor = all.units[time].state.clustering.clone();
        }
    }
    fn update_permutation<'a, I>(
        idiosyncratic: bool,
        clusterings: I,
        partition_distributions: &mut [SpParameters<CrpParameters>],
        tuning: &McmcTuning,
        rng: &mut Pcg64Mcg,
    ) -> u64
    where
        I: IntoIterator<Item = &'a Clustering> + Clone,
    {
        let mut permutation_n_acceptances = 0;
        let k = tuning.n_items_per_permutation_update;
        let mut log_target_current: f64 = clusterings
            .clone()
            .into_iter()
            .zip(partition_distributions.iter_mut())
            .map(|(clustering, partition_distribution)| partition_distribution.log_pmf(clustering))
            .sum();
        if idiosyncratic {
            for _ in 0..tuning.n_permutation_updates_per_scan {
                for (clustering, partition_distribution) in clusterings
                    .clone()
                    .into_iter()
                    .zip(partition_distributions.iter_mut())
                {
                    partition_distribution.permutation.partial_shuffle(k, rng);
                    let log_target_proposal = partition_distribution.log_pmf(clustering);
                    let log_hastings_ratio = log_target_proposal - log_target_current;
                    if 0.0 <= log_hastings_ratio
                        || rng.random_range(0.0..1.0_f64).ln() < log_hastings_ratio
                    {
                        log_target_current = log_target_proposal;
                        permutation_n_acceptances += 1;
                    } else {
                        partition_distribution.permutation.partial_shuffle_undo(k);
                    }
                }
            }
        } else {
            let mut permutation = partition_distributions.first().unwrap().permutation.clone();
            for _ in 0..tuning.n_permutation_updates_per_scan {
                permutation.partial_shuffle(k, rng);
                partition_distributions
                    .iter_mut()
                    .for_each(|partition_distribution| {
                        partition_distribution.permutation = permutation.clone();
                    });
                let log_target_proposal = clusterings
                    .clone()
                    .into_iter()
                    .zip(partition_distributions.iter_mut())
                    .map(|(clustering, partition_distribution)| {
                        partition_distribution.log_pmf(clustering)
                    })
                    .sum();
                let log_hastings_ratio = log_target_proposal - log_target_current;
                if 0.0 <= log_hastings_ratio
                    || rng.random_range(0.0..1.0_f64).ln() < log_hastings_ratio
                {
                    log_target_current = log_target_proposal;
                    permutation_n_acceptances += 1;
                } else {
                    permutation.partial_shuffle_undo(k);
                    partition_distributions
                        .iter_mut()
                        .for_each(|partition_distribution| {
                            partition_distribution.permutation = permutation.clone();
                        });
                }
            }
        }
        permutation_n_acceptances
    }
    fn update_anchor_shrinkage(
        pd: &mut SpParameters<CrpParameters>,
        clustering: &Clustering,
        reference: usize,
        shrinkage_prior_distribution: &GammaPDF,
        tuning: &McmcTuning,
        rng: &mut Pcg64Mcg,
    ) {
        if let Some(w) = tuning.shrinkage_slice_step_size {
            let fastrand_ = fastrand::Rng::with_seed(rng.random());
            let fastrand = &mut Some(fastrand_);
            let tuning_parameters = stepping_out::TuningParameters::new().width(w);
            let (_s_new, _) = stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                pd.shrinkage[reference].get(),
                |s| match ScalarShrinkage::new(s) {
                    None => f64::NEG_INFINITY,
                    Some(shrinkage) => {
                        pd.shrinkage.rescale_by_reference(reference, shrinkage);
                        shrinkage_prior_distribution.ln_pdf(s) + pd.log_pmf(clustering)
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
        }
    }
    fn update_anchor_grit(
        pd: &mut SpParameters<CrpParameters>,
        clustering: &Clustering,
        grit_prior_distribution: &BetaPDF,
        tuning: &McmcTuning,
        rng: &mut Pcg64Mcg,
    ) {
        if let Some(w) = tuning.shrinkage_slice_step_size {
            let fastrand_ = fastrand::Rng::with_seed(rng.random());
            let fastrand = &mut Some(fastrand_);
            let tuning_parameters = stepping_out::TuningParameters::new().width(w);
            let (_g_new, _) = stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                pd.grit.get(),
                |g| match Grit::new(g) {
                    None => f64::NEG_INFINITY,
                    Some(grit) => {
                        pd.grit = grit;
                        grit_prior_distribution.ln_pdf(g) + pd.log_pmf(clustering)
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
        }
    }
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
                    .zip(dists.par_iter_mut())
                    .zip(rngs.par_iter_mut())
                    .for_each(|((unit, partition_distribution), (r1, r2))| {
                        unit.state.mcmc_iteration(
                            &unit_mcmc_tuning,
                            &mut unit.data,
                            &unit.hyperparameters,
                            partition_distribution,
                            r1,
                            r2,
                        );
                    });
            } else {
                for time in 0..n_units {
                    let (middle, right) = all.units.split_at_mut(time).1.split_at_mut(1);
                    let (middle_pd, right_pd) = dists.split_at_mut(time).1.split_at_mut(1);
                    let unit = middle.first_mut().unwrap();
                    let next_partition = right.first().map(|x| &x.state.clustering);
                    let pd = middle_pd.first().unwrap();
                    let next_distribution = right_pd.first_mut();
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
                            pd,
                            next_partition,
                            next_distribution,
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
            // Update permutation
            if unit_mcmc_tuning.n_permutation_updates_per_scan > 0 {
                results.timers.permutation.tic();
                let n_acceptances = update_permutation(
                    global_mcmc_tuning.idiosyncratic_permutation,
                    all.units.iter().map(|x| &x.state.clustering),
                    &mut dists,
                    &unit_mcmc_tuning,
                    &mut rng,
                );
                if iteration_counter >= global_mcmc_tuning.burnin {
                    permutation_n_acceptances += n_acceptances
                }
                results.timers.permutation.toc();
            }
            // Update anchor
            if do_hierarchical_model {
                results.timers.anchor.tic();
                let mut state = dists[0].anchor.clone();
                let mut compute_log_likelihood = |item: usize, anchor: &Clustering| {
                    all.units
                        .par_iter()
                        .zip(dists.par_iter_mut())
                        .fold_with(0.0, |acc, (x, pd)| {
                            pd.anchor = anchor.clone();
                            acc + pd.log_pmf_partial(item, &x.state.clustering)
                        })
                        .sum::<f64>()
                };
                if global_mcmc_tuning.update_anchor {
                    update_partition_gibbs(
                        1,
                        &mut state,
                        &anchor_update_permutation,
                        &anchor_distribution,
                        &mut compute_log_likelihood,
                        &mut rng,
                    )
                }
                dists.iter_mut().for_each(|pd| {
                    pd.anchor = state.clone();
                });
                results.timers.anchor.toc();
                results.timers.permutation.tic();
                if unit_mcmc_tuning.n_permutation_updates_per_scan > 0 {
                    let _ = update_permutation(
                        true,
                        std::slice::from_ref(&dists[0].anchor),
                        std::slice::from_mut(&mut anchor_distribution),
                        &unit_mcmc_tuning,
                        &mut rng,
                    );
                }
                results.timers.permutation.toc();
                results.timers.shrinkage.tic();
                update_anchor_shrinkage(
                    &mut anchor_distribution,
                    &dists[0].anchor,
                    anchor_shrinkage_reference,
                    &shrinkage_prior_distribution,
                    &unit_mcmc_tuning,
                    &mut rng,
                );
                results.timers.shrinkage.toc();
                results.timers.grit.tic();
                update_anchor_grit(
                    &mut anchor_distribution,
                    &dists[0].anchor,
                    &grit_prior_distribution,
                    &unit_mcmc_tuning,
                    &mut rng,
                );
                results.timers.grit.toc();
            }
            // Helper
            fn compute_log_likelihood(
                units: &[Group],
                dists: &[SpParameters<CrpParameters>],
            ) -> f64 {
                units
                    .par_iter()
                    .zip(dists.par_iter())
                    .fold_with(0.0, |acc, (x, pd)| acc + pd.log_pmf(&x.state.clustering))
                    .sum::<f64>()
            }
            // Update shrinkage
            results.timers.shrinkage.tic();
            if let Some(w) = unit_mcmc_tuning.shrinkage_slice_step_size {
                if let Some(reference) = hyperparameters.shrinkage.reference {
                    let tuning_parameters = stepping_out::TuningParameters::new().width(w);
                    let (_s_new, n_evaluations) =
                        stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                            dists[1].shrinkage[reference].get(),
                            |s| match ScalarShrinkage::new(s) {
                                None => f64::NEG_INFINITY,
                                Some(shrinkage) => {
                                    dists
                                        .iter_mut()
                                        .skip(if do_hierarchical_model { 0 } else { 1 })
                                        .for_each(|pd| {
                                            pd.shrinkage.rescale_by_reference(reference, shrinkage);
                                        });
                                    shrinkage_prior_distribution.ln_pdf(s)
                                        + compute_log_likelihood(
                                            &all.units
                                                [(if do_hierarchical_model { 0 } else { 1 })..],
                                            &dists[(if do_hierarchical_model { 0 } else { 1 })..],
                                        )
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
                if !do_hierarchical_model {
                    update_anchor_shrinkage(
                        &mut dists[0],
                        &all.units[0].state.clustering,
                        anchor_shrinkage_reference,
                        &shrinkage_prior_distribution,
                        &unit_mcmc_tuning,
                        &mut rng,
                    );
                }
            }
            results.timers.shrinkage.toc();
            // Update grit
            results.timers.grit.tic();
            if let Some(w) = unit_mcmc_tuning.grit_slice_step_size {
                let tuning_parameters = stepping_out::TuningParameters::new().width(w);
                let (_g_new, n_evaluations) =
                    stepping_out::univariate_slice_sampler_stepping_out_and_shrinkage(
                        dists[1].grit.get(),
                        |g| match Grit::new(g) {
                            None => f64::NEG_INFINITY,
                            Some(grit) => {
                                dists
                                    .iter_mut()
                                    .skip(if do_hierarchical_model { 0 } else { 1 })
                                    .for_each(|pd| {
                                        pd.grit = grit;
                                    });
                                grit_prior_distribution.ln_pdf(g)
                                    + compute_log_likelihood(
                                        &all.units[(if do_hierarchical_model { 0 } else { 1 })..],
                                        &dists[(if do_hierarchical_model { 0 } else { 1 })..],
                                    )
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
                if !do_hierarchical_model {
                    update_anchor_grit(
                        &mut dists[0],
                        &all.units[0].state.clustering,
                        &grit_prior_distribution,
                        &unit_mcmc_tuning,
                        &mut rng,
                    );
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
                    Some(&dists[1].anchor)
                } else {
                    None
                },
                if do_hierarchical_model {
                    Some(&dists[1].permutation)
                } else {
                    None
                },
                if let Some(reference) = hyperparameters.shrinkage.reference {
                    dists[1].shrinkage[reference].get()
                } else {
                    f64::NAN
                },
                dists[1].grit.get(),
                log_likelihood,
            );
        }
    }
    let denominator = (global_mcmc_tuning.n_saves() * global_mcmc_tuning.thinning) as f64;
    results.finalize(
        permutation_n_acceptances as f64
            / (if global_mcmc_tuning.idiosyncratic_permutation {
                n_units as f64
            } else {
                1.0
            } * unit_mcmc_tuning.n_permutation_updates_per_scan as f64
                * denominator),
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
