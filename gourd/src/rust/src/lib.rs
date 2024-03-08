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
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use roxido::*;
use slice_sampler::univariate::stepping_out;
use statrs::distribution::{Beta, Continuous, Gamma};
use std::ptr::NonNull;
use walltime::TicToc;

#[roxido]
fn new_UpParameters(n_items: RObject) -> RObject {
    let p = UpParameters::new(n_items.usize().stop());
    R::encode(p, &"up".to_r(pc), true, pc)
}

#[roxido]
fn new_JlpParameters(concentration: RObject, permutation: RObject) -> RObject {
    let permutation = mk_permutation(permutation, pc);
    let concentration = Concentration::new(concentration.f64().stop())
        .unwrap_or_else(|| stop!("Invalid concentration value"));
    let p = JlpParameters::new(permutation.n_items(), concentration, permutation).unwrap();
    R::encode(p, &"jlp".to_r(pc), true, pc)
}

#[roxido]
fn new_CrpParameters(n_items: RObject, concentration: RObject, discount: RObject) -> RObject {
    let n_items = n_items.usize().stop();
    let discount =
        Discount::new(discount.f64().stop()).unwrap_or_else(|| stop!("Invalid discount value"));
    let concentration = Concentration::new_with_discount(concentration.f64().stop(), discount)
        .unwrap_or_else(|| stop!("Invalid concentration value"));
    let p = CrpParameters::new_with_discount(n_items, concentration, discount).unwrap();
    R::encode(p, &"crp".to_r(pc), true, pc)
}

#[roxido]
fn new_SpParameters(
    anchor: RObject,
    shrinkage: RObject,
    permutation: RObject,
    grit: RObject,
    baseline: RObject,
) -> RObject {
    let anchor = mk_clustering(anchor, pc);
    let shrinkage = mk_shrinkage(shrinkage, pc);
    let permutation = mk_permutation(permutation, pc);
    let grit = mk_grit(grit);
    let baseline = baseline.external_ptr().stop();
    macro_rules! distr_macro {
        ($tipe:ty, $label:literal) => {{
            let p = NonNull::new(baseline.address() as *mut $tipe).unwrap();
            let baseline = unsafe { p.as_ref().clone() };
            R::encode(
                SpParameters::new(anchor, shrinkage, permutation, grit, baseline).unwrap(),
                &$label.to_r(pc),
                true,
                pc,
            )
        }};
    }
    let tag = baseline.tag();
    let name = tag.to_str(pc).stop();
    match name {
        "up" => distr_macro!(UpParameters, "sp-up"),
        "jlp" => distr_macro!(JlpParameters, "sp-jlp"),
        "crp" => distr_macro!(CrpParameters, "sp-crp"),
        _ => stop!("Unsupported distribution: {}", name),
    }
}

fn mk_clustering(partition: RObject, pc: &mut Pc) -> Clustering {
    let partition = partition.vector().stop().to_integer(pc);
    Clustering::from_slice(partition.slice())
}

fn mk_shrinkage(shrinkage: RObject, pc: &mut Pc) -> Shrinkage {
    let shrinkage = shrinkage.vector().stop().to_double(pc);
    Shrinkage::from(shrinkage.slice()).unwrap()
}

fn mk_permutation(permutation: RObject, pc: &mut Pc) -> Permutation {
    let permutation = permutation.vector().stop().to_integer(pc);
    let slice = permutation.slice();
    let vector = slice.iter().map(|x| *x as usize).collect();
    Permutation::from_vector(vector).unwrap()
}

fn mk_grit(grit: RObject) -> Grit {
    let b = grit.f64().stop();
    match Grit::new(b) {
        Some(grit) => grit,
        None => stop!("Out of range."),
    }
}

#[derive(PartialEq, Copy, Clone)]
enum RandomizeShrinkage {
    Fixed,
    Common,
    Cluster,
    Idiosyncratic,
}

#[roxido]
fn prPartition(partition: RObject, prior: RObject) -> RObject {
    let partition = partition
        .matrix()
        .stop_str("'partition' should be a matrix.")
        .to_integer(pc);
    let matrix = partition.slice();
    let np = partition.nrow();
    let ni = partition.ncol();
    let mut log_probs_rval = R::new_vector_double(np, pc);
    let log_probs_slice = log_probs_rval.slice_mut();
    let prior = prior.external_ptr().stop();
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
    let tag = prior.tag();
    let name = tag.to_str(pc).stop();
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
fn samplePartition(
    n_samples: RObject,
    n_items: RObject,
    prior: RObject,
    randomize_permutation: RObject,
    randomize_shrinkage: RObject,
    max: RObject,
    shape1: RObject,
    shape2: RObject,
    n_cores: RObject,
) -> RObject {
    let n_cores = {
        let x = n_cores
            .usize()
            .stop_str("'nCores' should be a scalar integer");
        if x == 0 {
            num_cpus::get()
        } else {
            x
        }
    };
    let np = n_samples
        .usize()
        .map(|x| x.max(1))
        .stop_str("'nSamples' should be a scalar integer");
    let np_per_core = np / n_cores;
    let np_extra = np % n_cores;
    let ni = n_items
        .usize()
        .map(|x| x.max(1))
        .stop_str("'nItems' should be a scalar integer");
    let chunk_size = np_per_core * ni;
    let mut matrix_rval = R::new_matrix_integer(ni, np, pc);
    let mut stick = matrix_rval.slice_mut();
    let randomize_permutation = randomize_permutation
        .bool()
        .stop_str("'randomizePermutation' should be a scalar logical");
    let randomize_shrinkage = match randomize_shrinkage.to_str(pc) {
        Ok("fixed") => RandomizeShrinkage::Fixed,
        Ok("common") => RandomizeShrinkage::Common,
        Ok("cluster") => RandomizeShrinkage::Cluster,
        Ok("idiosyncratic") => RandomizeShrinkage::Idiosyncratic,
        _ => stop!("Unrecognized randomize_shrinkage value"),
    };
    let max = ScalarShrinkage::new(max.f64().stop_str("'max' should be a scalar double"))
        .unwrap_or_else(|| stop!("'max' should be non-negative"));
    let shape1 = Shape::new(shape1.f64().stop_str("'shape1' should be a scalar double"))
        .unwrap_or_else(|| stop!("'shape1' should be greater than 0"));
    let shape2 = Shape::new(shape2.f64().stop_str("'shape2' should be a scalar double"))
        .unwrap_or_else(|| stop!("'shape2' should be greater than 0"));
    let prior = prior.external_ptr().stop();
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
    let tag = prior.tag();
    let name = tag.to_str(pc).stop();
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
fn rngs_new() -> RObject {
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let mut seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    rng.fill(&mut seed);
    let rng2 = Pcg64Mcg::from_seed(seed);
    let mut result = R::new_list(2, pc);
    result
        .set(0, &R::encode(rng, &"rng".to_r(pc), true, pc))
        .stop();
    result
        .set(1, &R::encode(rng2, &"rng".to_r(pc), true, pc))
        .stop();
    result
}

#[roxido]
fn state_encode(state: RObject) -> RObject {
    let state = State::from_r(state, pc).stop();
    R::encode(state, &"state".to_r(pc), true, pc)
}

#[roxido]
fn state_decode(state: RObject) -> RObject {
    state.external_ptr().stop().decode_ref::<State>().to_r(pc)
}

#[roxido]
fn monitor_new() -> RObject {
    R::encode(Monitor::<u32>::new(), &"monitor".to_r(pc), true, pc)
}

#[roxido]
fn monitor_rate(monitor: RObject) -> RObject {
    monitor
        .external_ptr()
        .stop()
        .decode_ref::<Monitor<u32>>()
        .rate()
}

#[roxido]
fn monitor_reset(monitor: RObject) -> RObject {
    let mut monitor = monitor.external_ptr().stop();
    monitor.decode_mut::<Monitor<u32>>().reset();
}

#[roxido]
fn hyperparameters_encode(hyperparameters: RObject) -> RObject {
    let hp = Hyperparameters::from_r(hyperparameters, pc).stop();
    R::encode(hp, &"hyperparameters".to_r(pc), true, pc)
}

#[roxido]
fn data_encode(data: RObject, missing_items: RObject) -> RObject {
    let mut data = Data::from_r(data, pc).stop();
    let missing_items = missing_items.vector().stop().to_integer(pc);
    let missing_items: Vec<_> = missing_items
        .slice()
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    data.declare_missing(missing_items);
    R::encode(data, &"data".to_r(pc), true, pc)
}

fn permutation_to_r(permutation: &Permutation, rval: &mut RObject<roxido::r::Vector, i32>) {
    for (x, y) in permutation.as_slice().iter().zip(rval.slice_mut()) {
        *y = i32::try_from(*x).unwrap() + 1;
    }
}

fn scalar_shrinkage_to_r(shrinkage: &ScalarShrinkage, rval: &mut RObject<roxido::r::Vector, f64>) {
    rval.slice_mut()[0] = shrinkage.get();
}

fn shrinkage_to_r(shrinkage: &Shrinkage, rval: &mut RObject<roxido::r::Vector, f64>) {
    for (x, y) in rval.slice_mut().iter_mut().zip(shrinkage.as_slice()) {
        *x = y.get()
    }
}

fn grit_to_r(grit: &Grit, rval: &mut RObject<roxido::r::Vector, f64>) {
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
fn all(all: RObject) -> RObject {
    let all = all.list().stop();
    let mut vec = Vec::with_capacity(all.len());
    let mut n_items = None;
    for i in 0..all.len() {
        let list = all.get(i).stop().list().stop();
        let (data, state, hyperparameters) =
            (list.get(0).stop(), list.get(1).stop(), list.get(2).stop());
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
    R::encode(all, &"all".to_r(pc), true, pc)
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

impl<RType, RMode> FromR<RType, RMode, String> for GlobalMcmcTuning {
    fn from_r(x: RObject<RType, RMode>, _pc: &mut Pc) -> Result<Self, String> {
        let x = x.list().map_err_msg()?;
        let mut map = x.make_map();
        let result = Self {
            n_iterations: map.get("n_iterations")?.usize()?,
            burnin: map.get("burnin")?.usize()?,
            thinning: map.get("thinning")?.usize()?,
            update_anchor: map.get("update_anchor")?.bool()?,
        };
        map.exhaustive()?;
        Ok(result)
    }
}

fn validation_data_from_r(
    validation_data: RObject,
    pc: &mut Pc,
) -> Result<Option<Vec<Data>>, String> {
    match validation_data.option() {
        Some(x) => {
            let x = x.list().stop();
            let n_units = x.len();
            let mut vd = Vec::with_capacity(n_units);
            for k in 0..n_units {
                vd.push(Data::from_r(x.get(k).stop(), pc)?);
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
    rval: RObject<roxido::r::Vector, roxido::r::List>,
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
    pub fn new(tuning: &GlobalMcmcTuning, n_items: usize, n_units: usize, pc: &mut Pc) -> Self {
        let n_saves = tuning.n_saves();
        let mut unit_partitions_rval = R::new_list(n_units, pc);
        let mut unit_partitions = Vec::with_capacity(n_units);
        for t in 0..n_units {
            let rval = R::new_matrix_integer(n_items, n_saves, pc);
            unit_partitions_rval.set(t, &rval).stop();
            unit_partitions.push(unsafe { rval.slice_mut_static() });
        }
        let anchors_rval = R::new_matrix_integer(n_items, n_saves, pc);
        let permutations_rval = R::new_matrix_integer(n_items, n_saves, pc);
        let shrinkages_rval = R::new_vector_double(n_saves, pc);
        let grits_rval = R::new_vector_double(n_saves, pc);
        let log_likelihoods_rval = R::new_vector_double(n_saves, pc);
        let mut rval = R::new_list(10, pc); // Extra 4 items for rates after looping.
        rval.set_names(
            &[
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
            ]
            .to_r(pc),
        )
        .stop();
        rval.set(0, &unit_partitions_rval).stop();
        rval.set(1, &anchors_rval).stop();
        rval.set(2, &permutations_rval).stop();
        rval.set(3, &shrinkages_rval).stop();
        rval.set(4, &grits_rval).stop();
        rval.set(5, &log_likelihoods_rval).stop();
        unsafe {
            Self {
                rval,
                counter: 0,
                n_items,
                unit_partitions,
                anchors: anchors_rval.slice_mut_static(),
                permutations: permutations_rval.slice_mut_static(),
                shrinkages: shrinkages_rval.slice_mut_static(),
                grits: grits_rval.slice_mut_static(),
                log_likelihoods: log_likelihoods_rval.slice_mut_static(),
                timers: Timers {
                    units: TicToc::new(),
                    anchor: TicToc::new(),
                    permutation: TicToc::new(),
                    shrinkage: TicToc::new(),
                    grit: TicToc::new(),
                },
            }
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
        pc: &mut Pc,
    ) {
        self.rval.set(6, &permutation_rate.to_r(pc)).stop();
        self.rval
            .set(7, &shrinkage_slice_evaluations_rate.to_r(pc))
            .stop();
        self.rval
            .set(8, &grit_slice_evaluations_rate.to_r(pc))
            .stop();
        let mut wall_times = [
            self.timers.units.as_secs_f64(),
            self.timers.anchor.as_secs_f64(),
            self.timers.permutation.as_secs_f64(),
            self.timers.shrinkage.as_secs_f64(),
            self.timers.grit.as_secs_f64(),
        ]
        .to_r(pc);
        wall_times
            .set_names(&["units", "anchor", "permutation", "shrinkage", "grit"].to_r(pc))
            .stop();
        self.rval.set(9, &wall_times).stop();
    }
}

#[roxido]
fn fit_dependent(
    model_id: RObject,
    all_ptr: RObject,
    anchor_concentration: RObject,
    baseline_concentration: RObject,
    hyperparameters: RObject,
    unit_mcmc_tuning: RObject,
    global_mcmc_tuning: RObject,
    validation_data: RObject,
) -> RObject {
    let do_hierarchical_model = match model_id.to_str(pc).stop() {
        "hierarchical" => true,
        "temporal" => false,
        model_id => stop!("Unsupported model {}", model_id),
    };
    let mut all = all_ptr.external_ptr().stop();
    let all: &mut All = all.decode_mut();
    let validation_data = validation_data_from_r(validation_data, pc).stop();
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
        Concentration::new(baseline_concentration.f64().stop()).stop(),
    );
    let anchor_distribution = CrpParameters::new(
        all.n_items,
        Concentration::new(anchor_concentration.f64().stop()).stop(),
    );
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
                    let shrinkage_prior_distribution = Gamma::new(
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
                let grit_prior_distribution = Beta::new(
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
    n_updates: RObject,
    data: RObject,
    state: RObject,
    hyperparameters: RObject,
    monitor: RObject,
    partition_distribution: RObject,
    mcmc_tuning: RObject,
    permutation_bucket: RObject, // This is used to communicating temporary results back to R
    shrinkage_bucket: RObject,   // This is used to communicating temporary results back to R
    grit_bucket: RObject,        // This is used to communicating temporary results back to R
    rngs: RObject,
) -> RObject {
    let n_updates: u32 = n_updates.i32().stop().try_into().unwrap();
    let mut data = data.external_ptr().stop();
    let data: &mut Data = data.decode_mut();
    let mut state = state.external_ptr().stop();
    let state: &mut State = state.decode_mut();
    let hyperparameters = hyperparameters.external_ptr().stop();
    let hyperparameters: &Hyperparameters = hyperparameters.decode_ref();
    let mut monitor = monitor.external_ptr().stop();
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
    let mut permutation_bucket = permutation_bucket.vector().stop().integer().stop();
    let mut shrinkage_bucket = shrinkage_bucket.vector().stop().double().stop();
    let mut grit_bucket = grit_bucket.vector().stop().double().stop();
    let rngs = rngs.list().stop();
    let mut rng = rngs.get(0).stop().external_ptr().stop();
    let rng = rng.decode_mut::<Pcg64Mcg>();
    let mut rng2 = rngs.get(1).stop().external_ptr().stop();
    let rng2 = rng2.decode_mut::<Pcg64Mcg>();
    let mut partition_distribution = partition_distribution.external_ptr().stop();
    let tag = partition_distribution.tag();
    let prior_name = tag.to_str(pc).stop();
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
                permutation_to_r(&partition_distribution.permutation, &mut permutation_bucket);
            }
        }};
        ($tipe:ty, true, true, false, false) => {{
            let partition_distribution = partition_distribution.decode_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(u32::try_from(mcmc_tuning.n_permutation_updates_per_scan).unwrap(), |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, u32::try_from(mcmc_tuning.n_items_per_permutation_update).unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, &mut permutation_bucket);
                if let Some(w) = mcmc_tuning.shrinkage_slice_step_size {
                    dahl_randompartition::mcmc::update_scalar_shrinkage(1, partition_distribution, w, hyperparameters.shrinkage.shape, hyperparameters.shrinkage.rate, &state.clustering, rng);
                    scalar_shrinkage_to_r(&partition_distribution.shrinkage, &mut shrinkage_bucket);
                }
            }
        }};
        ($tipe:ty, true, false, true, true) => {{
            let partition_distribution = partition_distribution.decode_mut::<$tipe>();
            for _ in 0..n_updates {
                state.mcmc_iteration(&mcmc_tuning, data, hyperparameters, partition_distribution, rng, rng2);
                monitor.monitor(u32::try_from(mcmc_tuning.n_permutation_updates_per_scan).unwrap(), |n_updates| { dahl_randompartition::mcmc::update_permutation(n_updates, partition_distribution, u32::try_from(mcmc_tuning.n_items_per_permutation_update).unwrap(), &state.clustering, rng) });
                permutation_to_r(&partition_distribution.permutation, &mut permutation_bucket);
                if let Some(w) = mcmc_tuning.shrinkage_slice_step_size {
                    if let Some(reference) = &hyperparameters.shrinkage.reference {
                        dahl_randompartition::mcmc::update_shrinkage(1, partition_distribution, *reference, w, hyperparameters.shrinkage.shape, hyperparameters.shrinkage.rate, &state.clustering, rng);
                        shrinkage_to_r(&partition_distribution.shrinkage, &mut shrinkage_bucket);
                    }
                }
                if let Some(w) = mcmc_tuning.grit_slice_step_size {
                    dahl_randompartition::mcmc::update_grit(1, partition_distribution, w, hyperparameters.grit.shape1, hyperparameters.grit.shape2, &state.clustering, rng);
                    grit_to_r(&partition_distribution.grit, &mut grit_bucket);
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
fn log_likelihood_contributions(state: RObject, data: RObject) -> RObject {
    let state = state.external_ptr().stop();
    let state: &State = state.decode_ref();
    let data = data.external_ptr().stop();
    let data = data.decode_ref();
    state.log_likelihood_contributions(data).iter().to_r(pc)
}

#[roxido]
fn log_likelihood_contributions_of_missing(state: RObject, data: RObject) -> RObject {
    let state = state.external_ptr().stop();
    let state: &State = state.decode_ref();
    let data = data.external_ptr().stop();
    let data = data.decode_ref();
    state
        .log_likelihood_contributions_of_missing(data)
        .iter()
        .to_r(pc)
}

#[roxido]
fn log_likelihood_of(state: RObject, data: RObject, items: RObject) -> RObject {
    let state = state.external_ptr().stop();
    let state: &State = state.decode_ref();
    let data = data.external_ptr().stop();
    let data = data.decode_ref();
    let items: Vec<_> = items
        .vector()
        .stop()
        .to_integer(pc)
        .slice()
        .iter()
        .map(|x| usize::try_from(*x - 1).unwrap())
        .collect();
    state.log_likelihood_of(data, items)
}

#[roxido]
fn log_likelihood(state: RObject, data: RObject) -> RObject {
    let state = state.external_ptr().stop();
    let state: &State = state.decode_ref();
    let data = data.external_ptr().stop();
    let data = data.decode_ref();
    state.log_likelihood(data)
}

#[roxido]
fn sample_multivariate_normal(n_samples: RObject, mean: RObject, precision: RObject) -> RObject {
    let n_samples = n_samples.usize().stop();
    let mean = mean.vector().stop().to_double(pc);
    let mean = mean.slice();
    let n = mean.len();
    let mean = DVector::from_iterator(n, mean.iter().cloned());
    let precision = precision.matrix().stop().to_double(pc);
    let precision = DMatrix::from_iterator(n, n, precision.slice().iter().cloned());
    let mut rval = R::new_matrix_double(n, n_samples, pc);
    let slice = rval.slice_mut();
    let mut rng = Pcg64Mcg::from_seed(R::random_bytes::<16>());
    let x = sample_multivariate_normal_repeatedly(n_samples, mean, precision, &mut rng).unwrap();
    slice.clone_from_slice(x.as_slice());
    rval.transpose(pc)
}

fn validate_list<RType, RMode>(
    x: RObject<RType, RMode>,
    expected_names: &[&str],
    arg_name: &str,
) -> RObject<roxido::r::Vector, roxido::r::List> {
    let list = x
        .list()
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
