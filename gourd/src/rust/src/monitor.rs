use num_traits::cast::AsPrimitive;
use num_traits::int::PrimInt;
use num_traits::Zero;
use rand_distr::num_traits;
use std::ops::AddAssign;

#[derive(Debug)]
pub struct Monitor<A> {
    acceptance_counter: A,
    attempts_counter: A,
}

impl<A: PrimInt + AddAssign + AsPrimitive<f64>> Monitor<A> {
    pub fn new() -> Self {
        Self {
            acceptance_counter: Zero::zero(),
            attempts_counter: Zero::zero(),
        }
    }

    pub fn monitor<T: FnMut(A) -> A>(&mut self, n_attempts: A, mut f: T) {
        self.acceptance_counter += f(n_attempts);
        self.attempts_counter += n_attempts;
    }

    pub fn rate(&self) -> f64 {
        let num: f64 = self.acceptance_counter.as_();
        let den: f64 = self.attempts_counter.as_();
        num / den
    }

    pub fn reset(&mut self) {
        self.acceptance_counter = Zero::zero();
        self.attempts_counter = Zero::zero();
    }
}
