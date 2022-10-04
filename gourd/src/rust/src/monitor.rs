use roxido::*;

#[derive(Debug)]
pub struct Monitor {
    pub permutation_acceptance_counter: u32,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            permutation_acceptance_counter: 0,
        }
    }

    pub fn to_r(&self, pc: &mut Pc) -> Rval {
        let result = Rval::new_list(5, pc);
        result.set_list_element(
            0,
            Rval::new(
                i32::try_from(self.permutation_acceptance_counter).unwrap(),
                pc,
            ),
        );
        result
    }

    pub fn reset(&mut self) {
        self.permutation_acceptance_counter = 0;
    }
}
