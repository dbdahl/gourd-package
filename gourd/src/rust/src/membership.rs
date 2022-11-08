#[derive(Debug)]
pub struct MembershipGenerator {
    cum_sizes: Vec<usize>,
}

impl MembershipGenerator {
    pub fn new(sizes: &[usize]) -> Self {
        let mut cum_sizes = vec![0; sizes.len() + 1];
        let mut acc = 0;
        for (x, y) in cum_sizes.iter_mut().skip(1).zip(sizes) {
            acc += *y;
            *x = acc;
        }
        Self { cum_sizes }
    }

    pub fn n_items(&self) -> usize {
        *self.cum_sizes.last().unwrap()
    }

    pub fn get(&self, index: usize) -> Vec<usize> {
        (self.cum_sizes[index]..self.cum_sizes[index + 1]).collect()
    }

    pub fn generate<'a, T>(&self, indices: T) -> Vec<usize>
    where
        T: IntoIterator<Item = &'a usize>,
    {
        let mut x = Vec::new();
        for &i in indices {
            x.extend(self.cum_sizes[i]..self.cum_sizes[i + 1]);
        }
        x
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_generator() {
        let sizes = vec![6, 1, 3, 2, 4];
        let generator = MembershipGenerator::new(&sizes[..]);
        let guess = generator.generate(&[2, 1, 4, 0][..]);
        let truth = vec![7, 8, 9, 6, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5];
        assert_eq!(guess, truth);
    }
}
