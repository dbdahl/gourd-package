#[derive(Debug)]
pub struct MembershipGenerator {
    sizes: Vec<usize>,
    cum_sizes: Vec<usize>,
}

impl MembershipGenerator {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut cum_sizes = vec![0; sizes.len() + 1];
        let mut acc = 0;
        for (x, y) in cum_sizes.iter_mut().skip(1).zip(sizes.iter()) {
            acc += *y;
            *x = acc;
        }
        Self { sizes, cum_sizes }
    }

    #[allow(dead_code)]
    pub fn n_items(&self) -> usize {
        self.cum_sizes.len() - 1
    }

    pub fn n_observations(&self) -> usize {
        *self.cum_sizes.last().unwrap()
    }

    pub fn size_of_item(&self, item: usize) -> usize {
        self.sizes[item]
    }

    pub fn indices_of_item(&self, item: usize) -> Vec<usize> {
        (self.cum_sizes[item]..self.cum_sizes[item + 1]).collect()
    }

    pub fn indices_of_items<'a, T>(&self, items: T) -> Vec<usize>
    where
        T: IntoIterator<Item = &'a usize>,
    {
        let mut x = Vec::new();
        for &index in items {
            x.extend(self.cum_sizes[index]..self.cum_sizes[index + 1]);
        }
        x
    }
}

mod tests {

    #[test]
    fn test_generator() {
        let sizes = vec![6, 1, 3, 2, 4];
        let generator = super::MembershipGenerator::new(sizes);
        let guess = generator.indices_of_items(&[2, 1, 4, 0][..]);
        let truth = vec![7, 8, 9, 6, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5];
        assert_eq!(guess, truth);
    }
}
