use std::ops::Range;
use std::slice::Iter;

#[derive(Debug)]
pub struct MembershipGenerator {
    indices: Vec<usize>,
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
        Self {
            indices: (0..acc).collect(),
            sizes,
            cum_sizes,
        }
    }

    #[allow(dead_code)]
    pub fn n_items(&self) -> usize {
        self.cum_sizes.len() - 1
    }

    pub fn n_observations(&self) -> usize {
        *self.cum_sizes.last().unwrap()
    }

    #[allow(dead_code)]
    pub fn size_of_item(&self, item: usize) -> usize {
        self.sizes[item]
    }

    pub fn indices_of_item(&self, item: usize) -> &[usize] {
        &self.indices[self.cum_sizes[item]..self.cum_sizes[item + 1]]
    }

    pub fn indices_of_items<'a, 'b>(&'a self, items: Iter<'b, usize>) -> IndicesIterator<'a, 'b> {
        IndicesIterator::new(items, self)
    }
}

#[derive(Clone)]
pub struct IndicesIterator<'a, 'b> {
    items: Iter<'b, usize>,
    inner: Range<usize>,
    membership_generator: &'a MembershipGenerator,
}

impl<'a, 'b> IndicesIterator<'a, 'b> {
    fn new(items: Iter<'b, usize>, membership_generator: &'a MembershipGenerator) -> Self {
        let inner = 0_usize..0;
        Self {
            items,
            inner,
            membership_generator,
        }
    }
}

impl<'a, 'b> Iterator for IndicesIterator<'a, 'b> {
    type Item = &'a usize;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.inner.next() {
            Some(i) => Some(&self.membership_generator.indices[i]),
            None => match self.items.next() {
                Some(j) => {
                    self.inner = self.membership_generator.cum_sizes[*j]
                        ..self.membership_generator.cum_sizes[*j + 1];
                    self.next()
                }
                None => None,
            },
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len: usize = self
            .items
            .clone()
            .map(|x| self.membership_generator.sizes[*x])
            .sum();
        (len, Some(len))
    }
}

impl<'a, 'b> ExactSizeIterator for IndicesIterator<'a, 'b> {}

mod tests {

    #[test]
    fn test_generator() {
        let sizes = vec![6, 1, 3, 2, 4];
        let generator = super::MembershipGenerator::new(sizes);
        let guess = generator.indices_of_items([2, 1, 4, 0].iter());
        let truth = vec![7, 8, 9, 6, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5];
        assert_eq!(guess.copied().collect::<Vec<_>>(), truth);
    }
}
