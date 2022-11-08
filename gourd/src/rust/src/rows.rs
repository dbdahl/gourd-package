use std::ops::Range;
use std::slice::Iter;

struct RowsIteratorGenerator {
    cum_sizes: Vec<usize>,
}

impl RowsIteratorGenerator {
    pub fn new(sizes: &[usize]) -> Self {
        let mut cum_sizes = vec![0; sizes.len() + 1];
        let mut acc = 0;
        for (x, y) in cum_sizes.iter_mut().skip(1).zip(sizes) {
            acc += *y;
            *x = acc;
        }
        Self { cum_sizes }
    }
    pub fn iter<'a>(&'a self, indices: Iter<'a, usize>) -> RowsIterator {
        RowsIterator::new(indices, &self.cum_sizes[..])
    }
}

struct RowsIterator<'a> {
    cum_sizes: &'a [usize],
    outer: Iter<'a, usize>,
    inner: Range<usize>,
}

impl<'a> RowsIterator<'a> {
    pub fn new(indices: Iter<'a, usize>, cum_sizes: &'a [usize]) -> Self {
        let outer = indices.into_iter();
        Self {
            cum_sizes,
            outer,
            inner: 0..0_usize,
        }
    }
}

impl<'a> Iterator for RowsIterator<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.inner.next() {
            Some(index) => Some(index),
            None => match self.outer.next() {
                Some(&i) => {
                    self.inner = self.cum_sizes[i]..self.cum_sizes[i + 1];
                    self.next()
                }
                None => None,
            },
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_generator() {
        let sizes = vec![6, 1, 3, 2, 4];
        let generator = RowsIteratorGenerator::new(&sizes[..]);
        let iter = generator.iter([2, 1, 4, 0].iter());
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![7, 8, 9, 6, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_general() {
        let indices = vec![2, 1, 4, 0];
        let cum_sizes = vec![0, 6, 7, 10, 12, 16];
        let iter = RowsIterator::new(indices[..].iter(), &cum_sizes);
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![7, 8, 9, 6, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_empty() {
        let indices = vec![0; 0];
        let cum_sizes = vec![0, 6, 7, 10, 12, 16];
        let iter = RowsIterator::new(indices[..].iter(), &cum_sizes);
        assert_eq!(iter.collect::<Vec<_>>(), vec![]);
    }
}
