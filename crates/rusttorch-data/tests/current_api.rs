use std::convert::Infallible;

use rusttorch_data::{DataLoader, Dataset, SequentialSampler};

struct Rows([usize; 3]);

impl Dataset for Rows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Result<usize, Infallible> {
        Ok(self.0[index])
    }
}

#[test]
fn direct_package_matches_the_existing_facade_loader() {
    let rows = Rows([2, 3, 5]);
    let batches = DataLoader::new(&rows, SequentialSampler::new(3), 2, false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches, vec![vec![2, 3], vec![5]]);
}
