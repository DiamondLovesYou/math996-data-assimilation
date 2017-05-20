
extern crate ndarray as nd;
extern crate na_discrete_filtering as na_df;

use nd::{ArrayView, ArrayViewMut, Ix1};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct L95Model {
  pub forcing: f64,
}
impl na_df::Model<f64> for L95Model {
  fn workspace_size(&self) -> usize { 0 }
  fn run_model(&self, _step: u64,
               _: ArrayViewMut<f64, Ix1>,
               x: ArrayView<f64, Ix1>,
               mut out: ArrayViewMut<f64, Ix1>) {
    let n = x.dim() as isize;

    let i = (1..n+1).zip(n-2..n-2+n).zip(n-1..n-1+n);
    for (e, ((i1, i2), i3)) in i.enumerate() {
      out[e] = (x[(i1 % n) as usize] - x[(i2 % n) as usize]) * x[(i3 % n) as usize] - x[e] + self.forcing;
    }
  }
}
