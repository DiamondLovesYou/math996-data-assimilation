
#[macro_use]
extern crate ndarray as nd;

use nd::{Array, ArrayView, ArrayViewMut, Ix1};
use nd::linalg::general_mat_mul;

fn lorenz95(_: f64, x: ArrayView<f64, Ix1>,
            mut out: ArrayViewMut<f64, Ix1>) {
  const F: f64 = 8.0;

  let n = x.dim() as isize;

  let i = (1..n+1).zip(n-2..n-2+n).zip(n-1..n-1+n);
  for (e, ((i1, i2), i3)) in i.enumerate() {
    out[e] = (x[i1 % n] - x[i2 % n]) * x[i3 % n] - x[e] + F;
  }
}
fn main() {
}
