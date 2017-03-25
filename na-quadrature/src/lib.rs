
//! TODO: this crate allocates/deallocates often.

#![feature(loop_break_value)]

extern crate alga;
extern crate num_traits;
extern crate ndarray as nd;
extern crate linxal;
extern crate rand;

pub mod rke;

/*pub fn lorenz95(_: f64, y: ArrayView<f64, Ix1>, mut yp: ArrayViewMut<f64, Ix1>) {
  debug_assert!(y.len() == 40);
  debug_assert!(yp.len() == 40);

  for i in 2..39 {
    yp[i] = -y[i] + y[i - 1] * (y[i + 1] - y[i - 2]) + 8;
  }

  y[0] = -y[0] + y[39] * (y[1] - y[38]) + 8;
  y[1] = -y[1] + y[0] * (y[2] - y[39]) + 8;
  y[39] = -y[39] + x[38] * (y[0] - y[37]) + 8;
}*/