
#![feature(loop_break_value)]

extern crate alga;
extern crate num_traits;
#[macro_use]
extern crate ndarray as nd;
extern crate linxal;
extern crate rand;

use nd::{ArrayBase, Data, DataOwned, Ix1, Ix2, Ix3, arr2};
use alga::general::{Real, Identity, Multiplicative, Additive};

use num_traits::*;
use linxal::prelude::*;
use linxal::types::LinxalScalar;
use linxal::factorization::cholesky::*;
use std::ops::Range;

pub mod rke;

pub fn model_truth<F, E>(f: F, n: usize,
                     x: E,
                     y: Array<E, Ix1>,
                     yp: Array<E, Ix1>,
                     tol: E, h: E,
                     thresh: Array<E, Ix1>,) -> Array<E, Ix2>
  where F: Fn(E, ArrayView<E, Ix1>, ArrayViewMut<E, Ix1>),
        E: Float,
{
  let yp_size = yp.len();

  let dest: Array<E, Ix2> = ArrayBase::zeros((n, yp_size + 1));

  let mut state = rke::new(space[i], y, yp, h, tol, thresh, true);
  let dest = state
    .iter(f)
    .take(n)
    .fold(dest, |mut dest, (state, result)| {
      let ycoeff = result
        .expect("integration failed")
        .unwrap();

      {
        let mut z = dest.column_mut(idx);
        z[0] = state.x;

        nquad::rke::y_value(state.x - state.previous_h,
                            state.x, state.previous_h,
                            &ycoeff,
                            z.slice_mut(s![1..]).view_mut());
      }

      dest
    })
    .unwrap();

  dest
}

pub fn lorenz63(_: f64, y: ArrayView<f64, Ix1>, mut yp: ArrayViewMut<f64, Ix1>) {
  debug_assert!(y.len() == 3);
  debug_assert!(yp.len() == 3);

  const RHO: f64 = 28.0;
  const SIGMA: f64 = 10.0;
  const BETA: f64 = 8.0 / 3.0;

  yp[0] = sigma * (y[1] - y[0]);
  yp[1] = y[0] * (RHO - y[2]) - y[1];
  yp[2] = y[0] * y[1] - BETA * y[2];
}
pub fn lorenz95(_: f64, y: ArrayView<f64, Ix1>, mut yp: ArrayViewMut<f64, Ix1>) {
  debug_assert!(y.len() == 40);
  debug_assert!(yp.len() == 40);

  for i in 2..39 {
    yp[i] = -y[i] + y[i - 1] * (y[i + 1] - y[i - 2]) + 8;
  }

  y[0] = -y[0] + y[39] * (y[1] - y[38]) + 8;
  y[1] = -y[1] + y[0] * (y[2] - y[39]) + 8;
  y[39] = -y[39] + x[38] * (y[0] - y[37]) + 8;
}