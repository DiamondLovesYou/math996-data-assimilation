
use nd::{RcArray, Array, ArrayView, ArrayViewMut, Ix1, Ix2, Axis};
use rand;
use num_traits::{Float, One};
use rayon::prelude::*;
use nd_par::prelude::*;
use linxal::types::LinxalScalar;
use nd;
use na_df::utils::{extend_dim_ref, extend_dim_mut};

use std::ops::{Add, Sub, Mul, Div,
               AddAssign,};
use std::ops::Range;

pub struct Data<E> {
  pub x: RcArray<E, Ix1>,
  pub truth: RcArray<E, Ix2>,
  pub observations: RcArray<E, Ix2>,
}

pub fn generate_model_truth_and_observation<F, E, D, R>
(f: F, n: usize,
 x: Range<E>,
 y: Array<E, Ix1>,
 yp: Array<E, Ix1>,
 tol: E::RealPart, h: E::RealPart,
 thresh: Array<E::RealPart, Ix1>,
 sigma: ArrayView<E, Ix1>,
 gamma: ArrayView<E, Ix1>,
 obs_op: ArrayView<E, Ix2>,
 dist: &D,
 mut rand: &mut R,
) -> Data<E>
  where F: Fn(E, ArrayView<E, Ix1>, ArrayViewMut<E, Ix1>),
        E: Float + LinxalScalar + From<<E as LinxalScalar>::RealPart>,
        E: Sync + Send,
        E: AddAssign<E>,
        E: Mul<<E as LinxalScalar>::RealPart, Output = E> + Add<<E as LinxalScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Add<E, Output = E>,
        E: Div<<E as LinxalScalar>::RealPart, Output = E> + Sub<<E as LinxalScalar>::RealPart, Output = E>,
        E: Div<E, Output = E> + Sub<E, Output = E>,
        E::RealPart: nd::ScalarOperand,
        D: rand::distributions::IndependentSample<E>,
        R: rand::Rng,
{
  use na_q::rke;
  use nd::linalg::general_mat_mul;

  let yp_size = yp.len();
  assert_eq!(yp_size, sigma.len());
  assert_eq!(obs_op.dim().0, gamma.len());
  assert_eq!(yp_size, obs_op.dim().1);

  let xes: Array<E, Ix1> = Array::linspace(x.start, x.end, n);
  let truth = {
    let dest: Array<E, Ix2> = Array::zeros((n, yp_size));

    let mut state = rke::new(x.start, y, yp, h, tol, thresh, true);
    state
      .iter(f)
      .stepping_iter(xes.view())
      .fold(dest, |mut dest, (state, result)| {
        let (idx, _x, ycoeff) = result
          .expect("integration failed");
        let ycoeff = ycoeff.unwrap();

        {
          let mut z = dest.row_mut(idx);
          let step_size = state.step_size();

          rke::y_value(state.x() - step_size,
                       state.x(), step_size,
                       &ycoeff,
                       z.view_mut());

          for (mut z_mut, sigma) in z.iter_mut().zip(sigma.iter()) {
            *z_mut += *sigma * dist.ind_sample(&mut rand);
          }
        }

        dest
      })
  };

  let mut r_mat = Array::zeros(obs_op.dim().0);
  for (i, mut dest) in r_mat.indexed_iter_mut() {
    *dest = gamma[i] * dist.ind_sample(&mut rand);
  }
  let r_mat = r_mat;

  let mut observations = Array::zeros((n, obs_op.dim().0));
  {
    let obs_iter = observations.axis_iter_mut(Axis(0)).into_par_iter();
    truth
      .axis_iter(Axis(0))
      .into_par_iter()
      .zip(obs_iter)
      .into_par_iter()
      .for_each(|(truth, mut obs)| {
        obs.assign(&r_mat);

        let t2 = extend_dim_ref(&truth, false);
        let mut obs2 = extend_dim_mut(&mut obs, false);

        general_mat_mul(One::one(),
                        &obs_op,
                        &t2,
                        One::one(),
                        &mut obs2);
      });
  }

  Data {
    x: xes.to_shared(),
    truth: truth.to_shared(),
    observations: observations.to_shared(),
  }
}