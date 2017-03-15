
extern crate rand;
#[macro_use]
extern crate ndarray as nd;
extern crate num_traits;
extern crate na_discrete_filtering as na_df;
extern crate gnuplot;

use num_traits::*;

use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, Ix1, Ix2, Ix3,
         arr1, arr2,
         Axis};
use na_df::{Algorithm, Workspace};
use na_df::kalman::etkf::*;

use rand::Rng;
use rand::isaac::Isaac64Rng;
use rand::SeedableRng;

fn main() {
  use na_df::State;

  const STEPS: usize = 10000;
  let steps = 0..STEPS;
  let alpha = 2.5;
  let gamma = 1.0;
  let sigma = 0.3;
  let c0 = 0.09;
  let m0 = 0.0;
  let ensemble_count = 10;

  let mut rand = Isaac64Rng::from_seed(&[1]);

  let obs_op: Array<f64, Ix2> = arr2(&[[1.0]]);

  let mut truth: Array<f64, Ix1> = ArrayBase::zeros(STEPS);
  truth[0] = m0 + c0.sqrt() * rand.gen_range(-0.5, 0.5);
  steps.skip(1)
    .fold(truth.view_mut(), |mut v, i| {
      v[i] = alpha * v[i - 1].sin() + sigma * rand.gen_range(-0.5, 0.5);

      v
    });
  let observations: Array<f64, Ix1> = truth
    .iter()
    .fold(None, |prev, v| {
      if let Some(prev) = prev {
        let obs = &obs_op[[0,0]] * v + gamma * rand.gen_range(-0.5, 0.5);
        Some(stack![Axis(0), prev, arr1(&[obs])])
      } else {
        Some(arr1(&[0.0]))
      }
    })
    .unwrap();

  let initial_mean: Array<f64, Ix1> = arr1(&[10.0 * rand.gen_range(-0.5, 0.5)]);
  let initial_covariance: Array<f64, Ix2> = arr2(&[[10.0 * c0]]);

  let v_gamma: Array<_, Ix1> = arr1(&[gamma]);
  let v_sigma: Array<_, Ix1> = arr1(&[sigma]);

  let init = Init {
    initial_mean: initial_mean.view(),
    initial_covariance: initial_covariance.view(),
    observation_operator: obs_op.view(),

    gamma: v_gamma.view(),
    sigma: v_sigma.view(),
    ensemble_count: ensemble_count,
  };

  let model = |v: ArrayView<f64, Ix1>,
               mut dest: ArrayViewMut<f64, Ix1>| {
    dest[0] = alpha * v[0].sin();
  };
  let next_observation = |idx| { Some(arr1(&[observations[idx as usize]])) };

  let mut model = Model::new(model, next_observation);
  let algo = Algo::init(&init, &mut rand, &mut model, STEPS as u64);
  let mut workspace: OwnedWorkspace<_> =
    OwnedWorkspace::alloc(init, &mut rand, STEPS as u64);

  let mut means: Array<f64, Ix2> =
    ArrayBase::zeros((STEPS, 1));
  let mut covariances: Array<f64, Ix3> =
    ArrayBase::zeros((STEPS, 1, 1));
  let mut ensembles: Array<f64, Ix3> =
    ArrayBase::zeros((STEPS, ensemble_count, 1));

  for i in 0..STEPS as u64 {
    algo.next_step(i, STEPS as u64,
                   &mut rand,
                   &mut workspace,
                   &mut model)
      .expect("algorithm step failed");

    let state: na_df::kalman::etkf::State<_> =
      State::current(&workspace);

    means.subview_mut(Axis(0), i as usize).assign(&state.mean);
    covariances.subview_mut(Axis(0), i as usize).assign(&state.covariance);
    ensembles.subview_mut(Axis(0), i as usize).assign(&state.ensembles);

    //println!("step = {}, calls = {}", i, model.calls);
    //println!("{:?}", state.state);
  }

  make_plots(truth, observations.clone(),
             means, covariances, ensembles);
}

