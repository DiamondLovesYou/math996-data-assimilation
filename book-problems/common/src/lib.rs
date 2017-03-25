extern crate rand;
#[macro_use]
extern crate ndarray as nd;
extern crate num_traits;
extern crate na_discrete_filtering as na_df;
extern crate util;
extern crate plot_helper;

use nd::{Array, ArrayBase, ArrayView, ArrayViewMut,
         Ix1, Ix2,
         arr1, arr2,
         Axis};
use na_df::kalman::*;
use na_df::utils::{extend_dim_ref};
use na_df::{Observer, Model, ModelStats};

use rand::isaac::Isaac64Rng;
use rand::SeedableRng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

pub use util::{ModelTruth, StateSteps};
pub use plot_helper::make_ensemble_plots;

#[derive(Copy, Clone, Debug)]
pub struct SinMapSetup {
  pub alpha: f64,
  pub gamma: f64,
  pub sigma: f64,
  pub c0: f64,
  pub m0: f64,
  pub ensemble_count: usize,
  pub rand_seed: u64,
  pub steps: usize,
}
impl Default for SinMapSetup {
  fn default() -> Self {
    SinMapSetup {
      alpha: 2.5,
      gamma: 1.0,
      sigma: 0.3,
      c0: 0.09,
      m0: 0.0,
      rand_seed: 1,
      ensemble_count: 10,
      steps: 1000,
    }
  }
}

impl Into<SinMapData> for SinMapSetup {
  fn into(self) -> SinMapData {
    let mut rand = Isaac64Rng::from_seed(&[self.rand_seed]);
    let normal = Normal::new(0.0, 1.0);

    let obs_op: Array<f64, Ix2> = arr2(&[[1.0]]);

    let mut truth: Array<f64, Ix1> = ArrayBase::zeros(self.steps);
    truth[0] = self.m0 + self.c0.sqrt() * normal.ind_sample(&mut rand);
    (0..self.steps)
      .skip(1)
      .fold(truth.view_mut(), |mut v, i| {
        v[i] = self.alpha * v[i - 1].sin();
        v[i] += self.sigma * normal.ind_sample(&mut rand);

        v
      });
    let observations: Array<f64, Ix1> = truth
      .iter()
      .fold(None, |prev, v| {
        if let Some(prev) = prev {
          let obs = &obs_op[[0,0]] * v + self.gamma * normal.ind_sample(&mut rand);
          Some(stack![Axis(0), prev, arr1(&[obs])])
        } else {
          Some(arr1(&[0.0]))
        }
      })
      .unwrap();

    let initial_mean: Array<f64, Ix1> = arr1(&[10.0 * normal.ind_sample(&mut rand)]);
    let initial_covariance: Array<f64, Ix2> = arr2(&[[10.0 * self.c0]]);

    let v_gamma: Array<_, Ix1> = arr1(&[self.gamma]);
    let v_sigma: Array<_, Ix1> = arr1(&[self.sigma]);

    SinMapData {
      rand: Some(rand),
      params: self,
      truth: truth,
      observations: observations,

      initial_mean: initial_mean,
      initial_covariance: initial_covariance,
      observation_operator: obs_op,
      gamma: v_gamma,
      sigma: v_sigma,
      ensemble_count: self.ensemble_count,
    }
  }
}

#[derive(Clone)]
pub struct SinMapData {
  pub rand: Option<Isaac64Rng>,
  pub params: SinMapSetup,
  pub truth: Array<f64, Ix1>,
  pub observations: Array<f64, Ix1>,

  pub initial_mean: Array<f64, Ix1>,
  pub initial_covariance: Array<f64, Ix2>,
  pub observation_operator: Array<f64, Ix2>,
  pub gamma: Array<f64, Ix1>,
  pub sigma: Array<f64, Ix1>,
  pub ensemble_count: usize,
}

#[derive(Copy, Clone)]
pub struct SinMapDataModel<'a>(&'a SinMapData);
impl<'a> Model<f64> for SinMapDataModel<'a> {
  fn workspace_size() -> usize { 0 }
  fn run(&self, _step: u64,
         _workspace: ArrayViewMut<f64, Ix1>,
         mean: ArrayView<f64, Ix1>,
         mut out: ArrayViewMut<f64, Ix1>) {
    out[0] = self.0.params.alpha * mean[0].sin();
  }
}
#[derive(Clone, Copy)]
pub struct SinMapDataObservation<'a>(&'a SinMapData);
impl<'a> Observer<f64> for SinMapDataObservation<'a> {
  fn observe_into(&self, step: u64, mut out: ArrayViewMut<f64, Ix1>) -> bool {
    out[0] = self.0.observations[step as usize];
    true
  }
}
pub type SinMapModel<'a> = (ModelStats<SinMapDataModel<'a>>, SinMapDataObservation<'a>);
impl SinMapData {
  pub fn init(&self) -> EnsembleInit<f64> {
    EnsembleInit {
      initial_mean: self.initial_mean.view(),
      initial_covariance: self.initial_covariance.view(),
      observation_operator: self.observation_operator.view(),

      gamma: self.gamma.view(),
      sigma: self.sigma.view(),
      model_workspace_size: 0,
      ensemble_count: self.ensemble_count,
    }
  }
  pub fn model(&self) -> SinMapModel {
    (From::from(SinMapDataModel(self)), SinMapDataObservation(self))
  }
}


impl ModelTruth<f64> for SinMapData {
  fn truth(&self) -> ArrayView<f64, Ix2> {
    extend_dim_ref(&self.truth, false)
  }
  fn observations(&self) -> ArrayView<f64, Ix2> {
    extend_dim_ref(&self.observations, false)
  }
}


