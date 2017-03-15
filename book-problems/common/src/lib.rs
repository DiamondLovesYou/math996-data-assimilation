#![feature(unboxed_closures)]
#![feature(fn_traits)]

extern crate rand;
#[macro_use]
extern crate ndarray as nd;
extern crate num_traits;
extern crate na_discrete_filtering as na_df;
extern crate gnuplot;

use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, RcArray,
         Ix1, Ix2, Ix3,
         arr1, arr2,
         Axis};
use na_df::kalman::*;

use rand::isaac::Isaac64Rng;
use rand::SeedableRng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

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
      truth: truth.to_shared(),
      observations: observations.to_shared(),

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
  pub truth: RcArray<f64, Ix1>,
  pub observations: RcArray<f64, Ix1>,

  pub initial_mean: Array<f64, Ix1>,
  pub initial_covariance: Array<f64, Ix2>,
  pub observation_operator: Array<f64, Ix2>,
  pub gamma: Array<f64, Ix1>,
  pub sigma: Array<f64, Ix1>,
  pub ensemble_count: usize,
}

#[derive(Copy, Clone)]
pub struct SinMapDataModel<'a>(&'a SinMapData);
impl<'a> SinMapDataModel<'a> {
  fn run(&self, v: ArrayView<f64, Ix1>, mut dest: ArrayViewMut<f64, Ix1>) {
    dest[0] = self.0.params.alpha * v[0].sin();
  }
}
impl<'a, 'b, 'c> FnOnce<(ArrayView<'b, f64, Ix1>, ArrayViewMut<'c, f64, Ix1>)> for SinMapDataModel<'a> {
  type Output = ();
  extern "rust-call" fn call_once(self,
                                  (v, dest): (ArrayView<'b, f64, Ix1>, ArrayViewMut<'c, f64, Ix1>))
                                  -> Self::Output
  {
    self.run(v, dest);
    ()
  }
}
impl<'a, 'b, 'c> FnMut<(ArrayView<'b, f64, Ix1>, ArrayViewMut<'c, f64, Ix1>)> for SinMapDataModel<'a> {
  extern "rust-call" fn call_mut(&mut self,
                                 (v, dest): (ArrayView<'b, f64, Ix1>, ArrayViewMut<'c, f64, Ix1>))
                                 -> Self::Output
  {
    self.run(v, dest)
  }
}
impl<'a, 'b, 'c> Fn<(ArrayView<'b, f64, Ix1>, ArrayViewMut<'c, f64, Ix1>)> for SinMapDataModel<'a> {
  extern "rust-call" fn call(&self,
                             (v, dest): (ArrayView<'b, f64, Ix1>, ArrayViewMut<'c, f64, Ix1>))
                             -> Self::Output
  {
    self.run(v, dest)
  }
}
#[derive(Clone, Copy)]
pub struct SinMapDataObservation<'a>(&'a SinMapData);
impl<'a> SinMapDataObservation<'a> {
  fn run(&self, idx: u64) -> Option<Array<f64, Ix1>> {
    Some(arr1(&[self.0.observations[idx as usize]]))
  }
}
impl<'a> FnOnce<(u64,)> for SinMapDataObservation<'a> {
  type Output = Option<Array<f64, Ix1>>;
  extern "rust-call" fn call_once(self, (i,): (u64,)) -> Self::Output {
    self.run(i)
  }
}
impl<'a> FnMut<(u64,)> for SinMapDataObservation<'a> {
  extern "rust-call" fn call_mut(&mut self, (i,): (u64,)) -> Self::Output {
    self.run(i)
  }
}
impl<'a> Fn<(u64,)> for SinMapDataObservation<'a> {
  extern "rust-call" fn call(&self, (i,): (u64,)) -> Self::Output {
    self.run(i)
  }
}
pub type SinMapModel<'a> = Model<f64, SinMapDataModel<'a>, SinMapDataObservation<'a>>;
impl SinMapData {
  pub fn init(&self) -> EnsembleInit<f64> {
    EnsembleInit {
      initial_mean: self.initial_mean.view(),
      initial_covariance: self.initial_covariance.view(),
      observation_operator: self.observation_operator.view(),

      gamma: self.gamma.view(),
      sigma: self.sigma.view(),
      ensemble_count: self.ensemble_count,
    }
  }
  pub fn model(&self) -> SinMapModel {
    Model::new(SinMapDataModel(self),
               SinMapDataObservation(self))
  }
}

pub trait ModelTruth<E> {
  fn truth(&self) -> ArrayView<E, Ix1>;
  fn observations(&self) -> ArrayView<E, Ix1>;
}
impl ModelTruth<f64> for SinMapData {
  fn truth(&self) -> ArrayView<f64, Ix1> { self.truth.view() }
  fn observations(&self) -> ArrayView<f64, Ix1> { self.observations.view() }
}

#[derive(Clone, Debug)]
pub struct StateSteps {
  means: Array<f64, Ix2>,
  covariances: Array<f64, Ix3>,
  ensembles: Array<f64, Ix3>,
}
impl StateSteps {
  pub fn new(steps: usize, ensemble_count: usize) -> StateSteps {
    StateSteps {
      means: ArrayBase::zeros((steps, 1)),
      covariances: ArrayBase::zeros((steps, 1, 1)),
      ensembles: ArrayBase::zeros((steps, ensemble_count, 1)),
    }
  }

  pub fn store_state<WS>(&mut self, step: usize, ws: &WS)
    where WS: EnsembleWorkspace<f64>,
  {
    self.means
      .subview_mut(Axis(0), step)
      .assign(&ws.mean_view());
    self.covariances
      .subview_mut(Axis(0), step)
      .assign(&ws.covariance_view());
    self.ensembles
      .subview_mut(Axis(0), step)
      .assign(&ws.ensembles_view());
  }
}

pub fn make_ensemble_plots<T>(source: &T, states: &StateSteps,
                              js: u64, what: &str)
  where T: ModelTruth<f64>,
{
  use gnuplot::*;

  let truth = source.truth();
  let observations = source.observations();

  let means = states.means.view();
  let covariances = states.covariances.view();

  let js = ::std::cmp::min(means.dim().0 as u64 - 1, js);
  let js_space = 0..js;

  let mut all = Figure::new();
  all.set_terminal("wxt", "");

  {
    let mut axis = all.axes2d();
    let title = format!("{}, Ex. 1.3", what);
    axis.set_title(&title[..], &[]);
    axis.set_x_label("iteration, j", &[]);
    axis.lines(js_space.clone(), (0..js).map(|i| truth[i as usize] ),
               &[PlotOption::Caption("truth"),]);
    axis.lines(js_space.clone(), (0..js).map(|i| means[[i as usize, 0]] ),
               &[PlotOption::Caption("ensemble mean"),
                 PlotOption::Color("magenta"),]);
    axis.lines(js_space.clone(),
               (0..js).map(|i| {
                 means[[i as usize, 0]] + covariances[[i as usize, 0, 0]].sqrt()
               }),
               &[PlotOption::Caption("error"),
                 PlotOption::Color("red"),
                 PlotOption::LineStyle(DashType::Dash)]);
    axis.points((1..js).map(|i| i as f64 ),
                (0..js - 1).map(|i| {
                  observations[i as usize]
                }),
                &[PlotOption::Caption("observation"),
                  PlotOption::Color("black"),
                  PlotOption::PointSymbol('x'),]);
    axis.lines(js_space.clone(),
               (0..js).map(|i| {
                 means[[i as usize, 0]] - covariances[[i as usize, 0, 0]].sqrt()
               }),
               &[PlotOption::Caption("error"),
                 PlotOption::Color("red"),
                 PlotOption::LineStyle(DashType::Dash)]);
  }

  all.show();
}
