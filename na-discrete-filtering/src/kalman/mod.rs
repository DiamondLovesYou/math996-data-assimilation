
use rand::Rng;
use rand::distributions::IndependentSample;

use linxal::types::{LinxalScalar};
use nd::prelude::*;
use nd_par::prelude::*;

use rayon::prelude::*;

use num_traits::Zero;

use std::ops::{AddAssign, MulAssign, Add};

pub use super::utils::{extend_dim_mut, extend_dim_ref};

pub mod etkf;
pub mod sirs;

#[derive(Debug, Clone)]
pub struct EnsembleInit<'a, E>
  where E: LinxalScalar,
{
  /// Possibly an estimate.
  pub initial_mean: ArrayView<'a, E, Ix1>,
  pub initial_covariance: ArrayView<'a, E, Ix2>,
  pub observation_operator: ArrayView<'a, E, Ix2>,
  pub gamma: ArrayView<'a, E::RealPart, Ix1>,
  pub sigma: ArrayView<'a, E::RealPart, Ix1>,
  pub model_workspace_size: usize,
  pub ensemble_count: usize,
}
impl<'a, E> EnsembleInit<'a, E>
  where E: LinxalScalar,
{ }
impl<'a, E,> ::Initializer for EnsembleInit<'a, E>
  where E: LinxalScalar,
{ }

#[derive(Debug)]
pub struct EnsembleState<'a, E>
  where E: LinxalScalar,
{
  pub mean: ArrayView<'a, E, Ix1>,
  pub covariance: ArrayView<'a, E, Ix2>,
  pub ensembles: ArrayView<'a, E, Ix2>,
}
pub trait EnsembleWorkspace<E> {
  fn mean_view(&self) -> ArrayView<E, Ix1>;
  fn covariance_view(&self) -> ArrayView<E, Ix2>;
  fn ensembles_view(&self) -> ArrayView<E, Ix2>;
}
impl<'a, E, WS> ::State<'a, WS> for EnsembleState<'a, E>
  where E: LinxalScalar,
        WS: EnsembleWorkspace<E>,
{
  fn current(ws: &'a WS) -> EnsembleState<'a, E> {
    EnsembleState {
      mean: ws.mean_view(),
      covariance: ws.covariance_view(),
      ensembles: ws.ensembles_view(),
    }
  }
}

pub struct Model<E, F1, F2>
  where F1: for<'r, 's> Fn(u64, ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
        F2: FnMut(u64) -> Option<Array<E, Ix1>>,
        E: LinxalScalar,
{
  /// Don't modify this; safe to read though.
  pub calls: u64,
  pub model: F1,

  /// Will be called at most once per iteration.
  pub next_observation: F2,
}

impl<E, F1, F2> Model<E, F1, F2>
  where F1: for<'r, 's> Fn(u64, ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
        F2: FnMut(u64) -> Option<Array<E, Ix1>>,
        E: LinxalScalar,
{
  pub fn new(model: F1, observation: F2) -> Model<E, F1, F2> {
    Model {
      calls: 0,
      model: model,
      next_observation: observation,
    }
  }
}

pub trait ResampleForcing<E>
  where E: LinxalScalar + From<f64>,
        E: MulAssign<<E as LinxalScalar>::RealPart>,
{
  fn forcing_view_mut(&mut self) -> ArrayViewMut<E, Ix2>;

  fn resample_forcing<R, S>(&mut self,
                            diag: ArrayView<E::RealPart, Ix1>,
                            sampler: S,
                            mut rand: R)
    where R: Rng,
          S: IndependentSample<f64>,
  {
    let mut r = self.forcing_view_mut();
    for i in 0..r.dim().0 {
      for j in 0..r.dim().1 {
        r[[i,j]] = From::from(sampler.ind_sample(&mut rand));
        r[[i,j]] *= diag[i];
      }
    }
  }
}

#[derive(Debug)]
pub struct EnsemblePredictModelStuff<'a, E>
  where E: LinxalScalar,
{
  pub forcing: ArrayView<'a, E, Ix2>,
  pub transpose_ensemble_predict: bool,
  pub ensemble_predict: ArrayViewMut<'a, E, Ix2>,
  pub ensembles: ArrayView<'a, E, Ix2>,
  pub estimator: Option<ArrayViewMut<'a, E, Ix1>>,
  pub model_workspace: ArrayViewMut<'a, E, Ix2>,
}
pub trait EnsemblePredict<E>
  where E: LinxalScalar + Send + Sync + AddAssign<E> + Add<E> + Zero,
        ::rayon::par_iter::reduce::SumOp: ::rayon::par_iter::reduce::ReduceOp<E>,
{
  fn ensemble_predict_stuff(&mut self) -> EnsemblePredictModelStuff<E>;

  fn ensemble_predict<M>(&mut self, step: u64,
                         model: &mut ::ModelStats<M>)
    where M: ::Model<E>,
  {
    let EnsemblePredictModelStuff {
      forcing, mut ensemble_predict,
      ensembles, mut estimator,
      transpose_ensemble_predict,
      mut model_workspace,
    } = self.ensemble_predict_stuff();

    let n = if transpose_ensemble_predict {
      ensemble_predict.dim().1
    } else {
      ensemble_predict.dim().0
    };
    assert_eq!(n, forcing.dim().1);
    assert_eq!(model_workspace.dim().0, ensembles.dim().0);

    let outer_axis = if transpose_ensemble_predict {
      Axis(1)
    } else {
      Axis(0)
    };

    let predict_iter = ensemble_predict
      .axis_iter_mut(outer_axis)
      .into_par_iter();
    let iter = ensembles.axis_iter(Axis(0))
      .into_par_iter()
      .zip(model_workspace.axis_iter_mut(Axis(0)).into_par_iter())
      .zip(predict_iter)
      .zip(forcing.axis_iter(Axis(1)).into_par_iter());

    if let Some(ref mut estimator) = estimator {
      iter
        .zip(estimator.axis_iter_mut(Axis(0)).into_par_iter())
        .map(|((((ensemble, model_ws), out), forcing), estimator)| (ensemble, model_ws, out, forcing, estimator) )
        .for_each(|(ensemble, model_ws, mut out, forcing, mut estimator)| {
          model.model.run(step, model_ws, ensemble, out.view_mut());

          out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(forcing.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(mut out, forcing)| {
              out[()] += forcing[()];
            });

          estimator[()] = out.as_slice().unwrap()
            .into_par_iter()
            .map(|&v| v )
            .sum()
        })
    } else {
      iter
        .map(|(((ensemble, model_ws), out), forcing)| (ensemble, model_ws, out, forcing) )
        .for_each(|(ensemble, model_ws, mut out, forcing)| {
          model.model.run(step, model_ws, ensemble, out.view_mut());

          out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(forcing.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(mut out, forcing)| {
              out[()] += forcing[()];
            })
        });
    }

    model.calls += n as u64;
  }
}
