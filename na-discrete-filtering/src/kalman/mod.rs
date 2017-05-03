
use rand::Rng;
use rand::distributions::IndependentSample;

use linxal::types::{LinxalImplScalar};
use nd::Zip;
use nd::prelude::*;
use nd_par::prelude::*;

use rayon::prelude::*;
use rayon::iter::IndexedParallelIterator;

use num_traits::{Zero, NumCast};

use std::ops::{AddAssign, MulAssign, DivAssign, Add};

pub use super::utils::{extend_dim_mut, extend_dim_ref};

pub mod etkf;
pub mod sirs;
pub mod kf;

impl<'a, E, WS> ::State<'a, WS> for EnsembleState<'a, E>
  where E: LinxalImplScalar,
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
        E: LinxalImplScalar,
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
        E: LinxalImplScalar,
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
  where E: LinxalImplScalar + From<f64>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart>,
{
  fn forcing_view_mut(&mut self) -> ArrayViewMut<E, Ix2>;

  fn resample_forcing<R, S>(&mut self,
                            diag: ArrayView<E::RealPart, Ix1>,
                            sampler: &mut S,
                            mut rand: R)
    where R: Rng,
          S: IndependentSample<f64>,
  {
    let mut r = self.forcing_view_mut();
    for i in 0..r.dim().0 {
      for j in 0..r.dim().1 {
        r[[i,j]] = From::from(sampler.sample(&mut rand));
        r[[i,j]] *= diag[i];
      }
    }
  }
}

#[derive(Debug)]
pub struct EnsemblePredictModelStuff<'a, E>
  where E: LinxalImplScalar,
{
  pub forcing: ArrayView<'a, E, Ix2>,
  pub transpose_ensemble_predict: bool,
  pub ensemble_predict: ArrayViewMut<'a, E, Ix2>,
  pub ensembles: ArrayView<'a, E, Ix2>,
  pub estimator: Option<ArrayViewMut<'a, E, Ix1>>,
  pub model_workspace: ArrayViewMut<'a, E, Ix2>,
}
pub trait EnsemblePredict<E>
  where E: LinxalImplScalar + Send + Sync + AddAssign<E> + Add<E>,
        E: DivAssign<E>,
        E: Zero + NumCast,
        E: ::std::iter::Sum,
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

    let n_e: E = NumCast::from(n).unwrap();

    let (model_axis, ensembles_axis) = if transpose_ensemble_predict {
      (Axis(0), Axis(1))
    } else {
      (Axis(1), Axis(0))
    };

    ensembles.axis_iter(Axis(0))
      .into_par_iter()
      .zip(model_workspace.axis_iter_mut(Axis(0)).into_par_iter())
      .zip(ensemble_predict.axis_iter_mut(ensembles_axis).into_par_iter())
      .zip(forcing.axis_iter(Axis(1)).into_par_iter())
      .map(|(((ensemble, model_ws), out), forcing)| (ensemble, model_ws, out, forcing) )
      .with_min_len(64)
      .for_each(|(ensemble, model_ws, mut out, forcing)| {
        model.model.run(step, model_ws, ensemble, out.view_mut());

        out.axis_iter_mut(Axis(0))
          .into_par_iter()
          .zip(forcing.axis_iter(Axis(0)).into_par_iter())
          .with_min_len(16)
          .for_each(|(mut out, forcing)| {
            out[()] += forcing[()];
          });
      });

    if let Some(ref mut estimator) = estimator {
      Zip::from(ensemble_predict.axis_iter(model_axis))
        .and(estimator.axis_iter_mut(Axis(0)))
        .par_apply(|ensemble_predict, mut estimator| {
          estimator[()] = ensemble_predict.into_par_iter().map(|&v| v ).sum();
          estimator[()] /= n_e;
        });
    }

    model.calls += n as u64;
  }
}
