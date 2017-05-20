//! Common stuffs for ensembles/particles

use linxal::types::{LinxalImplScalar};

use nd::{ArrayView, ArrayViewMut, Ix1, Ix2, Axis, Zip};
use nd_par::prelude::*;

use std::ops::{Deref, AddAssign, Add, DivAssign, };
use num_traits::{Zero, NumCast};

#[derive(Debug, Clone)]
pub struct EnsembleCommonInit<'a, E>
  where E: LinxalImplScalar,
{
  /// Possibly an estimate.
  pub mean: ArrayView<'a, E, Ix1>,
  pub covariance: ArrayView<'a, E, Ix2>,
  pub observation_operator: ArrayView<'a, E, Ix2>,
  pub gamma: ArrayView<'a, E::RealPart, Ix1>,
  pub sigma: ArrayView<'a, E::RealPart, Ix1>,
  pub ensemble_count: usize,
}
impl<'a, E> EnsembleCommonInit<'a, E>
  where E: LinxalImplScalar,
{ }
impl<'a, E> ::Initializer for EnsembleCommonInit<'a, E>
  where E: LinxalImplScalar,
{ }

#[derive(Debug)]
pub struct EnsembleCommonState<'a, E>
  where E: LinxalImplScalar,
{
  pub state: ::State<'a, E>,
  pub ensembles: ArrayView<'a, E, Ix2>,
}
impl<'a, E> Deref for EnsembleCommonState<'a, E>
  where E: LinxalImplScalar,
{
  type Target = ::State<'a, E>;
  fn deref(&self) -> &::State<'a, E> {
    &self.state
  }
}

pub trait EnsembleWorkspace<E>
  where E: LinxalImplScalar,
{
  fn ensemble_state(&self) -> EnsembleCommonState<E>;
  fn mean_view(&self) -> ArrayView<E, Ix1> {
    self.ensemble_state().mean
  }
  fn covariance_view(&self) -> ArrayView<E, Ix2> {
    self.ensemble_state().covariance
  }
  fn ensembles_view(&self) -> ArrayView<E, Ix2> {
    self.ensemble_state().ensembles
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
      .for_each(|(ensemble, model_ws, mut out, forcing)| {
        model.model.run_model(step, model_ws, ensemble, out.view_mut());

        Zip::from(&mut out)
          .and(&forcing)
          .par_apply(|out, &forcing| {
            *out += forcing;
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
