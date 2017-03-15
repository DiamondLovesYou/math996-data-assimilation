
use rand::Rng;
use rand::distributions::IndependentSample;

use linxal::types::{LinxalScalar};
use nd::prelude::*;
use nd::{DataMut, ViewRepr};

use std::ops::{AddAssign, MulAssign,};

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
  where F1: for<'r, 's> Fn(ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
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
  where F1: for<'r, 's> Fn(ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
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
}
pub trait EnsemblePredict<E>
  where E: LinxalScalar + AddAssign<E>,
{
  fn ensemble_predict_stuff(&mut self) -> EnsemblePredictModelStuff<E>;

  fn ensemble_predict<F1, F2>(&mut self, model: &mut Model<E, F1, F2>)
    where F1: for<'r, 's> Fn(ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
          F2: FnMut(u64) -> Option<Array<E, Ix1>>,
  {
    let EnsemblePredictModelStuff {
      forcing, mut ensemble_predict,
      ensembles, mut estimator,
      transpose_ensemble_predict,
    } = self.ensemble_predict_stuff();

    let n = if transpose_ensemble_predict {
      ensemble_predict.dim().1
    } else {
      ensemble_predict.dim().0
    };
    assert_eq!(n, forcing.dim().1);

    for i in 0..n {
      let mut ensemble_dest = if transpose_ensemble_predict {
        ensemble_predict.column_mut(i)
      } else {
        ensemble_predict.row_mut(i)
      };
      (model.model)(ensembles.row(i).view(),
                    ensemble_dest.view_mut());
      model.calls += 1;

      let forcing = forcing.column(i);
      assert_eq!(ensemble_dest.dim(), forcing.dim());
      for j in 0..ensemble_dest.dim() {
        ensemble_dest[j] += forcing[j];

        if let Some(ref mut estimator) = estimator {
          estimator[j] += ensemble_dest[j];
        }
      }
    }
  }
}

pub fn extend_dim_mut<D>(d: &mut ArrayBase<D, Ix1>, t: bool)
                         -> ArrayBase<ViewRepr<&mut D::Elem>, Ix2>
  where D: DataMut,
{
  let d_dim = d.dim();
  let d_slice = d.as_slice_mut().expect("d_slice");
  let dim = if !t {
    (d_dim, 1)
  } else {
    (1, d_dim)
  };
  let d2 = ArrayBase::<ViewRepr<&mut D::Elem>, _>::from_shape(dim, d_slice)
    .expect("d2");

  d2
}