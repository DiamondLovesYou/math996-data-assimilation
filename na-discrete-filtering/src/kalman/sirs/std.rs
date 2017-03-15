
use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, Ix2, Ix1,
         Axis};
use linxal::types::{LinxalScalar};
use linxal::solve_linear::general::SolveLinear;
use num_traits::{NumCast, One, Zero, Float};
use rand::Rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Div,
               AddAssign, SubAssign,
               DivAssign, MulAssign,};

use {Algorithm};
use utils::{CholeskyLDL, Sqrt, Exp};

use super::super::{ResampleForcing, EnsembleWorkspace,
                   EnsemblePredictModelStuff, EnsemblePredict,
                   Model};

pub use super::super::etkf::{Init};

#[derive(Debug)]
pub struct Workspace<E>
  where E: LinxalScalar,
{
  mean: Array<E, Ix1>,
  covariance: Array<E, Ix2>,
  ensembles: Array<E, Ix2>,

  forcing: Array<E, Ix2>,

  /// Uhat
  ensemble_predict: Array<E, Ix2>,
  /// d
  ensemble_innovation: Array<E, Ix2>,
  /// Xhat
  centered_ensemble: Array<E, Ix2>,
}

impl<'a, E> ::Workspace<Init<'a, E>> for Workspace<E>
  where E: LinxalScalar + CholeskyLDL + From<f64> + NumCast + SolveLinear,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalScalar>::RealPart, Output = E> + Sub<<E as LinxalScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalScalar>::RealPart, Output = E> + Div<<E as LinxalScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalScalar>::RealPart> + DivAssign<<E as LinxalScalar>::RealPart>,
{
  fn alloc(i: Init<'a, E>, mut rand: &mut Rng, _: u64) -> Workspace<E> {
    use linxal::types::Symmetric;

    let Init {
      initial_mean,
      initial_covariance,
      ensemble_count,
      ..
    } = i;

    let normal = Normal::new(Zero::zero(), One::one());

    let n = initial_mean.dim();

    let mut m = ArrayBase::zeros(n);
    m.assign(&initial_mean);
    let mut c = ArrayBase::zeros(initial_covariance.dim());
    c.assign(&initial_covariance);

    let mut ensembles = ArrayBase::zeros((ensemble_count, n));

    let (_, d) = CholeskyLDL::compute(&initial_covariance,
                                      Symmetric::Lower)
      .expect("cholesky factorization failed");

    let d: Vec<E> = d.into_raw_vec();
    let scale: E = d.into_iter()
      .filter(|v| !v.is_zero() )
      .fold(E::one(), |p, v| p * v.mag() );

    {
      let mut first = ensembles.view_mut();
      let r = {
        let mut r: Array<E, Ix2> = ArrayBase::zeros((ensemble_count, n));
        for i in 0..ensemble_count {
          for j in 0..n {
            r[[i,j]] = From::from(normal.ind_sample(&mut rand));
          }
        }

        r
      };
      first.assign(&initial_mean.broadcast((ensemble_count,
                                            n)).unwrap());
      first.scaled_add(scale, &r);
    }

    let ec_e: E = NumCast::from(ensemble_count).unwrap();
    let m0 = ensembles
      .sum(Axis(0))
      .mapv_into(|v| v / ec_e);

    m.assign(&m0);

    Workspace {
      mean: m,
      covariance: c,
      ensembles: ensembles,

      forcing: ArrayBase::zeros((n, ensemble_count)),

      ensemble_predict: ArrayBase::zeros((n, ensemble_count)),
      ensemble_innovation: ArrayBase::zeros((n, ensemble_count)),
      centered_ensemble: ArrayBase::zeros((n, ensemble_count)),
    }
  }
}

impl<E> ResampleForcing<E> for Workspace<E>
  where E: LinxalScalar + From<f64>,
        E: MulAssign<<E as LinxalScalar>::RealPart>,
{
  fn forcing_view_mut(&mut self) -> ArrayViewMut<E, Ix2> { self.forcing.view_mut() }
}

impl<E> EnsembleWorkspace<E> for Workspace<E>
  where E: LinxalScalar,
{
  fn mean_view(&self) -> ArrayView<E, Ix1> { self.mean.view() }
  fn covariance_view(&self) -> ArrayView<E, Ix2> { self.covariance.view() }
  fn ensembles_view(&self) -> ArrayView<E, Ix2> { self.ensembles.view() }
}
impl<E> EnsemblePredict<E> for Workspace<E>
  where E: LinxalScalar + AddAssign<E>,
{
  fn ensemble_predict_stuff(&mut self) -> EnsemblePredictModelStuff<E> {
    EnsemblePredictModelStuff {
      forcing: self.forcing.view(),
      transpose_ensemble_predict: true,
      ensemble_predict: self.ensemble_predict.view_mut(),
      ensembles: self.ensembles.view(),
      estimator: None,
    }
  }
}

pub struct Algo<'a, E, F1, F2>
  where E: LinxalScalar,
{
  ensemble_count: usize,
  gamma: ArrayView<'a, E::RealPart, Ix1>,
  sigma: ArrayView<'a, E::RealPart, Ix1>,
  observation_operator: ArrayView<'a, E, Ix2>,

  _i: PhantomData<(F1, F2)>,
}

impl<'init, 'state, F1, F2, E>
Algorithm for Algo<'init, E, F1, F2>
  where F1: for<'r, 's> Fn(ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
        F2: FnMut(u64) -> Option<Array<E, Ix1>>,
        E: LinxalScalar + CholeskyLDL + From<f64> + PartialOrd,
        E: NumCast + SolveLinear + Exp + Sqrt,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalScalar>::RealPart, Output = E> + Sub<<E as LinxalScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalScalar>::RealPart, Output = E> + Div<<E as LinxalScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalScalar>::RealPart> + DivAssign<<E as LinxalScalar>::RealPart>,
{
  type Init = Init<'init, E>;
  type WS = Workspace<E>;
  type Model = Model<E, F1, F2>;

  fn init(i: &Init<'init, E>,
          _rand: &mut Rng,
          _model: &mut Model<E, F1, F2>,
          _: u64) -> Self
  {
    Algo {
      ensemble_count: i.ensemble_count,
      gamma: i.gamma.clone(),
      sigma: i.sigma.clone(),
      observation_operator: i.observation_operator.clone(),

      _i: PhantomData,
    }
  }

  fn next_step(&self,
               current_step: u64,
               _total_steps: u64,
               mut rand: &mut Rng,
               workspace: &mut Workspace<E>,
               model: &mut Model<E, F1, F2>)
               -> Result<(), ()>
  {
    use nd::linalg::general_mat_mul;

    let n = workspace.mean.dim();
    let neg_one = NumCast::from(-1).unwrap();

    let normal = Normal::new(Zero::zero(), One::one());

    // predict

    workspace.ensemble_predict.fill(Zero::zero());

    workspace.resample_forcing(self.sigma.view(),
                               normal,
                               &mut rand);
    workspace.ensemble_predict(model);

    let observation = (model.next_observation)(current_step);
    let observation = observation.expect("missing observation for step TODO");
    workspace.ensemble_innovation.assign(&observation);
    general_mat_mul(neg_one,
                    &self.observation_operator,
                    &workspace.ensemble_predict,
                    One::one(),
                    &mut workspace.ensemble_innovation);

    {
      let mut w = workspace.ensemble_innovation.view_mut();
      let neg_half: E = NumCast::from(-1.0f64/2.0).unwrap();
      for i in 0..n {
        let gamma = self.gamma[i];
        let gamma_sq = gamma * gamma;

        let mut sum: E = Zero::zero();
        for j in 0..self.ensemble_count {
          let v: E = neg_half * (w[[i, j]] * w[[i, j]] / gamma_sq);
          let v = v._exp();

          w[[i, j]] = v;
          sum += v;
        }

        // cumsum
        let mut cumsum = Zero::zero();
        for j in 0..self.ensemble_count {
          cumsum += w[[i, j]] / sum;
          w[[i, j]] = cumsum;
        }
      }
    }

    {
      let mut u = workspace.ensembles.view_mut();
      let uhat = workspace.ensemble_predict.view();
      let ws = workspace.ensemble_innovation.view();

      for i in 0..self.ensemble_count {
        let mut ix = ws.dim().1;
        for k in 0..ws.dim().1 {
          let mut norm: E = Zero::zero();

          for j in 0..ws.dim().0 {
            norm += ws[[j, k]] * ws[[j, k]].cj();
          }

          let norm = norm._sqrt();
          let sample = normal.ind_sample(&mut rand);
          let sample = NumCast::from(sample).unwrap();
          if norm > sample {
            ix = k;
            break;
          }
        }

        u.row_mut(i).assign(&uhat.subview(Axis(1), ix));
      }
    }

    // analyze
    {
      let mut m = workspace.mean.view_mut();
      let mut c = workspace.covariance.view_mut();
      let mut xhat = workspace.centered_ensemble.view_mut();
      let u = workspace.ensembles.view();

      // estimator update
      let scale: E::RealPart = NumCast::from(self.ensemble_count).unwrap();
      for i in 0..m.dim() {
        m[i] = Zero::zero();
        for k in 0..self.ensemble_count {
          m[i] += u[[k, i]];
        }

        m[i] /= scale;
      }

      // covariance update
      for i in 0..m.dim() {
        let mut r = xhat.row_mut(i);
        r.assign(&u.column(i));
        r.scaled_add(neg_one, &m.subview(Axis(0), i));
      }

      general_mat_mul(NumCast::from(scale.recip()).unwrap(),
                      &xhat, &xhat.t(),
                      Zero::zero(),
                      &mut c);
    }

    Ok(())
  }
}