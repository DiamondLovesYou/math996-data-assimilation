
use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, Ix2, Ix1,
         Axis};
use linxal::types::{LinxalImplScalar};
use linxal::eigenvalues::Eigen;
use linxal::solve_linear::general::SolveLinear;
use num_traits::{NumCast, One, Zero, Float};
use num_complex::Complex;
use rand::Rng;
use rand::distributions::{Sample};
use rand::distributions::normal::Normal;
use std::ops::{Add, Sub, Mul, Div,
               AddAssign, SubAssign,
               DivAssign, MulAssign,};

use {Algorithm, Result};
use utils::{Sqrt, Exp, Diagonal};
use rayon::prelude::*;
use nd_par::prelude::*;

use ensemble::{EnsemblePredict, EnsemblePredictModelStuff,
               EnsembleWorkspace, EnsembleCommonState,
               EnsembleCommonInit, };
use forcing::ResampleForcing;

pub use kalman::etkf::Init;

#[derive(Debug)]
pub struct Workspace<E>
  where E: LinxalImplScalar,
{
  mean: Array<E, Ix1>,
  covariance: Array<E, Ix2>,
  ensembles: Array<E, Ix2>,

  model_workspace: Array<E, Ix2>,

  forcing: Array<E, Ix2>,

  /// Uhat
  ensemble_predict: Array<E, Ix2>,
  /// d
  ensemble_innovation: Array<E, Ix2>,
  /// Xhat
  centered_ensemble: Array<E, Ix2>,
}

impl<'a, E> ::Workspace<Init<'a, E>> for Workspace<E>
  where E: LinxalImplScalar<Complex = Complex<<E as LinxalImplScalar>::RealPart>>,
        Complex<<E as LinxalImplScalar>::RealPart>: LinxalImplScalar,
        E: From<f64> + NumCast + SolveLinear + Eigen,
        E: Send + Sync,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalImplScalar>::RealPart, Output = E> + Sub<<E as LinxalImplScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalImplScalar>::RealPart, Output = E> + Div<<E as LinxalImplScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart> + DivAssign<<E as LinxalImplScalar>::RealPart>,
{
  fn alloc(i: Init<'a, E>, mut rand: &mut Rng, _: u64) -> Result<Workspace<E>> {
    let Init {
      common: EnsembleCommonInit {
        mean,
        covariance,
        ensemble_count,
        observation_operator,
        ..
      },
      model_workspace_size,
      ..
    } = i;
    let initial_mean = mean;
    let initial_covariance = covariance;

    let mut normal = Normal::new(Zero::zero(), One::one());

    let n = initial_mean.dim();

    let mut m = ArrayBase::zeros(n);
    m.assign(&initial_mean);
    let mut c = ArrayBase::zeros(initial_covariance.dim());
    c.assign(&initial_covariance);

    let mut ensembles = ArrayBase::zeros((ensemble_count, n));

    let sol = Eigen::compute_into(initial_covariance.to_owned(),
                                  false, false)
      .expect("can't eigendecomp initial_covariance");
    let d = sol.values.map(|v| v.re.sqrt() );

    {
      let mut first = ensembles.view_mut();
      let r = {
        let mut r: Array<E, Ix2> = ArrayBase::zeros((ensemble_count, n));
        for i in 0..ensemble_count {
          for j in 0..n {
            r[[i,j]] = From::from(normal.sample(&mut rand));
            r[[i,j]] *= d[j];
          }
        }

        r
      };
      first.assign(&initial_mean.broadcast((ensemble_count, n)).unwrap());
      first.scaled_add(One::one(), &r);
    }

    let ec_e: E = NumCast::from(ensemble_count).unwrap();
    let m0 = ensembles
      .sum(Axis(0))
      .mapv_into(|v| v / ec_e);

    m.assign(&m0);

    let w = Workspace {
      mean: m,
      covariance: c,
      ensembles: ensembles,

      model_workspace: ArrayBase::zeros((ensemble_count, model_workspace_size)),

      forcing: ArrayBase::zeros((n, ensemble_count)),

      ensemble_predict: ArrayBase::zeros((n, ensemble_count)),
      ensemble_innovation: ArrayBase::zeros((observation_operator.dim().0,
                                             ensemble_count)),
      centered_ensemble: ArrayBase::zeros((n, ensemble_count)),
    };
    Ok(w)
  }
}

impl<E> ResampleForcing<E> for Workspace<E>
  where E: LinxalImplScalar + From<f64>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart>,
{
  type Disc = ();
  fn forcing_view_mut(&mut self, _: ()) -> ArrayViewMut<E, Ix2> { self.forcing.view_mut() }
}

impl<E> EnsembleWorkspace<E> for Workspace<E>
  where E: LinxalImplScalar,
{
  fn ensemble_state(&self) -> EnsembleCommonState<E> {
    EnsembleCommonState {
      state: ::State {
        mean: self.mean.view(),
        covariance: self.covariance.view(),
      },
      ensembles: self.ensembles.view(),
    }
  }
}
impl<E> EnsemblePredict<E> for Workspace<E>
  where E: LinxalImplScalar + Send + Sync + AddAssign<E> + NumCast,
        E: DivAssign,
        E: ::std::iter::Sum,
{
  fn ensemble_predict_stuff(&mut self) -> EnsemblePredictModelStuff<E> {
    EnsemblePredictModelStuff {
      forcing: self.forcing.view(),
      transpose_ensemble_predict: true,
      ensemble_predict: self.ensemble_predict.view_mut(),
      ensembles: self.ensembles.view(),
      estimator: None,
      model_workspace: self.model_workspace.view_mut(),
    }
  }
}

pub struct Algo<'a, E>
  where E: LinxalImplScalar,
{
  ensemble_count: usize,
  gamma: ArrayView<'a, E::RealPart, Ix1>,
  sigma: ArrayView<'a, E::RealPart, Ix1>,
  observation_operator: ArrayView<'a, E, Ix2>,
}

impl<'init, 'state, E, M, Ob>
Algorithm<M, Ob> for Algo<'init, E>
  where M: ::Model<E>,
        Ob: ::Observer<E, Ix1> + Send + Sync,
        E: LinxalImplScalar<Complex = Complex<<E as LinxalImplScalar>::RealPart>>,
        Complex<<E as LinxalImplScalar>::RealPart>: LinxalImplScalar,
        E: From<f64> + PartialOrd + Eigen,
        E: NumCast + SolveLinear + Exp + Sqrt + ::rand::Rand,
        E: Send + Sync,
        <E as LinxalImplScalar>::RealPart: Send + Sync,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalImplScalar>::RealPart, Output = E> + Sub<<E as LinxalImplScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalImplScalar>::RealPart, Output = E> + Div<<E as LinxalImplScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart> + DivAssign<<E as LinxalImplScalar>::RealPart>,
        E: ::std::iter::Sum,
{
  type Init = Init<'init, E>;
  type WS = Workspace<E>;

  fn init(i: &Init<'init, E>,
          _rand: &mut Rng,
          _model: &mut ::ModelStats<M>,
          _observer: &Ob,
          _: u64) -> Result<Self>
  {
    Ok(Algo {
      ensemble_count: i.ensemble_count,
      gamma: i.gamma.clone(),
      sigma: i.sigma.clone(),
      observation_operator: i.observation_operator.clone(),
    })
  }

  fn next_step(&self,
               current_step: u64,
               _total_steps: u64,
               mut rand: &mut Rng,
               workspace: &mut Workspace<E>,
               model: &mut ::ModelStats<M>,
               observer: &Ob)
               -> Result<()>
  {
    use nd::linalg::general_mat_mul;

    fn max<T: PartialOrd>(a: T, b: T) -> T { if a > b { a } else { b } }

    let neg_one = NumCast::from(-1).unwrap();
    let neg_half: E = NumCast::from(-1.0f64/2.0).unwrap();

    let mut normal = Normal::new(Zero::zero(), One::one());

    // predict

    workspace.ensemble_predict.fill(Zero::zero());
    workspace.resample_forcing((),
                               Diagonal::from(self.sigma.view()),
                               &mut normal,
                               &mut rand);
    workspace.ensemble_predict(current_step, model);

    workspace.ensemble_innovation
      .axis_iter_mut(Axis(1))
      .into_par_iter()
      .for_each(|mut d| {
        assert!(observer.observe_into(current_step, d.view_mut()));
      });
    general_mat_mul(neg_one,
                    &self.observation_operator,
                    &workspace.ensemble_predict,
                    One::one(),
                    &mut workspace.ensemble_innovation);

    {
      workspace.ensemble_innovation
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(self.gamma.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut w, gamma)| {
          let gamma_sq = gamma[()] * gamma[()];

          let sum: E = w
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .fold(|| E::zero(),
                  |sum, mut w| {
                    let v: E = neg_half * w[()] * w[()] / gamma_sq;
                    let v = v._exp();

                    w[()] = v;

                    sum + v
                  })
            .sum();

          let _cumsum: E = w
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .fold(|| E::zero(),
                  |cumsum, mut w| {
                    let c = w[()] / sum;
                    w[()] = c;

                    cumsum + c
                  })
            .sum();
        });
    }

    {
      let mut u = workspace.ensembles.view_mut();
      let uhat = workspace.ensemble_predict.view();
      let ws = workspace.ensemble_innovation.view();

      for i in 0..self.ensemble_count {
        let mut ix = ws.dim().1 - 1;
        for k in 0..ws.dim().1 {
          let mut norm: Option<E> = None;

          for j in 0..ws.dim().0 {
            if let Some(ref mut v) = norm {
              *v = max(*v, ws[[j, k]]);
            } else {
              norm = Some(ws[[j, k]]);
            }
          }
          let norm = norm.unwrap();

          let sample = rand.next_f64();
          let sample = NumCast::from(sample).unwrap();
          if norm > sample {
            ix = k;
            break;
          }
        }

        u.row_mut(i)
          .assign(&uhat.subview(Axis(1),
                                ix));
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