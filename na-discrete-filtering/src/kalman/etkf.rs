
use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, Ix2, Ix1,
         Axis};
use nd::linalg::general_mat_mul;

use linxal::types::{LinxalImplScalar};
use linxal::solve_linear::general::SolveLinear;
use linxal::solve_linear::symmetric::SymmetricSolveLinear;
use linxal::eigenvalues::general::Eigen;
use num_traits::{NumCast, One, Zero, Float};
use num_complex::Complex;
use rand::Rng;
use rand::distributions::{Sample, IndependentSample};
use rand::distributions::normal::Normal;
use std::ops::{Add, Sub, Mul, Div,
               AddAssign, SubAssign,
               DivAssign, MulAssign,
               Deref, DerefMut};

use {Algorithm, Workspace, Result};
use utils::{SolutionHelper, Diagonal};
use rayon::prelude::*;
use nd_par::prelude::*;

use super::{extend_dim_mut, extend_dim_ref, };
use ensemble::{EnsemblePredict, EnsemblePredictModelStuff,
               EnsembleWorkspace, EnsembleCommonState,
               EnsembleCommonInit, };
use forcing::ResampleForcing;

pub use super::super::ensemble::EnsembleCommonState as State;
pub use super::Model;

pub struct Init<'a, E>
where E: LinxalImplScalar,
{
  pub common: EnsembleCommonInit<'a, E>,
  pub model_workspace_size: usize,
}
impl<'a, E> ::Initializer for Init<'a, E>
  where E: LinxalImplScalar,
{ }
impl<'a, E> Deref for Init<'a, E>
  where E: LinxalImplScalar,
{
  type Target = EnsembleCommonInit<'a, E>;
  fn deref(&self) -> &Self::Target {
    &self.common
  }
}
impl<'a, E> DerefMut for Init<'a, E>
  where E: LinxalImplScalar,
{
  fn deref_mut(&mut self) -> &mut EnsembleCommonInit<'a, E> {
    &mut self.common
  }
}

#[derive(Debug)]
pub struct OwnedWorkspace<E>
  where E: LinxalImplScalar,
{
  mean: Array<E, Ix1>,
  covariance: Array<E, Ix2>,
  ensembles: Array<E, Ix2>,

  model_workspace: Array<E, Ix2>,

  /// d
  innovation: Array<E, Ix1>,
  kalman_gain: Array<E, Ix2>,
  s: Array<E, Ix2>,
  forcing: Array<E, Ix2>,

  /// Uhat
  ensemble_predict: Array<E, Ix2>,
  /// mhat
  estimator_predict: Array<E, Ix1>,
  /// Xhat
  centered_ensemble: Array<E, Ix2>,
  /// chat
  covariance_predict: Array<E, Ix2>,
  /// T
  sqrt_transform: Array<E, Ix2>,
  sqrt_transform_intermediate: Array<E, Ix2>,
  /// X
  transformed_centered_ensemble: Array<E, Ix2>,
}
impl<'a, E> Workspace<Init<'a, E>> for OwnedWorkspace<E>
  where E: LinxalImplScalar<Complex = Complex<<E as LinxalImplScalar>::RealPart>>,
        Complex<<E as LinxalImplScalar>::RealPart>: LinxalImplScalar,
        E: From<f64> + NumCast + SolveLinear + Eigen,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalImplScalar>::RealPart, Output = E> + Sub<<E as LinxalImplScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalImplScalar>::RealPart, Output = E> + Div<<E as LinxalImplScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart> + DivAssign<<E as LinxalImplScalar>::RealPart>,
{
  fn alloc(i: Init<'a, E>, mut rand: &mut Rng, _: u64) -> Result<OwnedWorkspace<E>> {
    let Init {
      common: EnsembleCommonInit {
        mean,
        covariance,
        observation_operator,
        ensemble_count,
        ..
      },
      model_workspace_size,
      ..
    } = i;

    let mut normal = Normal::new(Zero::zero(), One::one());

    let n = mean.dim();

    let mut m = mean.to_owned();
    let mut c = covariance.to_owned();

    let mut ensembles = ArrayBase::zeros((ensemble_count, n));

    let sol = Eigen::compute_into(covariance.to_owned(),
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
      first.assign(&mean.broadcast((ensemble_count, n)).unwrap());
      first.scaled_add(One::one(), &r);
    }

    let ec_e: E = NumCast::from(ensemble_count).unwrap();
    let m0 = ensembles
      .sum(Axis(0))
      .mapv_into(|v| v / ec_e);

    m.assign(&m0);

    let r = OwnedWorkspace {
      mean: m,
      covariance: c,
      ensembles: ensembles,

      model_workspace: Array::zeros((ensemble_count, model_workspace_size)),

      innovation: ArrayBase::zeros(observation_operator.dim().0),
      s: ArrayBase::zeros((observation_operator.dim().0,
                           observation_operator.dim().0)),
      kalman_gain: ArrayBase::zeros((n, observation_operator.dim().0)),
      forcing: ArrayBase::zeros((n, ensemble_count)),

      ensemble_predict: ArrayBase::zeros((ensemble_count, n)),
      estimator_predict: ArrayBase::zeros(n),
      centered_ensemble: ArrayBase::zeros((n, ensemble_count)),
      covariance_predict: ArrayBase::zeros((n, n)),
      sqrt_transform: ArrayBase::zeros((ensemble_count, ensemble_count)),
      sqrt_transform_intermediate: ArrayBase::zeros((ensemble_count, ensemble_count)),
      transformed_centered_ensemble: ArrayBase::zeros((n, ensemble_count)),
    };
    Ok(r)
  }
}
impl<E> EnsembleWorkspace<E> for OwnedWorkspace<E>
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
impl<E> ResampleForcing<E> for OwnedWorkspace<E>
  where E: LinxalImplScalar + From<f64>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart>,
{
  type Disc = ();
  fn forcing_view_mut(&mut self, _: ()) -> ArrayViewMut<E, Ix2> { self.forcing.view_mut() }
}

impl<E> EnsemblePredict<E> for OwnedWorkspace<E>
  where E: LinxalImplScalar + Send + Sync + AddAssign<E> + NumCast,
        E: DivAssign<E>,
        E: ::std::iter::Sum,
{
  fn ensemble_predict_stuff(&mut self) -> EnsemblePredictModelStuff<E> {
    EnsemblePredictModelStuff {
      forcing: self.forcing.view(),
      transpose_ensemble_predict: false,
      ensemble_predict: self.ensemble_predict.view_mut(),
      ensembles: self.ensembles.view(),
      estimator: Some(self.estimator_predict.view_mut()),
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

impl<'init, 'state, E, M, Ob> Algorithm<M, Ob> for Algo<'init, E>
  where M: ::Model<E>,
        Ob: ::Observer<E>,
        E: LinxalImplScalar<Complex = Complex<<E as LinxalImplScalar>::RealPart>>,
        Complex<<E as LinxalImplScalar>::RealPart>: LinxalImplScalar,
        E: From<f64> + NumCast + SolveLinear + SymmetricSolveLinear + Send + Sync + Eigen + Float,
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
  type Init  = Init<'init, E>;
  type WS    = OwnedWorkspace<E>;

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

  /// This could use some cleaning.
  fn next_step(&self,
               current_step: u64,
               _total_steps: u64,
               mut rand: &mut Rng,
               workspace: &mut OwnedWorkspace<E>,
               model: &mut ::ModelStats<M>,
               observer: &Ob)
               -> Result<()>
  {
    use linxal::solve_linear::SolveLinear;

    const MIN_GROUP: usize = 64;

    let n = workspace.mean.dim();
    let obs_size = self.observation_operator.dim().0;
    let ec_e: <E as LinxalImplScalar>::RealPart =
      NumCast::from(self.ensemble_count).unwrap();
    let s = (ec_e - <E as LinxalImplScalar>::RealPart::one())
      .sqrt();

    let mut normal = Normal::new(Zero::zero(), One::one());

    // predict

    workspace.ensemble_predict
      .fill(Zero::zero());
    workspace.estimator_predict
      .fill(Zero::zero());
    workspace.resample_forcing((), Diagonal::Multiple(self.sigma.view()),
                               &mut normal,
                               &mut rand);
    workspace.ensemble_predict(current_step, model);

    workspace.centered_ensemble
      .fill(Zero::zero());
    let estimator = workspace.estimator_predict.view();
    for i in 0..self.ensemble_count {
      let ensemble = workspace.ensemble_predict.row(i);

      let mut dest = workspace.centered_ensemble.column_mut(i);
      dest.assign(&ensemble);
      dest -= &estimator;
      dest.par_mapv_inplace(|v| v / s);
    }

    {
      let mut chat = workspace.covariance_predict.view_mut();
      general_mat_mul(One::one(),
                      &workspace.centered_ensemble,
                      &workspace.centered_ensemble.t(),
                      Zero::zero(),
                      &mut chat);
    }

    {
      let mut z = workspace.sqrt_transform_intermediate.view_mut();

      {
        let t = workspace.sqrt_transform.view_mut();
        let (mut left, _) = t.split_at(Axis(1), obs_size);
        let mut left_ext = left.view_mut();

        general_mat_mul(One::one(),
                        &self.observation_operator,
                        &workspace.centered_ensemble,
                        Zero::zero(),
                        &mut left_ext.view_mut().reversed_axes());

        left_ext
          .axis_iter_mut(Axis(1))
          .into_par_iter()
          .zip(self.gamma.axis_iter(Axis(0)).into_par_iter().map(|v| v[()] * v[()]))
          .with_min_len(MIN_GROUP)
          .for_each(|(mut l, gamma)| {
            l.mapv_inplace(|v| v / gamma);
          });

        general_mat_mul(One::one(),
                        &left_ext,
                        &self.observation_operator,
                        Zero::zero(),
                        &mut z.slice_mut(s![.., ..n as isize]));
      }

      {
        let mut t = workspace.sqrt_transform.view_mut();
        general_mat_mul(One::one(),
                        &z.slice(s![.., ..n as isize]),
                        &workspace.centered_ensemble,
                        Zero::zero(),
                        &mut t);
        // copy `t` to `z`.
        z.assign(&t);
        z.par_map_inplace(|v| { *v += One::one(); } );

        let mut sol = Eigen::compute_into(z.to_owned(),
                                          true, false)
          .expect("can't eigendecomp");
        let (d, v) = sol.values_and_left_vectors();
        let d = d.map(|v| {
          if v.re < Zero::zero() {
            Zero::zero()
          } else {
            v.re.sqrt()
          }
        });
        t.fill(Zero::zero());
        t.axis_iter_mut(Axis(1))
          .into_par_iter()
          .zip(v.t().axis_iter(Axis(1)).into_par_iter())
          .zip(d.axis_iter(Axis(0)).into_par_iter())
          .with_min_len(MIN_GROUP)
          .for_each(|((mut t, v), lambda)| {
            t.assign(&v);
            t.mapv_inplace(|s| {
              if !lambda[()].is_zero() {
                s * lambda[()]
              } else {
                Zero::zero()
              }
            });
          });
      }
    }


    // consider workspace.sqrt_transform_intermediate to be garbage
    // from here on.
    // compute X from xhat * T
    {
      let l = workspace.centered_ensemble.view();
      let r = workspace.sqrt_transform.view();
      let mut t = workspace.transformed_centered_ensemble.view_mut();
      general_mat_mul(One::one(),
                      &l, &r,
                      Zero::zero(),
                      &mut t);
    }

    // analyze
    {
      let mut d = workspace.innovation.view_mut();
      assert!(observer.observe_into(current_step, d.view_mut()));
      let mut d2 = extend_dim_mut(&mut d, false);

      let mhat = workspace.estimator_predict.view();
      let mhat2 = extend_dim_ref(&mhat, false);
      let neg_one = NumCast::from(-1).unwrap();
      general_mat_mul(neg_one,
                      &self.observation_operator,
                      &mhat2,
                      One::one(),
                      &mut d2);
    }

    {
      let mut chat_ht = workspace.kalman_gain.view_mut();
      let chat = workspace.covariance_predict.view();
      let h = self.observation_operator.view();

      general_mat_mul(One::one(),
                      &chat, &h.t(),
                      Zero::zero(),
                      &mut chat_ht);

      let mut s = workspace.s.view_mut();
      s.fill(Zero::zero());
      for i in 0..self.observation_operator.dim().0 {
        s[[i, i]] = E::one() * self.gamma[i] * self.gamma[i];
      }

      general_mat_mul(One::one(),
                      &h, &chat_ht.view(),
                      One::one(),
                      &mut s);
      SolveLinear::compute_multi_into(s.view_mut().reversed_axes(),
                                      chat_ht.view_mut().reversed_axes())
        .expect("analysis solve failed");
    }

    {
      let mut m = workspace.mean.view_mut();
      m.assign(&workspace.estimator_predict);
      let mut m2 = extend_dim_mut(&mut m, false);
      let k = workspace.kalman_gain.view();
      let d = workspace.innovation.view();

      general_mat_mul(One::one(),
                      &k, &extend_dim_ref(&d, false),
                      One::one(),
                      &mut m2);
    }

    {
      let mut e = workspace.ensembles.view_mut();
      let m = workspace.mean
        .broadcast((self.ensemble_count, workspace.mean.dim()))
        .unwrap();
      let x = workspace.transformed_centered_ensemble
        .view();

      e.assign(&m);
      e.scaled_add(NumCast::from(s).unwrap(), &x.t());
    }

    {
      let mut c = workspace.covariance.view_mut();
      let x = workspace.transformed_centered_ensemble.view();
      general_mat_mul(One::one(),
                      &x, &x.t(),
                      Zero::zero(),
                      &mut c);
    }

    Ok(())
  }
}