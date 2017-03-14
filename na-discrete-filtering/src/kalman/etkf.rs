
use nd::{Array, ArrayBase, ArrayView, ArrayViewMut, Ix2, Ix1,
         Axis, DataMut, ViewRepr};
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

use {Initializer, Algorithm, Workspace};
use utils::CholeskyLDL;

#[derive(Debug, Clone)]
pub struct Init<'a, E>
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
impl<'a, E> Init<'a, E>
  where E: LinxalScalar,
{ }
impl<'a, E,> Initializer for Init<'a, E>
  where E: LinxalScalar,
{ }

#[derive(Debug)]
pub struct OwnedWorkspace<E>
  where E: LinxalScalar,
{
  mean: Array<E, Ix1>,
  covariance: Array<E, Ix2>,
  ensembles: Array<E, Ix2>,

  ///
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
  where E: LinxalScalar + CholeskyLDL + From<f64> + NumCast + SolveLinear,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalScalar>::RealPart, Output = E> + Sub<<E as LinxalScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalScalar>::RealPart, Output = E> + Div<<E as LinxalScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalScalar>::RealPart> + DivAssign<<E as LinxalScalar>::RealPart>,
{
  fn alloc(i: Init<'a, E>, mut rand: &mut Rng, _: u64) -> OwnedWorkspace<E> {
    use linxal::types::Symmetric;

    let Init {
      initial_mean,
      initial_covariance,
      observation_operator,
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

    let (_, d) = CholeskyLDL::compute(&initial_covariance, Symmetric::Lower)
      .expect("cholesky factorization failed");

    let d: Vec<E> = d.into_raw_vec();
    let scale: E = d.into_iter()
      .filter(|v| !v.is_zero() )
      .fold(E::one(), |p, v| p * v.mag() );

    {
      let mut first = ensembles.view_mut();
      let mut r = {
        let mut r: Array<E, Ix2> = ArrayBase::zeros((ensemble_count, n));
        for i in 0..ensemble_count {
          for j in 0..n {
            r[[i,j]] = From::from(normal.ind_sample(&mut rand));
          }
        }

        r
      };
      r.mapv_inplace(|v| v * scale);
      let t = &initial_mean.broadcast((ensemble_count, n)).unwrap() + &r;
      first.assign(&t);
    }

    let ec_e: E = NumCast::from(ensemble_count).unwrap();
    let m0 = ensembles
      .sum(Axis(0))
      .mapv_into(|v| v / ec_e);

    m.assign(&m0);

    OwnedWorkspace {
      mean: m,
      covariance: c,
      ensembles: ensembles,

      innovation: ArrayBase::zeros(n),
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
    }
  }
}

#[derive(Debug)]
pub struct State<'a, E>
  where E: LinxalScalar,
{
  pub mean: ArrayView<'a, E, Ix1>,
  pub covariance: ArrayView<'a, E, Ix2>,
  pub ensembles: ArrayView<'a, E, Ix2>,
}
impl<'a, E> ::State<'a> for State<'a, E>
  where E: LinxalScalar,
{
  type Workspace = OwnedWorkspace<E>;
  fn current(ws: &'a OwnedWorkspace<E>) -> State<'a, E> {
    State {
      mean: ws.mean.view(),
      covariance: ws.covariance.view(),
      ensembles: ws.ensembles.view(),
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

pub struct Algo<'a, E, F1, F2>
  where E: LinxalScalar,
{
  ensemble_count: usize,
  gamma: ArrayView<'a, E::RealPart, Ix1>,
  sigma: ArrayView<'a, E::RealPart, Ix1>,
  observation_operator: ArrayView<'a, E, Ix2>,

  _i: PhantomData<(F1, F2)>,
  _s: PhantomData<()>,
  _m: PhantomData<()>,
}

impl<'init, 'state, F1, F2, E>
Algorithm for Algo<'init, E, F1, F2>
  where F1: for<'r, 's> Fn(ArrayView<'r, E, Ix1>, ArrayViewMut<'s, E, Ix1>),
        F2: FnMut(u64) -> Option<Array<E, Ix1>>,
        E: LinxalScalar + CholeskyLDL + From<f64> + NumCast + SolveLinear,
        E: Add<E, Output = E> + Sub<E, Output = E>,
        E: AddAssign<E> + SubAssign<E>,
        E: Add<<E as LinxalScalar>::RealPart, Output = E> + Sub<<E as LinxalScalar>::RealPart, Output = E>,
        E: Mul<E, Output = E> + Div<E, Output = E>,
        E: MulAssign<E> + DivAssign<E>,
        E: Mul<<E as LinxalScalar>::RealPart, Output = E> + Div<<E as LinxalScalar>::RealPart, Output = E>,
        E: MulAssign<<E as LinxalScalar>::RealPart> + DivAssign<<E as LinxalScalar>::RealPart>,
{
  type Init  = Init<'init, E>;
  type WS    = OwnedWorkspace<E>;
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
      _s: PhantomData,
      _m: PhantomData,
    }
  }

  /// This could use some cleaning.
  fn next_step(&self,
               current_step: u64,
               _total_steps: u64,
               mut rand: &mut Rng,
               workspace: &mut OwnedWorkspace<E>,
               model: &mut Model<E, F1, F2>)
               -> Result<(), ()>
  {
    use linxal::solve_linear::SolveLinear;
    use linxal::factorization::cholesky::*;
    use linxal::types::Symmetric;

    use nd::linalg::general_mat_mul;

    let n = workspace.mean.dim();
    let ec_e: <E as LinxalScalar>::RealPart = NumCast::from(self.ensemble_count).unwrap();
    let s = (ec_e - <E as LinxalScalar>::RealPart::one()).sqrt();

    let normal = Normal::new(Zero::zero(), One::one());

    // predict

    workspace.ensemble_predict
      .fill(Zero::zero());
    workspace.estimator_predict
      .fill(Zero::zero());
    {
      {
        let mut r = workspace.forcing.view_mut();
        for i in 0..r.dim().0 {
          for j in 0..r.dim().1 {
            r[[i,j]] = From::from(normal.ind_sample(&mut rand));
            r[[i,j]] *= self.sigma[i];
          }
        }
      }
      let r = workspace.forcing.view();

      let mut estimator = workspace.estimator_predict.view_mut();
      for i in 0..self.ensemble_count {
        let mut ensemble_dest = workspace.ensemble_predict.row_mut(i);
        (model.model)(workspace.ensembles.row(i).view(),
                      ensemble_dest.view_mut());
        model.calls += 1;

        let forcing = r.column(i);
        for j in 0..n {
          ensemble_dest[j] += forcing[j];

          estimator[j] += ensemble_dest[j];
        }
      }
      estimator.mapv_inplace(|v| v / ec_e);
    }

    workspace.centered_ensemble
      .fill(Zero::zero());
    let estimator = workspace.estimator_predict.view();
    for i in 0..self.ensemble_count {
      let ensemble = workspace.ensemble_predict.row(i);

      let mut dest = workspace.centered_ensemble.column_mut(i);
      dest.assign(&ensemble);
      dest -= &estimator;
      dest.mapv_inplace(|v| v / s );
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
        let (mut left, _) = t.split_at(Axis(1), n);
        let mut left_ext = left.view_mut();

        general_mat_mul(One::one(),
                        &workspace.centered_ensemble.t(),
                        &self.observation_operator.t(),
                        Zero::zero(),
                        &mut left_ext);


        for i in 0..n {
          let gamma = self.gamma[i] * self.gamma[i];
          left_ext.subview_mut(Axis(1), i)
            .mapv_inplace(|v| v * gamma.recip());
        }

        general_mat_mul(One::one(),
                        &left_ext,
                        &self.observation_operator,
                        Zero::zero(),
                        &mut z.slice_mut(s![.., ..1]));
      }

      {
        let mut t = workspace.sqrt_transform.view_mut();
        t.fill(Zero::zero());
        for i in 0..t.dim().0 {
          t[[i,i]] = One::one();
        }

        general_mat_mul(One::one(),
                        &z.slice(s![.., ..1]),
                        &workspace.centered_ensemble,
                        One::one(),
                        &mut t);
        // copy `t` to `z`.
        z.assign(&t);
      }

      // Now, invert t and then factorize via cholesky
      let mut t = workspace.sqrt_transform.view_mut();
      t.fill(Zero::zero());
      for i in 0..self.ensemble_count {
        t[[i, i]] = One::one();
      }
      SolveLinear::compute_multi_into(z.view_mut(),
                                      t.view_mut())
        .expect("inversion failed (singular T_j? -- this is impossible(TM))");

      // the inverse is now in `t`. now we upper cholesky factorize `t`:
      Cholesky::compute_into(t.view_mut(),
                             Symmetric::Upper)
        .expect("cholesky factorization failed");
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
    let observation = (model.next_observation)(current_step);
    let observation = observation.expect("TODO");
    {
      let mut d = workspace.innovation.view_mut();
      d.assign(&observation);

      let mut d2 = extend_dim_mut(&mut d, false);

      let mhat = workspace.estimator_predict.view();
      let neg_one = NumCast::from(-1).unwrap();
      general_mat_mul(neg_one,
                      &self.observation_operator,
                      &mhat.broadcast((mhat.dim(), 1)).unwrap(),
                      One::one(),
                      &mut d2);
    }

    {
      let mut chat_ht = workspace.kalman_gain.view_mut();
      let chat = workspace.covariance_predict.view();
      let h = self.observation_operator.view();

      general_mat_mul(One::one(),
                      &chat, &h,
                      Zero::zero(),
                      &mut chat_ht);

      let mut s = workspace.s.view_mut();
      s.fill(Zero::zero());
      for i in 0..self.observation_operator.dim().1 {
        s[[i,i]] = E::one() * self.gamma[i] * self.gamma[i];
      }

      general_mat_mul(One::one(),
                      &h, &chat_ht.view(),
                      One::one(),
                      &mut s);

      SolveLinear::compute_multi_into(s, chat_ht)
        .expect("analysis solve failed");
    }

    {
      let mut m = workspace.mean.view_mut();
      m.assign(&workspace.estimator_predict);
      let mut m2 = extend_dim_mut(&mut m, false);
      let k = workspace.kalman_gain.view();
      let d = workspace.innovation
        .broadcast((workspace.innovation.dim(),
                    1))
        .unwrap();

      general_mat_mul(One::one(),
                      &k, &d,
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

fn extend_dim_mut<D>(d: &mut ArrayBase<D, Ix1>, t: bool)
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