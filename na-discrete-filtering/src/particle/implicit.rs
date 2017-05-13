
use linxal::types::{LinxalImplScalar};

use nd::{Array, ArrayView, ArrayViewMut, Ix1, Ix2, Axis, Zip, aview1};

use rand::Rng;
use rand::distributions::{Normal, IndependentSample};

use ensemble::{EnsembleCommonInit};
use error::{Result, Error};
use {ModelStats, LinearizedObservationOperator, Observer};

pub struct Init<'a, E, Obs>
  where E: LinxalImplScalar,
        Obs: LinearizedObservationOperator<E>,
{
  pub mean: ArrayView<'a, E, Ix1>,
  pub covariance: ArrayView<'a, E, Ix2>,
  pub particle_count: usize,
  pub obs_op: Obs,
  pub ksi: ArrayView<'a, E, Ix1>,
  // k x k diagonal
  pub cap_q: ArrayView<'a, E, Ix1>,
  pub delta: E,
}

pub struct Workspace<'a, E>
  where E: LinxalImplScalar,
{
  pub mean: Array<E, Ix1>,
  pub covariance: Array<E, Ix2>,
  pub particles: Array<E, Ix2>,

  pub observation: Array<E, Ix1>,

  pub fm_ws: Array<E, Ix2>,
  pub gm_ws: Array<E, Ix2>,

  // vector of k N(0, 1) Gaussian variables, per particle.
  pub ksi: Array<E, Ix2>,

  // vector of k N(0, delta) Gaussian variables. Resampled every step.
  pub cap_v: Array<E, Ix2>,
  // vector of k N(0, 1) Gaussian variables. Resampled every step.
  pub cap_w: Array<E, Ix1>,

  /// diagonal matrix, m x m, per particle. Stored as (G^* * G)^-1
  pub cap_gn: Array<E, Ix2>,
  /// vector, m, per particle
  pub cap_fn: Array<E, Ix2>,
}

enum ResamplerDisc {
  V, W,
}

impl<E> ResampleForcing<E> for Workspace<E>
  where E: LinxalImplScalar + From<f64>,
        E: MulAssign<<E as LinxalImplScalar>::RealPart>,
{
  type Disc = ResamplerDisc;
  fn forcing_view_mut(&mut self, disc: ResamplerDisc) -> ArrayViewMut<E, Ix2> {
    match disc {
      ResamplerDisc::V => {
        self.cap_v.view_mut()
      },
      ResamplerDisc::W => {
        self.cap_w.view_mut()
      },
    }
  }
}

pub struct Algo<FM, GM, E, ObsOp, Obs>
  where E: LinxalImplScalar,
        FM: Model<E>,
        GM: Model<E>,
        ObsOp: LinearizedObservationOperator<E>,
        Obs: Observer<E>,
{
  pub particle_count: usize,
  pub obs_op: ObsOp,
  pub f: ModelStats<FM>,
  pub g: ModelStats<GM>,
  pub observer: Obs,
  pub delta: E,
  pub sqrt_delta: E,
  /// k x k, diagonal
  pub cap_q_sqrd_inv: Array<E, Ix1>
}
impl<FM, GM, E, ObsOp, Obs> Algo<FM, GM, E, ObsOp, Obs>
  where E: LinxalImplScalar + NumCast,
{
  pub fn new<R>(init: Init<E, Obs>,
                f: FM,
                g: GM,
                rng: &mut R) -> Result<(Algo<FM, GM, E, ObsOp, Obs>,
                                        Workspace<E>)>
    where R: Rng,
  {
    let particle_count = init.particle_count;
    let m = init.mean.dim();
    let k = <ObsOp as LinearizedObservationOperator<E>>::space_dim();

    let ksi = Array::zeros((particle_count, k));
  }

  pub fn step(&mut self, t: u64, ws: &mut Workspace<E>) -> Result<()> {
    use linxal::factorization::Cholesky;
    use linxal::util::external::conj_t;
    use linxal::types::Symmetric;

    fn max<T: PartialOrd>(a: T, b: T) -> T { if a > b { a } else { b } }

    let particle_count = self.particle_count;
    let m = ws.mean.dim();
    let k = <Obs as LinearizedObservationOperator<E>>::space_dim();
    let neg_one = NumCast::from(-1).unwrap();
    let neg_half: E = NumCast::from(-1.0f64/2.0).unwrap();

    let mut normal = Normal::new(Zero::zero(), One::one());

    Zip::from(ws.cap_fn.axis_iter_mut(Axis(0)))
      .and(ws.fm_ws.axis_iter_mut(Axis(0)))
      .and_broadcast(ws.mean.view())
      .par_apply(|mut cap_fn, mut ws, mean| {
        self.f.model.run_model(t, ws, mean, cap_fn.view_mut());
        for mut v in cap_fn.iter_mut() {
          v *= self.delta;
        }
      });
    self.f.calls += particle_count as u64;

    Zip::from(ws.cap_gn.axis_iter_mut(Axis(0)))
      .and(ws.gm_ws.axis_iter_mut(Axis(0)))
      .and_broadcast(ws.mean.view())
      .par_apply(|mut cap_gn, mut ws, mean| {
        self.f.model.run_model(t, ws, mean, cap_gn.view_mut());
        for mut v in cap_gn.iter_mut() {
          *v *= self.sqrt_delta;
          *v = ((*v).cj() * (*v)).recip();
        }
      });
    self.g.calls += particle_count as u64;

    for mut ksi in ws.ksi.axis_iter_mut(Axis(0)) {
      for mut i in ksi.axis_iter_mut(Axis(0)) {
        i[()] = NumCast::from(normal.sample(&mut rng)).unwrap();
      }
    }

    assert!(self.observer.observe_into(t, ws.observation.view_mut()));

    ws.particles.fill(E::zero());
    let cap_x0 = aview1(&[E::zero(); 1]).broadcast(m).unwrap();
    let mut obs_op_jacobi0 = Array::zeros((k, m));
    self.obs_op.eval_jacobian_at(cap_x0.view(),
                                 obs_op_jacobi0.view_mut())?;
    let obs_op_jacobi0 = obs_op_jacobi0;

    let b_np1 = ws.observation.view();

    // `cap_x` is the particle

    Zip::from(ws.ksi.axis_iter(Axis(0)))
      .and(ws.cap_gn.axis_iter(Axis(0)))
      .and(ws.cap_fn.axis_iter(Axis(0)))
      .and(ws.particles.axis_iter_mut(Axis(0)))
      .par_apply(|ksi, cap_gn_sqrd_inv, cap_fn, mut cap_xn| {
        let mut cap_xj = Array::zeros(m);
        let mut obs_op_jacobi = Array::zeros((k, m));
        obs_op_jacobi.assign(&obs_op_jacobi0);

        let mut sigma = Array::zeros((k, m));
        let mut mhatj = Array::zeros(m);
        let mut zj    = Array::zeros(m);

        for i in 0.. {
          assert!(i < 16, "not converging to X^(n+1)");

          let obs_op_jacobi_conj_t = conj_t(&obs_op_jacobi);
          // First, calculate Sigma_j, then factorize via Cholesky.
          sigma.assign(&obs_op_jacobi_conj_t);
          sigma *= &obs_op_jacobi;
          Zip::from(sigma.axis_iter_mut(Axis(0)))
            .and(self.cap_q_sqrd_inv.axis_iter(Axis(0)))
            .and(cap_gn_sqrd_inv.axis_iter(Axis(0)))
            .par_apply(|mut sigma, cap_q, cap_gn_sqrd_inv| {
              sigma.mapv_inplace(|v| v * cap_q + cap_gn_sqrd_inv );
            });

          let cf = Cholesky::compute(&sigma, Symmetric::Lower)
            .expect("failed to compute Cholesky factors");

          // calculate m hat by solving
          mhatj.assign(&cap_xn);
          mhatj.scaled_add(E::one(), &cap_fn);
          Zip::from(mhatj.axis_iter_mut(Axis(0)))
            .and(self.cap_q_sqrd_inv.axis_iter(Axis(0)))
            .par_apply(|mut mhatj, cap_q_sqrd_inv| {
              mhatj[()] *= cap_q_sqrd_inv[()];
            });

          zj.assign(&b_np1);

          let stop = cap_xj.subview(Axis(0), 0)
            .partial_eq_within_std_tol(&cap_xj.subview(Axis(0), 1));
          if stop {
            break;
          }

          self.obs_op.eval_jacobian_at(cap_xj.subview(Axis(0), (i + 1) % 2),
                                       obs_op_jacobi.view_mut());
        }


      });

    Ok(())
  }
}