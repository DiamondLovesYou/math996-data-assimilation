
use std::f64::EPSILON;
use std::f64::consts::PI;
use std::iter::{Sum, Product};
use std::ops::{AddAssign, SubAssign,
               DivAssign, MulAssign,};

use linxal::types::{LinxalImplScalar};

use nd::{Array, Ix1, Ix2, Axis, Zip, aview1, aview_mut1,
         Data, ArrayBase};
use nd::linalg::{general_mat_vec_mul, general_mat_mul};
use nd_par::prelude::*;
use linxal::solve_linear::general::SolveLinear;
use linxal::factorization::Cholesky;
use linxal::eigenvalues::{Eigen, Solution};

use rand::Rng;
use rand::distributions::{Normal, IndependentSample,
                          Weighted, WeightedChoice};
use num_traits::{Zero, One, NumCast};
use num_complex::Complex;

use error::{Result};
use {ModelStats, LinearizedOperator, Observer, Model};
use utils::{Sqrt, Exp, par_conj_t, PartialEqWithinTol,
            make_2d_randn, Diagonal};

const A0: Axis = Axis(0);
const A1: Axis = Axis(1);

pub struct Init<D1, D2, D3, E, Obs, ObsOp>
  where E: LinxalImplScalar,
        Obs: Observer<E, Ix1>,
        ObsOp: LinearizedOperator<E, Ix2>,
        D1: Data<Elem = E>,
        D2: Data<Elem = E>,
        D3: Data<Elem = E>,
{
  pub mean: ArrayBase<D1, Ix1>,
  pub covariance: ArrayBase<D2, Ix2>,
  pub particle_count: usize,
  pub obs_op: ObsOp,
  pub observer: Obs,
  // k x k diagonal
  pub cap_q: ArrayBase<D3, Ix1>,
  pub total_steps: u64,
}

#[derive(Debug)]
pub struct Workspace<E>
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

  /// diagonal matrix, m x m, per particle. Stored as (G^* * G)^-1
  pub cap_gn: Array<E, Ix2>,
  /// vector, m, per particle
  pub cap_fn: Array<E, Ix2>,
}

#[derive(Debug, Clone)]
pub struct Diagnostics<E>
  where E: LinxalImplScalar,
{
  pub particle_weights: Array<E, Ix1>,
  pub histogram: Array<usize, Ix1>,
}

pub struct Algo<FM, GM, E, ObsOp, Obs>
  where E: LinxalImplScalar,
        FM: Model<E>,
        GM: Model<E>,
        ObsOp: LinearizedOperator<E, Ix2>,
        Obs: Observer<E, Ix1>,
{
  step: u64,
  pub total_steps: u64,
  pub particle_count: usize,
  pub obs_op: ObsOp,
  pub f: ModelStats<FM>,
  pub g: ModelStats<GM>,
  pub observer: Obs,
  pub delta: E,
  pub sqrt_delta: E,
  /// k x k, diagonal
  pub cap_q_sqrd_inv: Array<E, Ix1>,
  pub cap_q_sqrd:     Array<E, Ix1>,
}
impl<FM, GM, E, ObsOp, Obs> Algo<FM, GM, E, ObsOp, Obs>
  where E: LinxalImplScalar,
        FM: Model<E>,
        GM: Model<E>,
        ObsOp: LinearizedOperator<E, Ix2>,
        Obs: Observer<E, Ix1>,
{

}

impl<FM, GM, E, ObsOp, Obs> Algo<FM, GM, E, ObsOp, Obs>
  where E: LinxalImplScalar<Complex = Complex<<E as LinxalImplScalar>::RealPart>>,
        E: NumCast + Sqrt + Exp + SolveLinear + Sum + Product,
        E: Send + Sync + MulAssign + DivAssign + SubAssign + AddAssign,
        E: Cholesky + Eigen,
        E: PartialEqWithinTol<E, E>,
        E::Complex: Send + Sync,
        Complex<<E as LinxalImplScalar>::RealPart>: LinxalImplScalar,
        <E as LinxalImplScalar>::RealPart: Send + Sync + Sqrt,
        <E as LinxalImplScalar>::RealPart: MulAssign,
        FM: Model<E>,
        GM: Model<E>,
        ObsOp: LinearizedOperator<E, Ix2>,
        Obs: Observer<E, Ix1>,
{
  pub fn new<R, D1, D2, D3>(init: Init<D1, D2, D3, E, Obs, ObsOp>,
                            f: FM,
                            g: GM,
                            mut rng: &mut R) -> Result<(Algo<FM, GM, E, ObsOp, Obs>,
                                                        Workspace<E>)>
    where R: Rng,
          D1: Data<Elem = E>,
          D2: Data<Elem = E>,
          D3: Data<Elem = E>,
  {
    let particle_count = init.particle_count;
    let m = init.mean.dim();
    let k = init.obs_op.operator_output_dim();
    let one: E = One::one();

    let ksi = Array::zeros((particle_count, m));
    // initialize the workspace and algo structures, then run a single
    // step to initialize the particles

    let Init {
      mean,
      covariance,
      particle_count,
      obs_op,
      cap_q,
      observer,
      total_steps,
    } = init;

    let delta = one / NumCast::from(total_steps).unwrap();

    let mut cap_q_sqrd = cap_q.to_owned();
    cap_q_sqrd.par_mapv_inplace(|v| v.cj() * v );
    let mut cap_q_sqrd_inv = cap_q_sqrd.to_owned();
    cap_q_sqrd_inv.par_mapv_inplace(|v| one / v );

    let mut particles = Array::zeros((particle_count, m));
    let sol = Eigen::compute_into(covariance.to_owned(),
                                  false, false)
      .expect("can't eigendecomp initial_covariance");
    let Solution {
      values,
      ..
    } = sol;
    let mut d = Array::zeros(values.dim());
    Zip::from(&mut d)
      .and(&values)
      .par_apply(|mut d, &eigen| {
        *d = eigen.re;
        *d = d._sqrt();
      });

    let r = make_2d_randn((particle_count, m),
                          Diagonal::from(d.view()),
                          &mut rng);

    Zip::from(&mut particles)
      .and(&r)
      .and(&mean.broadcast((particle_count, m)).unwrap())
      .par_apply(|mut p, &r, &m| {
        *p = m + E::from_real(r);
      });

    let algo = Algo {
      step: 0,
      total_steps: total_steps,
      particle_count: particle_count,
      obs_op: obs_op,
      observer: observer,
      f: From::from(f),
      g: From::from(g),
      delta: delta,
      sqrt_delta: delta._sqrt(),
      cap_q_sqrd_inv: cap_q_sqrd_inv,
      cap_q_sqrd: cap_q_sqrd,
    };

    let workspace = Workspace {
      mean: mean.to_owned(),
      covariance: covariance.to_owned(),
      particles: particles,

      observation: Array::zeros(k),

      fm_ws: Array::zeros((particle_count, algo.f.model.workspace_size())),
      gm_ws: Array::zeros((particle_count, algo.g.model.workspace_size())),

      ksi: ksi,

      cap_gn: Array::zeros((particle_count, m)),
      cap_fn: Array::zeros((particle_count, m)),
    };

    Ok((algo, workspace))
  }

  pub fn current_step(&self) -> u64 {
    self.step
  }


  pub fn step<R>(&mut self, ws: &mut Workspace<E>,
                 mut diag: Option<&mut Option<Diagnostics<E>>>,
                 mut rng: &mut R) -> Result<()>
    where R: Rng,
  {
    use linxal::factorization::Cholesky;
    use linxal::types::Symmetric;

    let t = self.step;
    self.step += 1;
    let particle_count = self.particle_count;
    let m = ws.mean.dim();
    let k = self.obs_op.operator_output_dim();
    assert_eq!(m, self.obs_op.operator_input_dim());
    let neg_one: E = NumCast::from(-1).unwrap();
    let one: E = One::one();

    let normal = Normal::new(Zero::zero(), One::one());

    Zip::from(ws.cap_fn.axis_iter_mut(A0))
      .and(ws.fm_ws.axis_iter_mut(A0))
      .and(ws.particles.axis_iter(A0))
      .par_apply(|mut cap_fn, ws, mean| {
        self.f.model.run_model(t, ws, mean,
                               cap_fn.view_mut());
        Zip::from(&mut cap_fn)
          .par_apply(|mut cap_fn| {
            *cap_fn *= self.delta;
          });
      });
    self.f.calls += particle_count as u64;

    Zip::from(ws.cap_gn.axis_iter_mut(A0))
      .and(ws.gm_ws.axis_iter_mut(A0))
      .and(ws.particles.axis_iter(A0))
      .par_apply(|mut cap_gn, ws, mean| {
        self.g.model.run_model(t, ws, mean, cap_gn.view_mut());
        Zip::from(&mut cap_gn)
          .par_apply(|mut cap_gn| {
            *cap_gn *= cap_gn.cj();
            *cap_gn = self.delta / *cap_gn;
          });
      });
    self.g.calls += particle_count as u64;

    //println!("G_n: {:?}", ws.cap_gn);

    for mut ksi in ws.ksi.axis_iter_mut(A0) {
      for mut i in ksi.axis_iter_mut(A0) {
        i[()] = NumCast::from(normal.ind_sample(&mut rng)).unwrap();
      }
    }

    assert!(self.observer.observe_into(t, ws.observation.view_mut()));

    ws.particles.fill(E::zero());
    let cap_x0 = [E::zero(); 1];
    let cap_x0 = aview1(&cap_x0);
    let cap_x0 = cap_x0.broadcast(m).unwrap();
    let mut obs_op_jacobi0 = Array::zeros((k, m));
    self.obs_op.eval_jacobian_at(cap_x0.view(),
                                 obs_op_jacobi0.view_mut())?;
    let obs_op_jacobi0 = obs_op_jacobi0;

    let b_np1 = ws.observation.view();
    let mut z0 = b_np1.to_owned();
    {
      let mut t = Array::zeros(b_np1.dim());
      self.obs_op.eval_at(cap_x0.view(), t.view_mut())?;
      z0 -= &t;
    }
    let z0 = z0;

    let mut weights = Array::zeros(self.particle_count);
    let weight_div = (2.0 * PI).powf(m as f64 / 2.0);
    let weight_div = NumCast::from(weight_div).unwrap();

    // `cap_xn` is the particle
    Zip::from(ws.ksi.axis_iter(A0))
      .and(ws.cap_gn.axis_iter(A0))
      .and(ws.cap_fn.axis_iter(A0))
      .and(ws.particles.axis_iter_mut(A0))
      .and(weights.axis_iter_mut(A0))
      .apply(|ksi, cap_gn, cap_fn,
              mut cap_xn, mut weight| {
        let mut cap_xj = Array::zeros(m);
        let mut obs_op_jacobis = obs_op_jacobi0
          .broadcast((2, k, m))
          .unwrap()
          .to_owned();

        let (last_obs_op_jacobi, obs_op_jacobi) =
          obs_op_jacobis
            .view_mut()
            .split_at(A0, 1);
        let mut last_obs_op_jacobi = last_obs_op_jacobi.into_subview(A0, 0);
        let mut obs_op_jacobi = obs_op_jacobi.into_subview(A0, 0);

        let mut obs_op_jacobis_conj = Array::zeros((2, m, k));
        let (last_obs_op_jacobi_conj, obs_op_jacobi_conj) =
          obs_op_jacobis_conj
            .view_mut()
            .split_at(A0, 1);
        let mut last_obs_op_jacobi_conj = last_obs_op_jacobi_conj
          .into_subview(A0, 0);
        let mut obs_op_jacobi_conj = obs_op_jacobi_conj
          .into_subview(A0, 0);

        par_conj_t(&mut obs_op_jacobi_conj.view_mut(),
                   &obs_op_jacobi.view());


        let mut sigma_invs = Array::zeros((2, m, m));
        let (last_sigma_inv, sigma_inv) = sigma_invs.view_mut()
          .split_at(A0, 1);
        let mut last_sigma_inv = last_sigma_inv.into_subview(A0, 0);
        let mut sigma_inv      = sigma_inv.into_subview(A0, 0);


        let mut zjs    = Array::zeros((2, k));
        let (last_zj, zj) = zjs.view_mut()
          .split_at(A0, 1);
        let mut last_zj = last_zj.into_subview(A0, 0);
        let mut zj      = zj.into_subview(A0, 0);
        zj.assign(&z0);


        let mut mhatjs = Array::zeros((2, m));
        let (last_mhatj, mhatj_b) = mhatjs.view_mut()
          .split_at(A0, 1);
        let mut last_mhatj = last_mhatj
          .into_subview(A0, 0);
        let mut mhatj_b = mhatj_b
          .into_subview(A0, 0);

        let mut mhatj_t = Array::zeros(zj.dim());

        let mut lastcf: Option<Array<E, Ix2>> = None;

        let mut iterations = 0;
        for i in 0.. {
          iterations = i;

          if i != 0 {
            self.obs_op.eval_jacobian_at(cap_xj.view(),
                                         obs_op_jacobi.view_mut())
              .expect("observation operator jacobi error");

            self.obs_op.eval_at(cap_xj.view(), zj.view_mut())
              .expect("observation operator error");
            Zip::from(&mut zj)
              .and(&b_np1)
              .par_apply(|z, &b| {
                *z = b - *z;
              });

            general_mat_vec_mul(One::one(),
                                &obs_op_jacobi.view(),
                                &cap_xj.view(),
                                One::one(),
                                &mut zj);

            par_conj_t(&mut obs_op_jacobi_conj.view_mut(),
                       &obs_op_jacobi.view());
          }

          // First, calculate Sigma_j, then factorize via Cholesky.
          let mut t = Array::zeros((k, k));
          t.diag_mut().assign(&self.cap_q_sqrd_inv);
          let mut t2 = Array::zeros((m, k));
          general_mat_mul(One::one(),
                          &obs_op_jacobi_conj,
                          &t,
                          Zero::zero(),
                          &mut t2);
          general_mat_mul(One::one(),
                          &t2,
                          &obs_op_jacobi,
                          Zero::zero(),
                          &mut sigma_inv);

          Zip::from(sigma_inv.diag_mut())
            .and(&cap_gn)
            .par_apply(|sigma, &cap_gn| {
              *sigma += cap_gn;
            });

          let cf = Cholesky::compute(&sigma_inv, Symmetric::Lower)
            .unwrap_or_else(|e| {
              panic!("iteration {}: failed to compute Cholesky factors: {:?}",
                     i, e);
            });

          // calculate m hat by solving
          Zip::from(&mut mhatj_t)
            .and(&self.cap_q_sqrd_inv)
            .and(&zj)
            .par_apply(|mhatj, &cap_q_sqrd_inv, &zj| {
              *mhatj = cap_q_sqrd_inv * zj;
            });

          Zip::from(&mut mhatj_b)
            .and(&cap_xn)
            .and(&cap_fn)
            .and(&cap_gn)
            .par_apply(|mhatj, &cap_xn, &cap_fn, &cap_gn| {
              *mhatj = (cap_xn + cap_fn) * cap_gn;
            });

          general_mat_vec_mul(One::one(),
                              &obs_op_jacobi_conj,
                              &mhatj_t,
                              One::one(),
                              &mut mhatj_b);

          let mut mhatj = SolveLinear::compute(&sigma_inv,
                                               &mhatj_b)
            .expect("particle solve failed");

          // store mhatj in mhatj_b
          // reuse mhatj for cap_xjp1
          mhatj_b.assign(&mhatj);

          general_mat_vec_mul(One::one(),
                              &cf,
                              &ksi,
                              One::one(),
                              &mut mhatj);
          let cap_xjp1 = mhatj;

          let tol: E = NumCast::from(EPSILON * 10.0).unwrap();
          let stop = cap_xj.view()
            .partial_eq_within_tol(&cap_xjp1.view(), tol);

          if i + 1 >= 16 && !stop {
            println!("cap_xn: {:?}", cap_xn);
            println!("ksi: {:?}", ksi);
            println!("X_j: {:?}", cap_xj);
            println!("X_{{j+1}}: {:?}", cap_xjp1);
            println!("b^{{n+1}}: {:?}", b_np1);
            println!("lastcf: {:?}", lastcf);

            panic!("not converging to X^(n+1)");
          }

          if stop {
            // use the last iteration's values below!
            // don't copy cap_xj into cap_xn yet, we need
            // the previous value still to calculate phi below.
            break;
          } else {
            cap_xj.assign(&cap_xjp1);
            lastcf = Some(cf);

            last_zj.assign(&zj);
            last_sigma_inv.assign(&sigma_inv);
            last_mhatj.assign(&mhatj_b);
            last_obs_op_jacobi.assign(&obs_op_jacobi);
            last_obs_op_jacobi_conj.assign(&obs_op_jacobi_conj);
          }
        }

        if iterations > 1 {
          println!("completed implicit pf forward iteration in {} steps",
                   iterations + 1);
        }

        let capj: E = lastcf.as_ref().unwrap()
          .diag()
          .iter()
          .map(|&v| one / v )
          .product();
        let capj = capj.mag();
        let capj = E::from_real(capj);
        //println!("|J| = {}", capj);

        let mut cap_k = Array::zeros((k, k));
        let mut t = Array::zeros((m, m));
        Zip::from(&mut t.diag_mut())
          .and(&cap_gn)
          .apply(|t, &cap_gn| {
            *t = one / cap_gn;
          });
        let mut t2 = Array::zeros((k, m));
        general_mat_mul(One::one(),
                        &last_obs_op_jacobi,
                        &t,
                        Zero::zero(),
                        &mut t2);
        general_mat_mul(One::one(),
                        &t2,
                        &last_obs_op_jacobi_conj,
                        Zero::zero(),
                        &mut cap_k);

        Zip::from(cap_k.diag_mut())
          .and(&self.cap_q_sqrd)
          .par_apply(|cap_k, &cap_q_sqrd| {
            *cap_k += cap_q_sqrd;
          });

        let eye = Array::eye(k);
        let cap_k_inv = SolveLinear::compute_multi(&cap_k, &eye)
          .expect("cap_k inversion");

        // compute phi:
        let mut lrsides = last_zj.to_owned();
        general_mat_vec_mul(neg_one,
                            &last_obs_op_jacobi,
                            &cap_xn,
                            One::one(),
                            &mut lrsides);
        general_mat_vec_mul(neg_one,
                            &last_obs_op_jacobi,
                            &cap_fn,
                            One::one(),
                            &mut lrsides);
        let rside = lrsides
          .into_shape((k, 1))
          .unwrap();
        let mut lside = Array::zeros((1, k));
        par_conj_t(&mut lside.view_mut(),
                   &rside.view());
        let mut out = Array::zeros((1, k));
        general_mat_mul(One::one(),
                        &lside,
                        &cap_k_inv,
                        Zero::zero(),
                        &mut out);
        let phi = {
          let mut dest: [E; 1] = [Zero::zero(); 1];
          {
            let mut dest_nd = aview_mut1(&mut dest[..])
              .into_shape((1, 1))
              .unwrap();

            general_mat_mul(One::one(),
                            &out,
                            &rside,
                            Zero::zero(),
                            &mut dest_nd);
          }

          dest[0] / NumCast::from(2.0).unwrap()
        };

        let neg_phi: E = neg_one * phi;
        let phi_exp: E = neg_phi._exp();

        /*let expected_phi = {
          let b = b_np1.view().into_shape((k, 1)).unwrap();
          let mut b_conj = Array::zeros(b.t().dim());
          par_conj_t(&mut b_conj.view_mut(),
                     &b.view());

          let mut out = Array::zeros((1, 1));
          let one_forth = one / NumCast::from(4.0f64).unwrap();
          general_mat_mul(one_forth,
                          &b_conj, &b,
                          Zero::zero(),
                          &mut out);

          out[(0, 0)]
        };
        println!("phi: {}, expected phi: {}", phi, expected_phi);*/

        weight[()] = phi_exp * capj / weight_div;
        cap_xn.assign(&cap_xj);
      });

    // resample:
    let weight_sum = weights.iter().map(|&v| v ).sum();
    weights.par_mapv_inplace(|v| v / weight_sum );

    // update the mean:
    {
      let scale: E = NumCast::from(self.particle_count).unwrap();
      Zip::from(ws.mean.axis_iter_mut(A0))
        .and(ws.particles.axis_iter(A1))
        .par_apply(|mut mean, particles| {
          mean[()] = particles.axis_iter(A0)
            .into_par_iter()
            .zip(weights.axis_iter(A0).into_par_iter())
            .map(|(v, weight)| v[()] * weight[()] )
            .sum();
          mean[()] /= scale;
        });
    }

    let u32_max: E = NumCast::from(u32::max_value() - 1).unwrap();
    let mut weights_u32: Vec<_> = weights
      .iter()
      .map(|&v| {
        let v = v * u32_max;
        let v: u32 = NumCast::from(v).unwrap();
        v
      })
      .enumerate()
      .map(|(idx, v)| {
        Weighted {
          weight: v,
          item: idx,
        }
      })
      .collect();

    let wc = WeightedChoice::new(&mut weights_u32[..]);
    {
      let mut histogram = if let Some(ref mut diag) = diag {
        if let &mut &mut Some(ref mut diag) = diag {
          Some(&mut diag.histogram)
        } else {
          **diag = Some(Diagnostics {
            particle_weights: Array::zeros(weights.dim()),
            histogram: Array::zeros(self.particle_count),
          });
          diag.as_mut()
            .map(|d| &mut d.histogram )
        }
      } else {
        None
      };
      for i in 0..self.particle_count {
        let new_particle_idx = wc.ind_sample(&mut rng);

        if let Some(ref mut hist) = histogram {
          hist[new_particle_idx] += 1;
        }

        if i == new_particle_idx { continue; }

        let p = ws.particles.view_mut();
        // to get around Rust's ownership rules:
        let (mut to, from) = if i < new_particle_idx {
          let (left, right) = p.split_at(A0, i + 1);
          (left.into_subview(A0, i),
           right.into_subview(A0, new_particle_idx - i - 1))
        } else {
          let (left, right) = p.split_at(A0, i);
          (right.into_subview(A0, 0),
           left.into_subview(A0, new_particle_idx))
        };

        to.assign(&from);
      }
    }

    // update the covariance
    if false {
      let mut transformed_particles = ws.particles.clone();
      Zip::from(&mut transformed_particles)
        .and(&ws.mean.broadcast((self.particle_count, m)).unwrap())
        .par_apply(|tp, &mean| {
          *tp -= mean;
        });

      let dim = ws.particles.dim();
      let mut transformed_particles_star = Array::zeros((dim.1, dim.0));
      par_conj_t(&mut transformed_particles_star.view_mut(),
                 &transformed_particles.view());

      let s = one / NumCast::from(self.particle_count - 1).unwrap();
      general_mat_mul(s,
                      &transformed_particles,
                      &transformed_particles_star,
                      Zero::zero(),
                      &mut ws.covariance);
    }

    if let Some(ref mut diag) = diag {
      if let &mut &mut Some(ref mut diag) = diag {
        diag.particle_weights.assign(&weights);
      } else {
        **diag = Some(Diagnostics {
          particle_weights: weights,
          histogram: Array::zeros(self.particle_count),
        });
      }
    }

    Ok(())
  }
}

#[test]
fn null_model_high_dim() {
  use nd::{ArrayView, ArrayViewMut};
  use rand::{SeedableRng, Isaac64Rng};

  use {SimpleObserver};

  use gnuplot::*;

  use std::iter::repeat;

  const N: usize = 100;
  const PARTICLES: usize = 1000;

  struct NullModel;
  impl Model<f64> for NullModel {
    fn workspace_size(&self) -> usize { 0 }
    fn run_model(&self, _t: u64, _ws: ArrayViewMut<f64, Ix1>,
                 _mean: ArrayView<f64, Ix1>,
                 mut out: ArrayViewMut<f64, Ix1>) {
      out.fill(0.0);
    }
  }

  struct UnitModel;
  impl Model<f64> for UnitModel {
    fn workspace_size(&self) -> usize { 0 }
    fn run_model(&self, _t: u64, _ws: ArrayViewMut<f64, Ix1>,
                 _mean: ArrayView<f64, Ix1>,
                 mut out: ArrayViewMut<f64, Ix1>) {
      out.diag_mut().fill(1.0);
    }
  }

  let mut rand = Isaac64Rng::from_seed(&[1, 2, 3]);

  let h = Array::eye(N);
  let cap_q = Array::eye(N);
  let mean = Array::zeros(N);
  let covariance = Array::eye(N);

  let mut observation = make_2d_randn((N, 1),
                                      Diagonal::from(1.0),
                                      &mut rand);
  observation.fill(1.0);

  let init = Init {
    mean: mean,
    covariance: covariance,
    particle_count: PARTICLES,
    cap_q: cap_q.into_diag(),
    total_steps: 1,
    obs_op: h,
    observer: SimpleObserver(observation.t()),
  };

  let (mut algo, mut ws) = Algo::new(init,
                                     NullModel,
                                     UnitModel,
                                     &mut rand)
    .expect("init failed");

  let mut diag = None;
  algo.step(&mut ws, Some(&mut diag), &mut rand)
    .expect("step failed");

  let diag = diag.unwrap();
  let hist = &diag.histogram;
  println!("weights: {:?}", diag.particle_weights);
  println!("histogram: {:?}", hist);

  let mut fig = Figure::new();
  fig.set_terminal("wxt", "");
  {
    let mut a = fig.axes2d();
    a.fill_between(0..hist.len(),
                   repeat(0.0).take(hist.len()),
                   hist.iter().map(|&v| v ),
                   &[]);
  }
  fig.show();

  let mean_is_correct = ws.mean
    .view()
    .partial_eq_within_std_tol(&observation.subview(A1, 0));
  assert!(mean_is_correct, "{:?}",
          if !mean_is_correct {
            let o: Vec<_> = observation.subview(A1, 0)
              .axis_iter(A0)
              .zip(ws.mean.axis_iter(A0))
              .map(|(obs, mean)| (obs[()], mean[()]) )
              .collect();
            Some(o)
          } else {
            None
          });

  let first = diag.particle_weights[0];
  for weight in diag.particle_weights.iter().skip(1) {
    assert!(first.partial_eq_within_std_tol(weight));
  }
}
