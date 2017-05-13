use std::cell::{RefCell, Cell};
use std::f64::consts::{E, PI};
use std::io::stdin;

use num_traits::Float;
use num_complex::Complex;

use nd::{Array, ArrayView, ArrayViewMut, Ix1, Ix2, Ix3, Ix4, Axis};
use nd::{aview1};
use nd_par::prelude::*;
use rayon::prelude::*;

use gsl::{Value, SfLegendreNorm};
use gsl::legendre::associated_polynomials::{legendre_array_n, legendre_array_index,
                                            legendre_alp_deriv, legendre_alp};
use roots::{SimpleConvergency, find_root_newton_raphson};

use af;

/// https://climatedataguide.ucar.edu/climate-model-evaluation/common-spectral-model-grid-resolutions
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SpectralTruncation {
  M42, M63, M106, M170, M255, M382, M799,
}
impl SpectralTruncation {
  fn m(&self) -> usize {
    match self {
      &SpectralTruncation::M42 => 42,
      &SpectralTruncation::M63 => 63,
      &SpectralTruncation::M106 => 106,
      &SpectralTruncation::M170 => 170,
      &SpectralTruncation::M255 => 255,
      &SpectralTruncation::M382 => 382,
      &SpectralTruncation::M799 => 799,
    }
  }
  fn grid_size(&self) -> (usize, usize) {
    let j = match self {
      &SpectralTruncation::M42 => 64,
      &SpectralTruncation::M63 => 96,
      &SpectralTruncation::M106 => 160,
      &SpectralTruncation::M170 => 256,
      &SpectralTruncation::M255 => 320,
      &SpectralTruncation::M382 => 576,
      &SpectralTruncation::M799 => 800,
    };

    (j * 2, j)
  }
  /*fn diffusion_coefficient(&self) -> f64 {
    match self {
      &SpectralTruncation::M42 => 0.5 * (10.0).powi(16),
      &SpectralTruncation::M63 => 1.0 * (10.0).powi(15),
      &SpectralTruncation::M106 => 1.25 * (10.0).powi(14),
      &SpectralTruncation::M170 => 2.0 * (10.0).powi(13),
      &SpectralTruncation::M213 => 8.0 * (10.0).powi(12),
    }
  }
  fn time_step(&self) -> f64 {
    match self {
      &SpectralTruncation::M42 => 1200.0,
      &SpectralTruncation::M63 => 900.0,
      &SpectralTruncation::M106 => 600.0,
      &SpectralTruncation::M170 => 450.0,
      &SpectralTruncation::M213 => 360.0,
    }
  }*/
}
/// From, but using not the improved versions presented:
/// Fast and Accurate Computation of Gauss-Legendre and
/// Gauss-Jacobi Quadrature Nodes and Weights
/// N. Hale, A. Townsend
fn find_alp_roots_and_weights(lat_size: usize, mut cache: ArrayViewMut<f64, Ix3>) -> (Vec<f64>, Vec<f64>, Array<f64, Ix3>) {
  let mut roots = vec![0.0f64; lat_size];
  let mut weights = vec![0.0f64; lat_size];
  let n = lat_size as f64;
  let array_n = legendre_array_n(lat_size);

  let conv = SimpleConvergency {
    eps: E,
    max_iter: 32,
  };

  let mut cache: Array<f64, Ix3> = Array::zeros((lat_size, 2, array_n));

  cache.axis_iter_mut(Axis(0))
    .into_par_iter()
    .zip(roots.par_iter_mut())
    .zip(weights.par_iter_mut())
    .enumerate()
    .for_each(|(lat_k, ((mut cache, mut roots), mut weights))| {
      let (root, dp) = {
        let k = lat_k as f64;
        let phik = (k - 0.25) * PI / (n + 0.5);
        let xk1 = 1.0;
        let xk2 = (n - 1.0) / (8.0 * n * n * n);
        let xk3 = 1.0 / (384.0 * n * n * n * n);
        let xk3 = xk3 * (39.0 - 28.0 / (phik.sin() * phik.sin()));
        let xk = (xk1 - xk2 - xk3) * phik.cos();

        let idx = legendre_array_index(lat_k, lat_size);

        let last_x = Cell::new(-1.0);
        let results = RefCell::new(cache.view_mut());

        let calc = |x| {
          if last_x.get() == x { return; }
          let mut rb = results.borrow_mut();
          let (mut results, mut div) = rb.split_mut(Axis(0), 1);
          assert_eq!(legendre_alp_deriv(SfLegendreNorm::SphericalHarmonic,
                                        lat_size, x,
                                        results.as_mut_slice().unwrap(),
                                        div.as_mut_slice().unwrap()),
          Value::Success);
          last_x.set(x);
        };

        let f = |x| {
          calc(x);
          results.borrow()[(0, idx)]
        };
        let df = |x| {
          calc(x);
          div.borrow()[(1, idx)]
        };

        let r = find_root_newton_raphson(xk, &f, &df, &conv)
          .expect("failed to converge to a ALP root");

        (r, df(r))
      };

      *roots = root;
      let w = 2.0 / ((1.0 - root * root) * dp * dp);
      *weights = w;

      let (mut results, mut div) = cache.split_mut(Axis(0), 1);

      assert_eq!(legendre_alp_deriv(SfLegendreNorm::SphericalHarmonic,
                                    lat_size, root,
                                    results.as_mut_slice().unwrap(),
                                    div.as_mut_slice().unwrap()),
      Value::Success);
    });

  (roots, weights, cache)
}

/// parameters
const H0: f64 = 8000.0;
const WE_ZONAL_PHASE_SPEED: f64 = 7.848 * (1.0 / 1000000.0);
const WAVE_NUMBERS: &'static [f64] = &[4.0];
const EARTH_RADIUS: f64 = 6.37122 * 1000000.0;
const EARTH_ANG_VEL: f64 = 7.292 * (1.0 / 100000.0);
const GRAVITY: f64 = 9.80616;
const DIFFISION_COEFFICIENT: f64 = 8.0 * 1_000_000_000_000.0;
/// eta, delta, phi
const VERTEX_VARIABLES: usize = 3;
const DIAGNOSTIC_VARIABLES: usize = 2;
const MODEL_VARIABLES: usize = VERTEX_VARIABLES + DIAGNOSTIC_VARIABLES;
const FOURIER_COEFFICIENTS: usize = 6;
const SPECTRAL_COEFFICIENTS: usize = 3;

const U_IDX: usize = 0;
const V_IDX: usize = 1;

const ETA_V_IDX: usize = 0;
const DELTA_V_IDX: usize = 1;
const PHI_V_IDX: usize = 2;
const U_V_IDX: usize = 3;
const V_V_IDX: usize = 4;

const ETA_FC_IDX: usize = 0;
const DELTA_FC_IDX: usize = 1;
const PHI_FC_IDX: usize = 2;

const ETA_SC_IDX: usize = 0;
const DELTA_SC_IDX: usize = 1;

fn from_nd2_to_af_array<T>(parent_slice: &[T],
                           view: ArrayView<T, Ix2>) -> af::Array
  where T: af::HasAfEnum,
{
  let xs = view.dims().0;
  let ys = view.dims().1;
  let dims = af::Dim4::new(&[xs as u64, ys as u64, 1, 1]);
  let strides = view.strides();
  let mut af_strides = [1; 4];
  for (i, &stride) in strides.iter().enumerate() {
    af_strides[i] = stride as u64;
  }
  let af_strides = af::Dim4::new(&af_strides);
  af::Array::new_strided(parent_slice,
                         0, dims, af_strides)
}
fn from_nd1_to_af_array<T>(parent_slice: &[T],
                           view: ArrayView<T, Ix1>) -> af::Array
  where T: af::HasAfEnum,
{
  let xs = view.dims();
  let dims = af::Dim4::new(&[xs as u64, 1, 1, 1]);
  let strides = view.strides();
  let mut af_strides = [1; 4];
  for (i, &stride) in strides.iter().enumerate() {
    af_strides[i] = stride as u64;
  }
  let af_strides = af::Dim4::new(&af_strides);
  af::Array::new_strided(parent_slice,
                         0, dims, af_strides)
}

struct SweModel {
  spec_trunc: SpectralTruncation,
  weights: Vec<f64>,
  /// "Gaussian latitudes" or the integration nodes
  microjs: Vec<f64>,
  /// j x 2 x [array_n from GSL]
  /// derivative in the 1 in the last dim slot.
  /// Calculate the second derivative on the fly.
  /// Yes, this is a bit blunt. But fuck it; I have 32gb of RAM anyway.
  alpvs: Array<f64, Ix3>,
  delta_t: f64,
}
impl SweModel {
  #[inline(always)]
  fn alp_coeffs(&self, m: usize, n: usize, j: usize) -> (f64, f64) {
    let idx = legendre_array_index(n, m);
    let d = self.alpvs[(j, 1, idx)];
    let uj = self.microjs[j];
    (self.alpvs[(j, 0, idx)], (1.0 - uj * uj) * d)
  }
}
impl na_df::Model<f64> for SweModel {
  fn workspace_size(&self) -> usize {
    0
  }
  fn run_model(&self, step: u64,
               mut workspace: ArrayViewMut<f64, Ix1>,
               flat_state: ArrayView<f64, Ix1>,
               mut flat_out: ArrayViewMut<f64, Ix1>) {
    let (long_size, lat_size) = self.spec_trunc.grid_size();
    let state: ArrayView<_, Ix3> = flat_state
      .into_shape((2, lat_size, VERTEX_VARIABLES, long_size))
      .expect("grid reshape failed");
    let mut out: ArrayViewMut<_, Ix3> = flat_out
      .into_shape((2, lat_size, VERTEX_VARIABLES, long_size))
      .expect("out grid reshape failed");

    let m = self.spec_trunc.m();

    let state_slice = state.as_slice().unwrap();

    #[inline(always)]
    fn coriolis(n: usize, m: usize) -> f64 {
      if n == 1 && m == 0 {
        EARTH_ANG_VEL / (0.375).sqrt()
      } else {
        0.0
      }
    }

    #[inline(always)]
    fn gauss_lat(microj: usize) -> f64 {
      let u = microj as f64;
      u.sin()
    }

    let pure_i = C64::new(0.0, 1.0);
    let a0 = Axis(0);
    let a1 = Axis(1);
    let a2 = Axis(2);

    const A_FC_IDX: usize = 0;
    const B_FC_IDX: usize = A_FC_IDX + 1;
    const C_FC_IDX: usize = B_FC_IDX + 1;
    const D_FC_IDX: usize = C_FC_IDX + 1;
    const E_FC_IDX: usize = D_FC_IDX + 1;
    const ETA_FC_IDX: usize = E_FC_IDX + 1;
    const DELTA_FC_IDX: usize = ETA_FC_IDX + 1;
    const PHI_FC_IDX: usize = DELTA_FC_IDX + 1;

    let mut nonlinear_terms: Array<f64, Ix4> = Array::zeros((2, 9, lat_size, long_size));
    nonlinear_terms.axis_iter_mut(Axis(0))
      .into_par_iter()
      .zip(state.axis_iter(Axis(0)).into_par_iter())
      .for_each(|(mut nlt, state)| {
        let (mut a, mut nlt2) = nlt.view_mut().split_at(Axis(0), 1);
        let (mut b, mut nlt3) = nlt2.split_at(Axis(0), 1);
        let (mut c, mut nlt4) = nlt3.split_at(Axis(0), 1);
        let (mut d, mut nlt5) = nlt4.split_at(Axis(0), 1);
        let (mut e, mut nlt6) = nlt5.split_at(Axis(0), 1);
        let (mut eta, mut nlt7) = nlt6.split_at(a0, 1);
        let (mut delta, mut nlt8) = nlt7.split_at(a0, 1);
        let (mut phi, _) = nlt8.split_at(a0, 1);

        eta.assign(&s.subview(bx, ETA_V_IDX));
        delta.assign(&s.subview(bx, DELTA_V_IDX));
        phi.assign(&s.subview(bx, PHI_V_IDX));

        let ax = Axis(0);
        let bx = Axis(1);
        a.axis_iter_mut(bx)
          .into_par_iter()
          .zip(s.subview(bx, U_IDX).axis_iter(ax).into_par_iter())
          .zip(s.subview(bx, ETA_IDX).axis_iter(ax).into_par_iter())
          .for_each(|(mut out, a1, a2)| {
            Zip::from(out.axis_iter_mut(ax))
              .and(a1.axis_iter(ax))
              .and(a2.axis_iter(ax))
              .par_apply(|mut out, a1, a2|{
                out[()] = a1[()] * a2[()];
              });
          });
        b.axis_iter_mut(bx)
          .into_par_iter()
          .zip(s.subview(bx, V_IDX).axis_iter(ax).into_par_iter())
          .zip(s.subview(bx, ETA_IDX).axis_iter(ax).into_par_iter())
          .for_each(|(mut out, a1, a2)| {
            Zip::from(out.axis_iter_mut(ax))
              .and(a1.axis_iter(ax))
              .and(a2.axis_iter(ax))
              .par_apply(|mut out, a1, a2|{
                out[()] = a1[()] * a2[()];
              });
          });
        c.axis_iter_mut(bx)
          .into_par_iter()
          .zip(s.subview(bx, U_IDX).axis_iter(ax).into_par_iter())
          .zip(s.subview(bx, PHI_IDX).axis_iter(ax).into_par_iter())
          .for_each(|(mut out, a1, a2)| {
            Zip::from(out.axis_iter_mut(ax))
              .and(a1.axis_iter(ax))
              .and(a2.axis_iter(ax))
              .par_apply(|mut out, a1, a2|{
                out[()] = a1[()] * a2[()];
              });
          });
        d.axis_iter_mut(bx)
          .into_par_iter()
          .zip(s.subview(bx, V_IDX).axis_iter(ax).into_par_iter())
          .zip(s.subview(bx, ETA_IDX).axis_iter(ax).into_par_iter())
          .for_each(|(mut out, a1, a2)| {
            Zip::from(out.axis_iter_mut(ax))
              .and(a1.axis_iter(ax))
              .and(a2.axis_iter(ax))
              .par_apply(|mut out, a1, a2|{
                out[()] = a1[()] * a2[()];
              });
          });
        e.axis_iter_mut(bx)
          .into_par_iter()
          .zip(s.subview(bx, U_IDX).axis_iter(ax).into_par_iter())
          .zip(s.subview(bx, V_IDX).axis_iter(ax).into_par_iter())
          .enumerate()
          .for_each(|(lat_idx, ((mut out, a1), a2))| {

            let micro = (lat_idx as f64).sin();
            let micro_sq = micro * micro;

            Zip::from(out.axis_iter_mut(ax))
              .and(a1.axis_iter(ax))
              .and(a2.axis_iter(ax))
              .par_apply(|mut out, a1, a2|{
                out[()] = a1[()] * a1[()] + a2[()] * a2[()];
                out[()] /= 2.0 * (1 - micro_sq);
              });
          });

      });
    let nonlinear_terms: Array<f64, Ix4> = nonlinear_terms;
    let nlt_slice = nonlinear_terms.as_slice().unwrap();
    let mut ffts = Vec::with_capacity(2 * 9 * lat_size);
    for (i, nlt) in nonlinear_terms.axis_iter(a0).enumerate() {
      for (j, nlt) in nlt.axis_iter(a0).enumerate() {

        if i == 0 && (j < ETA_FC_IDX) { continue; }
        if i == 1 && j == ETA_FC_IDX { continue; }

        for nlt in nlt.axis_iter(a0) {
          let af_arr = from_nd1_to_af_array(nlt_slice, nlt.view());
          let out = af::fft(&af_arr, 1.0, m as _);
          ffts.push(out);
        }
      }
    }

    let ffts_refs: Vec<_> = ffts.iter().collect();
    af::eval_multiple(ffts_refs);

    // XXX wasted space.
    let mut nl_fc: Array<C64, Ix4> = Array::zeros((2, 9, lat_size, m));
    let mut i = 0;
    for (i, mut nlt) in nl_fc.axis_iter_mut(a0).enumerate() {
      for (j, mut nlt) in nlt.axis_iter_mut(a0).enumerate() {

        if i == 0 && (j < ETA_FC_IDX) { continue; }
        if i == 1 && j == ETA_FC_IDX { continue; }

        for mut nlt in nlt.axis_iter_mut(a0) {
          let fft = &ffts[i];
          fft.host(nlt.as_slice_mut().unwrap());
          i += 1;
        }
      }
    }
    let nl_fc: Array<C64, Ix4> = nl_fc;

    // XXX: many values unused. YOLO
    // the first row is the current state
    let mut spectral_coeffs: Array<Complex<f64>, Ix4> =
      Array::zeros((2, SPECTRAL_COEFFICIENTS, m, m));

    {
      // compute the next spectral coeffs
      let mut curr_sc = spectral_coeffs.subview_mut(ax, 1);
      let (mut sc_eta, mut rest) = curr_sc.split_at(ax, ETA_SC_IDX + 1);
      let (mut sc_delta, mut rest) = rest.split_at(ax, DELTA_SC_IDX + 1);
      let (mut sc_phi, _) = rest.split_at(ax, PHI_SC_IDX);

      let four_over_a4 = 4.0 / (EARTH_RADIUS * EARTH_RADIUS * EARTH_RADIUS * EARTH_RADIUS);

      for (m, n, mut sc_eta) in sc_eta.indexed_iter_mut() {
        if m > n { continue; }

        let im = pure_i * (m as f64);
        for (microj_idx, &microj) in self.microjs.iter().enumerate() {
          let (pmnuj, hmnuj) = self.alp_coeffs(m, n, microj_idx);
          let t = nl_fc[(0, ETA_FC_IDX, microj_idx, m)] * pmnuj;
          let t2 = 2.0 * self.delta_t / (EARTH_RADIUS * (1.0 - microj * microj));

          let t = t - im * t2 * nl_fc[(1, A_FC_IDX, microj_idx, m)] * pmnuj;
          let t = t + C64::new(t2, 0.0) * nl_fc[(1, B_FC_IDX, microj_idx, m)] * hmnuj;

          let t = t * self.weights[microj_idx];

          *sc_eta += t;
        }
      }
      // forcing:
      for (m, n, mut sc_eta) in sc_eta.indexed_iter_mut() {
        if m > n { continue; }

        let nf = n as f64;
        let del_sq_coeff = -nf * (nf + 1.0) / (EARTH_RADIUS * EARTH_RADIUS);
        let del_4_coeff = del_sq_coeff * del_sq_coeff;

        let im = pure_i * (m as f64);
        for (microj_idx, &microj) in self.microjs.iter().enumerate() {
          let (pmnuj, hmnuj) = self.alp_coeffs(m, n, microj_idx);

          let t = 2.0 * self.delta_t;
          let t = -t * DIFFISION_COEFFICIENT * del_4_coeff;
          let t = t * (*sc_eta + four_over_a4 * (*sc_eta));

          *sc_eta += t;
        }
      }

      for (m, n, mut sc_delta) in sc_delta.indexed_iter_mut() {
        if m > n { continue; }

        let nf = n as f64;
        let t3 = 2.0 * self.delta_t * (nf * (nf + 1.0)) / (EARTH_RADIUS * EARTH_RADIUS);

        let im = pure_i * (m as f64);
        for (microj_idx, &microj) in self.microjs.iter().enumerate() {
          let (pmnuj, hmnuj) = self.alp_coeffs(m, n, microj_idx);
          let t = nl_fc[(0, DELTA_FC_IDX, microj_idx, m)] * pmnuj;
          let t2 = 2.0 * self.delta_t / (EARTH_RADIUS * (1.0 - microj * microj));

          let t = t + im * t2 * nl_fc[(1, B_FC_IDX, microj_idx, m)] * pmnuj;
          let t = t + t2 * nl_fc[(1, A_FC_IDX, microj_idx, m)] * hmnuj;

          let t4 = nl_fc[(1, PHI_FC_IDX, microj_idx, m)];
          let t4 = t4 + nl_fc[(1, E_FC_IDX, microj_idx, m)];
          let t = t + t3 * t4 * pmnuj;

          let t = t * self.weights[microj_idx];

          *sc_delta += t;
        }
      }
      // forcing:
      for (m, n, mut sc_delta) in sc_delta.indexed_iter_mut() {
        if m > n { continue; }

        let nf = n as f64;
        let del_sq_coeff = -nf * (nf + 1.0) / (EARTH_RADIUS * EARTH_RADIUS);
        let del_4_coeff = del_sq_coeff * del_sq_coeff;

        let im = pure_i * (m as f64);
        for (microj_idx, &microj) in self.microjs.iter().enumerate() {
          let (pmnuj, hmnuj) = self.alp_coeffs(m, n, microj_idx);

          let t = 2.0 * self.delta_t;
          let t = -t * DIFFISION_COEFFICIENT * del_4_coeff;
          let t = t * (*sc_delta + four_over_a4 * (*sc_delta));

          *sc_delta += t;
        }
      }

      for (m, n, mut sc_phi) in sc_phi.indexed_iter_mut() {
        if m > n { continue; }

        for (microj_idx, &microj) in self.microjs.iter().enumerate() {
          let (pmnuj, hmnuj) = self.alp_coeffs(m, n, microj_idx);
          let t = nl_fc[(0, PHI_FC_IDX, microj_idx, m)] * pmnuj;
          let t2 = 2.0 * self.delta_t / (EARTH_RADIUS * (1.0 - microj * microj));

          let t = t - im * t2 * nl_fc[(1, C_FC_IDX, microj_idx, m)] * pmnuj;
          let t = t + t2 * nl_fc[(1, D_FC_IDX, microj_idx, m)] * hmnuj;
          let t = t - 2.0 * self.delta_t * GRAVITY * H0 * hl_fc[(1, DELTA_FC_IDX, microj_idx, m)] * pmnuj;

          let t = t * self.weights[microj_idx];

          *sc_phi += t;
        }
      }
      // forcing:
      for (m, n, mut sc_phi) in sc_phi.indexed_iter_mut() {
        if m > n { continue; }

        let nf = n as f64;
        let del_sq_coeff = -nf * (nf + 1.0) / (EARTH_RADIUS * EARTH_RADIUS);
        let del_4_coeff = del_sq_coeff * del_sq_coeff;

        let im = pure_i * (m as f64);
        for (microj_idx, &microj) in self.microjs.iter().enumerate() {
          let (pmnuj, hmnuj) = self.alp_coeffs(m, n, microj_idx);

          let t = 2.0 * self.delta_t;
          let t = -t * DIFFISION_COEFFICIENT * del_4_coeff;
          let t = t * (*sc_phi);

          *sc_phi += t;
        }
      }
    }


  }
}

fn run_for_trunc(trunc: SpectralTruncation) {
  let (lat_size, long_size) = trunc.grid_size();

  let (roots, weights) = find_alp_roots_and_weights(lat_size);
  let thetajs: Vec<_> = roots
    .into_iter()
    .map(|v| v.asin() )
    .collect();

  let mut grid_points: Array<f64, Ix3> = Array::zeros((long_size, lat_size, 2));
  grid_points.axis_iter_mut(Axis(0))
    .into_par_iter()
    .enumerate()
    .for_each(|(i, mut long)| {
      long.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut long_lat)| {
          let lambda = (i as f64) * PI / (long_size as f64);
          long_lat[0] = lambda;
          long_lat[1] = thetajs[j];
        });
    });
  let grid_points = grid_points;

  // in order: velocity field, geopotential, divergence, and vorticity.
  let grid_state: Array<f64, Ix4> =
    Array::zeros((lat_size, VERTEX_VARIABLES, long_size));


}
