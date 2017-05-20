
extern crate ninety_five as l95;
extern crate sixty_three as l63;
extern crate na_discrete_filtering as na_df;
extern crate ndarray as nd;
extern crate ndarray_parallel as nd_par;
extern crate rand;
extern crate num_traits;
extern crate gnuplot;
extern crate util;

use gnuplot::AxesCommon;

use std::ops::Deref;

use num_traits::{Zero, One, Float};

use rand::{Isaac64Rng, SeedableRng};
use rand::distributions::{Normal, IndependentSample, };

use nd::prelude::*;
use nd_par::prelude::*;
use nd::{Zip,};
use nd::linalg::{general_mat_vec_mul};

use util::StateSteps;
use util::progress::ReportingIterator;

use na_df::{Model, SimpleObserver, Algorithm, Workspace, particle};
use na_df::ensemble::EnsembleCommonInit;
use na_df::particle::std::{Algo, Init};
use na_df::utils::{Diagonal, make_2d_randn};

use l95::L95Model;
use l63::L63Model;

const A0: nd::Axis = nd::Axis(0);
const A1: nd::Axis = nd::Axis(1);

const STEPS: usize = 80_000;
const RNG_SEED: [u64; 3] = [1, 2, 3];
const TOL: f64 = 0.000000005;
const TAU: f64 = 0.0004;
const THRESH: &'static [f64] = &[TOL; 3];

const PLOT_MODULUS: usize = (STEPS / 10000) as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DataSet {
  Truth,
  Observation,
  Assimilation,
}

struct Run {
  rand: rand::Isaac64Rng,
  steps: usize,

  truth: Array<f64, Ix2>,
  observations: Array<f64, Ix2>,

  states: StateSteps,
}
impl Run {
  fn plot3d(&self, which: DataSet) {
    let mut fig = gnuplot::Figure::new();
    {
      let mut a = fig.axes3d();
      a.set_z_grid(true);
      a.set_y_grid(true);
      a.set_x_grid(true);
      match which {
        DataSet::Assimilation => {
          let x = self.means.subview(A1, 0);
          let x = x
            .axis_iter(A0)
            .enumerate()
            .filter(|&(idx, _)| {
              idx % PLOT_MODULUS == 0
            })
            .map(|(_, v)| { v[()] });
          let y = self.means.subview(A1, 1);
          let y = y
            .axis_iter(A0)
            .enumerate()
            .filter(|&(idx, _)| {
              idx % PLOT_MODULUS == 0
            })
            .map(|(_, v)| { v[()] });
          let z = self.means.subview(A1, 2);
          let z = z
            .axis_iter(A0)
            .enumerate()
            .filter(|&(idx, _)| {
              idx % PLOT_MODULUS == 0
            })
            .map(|(_, v)| { v[()] });
          a.lines(x, y, z, &[]);
        },
        DataSet::Truth => {
          a.lines(self.truth.subview(A1, 0).axis_iter(A0).map(|v| v[()] ),
                  self.truth.subview(A1, 1).axis_iter(A0).map(|v| v[()] ),
                  self.truth.subview(A1, 2).axis_iter(A0).map(|v| v[()] ),
                  &[]);
        },
        _ => unimplemented!(),
      }
    }
    fig.set_terminal("wxt", "");
    fig.show();
  }

  fn plot_axis(&self, which: DataSet, axis: usize) {
    let mut fig = gnuplot::Figure::new();
    {
      let mut a = fig.axes2d();
      let range = 0..self.steps;
      match which {
        DataSet::Assimilation => {
          a.lines(range,
                  self.means.subview(A1, axis).axis_iter(A0).map(|v| v[()] ),
                  &[]);
        },
        DataSet::Truth => {
          a.lines(range,
                  self.truth.subview(A1, axis).axis_iter(A0).map(|v| v[()] ),
                  &[]);
        },
        _ => unimplemented!(),
      }
    }
    fig.set_terminal("wxt", "");
    fig.show();
  }
}
impl Deref for Run
{
  type Target = StateSteps;
  fn deref(&self) -> &StateSteps { &self.states }
}

/// XXX duplicated from implicit-pf
fn for_model<M>(Params { n, tau, mu, seed, }: Params,
                f_model: M) -> Run
  where M: Model<f64>,
{
  let h = arr2(&[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ]);

  let m = n;
  let k = h.dim().0;

  let dt = tau;
  let dt_sqrt = dt.sqrt();

  let mut rand = rand::Isaac64Rng::from_seed(&seed[..]);

  let gamma = 0.2;
  let sigma = 2.0;

  let gamma = {
    let mut t = Array::zeros(k);
    t.fill(gamma);
    t
  };
  let gamma_dt_sqrt = {
    let mut t = gamma.to_owned();
    t.par_mapv_inplace(|v| v * dt_sqrt );
    t
  };
  let sigma = {
    let mut t = Array::zeros(m);
    t.fill(sigma);
    t
  };
  let sigma_dt_sqrt = {
    let mut t = sigma.to_owned();
    t.par_mapv_inplace(|v| v * dt_sqrt );
    t
  };

  let y = [1.0];
  let y = aview1(&y);
  let y = y.broadcast(n).unwrap();
  let mut workspace: Array<f64, Ix1> = Array::zeros(1);

  let mut truth: Array<f64, Ix2> = Array::zeros((STEPS as usize, n));
  truth.subview_mut(A0, 0)
    .assign(&y);

  let iter = ReportingIterator::new(1..STEPS, From::from("Truth gen"));
  for i in iter {
    let t = i as usize;
    let (prev, mut next) = truth.view_mut().split_at(A0, t);
    let prev2 = prev.subview(A0, t - 1);
    let mut next2 = next.subview_mut(A0, 0);

    f_model.run_model(i as _,
                      workspace.view_mut(),
                      prev2.view(),
                      next2.view_mut());

    let r = make_2d_randn((m, 1), Diagonal::from(sigma_dt_sqrt.view()),
                          &mut rand);
    Zip::from(&mut next2)
      .and(&prev2)
      .and(&r.subview(A1, 0))
      .apply(|next2, &prev2, &r| {
        *next2 *= dt;
        *next2 += prev2;
        *next2 += r;
      });
  }
  let truth = truth;
  let mut observations = Array::zeros((STEPS as _, k));

  let mut observed = Array::zeros(k);
  // create the data:

  let iter = ReportingIterator::new(1..STEPS, From::from("Data gen"));
  for step in iter {
    let j = step as usize;

    let (before, mut after) = observations
      .view_mut()
      .split_at(A0, j);

    let mut dest = after.subview_mut(A0, 0);
    let prev = before.subview(A0, j - 1);

    let r: Array<f64, Ix2> =
      make_2d_randn((k, 1),
                    Diagonal::from(gamma_dt_sqrt.view()),
                    &mut rand);

    general_mat_vec_mul(One::one(),
                        &h,
                        &truth.subview(A0, j),
                        Zero::zero(),
                        &mut observed);

    Zip::from(&mut dest)
      .and(&prev)
      .and(&observed)
      .and(&r.subview(A1, 0))
      .par_apply(|mut dest, &prev, &observed, &noise| {
        *dest = prev + dt * observed + noise;
      });
  }

  let observer = SimpleObserver(observations.clone());
  let mean = make_2d_randn((m, 1),
                           Diagonal::from(mu),
                           &mut rand)
    .into_shape(m)
    .unwrap();
  let covariance = Array::<f64, Ix2>::eye(n);
  let init = Init {
    common: EnsembleCommonInit {
      mean: y.view(),
      covariance: covariance.view(),
      observation_operator: h.view(),
      gamma: gamma.view(),
      sigma: sigma.view(),
      ensemble_count: 1000,
    },
    model_workspace_size: f_model.workspace_size(),
  };

  let mut model: na_df::ModelStats<_> = From::from(f_model);

  let mut r = Run {
    rand: rand,
    steps: STEPS,

    truth: truth,
    observations: observations,

    states: StateSteps::new(STEPS + 1, init.ensemble_count, 3),
  };

  let algo = Algo::init(&init,
                        &mut r.rand,
                        &mut model,
                        &observer,
                        STEPS as u64)
    .unwrap();
  let mut workspace: particle::std::Workspace<_> =
    particle::std::Workspace::alloc(init, &mut r.rand, STEPS as u64)
      .unwrap();
  r.states.store_state(0, &workspace);

  println!("Starting algorithm loop");
  for i in 0..STEPS as u64 {



  }

  println!("Running assimilation");

  let mut iter = ReportingIterator::new(0..r.steps - 1,
                                        From::from("Assimilation"));
  for i in iter {
    algo.next_step(i as _, STEPS as u64,
                   &mut r.rand,
                   &mut workspace,
                   &mut model,
                   &observer)
      .expect("algorithm step failed");

    r.states.store_state(i as usize + 1, &workspace);
  }

  r
}

fn sixty_three() {
  const N: usize = 3;
  const TAU: f64 = 0.001;

  let params = Params {
    n: N,
    tau: TAU,
    mu: (1.0 / 2.0).sqrt(),
    seed: vec![1, 2, 3],
  };

  let f_model: L63Model = L63Model::default();
  println!("Setup");
  let mut run = for_model(params, f_model);
  run.plot3d(DataSet::Truth);
  run.plot3d(DataSet::Assimilation);
}

pub fn main() {
  sixty_three();


/*  let current_exe = ::std::env::current_exe().unwrap();
  let out_dir = current_exe.parent().unwrap().join("../../graphs");



  if false {
    for i in 0..21 {
      let v = states.covariances.subview(Axis(0), i);
      println!("{:?}", v);
    }
  }

  let av = |idx: usize| { states.means.subview(Axis(1), idx) };
  let x = av(0); let xi = x.indexed_iter();
  let y = av(1); let yi = y.indexed_iter();
  let z = av(2); let zi = z.indexed_iter();

  let xu_error = || {
    xi.clone().map(|(j, m)| {
      let co = states.covariances
        .subview(Axis(0), j);

      let co_diag = co.diag();
      m + co_diag[0].sqrt();

      m
    })
  };
  let yu_error = || {
    yi.clone().map(|(j, m)| {
      let co = states.covariances
        .subview(Axis(0), j);
      let co_diag = co.diag();
      m + co_diag[1].sqrt();

      m
    })
  };
  let zu_error = || {
    zi.clone().map(|(j, m)| {
      let co = states.covariances
        .subview(Axis(0), j);
      let co_diag = co.diag();
      m + co_diag[2].sqrt();

      m
    })
  };

  let xb_error = || {
    xi.clone().map(|(j, m)| {
      let co = states.covariances
        .subview(Axis(0), j);
      let co_diag = co.diag();
      m - co_diag[0].sqrt();

      m
    })
  };
  let yb_error = || {
    yi.clone().map(|(j, m)| {
      let co = states.covariances
        .subview(Axis(0), j);
      let co_diag = co.diag();
      m - co_diag[1].sqrt();

      m
    })
  };
  let zb_error = || {
    zi.clone().map(|(j, m)| {
      let co = states.covariances
        .subview(Axis(0), j);
      let co_diag = co.diag();
      m - co_diag[2].sqrt();

      m
    })
  };

  let mut f = gnuplot::Figure::new();
  f.set_terminal("wxt",
                 format!("{}/sixty-three-etkf-observations.png",
                         out_dir.display()).as_str());
  {
    let mut a = f.axes3d();


    a.lines(x.iter(),
            y.iter(),
            z.iter(),
            &[gnuplot::PlotOption::Caption("ensemble mean"),
              gnuplot::PlotOption::Color("magenta"),]);

    a.lines(xu_error(), yu_error(), zu_error(),
            &[gnuplot::PlotOption::Caption("error (upper)"),
              gnuplot::PlotOption::Color("red"),
              gnuplot::PlotOption::LineStyle(gnuplot::DashType::Dash)]);
    a.points(data.observations.subview(Axis(1), 0).iter(),
             data.observations.subview(Axis(1), 1).iter(),
             data.observations.subview(Axis(1), 2).iter(),
             &[gnuplot::PlotOption::Caption("observation"),
               gnuplot::PlotOption::Color("black"),
               gnuplot::PlotOption::PointSymbol('x'),]);
    a.lines(xb_error(), yb_error(), zb_error(),
            &[gnuplot::PlotOption::Caption("error (lower)"),
              gnuplot::PlotOption::Color("red"),
              gnuplot::PlotOption::LineStyle(gnuplot::DashType::Dash)]);
  }
  f.show();

  let mut f = gnuplot::Figure::new();
  f.set_terminal("wxt",
                 format!("{}/sixty-three-etkf-error.png",
                         out_dir.display()).as_str());
  {
    let mut a = f.axes2d();
    a.lines(data.x.iter(), y,
            &[gnuplot::PlotOption::Caption("ensemble mean"),
              gnuplot::PlotOption::Color("magenta"),]);

    a.lines(data.x.iter(), yu_error(),
            &[gnuplot::PlotOption::Caption("error (upper)"),
              gnuplot::PlotOption::Color("red"),
              gnuplot::PlotOption::LineStyle(gnuplot::DashType::Dash)]);
    a.lines(data.x.iter(), yb_error(),
            &[gnuplot::PlotOption::Caption("error (lower)"),
              gnuplot::PlotOption::Color("red"),
              gnuplot::PlotOption::LineStyle(gnuplot::DashType::Dash)]);

    a.points(data.x.iter(),
             data.truth.subview(Axis(1), 2).iter(),
             &[gnuplot::PlotOption::Caption("truth"),
               gnuplot::PlotOption::Color("black"),
               gnuplot::PlotOption::PointSymbol('x'),]);
  }
  f.show();

  let mut f = gnuplot::Figure::new();
  f.set_terminal("wxt",
                 format!("{}/sixty-three-etkf-F-norm.png",
                         out_dir.display()).as_str());
  {
    let mut a = f.axes2d();
    let traces: Vec<_> = states.covariances
      .axis_iter(Axis(0))
      .into_par_iter()
      .map(|co| {
        co.diag().iter().sum()
      })
      .collect();
    let traces = Array::from_shape_vec(traces.len(), traces).unwrap();
    let mut avg_traces = Vec::new();
    let mut sum = 0.0;
    for (j, &tr) in traces.iter().enumerate() {
      sum += tr;
      avg_traces.push(sum / (j as f64));
    }

    a.lines(0..traces.len(), traces.iter(),
            &[gnuplot::PlotOption::Caption("trace"),
              gnuplot::PlotOption::Color("blue"),]);
    a.lines(0..traces.len(), avg_traces.iter(),
            &[gnuplot::PlotOption::Caption("avg trace"),
              gnuplot::PlotOption::Color("red"),]);
  }
  f.show();
  */
}

struct Params {
  n: usize,
  tau: f64,
  /// used in the G function.
  mu: f64,
  seed: Vec<u64>,
}