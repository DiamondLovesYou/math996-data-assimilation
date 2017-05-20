

extern crate ninety_five as l95;
extern crate sixty_three as l63;
extern crate na_discrete_filtering as na_df;
extern crate ndarray as nd;
extern crate ndarray_parallel as nd_par;
extern crate rand;
extern crate num_traits;
extern crate gnuplot;
extern crate util;

use gnuplot::{AxesCommon};

use rand::SeedableRng;

use num_traits::{Float, One, Zero};

use nd::prelude::*;
use nd_par::prelude::*;
use nd::{Zip};
use nd::linalg::{general_mat_vec_mul};

use util::progress::ReportingIterator;

use na_df::{Model, SimpleObserver};
use na_df::particle::implicit::{Init, Workspace, Algo};
use na_df::utils::{Diagonal, make_2d_randn};

use l95::L95Model;
use l63::L63Model;

const A0: nd::Axis = nd::Axis(0);
const A1: nd::Axis = nd::Axis(1);

const STEPS: usize = 80000;

struct Params {
  n: usize,
  tau: f64,
  /// used in the G function.
  mu: f64,
  seed: Vec<u64>,
}

#[derive(Debug, Clone)]
struct GModel {
  sigma: Array<f64, Ix1>,
}
impl na_df::Model<f64> for GModel {
  fn workspace_size(&self) -> usize { 0 }
  fn run_model(&self, _t: u64,
               _ws: ArrayViewMut<f64, Ix1>,
               mean: ArrayView<f64, Ix1>,
               mut out: ArrayViewMut<f64, Ix1>) {
    Zip::from(&mut out)
      .and(mean)
      .and(&self.sigma)
      .par_apply(|mut out, &mean, &sigma| {
        *out = sigma * mean;
      });
  }
}

const PLOT_MODULUS: usize = (STEPS / 10000) as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DataSet {
  Truth,
  Observation,
  Assimilation,
}

struct Run<FM>
  where FM: na_df::Model<f64>,
{
  rand: rand::Isaac64Rng,
  steps: usize,

  truth: Array<f64, Ix2>,
  observations: Array<f64, Ix2>,

  means: Array<f64, Ix2>,
  covariances: Array<f64, Ix3>,

  algo: Algo<FM, GModel, f64, Array<f64, Ix2>, SimpleObserver<nd::OwnedRepr<f64>>>,
  workspace: Workspace<f64>,
}
impl<FM> Run<FM>
  where FM: na_df::Model<f64>,
{
  fn run(&mut self) {
    println!("Running assimilation");

    let mut iter = ReportingIterator::new(0..self.steps - 1,
                                          From::from("Assimilation"));
    for i in iter {
      self.algo.step(&mut self.workspace, None, &mut self.rand)
        .unwrap_or_else(|_| {
          panic!("step {} failed!", i);
        });

      self.means.subview_mut(A0, (i + 1) as _)
        .assign(&self.workspace.mean);
      self.covariances.subview_mut(A0, (i + 1) as _)
        .assign(&self.workspace.covariance);
    }
  }

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

fn for_model<M>(Params { n, tau, mu, seed, }: Params,
                f_model: M) -> Run<M>
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

  let g_model = GModel {
    sigma: sigma.to_owned(),
  };

  let y = [1.0];
  let y = aview1(&y);
  let y = y.broadcast(n).unwrap();
  let mut workspace: Array<f64, Ix1> = Array::zeros(1);

  let mut truth: Array<f64, Ix2> = Array::zeros((STEPS as usize, n));
  truth.subview_mut(A0, 0)
    .assign(&y);

  let iter = ReportingIterator::new(1..STEPS, From::from("Truth gen"),);
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

  let iter = ReportingIterator::new(1..STEPS, From::from("Data gen"),);
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
  let capq: Array<f64, Ix2> = Array::eye(k);
  let initer = Init {
    mean: mean,
    covariance: covariance,
    particle_count: 20,
    cap_q: capq.into_diag(),
    total_steps: STEPS as _,
    obs_op: h,
    observer: observer,
  };

  let (mut algo, mut workspace) = Algo::new(initer,
                                            f_model,
                                            g_model,
                                            &mut rand)
    .expect("implicit pf init failed");

  algo.step(&mut workspace, None, &mut rand)
    .expect("bootstrap step failed");

  let r = Run {
    rand: rand,
    steps: STEPS,

    truth: truth,
    observations: observations,

    means: {
      let mut m = Array::zeros((STEPS as _, n));
      m.subview_mut(A0, 0).assign(&workspace.mean);
      m
    },
    covariances: {
      let mut c = Array::zeros((STEPS as _, n, n));
      c.subview_mut(A0, 0).assign(&workspace.covariance);
      c
    },

    algo: algo,
    workspace: workspace,
  };

  r
}

pub fn sixty_three() {
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
  run.run();
  run.plot3d(DataSet::Assimilation);
  //run.plot_axis(DataSet::Truth, 2);
  //run.plot_axis(DataSet::Assimilation, 2);
  //run.plot_axis(DataSet::Truth, 1);
  //run.plot_x_axis(DataSet::Assimilation);
}

pub fn ninety_five() {
}

pub fn main() {
  sixty_three();
  //ninety_five();
}