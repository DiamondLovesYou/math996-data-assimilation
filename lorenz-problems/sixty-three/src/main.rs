
extern crate ndarray as nd;
extern crate ndarray_rand as nd_rand;
extern crate ndarray_parallel as nd_par;
extern crate na_discrete_filtering as na_df;
extern crate na_quadrature as na_q;
extern crate util;
extern crate plot_helper;
extern crate rand;
extern crate gnuplot;

use nd::{Array, ArrayView, ArrayViewMut, Ix1, Ix2, Axis};
use nd::{arr2, arr1};
use nd_rand::RandomExt;
use rand::SeedableRng;
use rand::distributions::IndependentSample;

use util::{StateSteps};
use util::data::generate_model_truth_and_observation;

use na_df::{Algorithm, Workspace, Model};
use na_df::kalman::{sirs, etkf};
use na_df::kalman::{EnsembleInit};

use nd_par::prelude::*;

const STEPS: usize = 500;
const RNG_SEED: [u64; 1] = [1];
const TOL: f64 = 0.000000005;
const TAU: f64 = 0.0004;
const THRESH: &'static [f64] = &[TOL; 3];

fn lorenz63(_: f64, y: ArrayView<f64, Ix1>, mut yp: ArrayViewMut<f64, Ix1>) {
  debug_assert!(y.len() == 3);
  debug_assert!(yp.len() == 3);

  const RHO: f64 = 28.0;
  const SIGMA: f64 = 10.0;
  const BETA: f64 = 8.0 / 3.0;

  yp[0] = SIGMA * (y[1] - y[0]);
  yp[1] = y[0] * (RHO - y[2]) - y[1];
  yp[2] = y[0] * y[1] - BETA * y[2];
}

fn main() {
  let mut rand = rand::Isaac64Rng::from_seed(&RNG_SEED[..]);
  let normal = rand::distributions::normal::Normal::new(0.0, 1.0);

  let h = arr2(&[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
  ]);
  let scale = 1.0;
  let gamma = arr1(&[0.2 / scale; 2]);
  let sigma = arr1(&[2.0 / scale; 3]);

  let x = normal.ind_sample(&mut rand);
  let x = x..(x + STEPS as f64 * TAU);
  //let y: Array<f64, Ix1> = Array::random_using(3, normal, &mut rand);
  let y = arr1(&[1.0, 1.0, 1.0]);
  let yp = Array::zeros(3);
  let thresh = arr1(&(THRESH)[..]);
  let data = generate_model_truth_and_observation(lorenz63,
                                                  STEPS,
                                                  x.clone(),
                                                  y.clone(),
                                                  yp.clone(),
                                                  TOL,
                                                  TAU,
                                                  thresh.clone(),
                                                  sigma.view(),
                                                  gamma.view(),
                                                  h.view(),
                                                  &normal,
                                                  &mut rand);

  let initial_covariance = Array::<f64, Ix2>::eye(3) * 10.0;
  let init = EnsembleInit {
    initial_mean: y.view(),
    initial_covariance: initial_covariance.view(),
    observation_operator: h.view(),
    gamma: gamma.view(),
    sigma: sigma.view(),
    model_workspace_size: TruthModel::workspace_size(),
    ensemble_count: 300,
  };

  #[derive(Clone)]
  struct TruthModel;
  impl na_df::Model<f64> for TruthModel {
    fn workspace_size() -> usize { 3 + 3 }
    fn run(&self, i: u64,
           workspace: ArrayViewMut<f64, Ix1>,
           mean: ArrayView<f64, Ix1>,
           out: ArrayViewMut<f64, Ix1>) {
      use nd::aview1;
      use na_q::rke;

      let (mut initial_y, mut initial_yp) = workspace
        .split_at(Axis(0), 3);

      initial_y.assign(&mean);
      initial_yp.fill(0.0);
      let thresh: ArrayView<'static, _, _> = aview1(THRESH);
      let mut state: rke::State<_, _, _, _, _> =
        rke::new(i as f64 * TAU,
                 initial_y.view_mut(),
                 initial_yp.view_mut(),
                 TAU / 1.0,
                 TOL,
                 thresh.view(),
                 true);
      let x0 = [
        i as f64 * TAU,
        (i + 1) as f64 * TAU,
      ];
      let x = aview1(&x0[..]);

      let mut iter = state.iter(lorenz63).stepping_iter(x);
      let (s, result) = iter.next().unwrap();
      let (_, _, ycoeff) = result.expect("rke failed");
      let ycoeff = ycoeff.unwrap();

      if s.total_steps() > 1 && false {
        println!("rke required {} steps", s.total_steps());
      }

      rke::y_value(x[0], x[1], s.step_size(),
                   &ycoeff,
                   out);
    }
  }

  struct Observer<'a>(ArrayView<'a, f64, Ix2>);
  impl<'a> na_df::Observer<f64> for Observer<'a> {
    fn observe_into(&self, idx: u64,
                    mut out: ArrayViewMut<f64, Ix1>) -> bool {
      out.assign(&self.0.subview(Axis(0),
                                 idx as usize));
      true
    }
  }

  let mut model: na_df::ModelStats<_> = From::from(TruthModel);
  let observer = Observer(data.observations.view());

  let mut states = StateSteps::new(STEPS + 1, init.ensemble_count, 3);

  /*let algo = etkf::Algo::init(&init, &mut rand, &mut model,
                              &observer, STEPS as u64);
  let mut workspace: etkf::OwnedWorkspace<_> =
    etkf::OwnedWorkspace::alloc(init, &mut rand, STEPS as u64);
  states.store_state(0, &workspace);*/

  let algo = sirs::std::Algo::init(&init,
                                   &mut rand,
                                   &mut model,
                                   &observer,
                                   STEPS as u64);
  let mut workspace: sirs::std::Workspace<_> =
    sirs::std::Workspace::alloc(init, &mut rand, STEPS as u64);
  states.store_state(0, &workspace);

  println!("Starting algorithm loop");
  for i in 0..STEPS as u64 {
    algo.next_step(i, STEPS as u64,
                   &mut rand,
                   &mut workspace,
                   &mut model,
                   &observer)
      .expect("algorithm step failed");

    states.store_state(i as usize + 1, &workspace);
  }
  println!("Algorithm run done");
  let states = states;

  if false {
    for i in 0..21 {
      let v = states.covariances.subview(Axis(0), i);
      println!("{:?}", v);
    }
  }

  let current_exe = ::std::env::current_exe().unwrap();
  let out_dir = current_exe.parent().unwrap().join("../../graphs");

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
  f.set_terminal("pngcairo",
                 format!("{}/sixty-three-sirs-1.png",
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
             data.truth.subview(Axis(1), 0).iter().map(|_| 0.0 ),
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
  f.set_terminal("pngcairo",
                 format!("{}/sixty-three-sirs-2.png",
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
             data.truth.subview(Axis(1), 1).iter(),
             &[gnuplot::PlotOption::Caption("truth"),
               gnuplot::PlotOption::Color("black"),
               gnuplot::PlotOption::PointSymbol('x'),]);
  }
  f.show();

  let mut f = gnuplot::Figure::new();
  f.set_terminal("pngcairo",
                 format!("{}/sixty-three-sirs-3.png",
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
}