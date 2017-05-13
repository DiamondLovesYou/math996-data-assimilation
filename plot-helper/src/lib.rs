
extern crate util;
extern crate gnuplot;
extern crate ndarray as nd;

use util::{StateSteps, ModelTruth};

use nd::{ArrayView, Ix2, Axis,};

use gnuplot::{Figure, PlotOption, DashType, AxesCommon, Axes3D};

pub mod xdmf;

pub fn make_ensemble_plots<T>(source: &T, states: &StateSteps,
                              js: u64, what: &str)
  where T: ModelTruth<f64>,
{

  let current_exe = ::std::env::current_exe().unwrap();
  let pname = current_exe
    .file_stem()
    .unwrap();
  let pname = std::path::Path::new(pname).to_path_buf();
  let out_dir = current_exe.parent().unwrap().join("../../graphs");

  let truth = source.truth();
  let observations = source.observations();

  let means = states.means.view();
  let covariances = states.covariances.view();

  let js = ::std::cmp::min(means.dim().0 as u64 - 1, js);
  let js_space = 0..js;

  let mk_axis = |all: &mut Figure| {
    let mut axis = all.axes2d();
    let title = format!("{}, {}, Ex. 1.3", what, pname.display());
    axis.set_title(&title[..], &[]);
    axis.set_x_label("iteration, j", &[]);
    axis.lines(js_space.clone(), (0..js).map(|i| truth[(i as usize, 0)] ),
               &[PlotOption::Caption("truth"),]);
    axis.lines(js_space.clone(), (0..js).map(|i| means[[i as usize, 0]] ),
               &[PlotOption::Caption("ensemble mean"),
                 PlotOption::Color("magenta"),]);
    axis.lines(js_space.clone(),
               (0..js).map(|i| {
                 means[[i as usize, 0]] + covariances[[i as usize, 0, 0]].sqrt()
               }),
               &[PlotOption::Caption("error"),
                 PlotOption::Color("red"),
                 PlotOption::LineStyle(DashType::Dash)]);
    axis.points((1..js).map(|i| i as f64 ),
                (0..js - 1).map(|i| {
                  observations[(i as usize, 0)]
                }),
                &[PlotOption::Caption("observation"),
                  PlotOption::Color("black"),
                  PlotOption::PointSymbol('x'),]);
    axis.lines(js_space.clone(),
               (0..js).map(|i| {
                 means[[i as usize, 0]] - covariances[[i as usize, 0, 0]].sqrt()
               }),
               &[PlotOption::Caption("error"),
                 PlotOption::Color("red"),
                 PlotOption::LineStyle(DashType::Dash)]);
  };

  let mut all = Figure::new();
  let out_name = format!("{}/{}",
                         out_dir.display(),
                         pname.with_extension("png").display());
  all.set_terminal("pngcairo", &out_name[..]);
  mk_axis(&mut all);
  all.show();

  let skip_gui = ::std::env::args()
    .position(|v| v == "--no-gui" )
    .is_some();
  if skip_gui { return; }

  let mut all = Figure::new();
  all.set_terminal("wxt", "");
  mk_axis(&mut all);
  all.show();
}

pub fn make_3d_plot(name: &str, fsuffix: &str,
                    plots: &[(&str, ArrayView<f64, Ix2>)]) {
  let current_exe = ::std::env::current_exe().unwrap();
  let pname = current_exe
    .file_stem()
    .unwrap();
  let pname = std::path::Path::new(pname).to_path_buf();

  let mk_axis = |axis: &mut Axes3D, name: &str, v: ArrayView<f64, Ix2>| {
    let title = format!("\\`{}\\`, from \\`{}\\`", name, pname.display());
    axis.set_title(&title[..], &[]);
    axis.set_x_label("x", &[]);
    axis.set_y_label("y", &[]);
    axis.set_z_label("z", &[]);

    axis.lines(v.subview(Axis(1), 0),
               v.subview(Axis(1), 1),
               v.subview(Axis(1), 2),
               &[PlotOption::Caption("truth"),]);
  };

  let mut all = Figure::new();
  let out_name = format!("{}-{}.png", pname.display(),
                         fsuffix);
  all.set_terminal("pngcairo", &out_name[..]);
  for &(name, ref v) in plots.iter() {
    let mut axis = all.axes3d();
    mk_axis(axis, name, v.view());
  }
  all.show();

  let skip_gui = ::std::env::args()
    .position(|v| v == "--no-gui" )
    .is_some();
  if skip_gui { return; }

  let mut all = Figure::new();
  all.set_terminal("wxt", "");
  for &(name, ref v) in plots.iter() {
    let axis = all.axes3d();
    mk_axis(axis, name, v.view());
  }
  all.show();
}
