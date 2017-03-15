
extern crate rand;
extern crate ndarray as nd;
extern crate num_traits;
extern crate na_discrete_filtering as na_df;
extern crate gnuplot;
extern crate common;

use common::*;
use na_df::{Algorithm, Workspace};
use na_df::kalman::sirs;

fn main() {

  const STEPS: usize = 10000;

  let mut setup = SinMapSetup::default();
  setup.ensemble_count = 100;
  setup.steps = STEPS;

  let mut data: SinMapData = setup.into();

  let mut rand = data.rand.take().unwrap();

  let init = data.init();
  let mut model = data.model();

  let mut states = StateSteps::new(STEPS + 1, setup.ensemble_count);

  let algo = sirs::std::Algo::init(&init, &mut rand,
                                   &mut model, STEPS as u64);
  let mut workspace: sirs::std::Workspace<_> =
    sirs::std::Workspace::alloc(init,
                                &mut rand,
                                STEPS as u64);
  states.store_state(0, &workspace);

  for i in 0..STEPS as u64 {
    algo.next_step(i, STEPS as u64,
                   &mut rand,
                   &mut workspace,
                   &mut model)
      .expect("algorithm step failed");

    states.store_state(i as usize + 1, &workspace);
  }

  make_ensemble_plots(&data, &states, 21,
                      "Particle Filter (Standard)");
}
