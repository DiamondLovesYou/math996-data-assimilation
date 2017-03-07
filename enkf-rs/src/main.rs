
extern crate alga;
extern crate num_traits;
extern crate ndarray as nd;

use nd::{ArrayBase, Ix1, Ix2};
use alga::{Identity, Multiplicative, Additive};

use num_traits::*;

#[derive(Clone, Copy)]
struct EnKF<InputShape, ObsShape, ModelF>
  where ModelF: FnMut(ArrayBase<InputShape, Ix1>) -> ArrayBase<InputShape, Ix1>,
        InputShape: nd::Data,
        ObsShape: nd::Data,
{
  model: ModelF,
  observation_operator: ArrayBase<ObsShape, Ix2>,

  /// C(hat)_{j+1}
  covariance: Vec<ArrayBase<InputShape, Ix2>>,
  /// y_{j+1}
  state: Vec<ArrayBase<InputShape, Ix1>>,
  update: ArrayBase<InputShape, Ix2>,
}

impl<InputShape, ObsShape, ModelF> EnKF<InputShape, ObsShape, ModelF>
  where ModelF: FnMut(ArrayBase<InputShape, Ix1>) -> ArrayBase<InputShape, Ix1>,
        InputShape: nd::Data,
        InputShape::Elem: Identity<Additive> + Identity<Multiplicative> + Clone,
        ObsShape: nd::Data,
{
  fn new(model: ModelF,
         observation_operator: ArrayBase<ObsShape, Ix2>,
         initial_covariance: ArrayBase<InputShape, Ix2>,
         initial_state: ArrayBase<InputShape, Ix1>,
         ensemble_count: usize)
         -> EnKF<InputShape, ModelF>
  {

    let avg = avg(&initial_state);

    let zero = <<InputShape as nd::Data>::Elem as Identity<Additive>>::identity();
    let one = <<InputShape as nd::Data>::Elem as Identity<Multiplicative>>::identity();

    let mut initial_ensemble = Vec::with_capacity(ensemble_count);
    for i in 0..ensemble_count {
      let iter = (0..ensemble_count)
        .map(|j| {
          if i == j {
            one.clone()
          } else {
            zero.clone()
          }
        });
      let is = ArrayBase::from_iter(iter);

    }

    unreachable!();

    //EnKF {
    //  model: model,
    //  observation_operator,


    //}

  }
}

fn avg<S>(a: &ArrayBase<S, Ix1>) -> S::Elem
  where S: Data,
        S::Elem: Add<S::Elem, Output = S::Elem> + Div<S::Elem, Output = S::Elem>,
{
  a.iter().sum() / <S::Data as NumCast>::from(a.shape()[0]).unwrap()
}

fn main() {

}
