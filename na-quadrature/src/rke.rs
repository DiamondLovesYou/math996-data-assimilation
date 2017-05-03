
//! Fifth order explicit Runge-Kutta (by R. England).
//! Use to approximate first order ODEs.
//! TODO: create workspace structures so the inner iterator doesn't alloc dealloc so much.
//! After taking a look at optimized assembly, this looks terrible. I'll fix it later.

use linxal::types::*;
use nd::{ArrayBase, ArrayView, ArrayViewMut, Array, Data, DataMut, Ix2, Ix1,
         ScalarOperand,};
use num_traits::*;
use std::ops::*;
use rand::distributions::range::SampleRange;

pub trait BoundsHelper: LinxalImplScalar + Float + NumCast + From<f32> + SampleRange + ScalarOperand {}
impl<T> BoundsHelper for T
  where T: LinxalImplScalar + Float + NumCast + From<f32> + SampleRange + ScalarOperand
{}

pub fn new<Elem, ElemReal,
           S1, S2, S3>(x: Elem,
                       y: ArrayBase<S1, Ix1>,
                       yp: ArrayBase<S2, Ix1>,
                       h: Elem::RealPart,
                       tol: Elem::RealPart,
                       threshold: ArrayBase<S3, Ix1>,
                       want_ycoeff: bool)
                       -> State<Elem, ElemReal, S1, S2, S3>
  where S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + From<ElemReal>,
        ElemReal: BoundsHelper,
{
  State::new(x, y, yp, h, tol, threshold, want_ycoeff)
}

pub struct State<Elem, ElemReal, S1, S2, S3>
  where S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = ElemReal>,
        Elem: LinxalImplScalar<RealPart = ElemReal>,
        ElemReal: BoundsHelper,
{
  calls: u64,
  steps: u64,
  pub step_size: Option<Elem::RealPart>,
  pub h: Elem::RealPart,
  pub tolerance: Elem::RealPart,
  pub threshold: ArrayBase<S3, Ix1>,

  pub want_ycoeff: bool,

  x: Elem,
  y: ArrayBase<S1, Ix1>,
  yp: ArrayBase<S2, Ix1>,
}
impl<Elem, ElemReal, S1, S2, S3> State<Elem, ElemReal, S1, S2, S3>
  where S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        Elem: LinxalImplScalar<RealPart = ElemReal>,
        ElemReal: BoundsHelper,
{
  pub fn new(x: Elem,
             y: ArrayBase<S1, Ix1>,
             yp: ArrayBase<S2, Ix1>,
             h: Elem::RealPart,
             tol: Elem::RealPart,
             threshold: ArrayBase<S3, Ix1>,
             want_ycoeff: bool) -> State<Elem, ElemReal, S1, S2, S3>
  {
    State {
      calls: 0,
      steps: 0,

      step_size: None,
      h: h,
      tolerance: tol,
      threshold: threshold,
      want_ycoeff: want_ycoeff,

      x: x,
      y: y,
      yp: yp,
    }
  }
  pub fn x(&self) -> Elem { self.x }
  pub fn y(&self) -> &ArrayBase<S1, Ix1> { &self.y }
  pub fn yp(&self) -> &ArrayBase<S2, Ix1> { &self.yp }

  pub fn step_size_opt(&self) -> Option<ElemReal> { self.step_size }
  pub fn step_size(&self) -> ElemReal { self.step_size.unwrap() }
  pub fn total_model_calls(&self) -> u64 { self.calls }
  pub fn total_steps(&self) -> u64 { self.steps }

  pub fn iter<'a, ModelF>(&'a mut self, f: ModelF) -> InnerIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3>
    where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
  {
    InnerIterator {
      model: f,
      state: self,
    }
  }
}
pub struct InnerIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem> + 'a,
        S2: DataMut<Elem = Elem> + 'a,
        S3: Data<Elem = Elem::RealPart> + 'a,
        Elem: LinxalImplScalar<RealPart = ElemReal>,
        ElemReal: BoundsHelper,
{
  pub model: ModelF,
  pub state: &'a mut State<Elem, ElemReal, S1, S2, S3>,
}
impl<'a, ModelF, Elem, ElemReal, S1, S2, S3>
InnerIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + PartialOrd,
        ElemReal: BoundsHelper,
{
  fn feval<S4>(&mut self, x: Elem, y: &ArrayBase<S4, Ix1>) -> Array<Elem, Ix1>
    where S4: Data<Elem = Elem>,
  {
    // TODO: recycle outputs

    let mut yp: Array<Elem, Ix1>;
    yp = ArrayBase::zeros(self.yp().dim());
    (self.model)(x, y.view(), yp.view_mut());

    self.state.calls += 1;

    yp
  }
  fn h(&self) -> Elem::RealPart { self.state.h }
  fn x(&self) -> Elem { self.state.x }
  fn y(&self) -> &ArrayBase<S1, Ix1> { &self.state.y }
  fn yp(&self) -> &ArrayBase<S2, Ix1> { &self.state.yp }

  pub fn stepping_iter<S4>(self, x: ArrayBase<S4, Ix1>)
                           -> SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
    where S4: Data<Elem = Elem>,
  {
    SteppingIterator::new_inner(self, x)
  }
}
impl<'a, ModelF, Elem, ElemReal, S1, S2, S3>
Iterator for InnerIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + From<ElemReal> + PartialOrd,
        Elem: Mul<ElemReal, Output = Elem> + Add<ElemReal, Output = Elem>,
        Elem: Mul<Elem, Output = Elem> + Add<Elem, Output = Elem>,
        Elem: Div<ElemReal, Output = Elem> + Sub<ElemReal, Output = Elem>,
        Elem: Div<Elem, Output = Elem> + Sub<Elem, Output = Elem>,
        ElemReal: BoundsHelper,
{
  type Item = (&'a mut &'a mut State<Elem, ElemReal, S1, S2, S3>,
               Result<Option<Array<Elem, Ix2>>, ()>);
  fn next(&mut self) -> Option<Self::Item>
  {
    if self.state.steps == 0 {
      (self.model)(self.state.x, self.state.y.view(),
                   self.state.yp.view_mut());
      self.state.calls += 1;
    }

    fn min<T: PartialOrd>(a: T, b: T) -> T { if a<b { a } else { b } }
    fn max<T: PartialOrd>(a: T, b: T) -> T { if a>b { a } else { b } }

    let mut failed = false;

    let zero: Elem = Zero::zero();
    let one: Elem = One::one();
    let zero = zero.mag();
    let one  = one.mag();

    let two = one + one;
    let four = one + one + one + one;
    let six = four + two;
    let ten = six + four;
    let twelve = four + four + four;
    let sixteen = twelve + four;
    let sixty = twelve + twelve + twelve + twelve + twelve;
    let nintytwo = sixty + twelve + twelve + four + four;
    let nintysix = nintytwo + four;
    let onefourtyfour = sixty + sixty + twelve + twelve;
    let oneeighty = sixty + sixty + sixty;
    let onetwentyone = (twelve - one) * (twelve - one);

    let quat = (four).recip();
    let half = (one + one).recip();
    let eighth = (one + one + one + one + one + one + one + one).recip();
    let twelveth = twelve.recip();

    let tolaim = self.state.tolerance * (four - one) / (six - one);

    let result = loop {
      let t = self.x() + Elem::from(quat * self.h());
      let t2 = self.y() + &(self.yp() * quat * self.h());
      let k1 = self.feval(t, &t2);
      let t2 = self.y() + &((self.yp() + &k1) * eighth * self.h());
      let k2 = self.feval(t, &t2);
      let t = self.x() + Elem::from(half * self.h());
      let t2 = self.y() + &((&k2 * two - &k1) * half * self.h());
      let k3 = self.feval(t, &t2);

      let xmid = self.x() + Elem::from(half * self.h());
      let ymid = self.y() + &((self.yp() + &(&k2 * four) + &k3) * self.h() * twelveth);
      let k4 = self.feval(xmid, &ymid);

      let t = xmid + quat * self.h();
      let t2 = &ymid + &(&k4 * quat * self.h());
      let k5 = self.feval(t, &t2);
      let t2 = &ymid + &((&k4 + &k5) * eighth * self.h());
      let k6 = self.feval(t, &t2);
      let t = xmid + half * self.h();
      let t2 = &ymid + &((&k6 * two - &k5) * half * self.h());
      let k7 = self.feval(t, &t2);
      let y4th = &ymid + &((&k4 + &(&k6 * four) + &k7) * self.h() * twelveth);

      let y5th = {
        let t = self.yp() * one.neg() - &k1 * nintysix + &k2 * nintytwo;
        let t = &t - &(&k3 * onetwentyone) + &(&k4 * onefourtyfour) + &(&k5 * six) - &(&k6 * twelve);
        let y = self.y() + &(&t * twelveth * self.h());
        let t = self.x() + self.h();
        let k = self.feval(t, &y);

        let t = &(self.yp() * (twelve + two)) + &(&k2 * (sixty + four)) + &(&k3 * (sixteen + sixteen));
        let t = &t - &(&k4 * (four + four)) + &(&k6 * (sixty + four)) + &(&k7 * (sixteen - one)) - &k;

        self.y() + &(&t * self.h() * oneeighty.recip())
      };

      let mut first_wt = &self.y().mapv(|v| v.mag() ) + &ymid.mapv(|v| v.mag() );
      first_wt.mapv_inplace(|v| v * two );
      let second_wt = &y4th.mapv(|v| v.mag() ) + &y5th.mapv(|v| v.mag() );

      let mut wt = (&first_wt + &second_wt) * six.recip();
      debug_assert!(wt.len() == self.state.threshold.len());
      let mut delta = (&y5th - &y4th).mapv_into(|v| From::from(v.mag()) );

      for i in 0..wt.len() {
        wt[i] = if wt[i] > self.state.threshold[i] {
          wt[i]
        } else {
          self.state.threshold[i]
        };

        if wt[i] > zero {
          delta[i] = delta[i] / wt[i];
        }
      }

      let err = delta.iter()
        .fold(None, |err, &d| {
          err
            .map(move |v| {
              max(v, d)
            })
            .or_else(|| Some(d) )
        })
        .unwrap_or(Zero::zero())
        .mag();

      let first_step = self.state.steps == 0;
      self.state.steps += 1;

      if err > self.state.tolerance {
        if first_step {
          self.state.h = self.state.h * ten.recip();
        } else if !failed {
          failed = true;
          self.state.h = self.state.h * max(ten.recip(), (tolaim / err).powf(From::from(1.0/5.0))) * self.h();
        } else {
          self.state.h = self.state.h * two.recip();
        }

        let eps = Elem::eps();
        if self.h().abs() < twelve * twelve * eps * max(self.x().mag(), (self.x() + self.h()).mag()) {
          break Err(());
        }
      } else {
        self.state.x = self.state.x + self.h();
        self.state.step_size = Some(self.h());

        let x = self.x();
        let k8 = self.feval(x, &y5th);

        let result = if self.state.want_ycoeff {
          let mut ycoeff: Array<Elem, Ix2>;
          ycoeff = ArrayBase::zeros((self.y().len(), 6));

          let alpha = (self.yp() - &k8) * self.h();
          let beta = &y5th - &(&ymid * two) + self.y();
          let gamma = &y5th - self.y();

          ycoeff.column_mut(0)
            .assign(&ymid);
          ycoeff.column_mut(1)
            .assign(&(&k4 * self.h()));
          ycoeff.column_mut(2)
            .assign(&(&(&beta * four) + &(&alpha * half)));
          ycoeff.column_mut(3)
            .assign(&(&(&gamma * ten) - &((self.yp() + &(&k4 * (four + four)) + &k8) * self.h())));
          ycoeff.column_mut(4)
            .assign(&(&(&beta * (four + four).neg()) - &(&alpha * two)));
          ycoeff.column_mut(5)
            .assign(&(&(&gamma * (twelve * two).neg()) + &((self.yp() + &(&k4 * four) + &k8) * four * self.h())));

          Some(ycoeff)
        } else {
          None
        };

        self.state.y.assign(&y5th);
        self.state.yp.assign(&k8);

        let temp = max(ten.recip(), (err / tolaim).powf(From::from(1.0/5.0)))
          .recip();
        let temp = if failed {
          min(one, temp)
        } else {
          temp
        };

        self.state.h = self.state.h * temp;

        break Ok(result);
      }
    };

    Some((unsafe { ::std::mem::transmute(&mut self.state) }, result))
  }
}

pub struct SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem> + 'a,
        S2: DataMut<Elem = Elem> + 'a,
        S3: Data<Elem = Elem::RealPart> + 'a,
        S4: Data<Elem = Elem>,
        Elem: LinxalImplScalar<RealPart = ElemReal>,
        ElemReal: BoundsHelper,
{
  x: ArrayBase<S4, Ix1>,
  idx: Option<(usize, bool)>,
  inner: InnerIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3>,
}
impl<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4> SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        S4: Data<Elem = Elem>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + PartialOrd,
        ElemReal: BoundsHelper,
{
  pub fn new_inner(inner: InnerIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3>,
                   x: ArrayBase<S4, Ix1>) -> SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
  {
    debug_assert!((0..x.len()).skip(1).all(|i| x[i - 1] < x[i] ));

    SteppingIterator {
      x: x,
      idx: None,
      inner: inner,
    }
  }
}

impl<'a, ModelF, Elem, ElemReal,
     S1, S2, S3, S4>
SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        S4: Data<Elem = Elem>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + From<ElemReal> + PartialOrd,
        Elem: Mul<ElemReal, Output = Elem> + Add<ElemReal, Output = Elem>,
        Elem: Mul<Elem, Output = Elem> + Add<Elem, Output = Elem>,
        Elem: Div<ElemReal, Output = Elem> + Sub<ElemReal, Output = Elem>,
        Elem: Div<Elem, Output = Elem> + Sub<Elem, Output = Elem>,
        ElemReal: BoundsHelper,
{
  fn run<F1>(&mut self, idx: usize, dir: bool, end: Elem, end_test: F1)
                 -> Option<(&'a mut &'a mut State<Elem, ElemReal, S1, S2, S3>,
                            Result<(usize, Elem, Option<Array<Elem, Ix2>>), ()>)>
    where F1: Fn(Elem) -> bool,
  {
    let mut quit_next = false;

    loop {
      match self.inner.next() {
        None => { return None; }
        Some((state, Ok(ycoeff))) => {

          let h: Elem = From::from(self.inner.state.h);
          let x = self.inner.state.x();

          if quit_next {
            return Some((state, Ok((idx, end, ycoeff))));
          }

          if !end_test(x + h) {
            let next_step_size = (end - x).mag();

            if next_step_size <= self.inner.state.tolerance {
              return Some((state, Ok((idx, end, ycoeff))));
            } else {
              self.inner.state.h = next_step_size;
            }

            quit_next = true;
          }
        },
        Some((state, Err(()))) => {
          return Some((state, Err(())));
        }
      }
    }

  }
}
impl<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
Iterator for SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        S4: Data<Elem = Elem>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + From<ElemReal> + PartialEq + PartialOrd,
        Elem: Mul<ElemReal, Output = Elem> + Add<ElemReal, Output = Elem>,
        Elem: Mul<Elem, Output = Elem> + Add<Elem, Output = Elem>,
        Elem: Div<ElemReal, Output = Elem> + Sub<ElemReal, Output = Elem>,
        Elem: Div<Elem, Output = Elem> + Sub<Elem, Output = Elem>,
        ElemReal: BoundsHelper,
{
  type Item = (&'a mut &'a mut State<Elem, ElemReal, S1, S2, S3>,
               Result<(usize, Elem, Option<Array<Elem, Ix2>>), ()>);

  fn next(&mut self) -> Option<Self::Item> {
    let idx = match self.idx {
      Some((idx, true)) => idx,
      Some((_, false)) => {
        panic!("this iterator is irreversible once a direction is picked")
      },
      None => {
        self.inner.state.x = self.x[0];
        0
      },
    };
    if idx >= self.x.len() - 1 { return None; }

    let start = self.x[idx];
    let end   = self.x[idx + 1];
    assert!(start != end);

    match self.run(idx, true, end, |v| v <= end ) {
      Some((state, result)) => {
        self.idx = Some((idx + 1, true));
        Some((state, result))
      },
      None => { None },
    }
  }
}

impl<'a, ModelF, Elem, ElemReal,
     S1, S2, S3, S4>
DoubleEndedIterator for SteppingIterator<'a, ModelF, Elem, ElemReal, S1, S2, S3, S4>
  where ModelF: for<'r, 's> FnMut(Elem, ArrayView<'s, Elem, Ix1>, ArrayViewMut<'r, Elem, Ix1>),
        S1: DataMut<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        S3: Data<Elem = Elem::RealPart>,
        S4: Data<Elem = Elem>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + From<ElemReal> + PartialEq + PartialOrd,
        Elem: Mul<ElemReal, Output = Elem> + Add<ElemReal, Output = Elem>,
        Elem: Mul<Elem, Output = Elem> + Add<Elem, Output = Elem>,
        Elem: Div<ElemReal, Output = Elem> + Sub<ElemReal, Output = Elem>,
        Elem: Div<Elem, Output = Elem> + Sub<Elem, Output = Elem>,
        ElemReal: BoundsHelper,
{
  fn next_back(&mut self) -> Option<Self::Item> {
    let idx = match self.idx {
      Some((idx, false)) => idx,
      Some((_, true)) => {
        panic!("this iterator is irreversible once a direction is picked")
      },
      None => {
        self.inner.state.h = self.inner.state.h * Elem::one().mag().neg();
        self.inner.state.x = self.x[self.x.len() - 1];
        self.x.len()
      },
    };
    if idx == 0 { return None; }

    let start = self.x[idx];
    let end   = self.x[idx - 1];
    assert!(start != end);

    match self.run(idx, true, end, |v| v >= start ) {
      Some((state, result)) => {
        self.idx = Some((idx - 1, false));
        Some((state, result))
      },
      None => { None },
    }
  }
}

pub fn y_value<Elem, ElemReal, S1, S2>(xi: Elem, x: Elem, h: ElemReal,
                                       ycoeff: &ArrayBase<S1, Ix2>,
                                       dest: ArrayBase<S2, Ix1>)
  where S1: Data<Elem = Elem>,
        S2: DataMut<Elem = Elem>,
        Elem: LinxalImplScalar<RealPart = ElemReal> + Mul<ElemReal, Output = Elem>,
        ElemReal: BoundsHelper,
{
  let one: Elem = One::one();
  let two = one + one;

  let a = two.mag().recip() + (xi - x).mag() / h;

  let mut z = dest;
  z.assign(&ycoeff.column(5));
  z.mapv_inplace(|v| v * a );

  for i in (0..6).rev() {
    let t = &z + &ycoeff.column(i);
    z.assign(&t);
    if i != 0 {
      z.mapv_inplace(|v| v * a);
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use nd::*;

  #[test]
  fn t1() {
    let tol = 10.0.powf(-6.0);
    let thresh = (0..8)
      .map(|_| (10.0).powf(-7.0) );
    let thresh: Array1<f64> = ArrayBase::from_iter(thresh);

    let h = 0.1;
    let b = 500.0;
    let a = 0.0;

    fn f(_: f64, y: ArrayView<f64, Ix1>, mut yp: ArrayViewMut<f64, Ix1>) {
      yp[7] = -0.396 * (y[6] - 47.6);
      yp[6] = 0.508 * (y[5] - y[6]);
      yp[5] = 0.433 * (y[4] - y[5]);
      yp[4] = 0.873 * (y[2] - y[4]);
      yp[3] = y[1] + 0.025 * y[0] - 0.087 * y[3];
      yp[2] = y[3];
      yp[1] = yp[7] + 0.063 * y[7] - 0.0425 * y[1] - 0.961 * y[0];
      yp[0] = y[1];
    }

    let mut y: ArrayBase<Vec<f64>, _> = ArrayBase::zeros(8);
    y[2] = 50.0;
    for i in 4..7 {
      y[i] = 75.0;
    }

    let yp: Array1<f64> = ArrayBase::zeros(8);

    let pv = vec![0.0, 25.0, 50.0, 75.0, 100.0,
                  150.0, 200.0, 250.0, 300.0, 400.0, 500.0];
    let pv: Array1<f64> = ArrayBase::from_shape_vec(pv.len(), pv).unwrap();

    let mut state = new(a, y, yp, h, tol, thresh, true);
    let mut iter = state.iter(f).stepping_iter(pv);

    let mut z: Array1<f64> = ArrayBase::zeros(8);
    let mut last_h = h;
    for (state, result) in iter {
      let (idx, x, ycoeff) = result.unwrap();
      let ycoeff = ycoeff.unwrap();

      y_value(x - last_h, x,
              last_h, &ycoeff,
              z.view_mut());


      println!("t = {}, v = {}, h = {}, steps = {}, calls = {}",
               x, z[4], last_h, state.total_steps(),
               state.total_model_calls());

      last_h = state.h;
    }

    panic!("want to see output");
  }
}
