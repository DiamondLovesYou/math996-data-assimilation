use alga::general::{Identity, Additive, Real};
use nd::{ArrayBase, DataMut,
         Ix1, Ix2,
         ScalarOperand, ShapeError};

use std::ops::{Sub, Div, Mul};

pub trait Cholesky {
  type LMatrix: Sized;
  type DMatrix: Sized;
  /// Returns the upper triangular factor, R, of A = R^*R.
  fn l_lstar(self) -> Result<Self::LMatrix, ShapeError>;
  fn l_d_lstar(self) -> Result<(Self::LMatrix, Self::DMatrix), ShapeError>;
}

impl<S> Cholesky for ArrayBase<S, Ix2>
  where S: DataMut,
        S::Elem: Clone + ScalarOperand + Real,
        S::Elem: Sub<S::Elem, Output = S::Elem> + Mul<S::Elem, Output = S::Elem>,
        S::Elem: Div<S::Elem, Output = S::Elem>,
{
  type LMatrix = ArrayBase<S, Ix2>;
  type DMatrix = ArrayBase<S, Ix1>;

  fn l_lstar(self) -> Result<Self::LMatrix, ShapeError> {
    let dim = self.dim();
    let m = dim.0;
    let mut r = self.into_shape((m, m))?;
    let m = m as isize;

    for k in 0..m {
      for j in k + 1..m {
        let t = {
          let t1 = r.row(j as usize);
          let t2 = t1.slice(s![j..]);
          let t3 = r.row(k as usize);
          let t4 = t3.slice(s![j..]);

          &t2 - &(&t4 * (r[[k as usize, j as usize]] / r[[k as usize, k as usize]]))
        };

        let mut t2 = r.row_mut(j as usize);
        t2.slice_mut(s![j..])
          .assign(&t);
      }

      let t = r[[k as usize, k as usize]].sqrt().recip();

      let mut t2 = r.row_mut(k as usize);
      t2.slice_mut(s![k..])
        .mapv_inplace(|v| v * t );
    }

    let zero = <S::Elem as Identity<Additive>>::identity();
    for k in 0..m {
      let mut t = r.row_mut(k as usize);
      t.slice_mut(s![..k])
        .fill(zero.clone());
    }

    r.into_shape(dim)
  }

  /// XXX TODO
  fn l_d_lstar(self) -> Result<(Self::LMatrix, Self::DMatrix), ShapeError> {
    unreachable!();
  }
}

#[test]
fn cholesky_llstar() {
  use nd::arr2;

  let a = arr2(&[
    [4.0, 12.0, -16.0],
    [12.0, 37.0, -43.0],
    [-16.0, -43.0, 98.0],
  ]);

  let l = a.l_lstar().expect("cholesky factorization failed");

  let expected = arr2(&[
    [2.0, 6.0, -8.0],
    [0.0, 1.0, 5.0],
    [0.0, 0.0, 3.0],
  ]);
  assert_eq!(l, expected);
}