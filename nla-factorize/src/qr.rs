
use alga::general::{Real};
use nd::{ArrayBase, DataMut, Ix2,
         ScalarOperand};

use std::ops::{Sub, Div, Mul};

pub trait QR {
  type QMatrix: Sized;
  type RMatrix: Sized;
  /// Assumes full rank.
  fn modified_gram_schmidt(self) -> (Self::QMatrix, Self::RMatrix);
}

impl<S> QR for ArrayBase<S, Ix2>
  where S: DataMut,
        S::Elem: Clone + ScalarOperand + Real,
        S::Elem: Sub<S::Elem, Output = S::Elem> + Mul<S::Elem, Output = S::Elem>,
        S::Elem: Div<S::Elem, Output = S::Elem>,
{
  type QMatrix = ArrayBase<S, Ix2>;
  type RMatrix = ArrayBase<Vec<S::Elem>, Ix2>;

  fn modified_gram_schmidt(self) -> (Self::QMatrix, Self::RMatrix) {
    let mut dim = self.dim();
    if dim.0 < dim.1 {
      ::std::mem::swap(&mut dim.0, &mut dim.1);
    }
    let dim = dim;
    let n = dim.1;
    let mut r = ArrayBase::zeros((dim.0, dim.1));
    let mut a = self.into_shape((dim.0, dim.1)).unwrap();

    for i in 0..n {
      r[[i, i]] = {
        let r = a.row(i);
        r.dot(&r).sqrt()
      };

      a.row_mut(i)
        .mapv_inplace(|v| v / r[[i, i]] );
      for j in i+1..n {
        r[[i,j]] = {
          let t = a.row(i);
          t.t().dot(&a.row(j))
        };

        let t1 = {
          &a.row(j) - &(&a.row(i) * r[[i,j]])
        };

        a.row_mut(j)
          .assign(&t1);
      }
    }

    (a, r)
  }
}

#[test]
fn modified_gram_schmidt() {
  use nd::arr2;
  let a = arr2(&[
    [12.0, -51.0, 4.0,],
    [6.0, 167.0, -68.0,],
    [4.0, 24.0, -41.0,],
  ]);

  let expected_q = arr2(&[
    [6.0/7.0, -69.0/175.0, -58.0/175.0,],
    [3.0/7.0, 158.0/175.0, 6.0/175.0,],
    [-2.0/7.0, 6.0/35.0, -33.0/35.0,],
  ]);
  let expected_r = arr2(&[
    [14.0, 21.0, -14.0,],
    [0.0, 175.0, -70.0,],
    [0.0, 0.0, 35.0,],
  ]);

  let (q, r) = a.modified_gram_schmidt();
  assert_eq!(expected_r, r);
  assert_eq!(expected_q, q);
}