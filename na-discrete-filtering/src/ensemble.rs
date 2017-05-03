//! Common stuffs for ensembles/particles

#[derive(Debug, Clone)]
pub struct EnsembleCommonInit<'a, E>
  where E: LinxalImplScalar,
{
  /// Possibly an estimate.
  pub initial_mean: ArrayView<'a, E, Ix1>,
  pub initial_covariance: ArrayView<'a, E, Ix2>,
  pub observation_operator: ArrayView<'a, E, Ix2>,
  pub gamma: ArrayView<'a, E::RealPart, Ix1>,
  pub sigma: ArrayView<'a, E::RealPart, Ix1>,
  pub model_workspace_size: usize,
  pub ensemble_count: usize,
}
impl<'a, E> EnsembleInit<'a, E>
  where E: LinxalImplScalar,
{ }

#[derive(Debug)]
pub struct EnsembleState<'a, E>
  where E: LinxalImplScalar,
{
  pub mean: ArrayView<'a, E, Ix1>,
  pub covariance: ArrayView<'a, E, Ix2>,
  pub ensembles: ArrayView<'a, E, Ix2>,
}
pub trait EnsembleWorkspace<E> {
  fn ensemble_state(&self) -> &EnsembleState<E>;
  fn mean_view(&self) -> ArrayView<E, Ix1> {
    self.ensembles_states().mean.view()
  }
  fn covariance_view(&self) -> ArrayView<E, Ix2> {
    self.ensembles_states().covariance.view()
  }
  fn ensembles_view(&self) -> ArrayView<E, Ix2> {
    self.ensembles_states().ensembles.view()
  }
}
