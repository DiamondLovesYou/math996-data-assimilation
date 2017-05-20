
//! very very non programmable xdmf writer for timeseries grids.
//! for every time step, a new xdmf file is written and placed
//! next to the binary data file. once timestepping is complete,

use util::MeshGrid;
use nd::prelude::*;

use std::env::current_dir;
use std::path::{PathBuf};

use std::io::Write;
use std::fs::File;

const DATA_OUTPUT_SUBDIR: &'static str = "data/output/";
fn output_path(run_name: &'static str, filename: &str) -> PathBuf {
  let cdir = current_dir().unwrap();
  let outdir = cdir.join(DATA_OUTPUT_SUBDIR);
  let outdir = if outdir.exists() {
    outdir
  } else {
    cdir.join("../..")
      .join(DATA_OUTPUT_SUBDIR)
  };

  assert!(outdir.exists());

  let outdir = outdir.join(run_name);
  if !outdir.exists() {
    ::std::fs::create_dir_all(outdir.as_path())
      .expect("failed to create run data output dir!");
  }

  let outpath = outdir.join(filename);

  outpath.to_path_buf()
}

#[derive(Debug, Default)]
pub struct Xdmf {
  run_name: &'static str,
  complete: bool,
  steps: Vec<(u64, PathBuf)>,
}

impl Xdmf {
  pub fn new(run_name: &'static str) -> Xdmf {
    Xdmf {
      run_name: run_name,
      complete: false,
      steps: vec![],
    }
  }
  pub fn next_timestep(&mut self, step: u64, grid: &MeshGrid,
                       attributes: &[(&'static str,
                                      ArrayView<f64, Ix2>)]) {
    assert!(!self.complete);

    let xdmfname = format!("timestep-{}.xdmf", step);
    let xdmfpath = output_path(self.run_name, &xdmfname[..]);

    {
      let mut xdmf = File::create(xdmfpath.as_path())
        .expect("failed to create grid meta file");

      write!(xdmf, r#"
        <Grid Name="T@{step_i:}" GridType="Uniform">
            <Topology Reference="/Xdmf/Domain/Topology[1]"/>
            <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>"#,
             step_i = step)
        .unwrap();

      for &(name, ref state) in attributes.iter() {
        let outname = format!("timestep-{}-attr-{}.bin",
                              step, name);
        let outpath = output_path(self.run_name, &outname[..]);

        writeln!(xdmf, r#"
            <Attribute Name="{name}" Center="Node">
                <DataItem Format="Binary"
                 DataType="Float" Precision="8" Endian="Native"
                 Dimensions="{topology_x_dim:} {topology_y_dim:}">
                    {output_path:}
                </DataItem>
            </Attribute>"#,
               name = name,
               topology_x_dim = grid.dim().0,
               topology_y_dim = grid.dim().1,
               output_path = outpath.display())
          .unwrap();

        let state_slice = state.as_slice_memory_order()
          .expect("state isn't contiguous?");
        let state_slice_u8 = unsafe {
          ::std::slice::from_raw_parts(state_slice.as_ptr() as *const u8,
                                       8 * state_slice.len())
        };
        {
          let mut bin = File::create(outpath)
            .expect("failed to create grid output file");
          bin.write_all(state_slice_u8).expect("bin write failed");
        }
      }

      writeln!(xdmf, "        </Grid>").unwrap();
    }

    self.steps.push((step, xdmfpath));
  }

  pub fn run_complete(&mut self, grid: &MeshGrid) {
    self.complete = true;
    if self.steps.len() == 0 {
      return;
    }

    self.steps.sort_by_key(|v| v.0 );

    let xdmfpath = output_path(self.run_name, "run.xdmf");
    let mut xdmf = File::create(xdmfpath)
      .expect("failed to create final xdmf file");

    // >.< this *has* to be inline.
    write!(xdmf, r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
<Domain>
    <Topology name="topo" TopologyType="2DCoRectMesh"
        Dimensions="{topology_x_dim:} {topology_y_dim:}">
    </Topology>
    <Geometry name="geo" Type="ORIGIN_DXDYDZ">
        <DataItem Format="XML" Dimensions="3">
        0.0 0.0 0.0
        </DataItem>
        <DataItem Format="XML" Dimensions="3">
        {dx:} {dy:} 1.0
        </DataItem>
    </Geometry>

    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
        <Time TimeType="HyperSlab">
            <DataItem Format="XML" NumberType="Float" Dimensions="3">
            0.0 1.0 {step_count:}
            </DataItem>
        </Time>
"#,
           dx = grid.dx(),
           dy = grid.dy(),
           step_count = self.steps.len() - 1,
           topology_x_dim = grid.dim().0,
           topology_y_dim = grid.dim().1)
      .unwrap();

    for &(_, ref path) in self.steps.iter() {
      writeln!(xdmf,
               "\t\t<xi:include href=\"{path:}\" parse=\"xml\" />",
               path = path.display())
        .unwrap();
    }
    write!(xdmf, r#"
    </Grid>
  </Domain>
</Xdmf>
"#).unwrap();

  }
}
