use vm::{ObjId, Tensor, op::Conv};

impl Conv for crate::CpuVM {
    fn conv(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
        strides: [usize; 2],
        dilations: [usize; 2],
        pads: [usize; 4],
    ) {
        assert_eq!(y.dt(), x.dt());

        todo!()
    }
}
