use crate::{ObjId, Tensor, VirtualMachine};

pub trait Conv: VirtualMachine {
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
    );
}
