use crate::Context;
use vm::{Tensor, op::Conv};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Conv + ?Sized,
{
    pub fn conv(
        &self,
        y: &mut Tensor<VM>,
        x: &Tensor<VM>,
        w: &Tensor<VM>,
        b: &Tensor<VM>,
        d_patch: usize,
    ) {
        let strides = [d_patch; 2];
        let dilations = [1; 2];
        let pads = [0; 4];
        self.vm()
            .conv(self.stack(), y, x, w, b, strides, dilations, pads);
    }
}
