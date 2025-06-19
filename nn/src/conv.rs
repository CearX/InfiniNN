use crate::{Context, Mapping, NuralNetwork, WeightBias, WeightBiasData};
use digit_layout::DigitLayout;
use vm::op::Conv;
use vm::{Tensor, VirtualMachine, op};

#[derive(Clone)]
pub struct ConvLayer {
    pub dt_w: DigitLayout,
    pub bias: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
    pub d: usize,
    pub d_patch: usize,
}

pub trait Ops: op::Conv {}
impl<VM> Ops for VM where VM: Conv + ?Sized {}

impl<VM> NuralNetwork<VM> for ConvLayer
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = WeightBiasData;
    type Obj = WeightBias;
    type Sub = ();

    fn init(data: Self::Data, mapping: Mapping<VM, Self>) {
        data.map(mapping)
    }

    fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
        let &Self { dt_w, bias } = self;
        let Args {
            mut y,
            x,
            d,
            d_patch,
        } = args;

        let _dt = Tensor::check_dt_same(&[&y, &x]).unwrap();

        let w = ctx.get_mapped(WeightBias::Weight, dt_w, &[d, 3, d_patch, d_patch]);
        let b = if bias {
            ctx.get_mapped(WeightBias::Bias, dt_w, &[d])
        } else {
            ctx.workspace(dt_w, &[d])
        };
        ctx.conv(&mut y, &x, &w, &b, d_patch);
    }
}
