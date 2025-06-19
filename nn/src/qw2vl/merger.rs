use crate::{
    Context, Mapping, NuralNetwork, WeightBiasData,
    mlp::{self, Mlp},
    normalization::{self as norm, Normalization},
};
use std::borrow::Cow;
use vm::{Id, Tensor, VirtualMachine};

#[derive(Clone)]
pub struct Merger {
    pub post_norm: Normalization,
    pub mlp: Mlp,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
}

pub struct Data {
    pub post_norm: WeightBiasData,
    pub mlp: mlp::Data,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    PostNorm,
    Mlp,
}

impl Id for Sub {
    fn name(&self) -> Cow<str> {
        match self {
            Sub::PostNorm => "post-norm".into(),
            Sub::Mlp => "mlp".into(),
        }
    }
}

pub trait Ops: norm::Ops + mlp::Ops {}
impl<VM> Ops for VM where VM: norm::Ops + mlp::Ops + ?Sized {}

impl<VM> NuralNetwork<VM> for Merger
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = Data;
    type Obj = ();
    type Sub = Sub;

    fn init(data: Self::Data, mut mapping: Mapping<VM, Self>) {
        let Self::Data { post_norm, mlp } = data;
        mapping
            .trap::<Normalization>(Sub::PostNorm, post_norm)
            .trap::<Mlp>(Sub::Mlp, mlp);
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let Self { post_norm, mlp } = self;
        let Args { y, x } = args;

        let _dt = Tensor::check_dt_same(&[&y, &x]).unwrap();
        let &[n_x, _] = x.shape() else { panic!() };
        let &[n_y, _] = y.shape() else { panic!() };
        assert_eq!(n_x * 4, n_y);

        let x1 = ctx.workspace(x.dt(), x.shape());

        ctx.trap(
            Sub::PostNorm,
            post_norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        )
        .trap(
            Sub::Mlp,
            mlp,
            mlp::Args {
                y,
                x: x1,
                scale: 1.,
                residual: false,
            },
        );
    }
}
