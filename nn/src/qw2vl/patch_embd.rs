use crate::{
    Context, Mapping, NuralNetwork, WeightBiasData,
    conv::{self, ConvLayer},
};
use digit_layout::DigitLayout;
use std::borrow::Cow;
use vm::{
    Id, Tensor, VirtualMachine,
    op::{Add, Conv, Rearrange},
};

#[derive(Clone)]
pub struct PatchEmbd {
    pub dt_w: DigitLayout,
    pub bias: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub patch_embd: Tensor<'vm, VM>,
    pub raw: Tensor<'vm, VM>,
    pub d_patch: usize,
}

pub struct Data {
    pub patch_embd: WeightBiasData,
    pub patch_embd1: WeightBiasData,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    Conv,
    Conv1,
}

impl Id for Sub {
    fn name(&self) -> Cow<str> {
        match self {
            Sub::Conv => "conv".into(),
            Sub::Conv1 => "conv1".into(),
        }
    }
}

pub trait Ops: Conv + Add + Rearrange {}
impl<VM> Ops for VM where VM: Conv + Add + Rearrange + ?Sized {}

impl<VM> NuralNetwork<VM> for PatchEmbd
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
        let Self::Data {
            patch_embd,
            patch_embd1,
        } = data;
        mapping
            .trap::<ConvLayer>(Sub::Conv, patch_embd)
            .trap::<ConvLayer>(Sub::Conv1, patch_embd1);
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let &Self { dt_w, bias } = self;
        let Args {
            mut patch_embd,
            raw,
            d_patch,
        } = args;

        let _dt = Tensor::check_dt_same(&[&raw, &patch_embd]).unwrap();

        let &[n, _c, h, w] = raw.shape() else {
            panic!()
        };
        assert_eq!(n, 1);
        let &[patches, m] = patch_embd.shape() else {
            panic!()
        };
        let h_patches = h / d_patch;
        let w_patches = w / d_patch;
        assert_eq!(h_patches * w_patches, patches);

        let mut embd = ctx.workspace(dt_w, &[n, m, h_patches, w_patches]);
        ctx.trap(
            Sub::Conv,
            &ConvLayer { dt_w, bias },
            conv::Args {
                y: embd.clone(),
                x: raw.clone(),
                d: m,
                d_patch,
            },
        );

        let embd1 = ctx.workspace(dt_w, &[n, m, h_patches, w_patches]);
        ctx.trap(
            Sub::Conv1,
            &ConvLayer { dt_w, bias },
            conv::Args {
                y: embd1.clone(),
                x: raw,
                d: m,
                d_patch,
            },
        );

        ctx.add(&mut embd, &embd1);

        // reshape

        // [n, m, h_patches, w_patches] -> [n, h_patches, w_patches, m]
        let embd_ = embd.transpose(&[0, 2, 3, 1]);
        let mut embd = ctx.workspace(dt_w, embd_.shape());
        ctx.rearrange(&mut embd, &embd_);

        // [n, h_patches, w_patches, m] -> [n * h_patches/2, 2, w_patches/2, 2*m]
        let embd_ = embd.tile(1, &[h_patches / 2, 2]);
        let embd_ = embd_.merge(0, 2).unwrap();
        let embd_ = embd_.tile(2, &[w_patches / 2, 2]);
        let embd_ = embd_.merge(3, 2).unwrap();

        // [n * h_patches/2, 2, w_patches/2, 2*m] -> [n * h_patches/2, w_patches/2, 2, 2*m]
        let embd_ = embd_.transpose(&[0, 2, 1, 3]);
        let mut embd = ctx.workspace(dt_w, embd_.shape());
        ctx.rearrange(&mut embd, &embd_);

        // [n * h_patches/2, w_patches/2, 2, 2*m] -> [n, h_patches * w_patches, m]
        let embd_ = embd.tile(0, &[n, h_patches / 2]);
        let embd_ = embd_.merge(1, 3).unwrap();
        let embd_ = embd_.tile(2, &[2, m]);
        let embd_ = embd_.merge(1, 2).unwrap();

        // [n, h_patches * w_patches, m] -> [n * patches, m]
        let embd_ = embd_.merge(0, 2).unwrap();
        ctx.rearrange(&mut patch_embd, &embd_);
    }
}
