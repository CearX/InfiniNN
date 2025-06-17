use crate::{Context, Mapping};
use crate::{NuralNetwork, WeightBiasData};
use crate::{
    mlp::{self, Mlp},
    normalization::Normalization,
};
use std::ops::Deref;
use vm::{Tensor, VirtualMachine, op};

#[derive(Clone)]
pub struct Merger {
    pub norm: Normalization,
    pub mlp: Mlp,
}

pub struct Data {
    pub post_norm: WeightBiasData,
    pub mlp: mlp::Data,
}

// impl<VM> NuralNetwork<VM> for Merger
// where
//     VM: VirtualMachine + ?Sized,
// {
//     type Args<'vm>
//         = ()
//     where
//         VM: 'vm;
//     type Data = Data;
//     type Obj = ();
//     type Sub = ();

//     fn init(data: Self::Data, mapping: Mapping<VM, Self>) {}
//     fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {}
// }
