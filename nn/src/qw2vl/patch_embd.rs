use crate::{Context, Mapping};
use crate::{NuralNetwork, WeightBiasData, conv};
use std::ops::Deref;
use vm::{Tensor, VirtualMachine, op};
#[derive(Clone)]
pub struct Patchembd {
    pub patch_embd: conv::Conv,
}

pub struct Data {
    pub patch_embd: conv::Data,
    pub patch_embd1: conv::Data,
}

// impl<VM> NuralNetwork<VM> for Patchembd
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
