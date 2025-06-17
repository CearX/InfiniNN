use crate::{Context, Mapping};
use crate::{NuralNetwork, WeightBiasData};
use std::ops::Deref;
use vm::{Tensor, VirtualMachine, op};

#[derive(Clone)]
pub struct Conv {
    pub kernel_size: Vec<usize>,
}

pub struct Data {
    pub conv: WeightBiasData,
}

// impl<VM> NuralNetwork<VM> for Conv
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
