use crate::{
    Context, Mapping, NuralNetwork, WeightBiasData,
    mlp::{self, Activation, Mlp},
    normalization::{self as norm, Normalization},
    self_attn::{self, Request, SelfAttn},
};
use digit_layout::DigitLayout;
use std::borrow::Cow;
use vm::{Id, Tensor, VirtualMachine, op::AttnMask};

#[derive(Clone)]
pub struct TransformerBlk {
    pub norm: Normalization,
    pub self_attn: SelfAttn,
    pub mlp: Mlp,
}

impl TransformerBlk {
    pub fn llama(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
        epsilon: f32,
    ) -> Self {
        Self {
            norm: Normalization {
                ty: norm::Type::RmsNorm { epsilon },
                dt_w: dt_norm,
            },
            self_attn: SelfAttn {
                dt_w,
                nh,
                nkvh,
                dh,
                qkv_bias: false,
                use_rope: true,
                mask: AttnMask::Causal,
                o_bias: false,
            },
            mlp: Mlp {
                act: Activation::SwiGLU,
                dt_w,
                di,
                up_bias: false,
                down_bias: false,
            },
        }
    }

    pub fn qwen2(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
        epsilon: f32,
    ) -> Self {
        let mut llama = Self::llama(dt_norm, dt_w, nh, nkvh, dh, di, epsilon);
        llama.self_attn.qkv_bias = true;
        llama
    }

    pub fn qw2vl_vit(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
    ) -> Self {
        let mut gpt2 = Self::gpt2(dt_norm, dt_w, nh, nkvh, dh, di);
        gpt2.self_attn.use_rope = true;
        gpt2.self_attn.mask = AttnMask::None;
        gpt2
    }

    pub fn gpt2(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
    ) -> Self {
        Self {
            norm: Normalization {
                ty: norm::Type::LayerNorm,
                dt_w: dt_norm,
            },
            self_attn: SelfAttn {
                dt_w,
                nh,
                nkvh,
                dh,
                qkv_bias: true,
                use_rope: false,
                mask: AttnMask::Causal,
                o_bias: true,
            },
            mlp: Mlp {
                act: Activation::GeLU,
                dt_w,
                di,
                up_bias: true,
                down_bias: true,
            },
        }
    }
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub embed: Tensor<'vm, VM>, // [n, d]
    pub pos: Tensor<'vm, VM>,   // [d]
    pub n_sin: usize,
    pub n_cos: usize,
    pub reqs: Vec<Request<'vm, VM>>,
}

pub struct Data {
    pub pre_norm: WeightBiasData,
    pub self_attn: self_attn::Data,
    pub post_norm: WeightBiasData,
    pub mlp: mlp::Data,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    PreNorm,
    SelfAttn,
    PostNorm,
    Mlp,
}

impl Id for Sub {
    fn name(&self) -> Cow<str> {
        match self {
            Sub::PreNorm => "pre-norm".into(),
            Sub::SelfAttn => "self-attn".into(),
            Sub::PostNorm => "post-norm".into(),
            Sub::Mlp => "mlp".into(),
        }
    }
}

pub trait Ops: norm::Ops + self_attn::Ops + mlp::Ops {}
impl<VM> Ops for VM where VM: norm::Ops + self_attn::Ops + mlp::Ops + ?Sized {}

impl<VM> NuralNetwork<VM> for TransformerBlk
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
            pre_norm,
            self_attn,
            post_norm,
            mlp,
        } = data;
        mapping
            .trap::<Normalization>(Sub::PreNorm, pre_norm)
            .trap::<SelfAttn>(Sub::SelfAttn, self_attn)
            .trap::<Normalization>(Sub::PostNorm, post_norm)
            .trap::<Mlp>(Sub::Mlp, mlp);
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let Self {
            norm,
            self_attn,
            mlp,
        } = self;
        let Args {
            embed: x,
            pos,
            n_sin,
            n_cos,
            reqs,
        } = args;

        let x1 = ctx.workspace(x.dt(), x.shape());

        ctx.trap(
            Sub::PreNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        )
        .trap(
            Sub::SelfAttn,
            self_attn,
            self_attn::Args {
                y: x.clone(),
                x: x1.clone(),
                pos,
                n_sin,
                n_cos,
                reqs,
            },
        )
        .trap(
            Sub::PostNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        )
        .trap(
            Sub::Mlp,
            mlp,
            mlp::Args {
                y: x,
                x: x1,
                scale: 1.,
                residual: true,
            },
        );
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Data, TransformerBlk};
    use crate::{
        VirtualMachineExt, WeightBiasData, mlp,
        self_attn::{self, Request},
    };
    use digit_layout::{DigitLayout, types};
    use test_vm::{TestVM, test_data};
    use vm::{VirtualMachine, device_id};

    #[test]
    fn test() {
        const DEVICE: device_id = 0;

        const DT: DigitLayout = types::F16;
        const DT_NORM: DigitLayout = types::F32;
        const NH: usize = 64;
        const NKVH: usize = 8;
        const DH: usize = 128;
        const N: usize = 11;
        const D: usize = 2048;
        const MAX_CTX: usize = 4096;
        const DI: usize = 11008;

        let vm = TestVM::default();
        let pid = vm.register("transformer");

        vm.init::<TransformerBlk>(
            pid,
            DEVICE,
            Data {
                pre_norm: WeightBiasData {
                    weight: test_data(DT_NORM, &[D]),
                    bias: None,
                },
                self_attn: self_attn::Data {
                    qkv: WeightBiasData {
                        weight: test_data(DT, &[(NH + NKVH + NKVH) * DH, D]),
                        bias: None,
                    },
                    rope: Some([
                        test_data(DT, &[MAX_CTX, DH / 2]),
                        test_data(DT, &[MAX_CTX, DH / 2]),
                    ]),
                    output: WeightBiasData {
                        weight: test_data(DT, &[D, NH * DH]),
                        bias: None,
                    },
                },
                post_norm: WeightBiasData {
                    weight: test_data(DT_NORM, &[D]),
                    bias: None,
                },
                mlp: mlp::Data {
                    up: WeightBiasData {
                        weight: test_data(DT, &[D * DI * 2]),
                        bias: None,
                    },
                    down: WeightBiasData {
                        weight: test_data(DT, &[DI * D]),
                        bias: None,
                    },
                },
            },
        )
        .forward(
            pid,
            DEVICE,
            &TransformerBlk::llama(DT_NORM, DT, NH, NKVH, DH, DI, 1e-5),
            Args {
                embed: vm.workspace(DT, &[N, D]),
                pos: vm.workspace(types::U32, &[N]),
                n_sin: MAX_CTX,
                n_cos: MAX_CTX,
                reqs: vec![
                    Request {
                        k_cache: vm.workspace(DT, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(DT, &[MAX_CTX, NKVH, DH]),
                        n_seq: 7,
                        pos: 20,
                    },
                    Request {
                        k_cache: vm.workspace(DT, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(DT, &[MAX_CTX, NKVH, DH]),
                        n_seq: 1,
                        pos: 30,
                    },
                    Request {
                        k_cache: vm.workspace(DT, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(DT, &[MAX_CTX, NKVH, DH]),
                        n_seq: 3,
                        pos: 40,
                    },
                ],
            },
        );

        vm.unregister(pid)
    }
}
