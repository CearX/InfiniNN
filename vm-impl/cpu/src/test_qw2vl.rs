use crate::CpuVM;
use core::slice;
use digit_layout::types;
use ggus::{GGmlTokenType, GGuf, GGufMetaMapExt, ggml_quants::f16};
use memmap2::Mmap;
use nn::{
    VirtualMachineExt, WeightBiasData, conv,
    lm_output::{self, LmOutput},
    mlp,
    normalization::{Normalization, Type},
    qw2vl::merger::{self, Merger},
    qw2vl::patch_embd::{self, Patchembd},
    self_attn,
    token_embed::{self, TokenEmbed},
    transformer::{self, Repeat, Transformer},
    transformer_blk::{self, TransformerBlk},
};
use std::{env::var_os, fs::File, io::Write, ops::Deref, slice::from_raw_parts, sync::Arc};
use tokeneer::{Bpe, Tokeneer};
use vm::{VirtualMachine, op::RotaryType};

#[test]
fn test() {
    let Some(path) = var_os("TEST_MODEL") else {
        return;
    };
    let file = File::open(path).unwrap();
    let file = Arc::new(unsafe { Mmap::map(&file) }.unwrap());
    let gguf = GGuf::new(&file).unwrap();

    assert_eq!(gguf.general_architecture().unwrap(), "clip.vision");

    // image preprocess
    // todo
    // ✔️  clip.vision.patch_size····················u32: 14
    // ✔️  clip.vision.image_size····················u32: 560
    // ✔️  clip.vision.projection_dim················u32: 1536
    // ✔️  clip.vision.image_mean····················arr: [0.48145467, 0.4578275, 0.40821072]
    // ✔️  clip.vision.image_std·····················arr: [0.26862955, 0.2613026, 0.2757771]
    let image = vec![1.0; 3 * 336 * 476];

    let vm = CpuVM::default();

    // patch_embd

    let patch_embd = vm.register("patch_embd");
    let data = patch_embd::Data {
        patch_embd: conv::Data {
            conv: Data::mmap(&file, &gguf, &format!("v.patch_embd.weight")).as_weight(),
        },
        patch_embd1: conv::Data {
            conv: Data::mmap(&file, &gguf, &format!("v.patch_embd.weight.1")).as_weight(),
        },
    };
    // vm.init::<Patchembd>(patch_embd, 0, data);

    // qw2vl vit

    let qw2vl_vit = vm.register("qw2vl-vit");

    let nblk = gguf.llm_block_count().unwrap();
    let nh = gguf.llm_attention_head_count().unwrap();
    let d = gguf.llm_embedding_length().unwrap();
    let di = gguf.llm_feed_forward_length().unwrap();
    let epsilon = gguf.llm_attention_layer_norm_epsilon().unwrap();

    // let nctx = gguf.llm_context_length().unwrap();
    // let theta = gguf.llm_rope_freq_base().unwrap();
    // let dh = gguf.llm_rope_dimension_count().unwrap();
    let nctx = 34;
    let dh = 80;
    let theta = 1e4;

    let [sin, cos] = RotaryType::Normal { theta }.generate(nctx, dh);
    let sin = share_raw(&sin.into_iter().map(f16::from_f32).collect::<Vec<_>>());
    let cos = share_raw(&cos.into_iter().map(f16::from_f32).collect::<Vec<_>>());

    let data = (0..nblk)
        .map(|i| transformer_blk::Data {
            pre_norm: WeightBiasData {
                weight: Data::mmap(&file, &gguf, &format!("v.blk.{i}.ln1.weight")),
                bias: Some(Data::mmap(&file, &gguf, &format!("v.blk.{i}.ln1.bias"))),
            },
            self_attn: self_attn::Data {
                qkv: WeightBiasData {
                    weight: Data::mmap(&file, &gguf, &format!("v.blk.{i}.attn_qkv.weight")),
                    bias: Some(Data::mmap(
                        &file,
                        &gguf,
                        &format!("v.blk.{i}.attn_qkv.bias"),
                    )),
                },
                rope: Some([Data::share(sin.clone()), Data::share(cos.clone())]),
                output: WeightBiasData {
                    weight: Data::mmap(&file, &gguf, &format!("v.blk.{i}.attn_out.weight")),
                    bias: Some(Data::mmap(
                        &file,
                        &gguf,
                        &format!("v.blk.{i}.attn_out.bias"),
                    )),
                },
            },
            post_norm: WeightBiasData {
                weight: Data::mmap(&file, &gguf, &format!("v.blk.{i}.ln2.weight")),
                bias: Some(Data::mmap(&file, &gguf, &format!("v.blk.{i}.ln2.bias"))),
            },
            mlp: mlp::Data {
                up: WeightBiasData {
                    weight: Data::mmap(&file, &gguf, &format!("v.blk.{i}.ffn_up.weight")),
                    bias: Some(Data::mmap(&file, &gguf, &format!("v.blk.{i}.ffn_up.bias"))),
                },
                down: WeightBiasData {
                    weight: Data::mmap(&file, &gguf, &format!("v.blk.{i}.ffn_down.weight")),
                    bias: Some(Data::mmap(
                        &file,
                        &gguf,
                        &format!("v.blk.{i}.ffn_down.bias"),
                    )),
                },
            },
        })
        .collect();
    vm.init::<Transformer<Repeat<TransformerBlk>>>(qw2vl_vit, 0, data);

    // merger

    let merger = vm.register("merger");

    let data = merger::Data {
        post_norm: WeightBiasData {
            weight: Data::mmap(&file, &gguf, &format!("v.post_ln.weight")),
            bias: Some(Data::mmap(&file, &gguf, &format!("v.post_ln.bias"))),
        },
        mlp: mlp::Data {
            up: WeightBiasData {
                weight: Data::mmap(&file, &gguf, &format!("mm.0.weight")),
                bias: Some(Data::mmap(&file, &gguf, &format!("mm.0.bias"))),
            },
            down: WeightBiasData {
                weight: Data::mmap(&file, &gguf, &format!("mm.2.weight")),
                bias: Some(Data::mmap(&file, &gguf, &format!("mm.2.bias"))),
            },
        },
    };
    // vm.init::<Merger>(merger, 0, data);

    // forward
    // todo
}

enum Data {
    Generate(Arc<[u8]>),
    Mmap {
        _mmap: Arc<Mmap>,
        slice: &'static [u8],
    },
}

impl Data {
    fn mmap(mmap: &Arc<Mmap>, gguf: &GGuf, name: &str) -> Box<Self> {
        let info = gguf.tensors[name].to_info();
        let data = &gguf.data[info.offset()..][..info.nbytes()];
        let slice = unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) };
        Box::new(Self::Mmap {
            _mmap: mmap.clone(),
            slice,
        })
    }

    fn share(data: Arc<[u8]>) -> Box<Self> {
        Box::new(Self::Generate(data))
    }

    fn as_weight(self: Box<Self>) -> WeightBiasData {
        WeightBiasData {
            weight: self,
            bias: None,
        }
    }
}

impl Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Generate(data) => data,
            Self::Mmap { slice, .. } => slice,
        }
    }
}

fn share_raw<T: Copy>(data: &[T]) -> Arc<[u8]> {
    unsafe { from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
        .to_vec()
        .into()
}
