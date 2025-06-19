use crate::CpuVM;
use core::slice;
use digit_layout::types;
use ggus::{GGuf, GGufMetaMapExt, ggml_quants::f16};
use memmap2::Mmap;
use nn::{
    VirtualMachineExt, WeightBiasData, mlp,
    normalization::{Normalization, Type},
    qw2vl::merger::{self, Merger},
    qw2vl::patch_embd::{self, PatchEmbd},
    self_attn,
    transformer::{self, Repeat, Transformer},
    transformer_blk::{self, TransformerBlk},
};
use std::{env::var_os, fs::File, ops::Deref, slice::from_raw_parts, sync::Arc};
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

    let vm = CpuVM::default();

    // image preprocess
    // todo
    // ✔️  clip.vision.patch_size····················u32: 14
    // ✔️  clip.vision.image_size····················u32: 560
    // ✔️  clip.vision.projection_dim················u32: 1536
    // ✔️  clip.vision.image_mean····················arr: [0.48145467, 0.4578275, 0.40821072]
    // ✔️  clip.vision.image_std·····················arr: [0.26862955, 0.2613026, 0.2757771]
    let d_patch = 14;
    let d_proj = 1536;
    let _image_mean = [0.48145467, 0.4578275, 0.40821072];
    let _image_std = [0.26862955, 0.2613026, 0.2757771];
    let image = vm.workspace(types::F16, &[1, 3, 336, 476]);
    let &[_, _, h, w] = image.shape() else {
        panic!()
    };
    let h_patches = h / d_patch;
    let w_patches = w / d_patch;
    let patches = h_patches * w_patches;

    // patch_embd

    let patch_embd = vm.register("patch_embd");
    let data = patch_embd::Data {
        patch_embd: Data::mmap(&file, &gguf, &format!("v.patch_embd.weight")).as_weight(),
        patch_embd1: Data::mmap(&file, &gguf, &format!("v.patch_embd.weight.1")).as_weight(),
    };
    vm.init::<PatchEmbd>(patch_embd, 0, data);

    // qw2vl_vit

    let qw2vl_vit = vm.register("qw2vl-vit");

    let nblk = gguf.llm_block_count().unwrap();
    let nh = gguf.llm_attention_head_count().unwrap();
    let d = gguf.llm_embedding_length().unwrap();
    let di = gguf.llm_feed_forward_length().unwrap();
    let _epsilon = gguf.llm_attention_layer_norm_epsilon().unwrap();

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
    vm.init::<Merger>(merger, 0, data);

    // forward

    let dt_w = types::F16;
    let dt_norm = types::F32;
    let patches_embd = vm.workspace(dt_w, &[patches, d]);
    let image_embd = vm.workspace(dt_w, &[patches / 4, d_proj]);
    vm.forward(
        patch_embd,
        0,
        &PatchEmbd { dt_w, bias: false },
        patch_embd::Args {
            patch_embd: patches_embd.clone(),
            raw: image,
            d_patch,
        },
    )
    .forward(
        qw2vl_vit,
        0,
        &Transformer::repeat(
            TransformerBlk::qw2vl_vit(dt_norm, dt_w, nh, nh, dh, di),
            nblk,
        ),
        transformer::Args {
            embed: patches_embd.clone(),
            n_sin: nctx,
            n_cos: nctx,
            // 无 kv_cache
            reqs: vec![transformer::Request {
                kv_cache: vm.workspace(dt_w, &[1]),
                n_seq: 1,
                pos: 0,
            }],
        },
    )
    .forward(
        merger,
        0,
        &Merger {
            post_norm: Normalization {
                ty: Type::LayerNorm,
                dt_w,
            },
            mlp: mlp::Mlp {
                act: mlp::Activation::GeLU,
                dt_w,
                di,
                up_bias: true,
                down_bias: true,
            },
        },
        merger::Args {
            y: image_embd,
            x: patches_embd,
        },
    );
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
