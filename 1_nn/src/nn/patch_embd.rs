use super::{Context, Distribution, NNError, NuralNetwork, TPTensor, Tensor, macros::destruct};
use crate::macros::dims;
use arg::Arg;
use tensor::digit_layout::DigitLayout;

#[allow(dead_code)]
#[derive(Clone)]
pub struct PatchEmbd<T> {
    pub dt: DigitLayout,
    pub shape: [usize; 4],
    pub patch_embd: T,
    pub patch_embd1: T,
}

#[allow(dead_code)]
impl<T> PatchEmbd<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> PatchEmbd<TPTensor<T>> {
        let Self {
            dt,
            shape,
            patch_embd,
            patch_embd1,
        } = self;

        if dist.is_mono() {
            todo!();
        } else {
            PatchEmbd {
                dt,
                shape,
                patch_embd: TPTensor::from(patch_embd),
                patch_embd1: TPTensor::from(patch_embd1),
            }
        }
    }
}

impl<T> NuralNetwork<T> for PatchEmbd<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, b] = inputs);

        dims!([n, c, h, w] = x);
        let [n, c, height, width] = [n, c, h, w].map(|v| v.to_usize());
        let Self {
            dt,
            shape,
            patch_embd,
            patch_embd1,
        } = self;
        let [m, channel, h_k, w_k] = shape;
        assert_eq!(n, 1);
        assert_eq!(c, channel);
        assert_eq!(h_k, w_k);
        let w = ctx.load_external(
            "patch_embd",
            dt,
            [m.into(), channel.into(), h_k.into(), w_k.into()],
            patch_embd,
        );
        let w1 = ctx.load_external(
            "patch_embd1",
            dt,
            [m.into(), channel.into(), h_k.into(), w_k.into()],
            patch_embd1,
        );
        let tensors = ctx
            .call("", "conv", None, [x.clone(), w.clone(), b.clone()])
            .unwrap();
        destruct!([patch_embd] = tensors);
        let tensors = ctx.call("", "conv", None, [x, w1, b]).unwrap();
        destruct!([patch_embd1] = tensors);
        let tensors = ctx
            .call("", "add", None, [patch_embd, patch_embd1])
            .unwrap();
        destruct!([embd] = tensors);

        // reshape

        let h_p = (height / h_k) as u64; // h_patches
        let w_p = (width / w_k) as u64; // w_patches
        // [n, m, h_p, w_p] -> [n, h_p, w_p, m]
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "transpose",
                    Some(Arg::dict([(
                        "perm".into(),
                        Arg::arr([0, 2, 3, 1].map(Arg::from)),
                    )])),
                    [embd.clone()],
                )
                .unwrap()
        );
        destruct!(
            [embd] = ctx
                .call("", "rearrange", None, [embd.clone(), embd_],)
                .unwrap()
        );
        // [n, h_p, w_p, m] -> [n * h_p/2, 2, w_p/2, 2*m]
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "tile",
                    Some(Arg::dict([
                        ("axis".into(), Arg::int(1)),
                        ("tiles".into(), Arg::arr([h_p / 2, 2].map(Arg::from)),)
                    ])),
                    [embd.clone()],
                )
                .unwrap()
        );
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "merge",
                    Some(Arg::dict([
                        ("start".into(), Arg::int(0)),
                        ("len".into(), Arg::int(2),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "tile",
                    Some(Arg::dict([
                        ("axis".into(), Arg::int(2)),
                        ("tiles".into(), Arg::arr([w_p / 2, 2].map(Arg::from)),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "merge",
                    Some(Arg::dict([
                        ("start".into(), Arg::int(3)),
                        ("len".into(), Arg::int(2),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        // [n * h_p/2, 2, w_p/2, 2*m] -> [n * h_p/2, w_p/2, 2, 2*m]
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "transpose",
                    Some(Arg::dict([(
                        "perm".into(),
                        Arg::arr([0, 2, 1, 3].map(Arg::from)),
                    )])),
                    [embd_],
                )
                .unwrap()
        );
        destruct!(
            [embd] = ctx
                .call("", "rearrange", None, [embd.clone(), embd_],)
                .unwrap()
        );
        // [n * h_p/2, w_p/2, 2, 2*m] -> [n, h_p * w_p, m]
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "tile",
                    Some(Arg::dict([
                        ("axis".into(), Arg::int(0)),
                        ("tiles".into(), Arg::arr([n as u64, h_p / 2].map(Arg::from)),)
                    ])),
                    [embd.clone()],
                )
                .unwrap()
        );
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "merge",
                    Some(Arg::dict([
                        ("start".into(), Arg::int(1)),
                        ("len".into(), Arg::int(3),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "tile",
                    Some(Arg::dict([
                        ("axis".into(), Arg::int(2)),
                        ("tiles".into(), Arg::arr([2, m as u64].map(Arg::from)),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "merge",
                    Some(Arg::dict([
                        ("start".into(), Arg::int(1)),
                        ("len".into(), Arg::int(2),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        // [n, h_p * w_p, m] -> [n * patches, m]
        destruct!(
            [embd_] = ctx
                .call(
                    "",
                    "merge",
                    Some(Arg::dict([
                        ("start".into(), Arg::int(0)),
                        ("len".into(), Arg::int(2),)
                    ])),
                    [embd_],
                )
                .unwrap()
        );
        let image_embd = ctx
            .call("", "rearrange", None, [embd.clone(), embd_])
            .unwrap();

        Ok((ctx, image_embd))
    }
}
