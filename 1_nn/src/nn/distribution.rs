﻿use std::{any::Any, hash::Hash, rc::Rc};
use tensor::Tensor;

/// 分布式切分方式
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Distribution {
    pub start: usize,
    pub len: usize,
    pub total: usize,
}

impl Distribution {
    pub const MONO: Self = Self {
        start: 0,
        len: 1,
        total: 1,
    };

    pub fn new(start: usize, len: usize, total: usize) -> Self {
        assert!(0 < len && start + len <= total);
        Self { start, len, total }
    }

    #[inline]
    pub const fn is_mono(&self) -> bool {
        self.len == self.total
    }
}

#[derive(Clone)]
pub struct TPTensor<T> {
    pub act: Option<TPAction>,
    pub val: T,
}

impl<T> From<T> for TPTensor<T> {
    fn from(value: T) -> Self {
        Self {
            act: None,
            val: value,
        }
    }
}

#[derive(Clone)]
pub struct TPAction {
    pub wt: Rc<dyn WeightType>,
    pub dist: Distribution,
}

impl TPAction {
    pub fn new<WT: WeightType>(wt: WT, dist: Distribution) -> Self {
        Self {
            wt: Rc::new(wt),
            dist,
        }
    }
}

impl PartialEq for TPAction {
    fn eq(&self, other: &Self) -> bool {
        self.wt.check_eq(&*other.wt) && self.dist == other.dist
    }
}

impl Eq for TPAction {}

impl Hash for TPAction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.wt.type_id().hash(state);
        self.dist.hash(state);
    }
}

pub trait WeightType: Any {
    fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>);
    fn split_shape(&self, dist: Distribution, shape: &[usize]) -> Box<[usize]>;
    fn check_eq(&self, other: &dyn WeightType) -> bool;
}

pub mod weight_types {
    use super::*;

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct AttnQKV(pub usize);

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct FfnGateUp;

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct ColumnTPWeight;

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct RowTPWeight;

    macro_rules! impl_wt_eq {
        () => {
            fn check_eq(&self, other: &dyn WeightType) -> bool {
                (other as &dyn Any)
                    .downcast_ref::<Self>()
                    .is_some_and(|other| self.eq(other))
            }
        };
    }

    impl WeightType for AttnQKV {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            assert!(src.is_contiguous());
            let Distribution { start, len, total } = dist;

            let &Self(gqa) = self;
            assert_eq!(src.shape()[0] % (gqa + 2), 0);
            assert_eq!(src.shape()[0] / (gqa + 2) % total, 0);

            let src = *src.get();
            let shard = src.len() / (gqa + 2);
            let piece = shard / total;
            dst[..gqa * len * piece]
                .copy_from_slice(&src[gqa * start * piece..][..gqa * len * piece]);
            dst[gqa * len * piece..][..len * piece]
                .copy_from_slice(&src[gqa * shard..][start * piece..][..len * piece]);
            dst[(gqa + 1) * len * piece..]
                .copy_from_slice(&src[(gqa + 1) * shard..][start * piece..][..len * piece]);
        }

        fn split_shape(&self, dist: Distribution, shape: &[usize]) -> Box<[usize]> {
            let Distribution { len, total, .. } = dist;
            match *shape {
                [r] => [r / total * len].into(),
                [r, c] => [r / total * len, c].into(),
                [..] => unreachable!(),
            }
        }
    }

    impl WeightType for FfnGateUp {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            assert!(src.is_contiguous());
            let Distribution { start, len, total } = dist;

            assert_eq!(src.shape()[0] % 2, 0);
            assert_eq!(src.shape()[0] / 2 % total, 0);

            let src = *src.get();
            let shard = src.len() / 2;
            let piece = shard / total;
            dst[..len * piece].copy_from_slice(&src[start * piece..][..len * piece]);
            dst[len * piece..].copy_from_slice(&src[shard..][start * piece..][..len * piece]);
        }

        fn split_shape(&self, dist: Distribution, shape: &[usize]) -> Box<[usize]> {
            let Distribution { len, total, .. } = dist;
            match *shape {
                [r] => [r / total * len].into(),
                [r, c] => [r / total * len, c].into(),
                [..] => unreachable!(),
            }
        }
    }

    impl WeightType for ColumnTPWeight {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            assert!(src.is_contiguous());
            let Distribution { start, len, total } = dist;

            assert_eq!(src.shape()[0] % total, 0);

            let src = *src.get();
            let piece = src.len() / total;
            dst.copy_from_slice(&src[start * piece..][..len * piece]);
        }

        fn split_shape(&self, dist: Distribution, shape: &[usize]) -> Box<[usize]> {
            let Distribution { len, total, .. } = dist;
            match *shape {
                [r] => [r / total * len].into(),
                [r, c] => [r / total * len, c].into(),
                [..] => unreachable!(),
            }
        }
    }

    impl WeightType for RowTPWeight {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            assert!(src.is_contiguous());
            let Distribution { start, len, total } = dist;

            match src.layout().ndim() {
                1 => dst.copy_from_slice(src.get()),
                2 => {
                    use mem_rearrange::Rearranging;

                    assert_eq!(src.shape()[1] % total, 0);
                    let piece = src.shape()[1] / total;
                    let src = src
                        .as_deref()
                        .transform(|layout| layout.slice(1, start * piece, 1, len * piece))
                        .map(|slice| slice.as_ptr());
                    let mut dst = src.use_info().map(|len| {
                        assert_eq!(size_of_val(dst), len);
                        dst.as_mut_ptr()
                    });

                    let scheme =
                        Rearranging::new(dst.layout(), src.layout(), src.dt().nbytes()).unwrap();
                    unsafe { scheme.launch(*dst.get_mut(), *src.get()) }
                }
                _ => unreachable!(),
            }
        }

        fn split_shape(&self, dist: Distribution, shape: &[usize]) -> Box<[usize]> {
            let Distribution { len, total, .. } = dist;
            match *shape {
                [r] => [r].into(),
                [r, c] => [r, c / total * len].into(),
                [..] => unreachable!(),
            }
        }
    }
}
