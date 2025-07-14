use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Rearrange;

impl Operator for Rearrange {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }
        destruct!([x] = inputs);

        Ok(vec![TensorMeta::new(x.dt, x.shape().to_vec())])
    }
}
