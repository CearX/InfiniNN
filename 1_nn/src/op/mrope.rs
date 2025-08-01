use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Mrope;

impl Operator for Mrope {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        match inputs {
            [x, pos, sin, cos] => {
                dims!([_n, _d] = x);
                dims!([n_pos, d_pos] = pos);
                dims!([n_ctx_sin, dh_2_sin] = sin);
                dims!([n_ctx_cos, dh_2_cos] = cos);

                match d_pos.to_usize() {
                    2 => {
                        if args.is_some() {
                            return Err(OpError::ArgError);
                        }
                    }
                    3 => {
                        let Some(Arg::Arr(_mrope_section)) = args else {
                            return Err(OpError::ArgError);
                        };
                    }
                    _ => return Err(OpError::ShapeError),
                }

                // Check if context lengths match
                if n_ctx_sin != n_ctx_cos {
                    return Err(OpError::ShapeMismatch);
                }

                // Check if half embedding dimensions match
                if dh_2_sin != dh_2_cos {
                    return Err(OpError::ShapeMismatch);
                }

                let _n = make_eq(&[&x.shape[0], n_pos]).ok_or(OpError::ShapeMismatch)?;

                Ok(vec![TensorMeta::new(x.dt, [_n, _d.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
