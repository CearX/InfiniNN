﻿mod graph;
mod name;
mod tensor;

use crate::op::Operator;
use std::{borrow::Borrow, cell::RefCell, collections::HashMap, hash::Hash, rc::Rc};

pub use graph::Context;
pub use tensor::{Tensor, TensorMeta};

#[derive(Default)]
pub struct GraphBuilder {
    op_lib: Rc<OpLib>,
}

impl GraphBuilder {
    pub fn register_op(
        &mut self,
        name: impl Into<String>,
        op: impl Operator + 'static,
    ) -> &mut Self {
        assert!(
            self.op_lib
                .0
                .borrow_mut()
                .insert(name.into(), Rc::new(op))
                .is_none()
        );
        self
    }
}

#[repr(transparent)]
#[derive(Default)]
pub struct OpLib(RefCell<HashMap<String, Rc<dyn Operator>>>);

impl OpLib {
    pub fn get<Q>(&self, name: &Q) -> Option<Rc<dyn Operator>>
    where
        String: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.0.borrow().get(name).cloned()
    }
}
