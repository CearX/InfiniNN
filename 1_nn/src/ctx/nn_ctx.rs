﻿use super::{GraphBuilder, Tensor, TensorMeta, internal::GraphContext};
use crate::{Arg, Dim, NNError, NNGraph, NuralNetwork};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};
use tensor::digit_layout::DigitLayout;

impl GraphBuilder {
    pub fn build<T, NN: NuralNetwork<T>>(
        &self,
        nn: NN,
        inputs: impl IntoIterator<Item = TensorMeta>,
    ) -> Result<NNGraph<T>, NNError> {
        let (context, inputs) = self.new_context(inputs);
        let outputs = trap::<T, NN>("Ω".into(), &context, nn, inputs)?;
        Ok(context.take().into_graph(outputs))
    }
}

pub struct Context<'g, T> {
    path: String,
    graph: &'g GraphContext<T>,
    tensor_name: HashSet<String>,
    sub: NameDecorator,
    ops: NameDecorator,
}

impl<T> Context<'_, T> {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn trap<NN: NuralNetwork<T>>(
        &mut self,
        name: impl Display,
        nn: NN,
        inputs: impl IntoIterator<Item = Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        let Self {
            path, graph, sub, ..
        } = self;
        let name = sub.decorate(name.to_string());
        trap::<T, NN>(format!("{path}.{name}"), graph, nn, inputs)
    }

    pub fn load_external(
        &mut self,
        name: impl Display,
        dt: DigitLayout,
        shape: impl IntoIterator<Item = Dim>,
        item: T,
    ) -> Tensor<T> {
        let Self {
            path,
            graph,
            tensor_name,
            ..
        } = self;
        assert!(tensor_name.insert(name.to_string()));
        graph.load_external(format!("{path}.{name}"), dt, shape, item)
    }

    pub fn save_external(&mut self, name: impl Display, tensor: Tensor<T>, item: T) {
        let Self {
            path,
            graph,
            tensor_name,
            ..
        } = self;
        assert!(tensor_name.insert(name.to_string()));
        graph.save_external(format!("{path}.{name}"), tensor, item)
    }

    pub fn call(
        &mut self,
        name: impl Display,
        op: impl AsRef<str>,
        arg: Option<Arg>,
        inputs: impl IntoIterator<Item = Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        let Self {
            path, graph, ops, ..
        } = self;
        // 没有设置名字的，使用 op 名作为名字
        let mut name = name.to_string();
        if name.is_empty() {
            name = op.as_ref().into()
        }
        // 加序号去重
        let name = ops.decorate(name);
        // 连接到图
        graph
            .call(format!("{path}:{name}"), op, inputs, arg)
            .map_err(|err| NNError {
                name: format!("{path}:{name}"),
                err,
            })
    }
}

#[inline(always)]
fn trap<T, NN: NuralNetwork<T>>(
    path: String,
    context: &GraphContext<T>,
    nn: NN,
    inputs: impl IntoIterator<Item = Tensor<T>>,
) -> Result<Vec<Tensor<T>>, NNError> {
    nn.launch(
        inputs,
        Context {
            path,
            graph: context,
            tensor_name: Default::default(),
            sub: Default::default(),
            ops: Default::default(),
        },
    )
    .map(|(_, outputs)| outputs)
}

#[derive(Default)]
#[repr(transparent)]
struct NameDecorator(HashMap<String, usize>);

impl NameDecorator {
    pub fn decorate(&mut self, name: String) -> String {
        use std::collections::hash_map::Entry::*;
        match self.0.entry(name) {
            Occupied(mut entry) => {
                *entry.get_mut() += 1;
                format!("{}-{}", entry.key(), entry.get())
            }
            Vacant(entry) => {
                let ans = entry.key().clone();
                entry.insert(1);
                ans
            }
        }
    }
}
