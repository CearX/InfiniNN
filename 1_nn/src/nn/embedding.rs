use super::{Context, NNError, NuralNetwork, TPTensor, Tensor};
use arg::Arg;
use tensor::digit_layout::DigitLayout;

#[derive(Clone)]
pub struct Embedding<T> {
    pub dt: DigitLayout,
    pub d: usize,
    pub wte: Table<T>,
    pub wpe: Option<Table<T>>,
    pub img_info: Option<[usize; 3]>,
}

#[derive(Clone)]
pub struct Table<T> {
    pub row: usize,
    pub weight: T,
}

impl<T> Embedding<T> {
    pub fn tensor_parallel(self) -> Embedding<TPTensor<T>> {
        let Self {
            dt,
            d,
            wte,
            wpe,
            img_info,
        } = self;
        Embedding {
            dt,
            d,
            wte: Table {
                row: wte.row,
                weight: wte.weight.into(),
            },
            wpe: wpe.map(|Table { row, weight }| Table {
                row,
                weight: weight.into(),
            }),
            img_info,
        }
    }
}

impl<T> NuralNetwork<T> for Embedding<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            dt,
            d,
            wte,
            wpe,
            img_info,
        } = self;
        let mut inputs = inputs.into_iter();

        let Table { row, weight } = wte;
        let wte = ctx.load_external("wte", dt, [row.into(), d.into()], weight);
        let tokens = inputs.next().unwrap();

        let arg = img_info
            .as_ref()
            .map(|x| Arg::arr(x.iter().map(|&val| Arg::int(val))));

        let outputs = match wpe {
            Some(wpe) => {
                let Table { row, weight } = wpe;
                let wpe = ctx.load_external("wpe", dt, [row.into(), d.into()], weight);
                let pos = inputs.next().unwrap();
                ctx.call("", "embedding", arg, [wte, tokens, wpe, pos])
            }
            None => {
                // format
                ctx.call("", "embedding", arg, [wte, tokens])
            }
        };

        Ok((ctx, outputs?))
    }
}
