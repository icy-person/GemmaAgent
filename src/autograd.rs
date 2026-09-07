use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

#[derive(Clone)]
enum Op {
    Leaf,
    Add(Value, Value),
    Mul(Value, Value),
    MatMul(Value, Value),
    Transpose(Value),
    Log(Value),
    Neg(Value),
    Softmax(Value),
    Relu(Value),
    Silu(Value),
    ConcatRows(Vec<Value>),
    ConcatCols(Vec<Value>),
    Slice(Value, usize, usize),
    Gather(Value, usize),
    RowGather(Value, usize),
}

struct Node {
    r: usize,
    c: usize,
    d: Vec<f32>,
    g: Vec<f32>,
    op: Op,
}

impl Value {
    fn mk(r: usize, c: usize, d: Vec<f32>, op: Op) -> Self {
        assert_eq!(r.checked_mul(c), Some(d.len()), "invalid tensor shape");
        Self(Rc::new(RefCell::new(Node {
            r,
            c,
            g: vec![0.0; d.len()],
            d,
            op,
        })))
    }

    pub fn leaf(r: usize, c: usize, d: Vec<f32>) -> Self {
        Self::mk(r, c, d, Op::Leaf)
    }

    pub fn parameter(r: usize, c: usize, seed: &mut u64) -> Self {
        let mut d = Vec::with_capacity(r * c);
        for _ in 0..r * c {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((*seed >> 32) as u32) as f32 / u32::MAX as f32;
            d.push((u - 0.5) * 0.05);
        }
        Self::leaf(r, c, d)
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().d.clone()
    }

    pub fn grad(&self) -> Vec<f32> {
        self.0.borrow().g.clone()
    }

    pub fn shape(&self) -> (usize, usize) {
        let n = self.0.borrow();
        (n.r, n.c)
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().g.fill(0.0);
    }

    pub fn set_data(&self, data: Vec<f32>) {
        assert_eq!(data.len(), self.0.borrow().d.len(), "parameter length mismatch");
        self.0.borrow_mut().d = data;
    }

    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "add shape mismatch");
        let x = self.data();
        let y = rhs.data();
        Self::mk(
            self.shape().0,
            self.shape().1,
            x.iter().zip(y).map(|(a, b)| a + b).collect(),
            Op::Add(self.clone(), rhs.clone()),
        )
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "mul shape mismatch");
        let x = self.data();
        let y = rhs.data();
        Self::mk(
            self.shape().0,
            self.shape().1,
            x.iter().zip(y).map(|(a, b)| a * b).collect(),
            Op::Mul(self.clone(), rhs.clone()),
        )
    }

    pub fn scalar_mul(&self, scalar: f32) -> Self {
        assert!(scalar.is_finite());
        self.mul(&Self::leaf(
            self.shape().0,
            self.shape().1,
            vec![scalar; self.data().len()],
        ))
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        assert!(scalar.is_finite() && scalar != 0.0);
        self.scalar_mul(1.0 / scalar)
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        let (ar, ac) = self.shape();
        let (br, bc) = rhs.shape();
        assert_eq!(ac, br, "matmul shape mismatch: {ar}x{ac} · {br}x{bc}");
        let x = self.data();
        let y = rhs.data();
        let mut out = vec![0.0; ar * bc];
        for i in 0..ar {
            for k in 0..ac {
                let a = x[i * ac + k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..bc {
                    out[i * bc + j] += a * y[k * bc + j];
                }
            }
        }
        Self::mk(ar, bc, out, Op::MatMul(self.clone(), rhs.clone()))
    }

    pub fn transpose(&self) -> Self {
        let (rows, cols) = self.shape();
        let x = self.data();
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = x[i * cols + j];
            }
        }
        Self::mk(cols, rows, out, Op::Transpose(self.clone()))
    }

    pub fn log(&self) -> Self {
        Self::mk(
            self.shape().0,
            self.shape().1,
            self.data().into_iter().map(|x| x.max(1e-20).ln()).collect(),
            Op::Log(self.clone()),
        )
    }

    pub fn neg(&self) -> Self {
        Self::mk(
            self.shape().0,
            self.shape().1,
            self.data().into_iter().map(|x| -x).collect(),
            Op::Neg(self.clone()),
        )
    }

    pub fn relu(&self) -> Self {
        Self::mk(
            self.shape().0,
            self.shape().1,
            self.data().into_iter().map(|x| x.max(0.0)).collect(),
            Op::Relu(self.clone()),
        )
    }

    pub fn silu(&self) -> Self {
        Self::mk(
            self.shape().0,
            self.shape().1,
            self.data().into_iter().map(|x| x / (1.0 + (-x).exp())).collect(),
            Op::Silu(self.clone()),
        )
    }

    pub fn softmax(&self) -> Self {
        let (rows, cols) = self.shape();
        assert_eq!(rows, 1, "softmax expects a row vector");
        let x = self.data();
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = x.iter().map(|v| (*v - max).exp()).collect();
        let sum = exp.iter().sum::<f32>().max(1e-20);
        Self::mk(
            1,
            cols,
            exp.into_iter().map(|v| v / sum).collect(),
            Op::Softmax(self.clone()),
        )
    }

    pub fn concat_rows(xs: &[Self]) -> Self {
        assert!(!xs.is_empty());
        let cols = xs[0].shape().1;
        let mut data = Vec::new();
        for x in xs {
            assert_eq!(x.shape(), (1, cols), "concat_rows shape mismatch");
            data.extend(x.data());
        }
        Self::mk(xs.len(), cols, data, Op::ConcatRows(xs.to_vec()))
    }

    pub fn concat_cols(xs: &[Self]) -> Self {
        assert!(!xs.is_empty());
        let rows = xs[0].shape().0;
        let cols: usize = xs
            .iter()
            .map(|x| {
                assert_eq!(x.shape().0, rows, "concat_cols row mismatch");
                x.shape().1
            })
            .sum();
        let mut out = vec![0.0; rows * cols];
        let mut offset = 0;
        for x in xs {
            let width = x.shape().1;
            let data = x.data();
            for i in 0..rows {
                out[i * cols + offset..i * cols + offset + width]
                    .copy_from_slice(&data[i * width..(i + 1) * width]);
            }
            offset += width;
        }
        Self::mk(rows, cols, out, Op::ConcatCols(xs.to_vec()))
    }

    pub fn slice_cols(&self, start: usize, len: usize) -> Self {
        let (rows, cols) = self.shape();
        assert!(start <= cols && len <= cols - start);
        let data = self.data();
        let mut out = Vec::with_capacity(rows * len);
        for i in 0..rows {
            out.extend_from_slice(&data[i * cols + start..i * cols + start + len]);
        }
        Self::mk(rows, len, out, Op::Slice(self.clone(), start, len))
    }

    pub fn gather(&self, index: usize) -> Self {
        let (rows, cols) = self.shape();
        assert_eq!(rows, 1, "gather expects a row vector");
        assert!(index < cols);
        Self::mk(1, 1, vec![self.data()[index]], Op::Gather(self.clone(), index))
    }

    pub fn row(&self, row: usize) -> Self {
        let (rows, cols) = self.shape();
        assert!(row < rows);
        let data = self.data();
        Self::mk(
            1,
            cols,
            data[row * cols..(row + 1) * cols].to_vec(),
            Op::RowGather(self.clone(), row),
        )
    }

    pub fn backward(&self) {
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        topo(self, &mut seen, &mut order);
        self.0.borrow_mut().g.fill(1.0);
        for node in order.into_iter().rev() {
            back(&node);
        }
    }
}

fn topo(value: &Value, seen: &mut HashSet<usize>, order: &mut Vec<Value>) {
    if !seen.insert(value.id()) {
        return;
    }
    match value.0.borrow().op.clone() {
        Op::Leaf => {}
        Op::Add(a, b) | Op::Mul(a, b) | Op::MatMul(a, b) => {
            topo(&a, seen, order);
            topo(&b, seen, order);
        }
        Op::Transpose(a)
        | Op::Log(a)
        | Op::Neg(a)
        | Op::Softmax(a)
        | Op::Relu(a)
        | Op::Silu(a)
        | Op::Slice(a, _, _)
        | Op::Gather(a, _)
        | Op::RowGather(a, _) => topo(&a, seen, order),
        Op::ConcatRows(xs) | Op::ConcatCols(xs) => {
            for x in xs {
                topo(&x, seen, order);
            }
        }
    }
    order.push(value.clone());
}

fn accumulate(value: &Value, grad: &[f32]) {
    let mut node = value.0.borrow_mut();
    assert_eq!(node.g.len(), grad.len());
    for (dst, src) in node.g.iter_mut().zip(grad) {
        *dst += *src;
    }
}

fn back(value: &Value) {
    let grad = value.grad();
    match value.0.borrow().op.clone() {
        Op::Leaf => {}
        Op::Add(a, b) => {
            accumulate(&a, &grad);
            accumulate(&b, &grad);
        }
        Op::Mul(a, b) => {
            let x = a.data();
            let y = b.data();
            accumulate(&a, &grad.iter().zip(&y).map(|(g, y)| g * y).collect::<Vec<_>>());
            accumulate(&b, &grad.iter().zip(&x).map(|(g, x)| g * x).collect::<Vec<_>>());
        }
        Op::MatMul(a, b) => {
            let (ar, ac) = a.shape();
            let (_, bc) = b.shape();
            let x = a.data();
            let y = b.data();
            let mut ga = vec![0.0; ar * ac];
            let mut gb = vec![0.0; ac * bc];
            for i in 0..ar {
                for k in 0..ac {
                    for j in 0..bc {
                        let g = grad[i * bc + j];
                        ga[i * ac + k] += g * y[k * bc + j];
                        gb[k * bc + j] += x[i * ac + k] * g;
                    }
                }
            }
            accumulate(&a, &ga);
            accumulate(&b, &gb);
        }
        Op::Transpose(a) => {
            let (out_rows, out_cols) = value.shape();
            let mut ga = vec![0.0; grad.len()];
            for i in 0..out_rows {
                for j in 0..out_cols {
                    ga[j * out_rows + i] = grad[i * out_cols + j];
                }
            }
            accumulate(&a, &ga);
        }
        Op::Log(a) => {
            let data = a.data();
            accumulate(
                &a,
                &grad
                    .iter()
                    .zip(data)
                    .map(|(g, x)| g / x.max(1e-20))
                    .collect::<Vec<_>>(),
            );
        }
        Op::Neg(a) => {
            accumulate(&a, &grad.iter().map(|g| -g).collect::<Vec<_>>());
        }
        Op::Relu(a) => {
            let data = a.data();
            accumulate(
                &a,
                &grad
                    .iter()
                    .zip(data)
                    .map(|(g, x)| if x > 0.0 { *g } else { 0.0 })
                    .collect::<Vec<_>>(),
            );
        }
        Op::Silu(a) => {
            let data = a.data();
            let ga = data
                .iter()
                .zip(&grad)
                .map(|(x, g)| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    g * s * (1.0 + x * (1.0 - s))
                })
                .collect::<Vec<_>>();
            accumulate(&a, &ga);
        }
        Op::Softmax(a) => {
            let y = value.data();
            let dot = grad.iter().zip(&y).map(|(g, y)| g * y).sum::<f32>();
            let ga = y
                .iter()
                .zip(&grad)
                .map(|(y, g)| y * (g - dot))
                .collect::<Vec<_>>();
            accumulate(&a, &ga);
        }
        Op::ConcatRows(xs) => {
            let cols = value.shape().1;
            for (row, x) in xs.iter().enumerate() {
                accumulate(x, &grad[row * cols..(row + 1) * cols]);
            }
        }
        Op::ConcatCols(xs) => {
            let rows = value.shape().0;
            let cols = value.shape().1;
            let mut offset = 0;
            for x in xs {
                let width = x.shape().1;
                let mut gx = vec![0.0; rows * width];
                for row in 0..rows {
                    gx[row * width..(row + 1) * width]
                        .copy_from_slice(&grad[row * cols + offset..row * cols + offset + width]);
                }
                accumulate(&x, &gx);
                offset += width;
            }
        }
        Op::Slice(a, start, len) => {
            let (rows, cols) = a.shape();
            let mut ga = vec![0.0; rows * cols];
            for row in 0..rows {
                ga[row * cols + start..row * cols + start + len]
                    .copy_from_slice(&grad[row * len..(row + 1) * len]);
            }
            accumulate(&a, &ga);
        }
        Op::Gather(a, index) => {
            let mut ga = vec![0.0; a.data().len()];
            ga[index] = grad[0];
            accumulate(&a, &ga);
        }
        Op::RowGather(a, row) => {
            let cols = a.shape().1;
            let mut ga = vec![0.0; a.data().len()];
            ga[row * cols..(row + 1) * cols].copy_from_slice(&grad);
            accumulate(&a, &ga);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_backward() {
        let a = Value::leaf(1, 2, vec![2.0, 3.0]);
        let b = Value::leaf(2, 1, vec![5.0, 7.0]);
        let y = a.matmul(&b);
        y.backward();
        assert_eq!(a.grad(), vec![5.0, 7.0]);
        assert_eq!(b.grad(), vec![2.0, 3.0]);
    }

    #[test]
    fn transpose_backward() {
        let a = Value::leaf(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let y = a.transpose();
        y.backward();
        assert_eq!(a.grad(), vec![1.; 6]);
    }

    #[test]
    fn concat_cols_matrix_backward() {
        let a = Value::leaf(2, 1, vec![1., 2.]);
        let b = Value::leaf(2, 2, vec![3., 4., 5., 6.]);
        let y = Value::concat_cols(&[a.clone(), b.clone()]);
        y.backward();
        assert_eq!(a.grad(), vec![1., 1.]);
        assert_eq!(b.grad(), vec![1., 1., 1., 1.]);
    }

    #[test]
    fn slice_matrix_backward() {
        let a = Value::leaf(2, 4, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let y = a.slice_cols(1, 2);
        y.backward();
        assert_eq!(a.grad(), vec![0., 1., 1., 0., 0., 1., 1., 0.]);
    }

    #[test]
    fn row_gather_backward() {
        let a = Value::leaf(3, 2, vec![1., 2., 3., 4., 5., 6.]);
        let y = a.row(1);
        y.backward();
        assert_eq!(a.grad(), vec![0., 0., 1., 1., 0., 0.]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let y = Value::leaf(1, 3, vec![1., 2., 3.]).softmax();
        let sum: f32 = y.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silu_backward() {
        let x = Value::leaf(1, 1, vec![0.7]);
        let y = x.silu();
        y.backward();
        let s = 1.0 / (1.0 + (-0.7_f32).exp());
        let expected = s * (1.0 + 0.7 * (1.0 - s));
        assert!((x.grad()[0] - expected).abs() < 1e-6);
    }
}
