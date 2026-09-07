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
        assert_eq!(r * c, d.len());
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

    pub fn parameter(r: usize, c: usize, s: &mut u64) -> Self {
        let mut d = Vec::with_capacity(r * c);
        for _ in 0..r * c {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((*s >> 32) as u32) as f32 / u32::MAX as f32;
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

    pub fn set_data(&self, d: Vec<f32>) {
        assert_eq!(d.len(), self.0.borrow().d.len());
        self.0.borrow_mut().d = d;
    }

    pub fn add(&self, b: &Self) -> Self {
        let x = self.data();
        let y = b.data();
        assert_eq!(self.shape(), b.shape());
        Self::mk(
            self.shape().0,
            self.shape().1,
            x.iter().zip(y).map(|(a, b)| a + b).collect(),
            Op::Add(self.clone(), b.clone()),
        )
    }

    pub fn mul(&self, b: &Self) -> Self {
        let x = self.data();
        let y = b.data();
        assert_eq!(self.shape(), b.shape());
        Self::mk(
            self.shape().0,
            self.shape().1,
            x.iter().zip(y).map(|(a, b)| a * b).collect(),
            Op::Mul(self.clone(), b.clone()),
        )
    }

    pub fn scalar_mul(&self, s: f32) -> Self {
        self.mul(&Self::leaf(
            self.shape().0,
            self.shape().1,
            vec![s; self.0.borrow().d.len()],
        ))
    }

    pub fn div_scalar(&self, s: f32) -> Self {
        assert!(s.is_finite() && s != 0.0);
        self.scalar_mul(1.0 / s)
    }

    pub fn matmul(&self, b: &Self) -> Self {
        let (ar, ac) = self.shape();
        let (br, bc) = b.shape();
        assert_eq!(ac, br, "matmul shape mismatch: {ar}x{ac} · {br}x{bc}");
        let x = self.data();
        let y = b.data();
        let mut o = vec![0.0; ar * bc];
        for i in 0..ar {
            for k in 0..ac {
                let a = x[i * ac + k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..bc {
                    o[i * bc + j] += a * y[k * bc + j];
                }
            }
        }
        Self::mk(ar, bc, o, Op::MatMul(self.clone(), b.clone()))
    }

    pub fn transpose(&self) -> Self {
        let (r, c) = self.shape();
        let x = self.data();
        let mut o = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = x[i * c + j];
            }
        }
        Self::mk(c, r, o, Op::Transpose(self.clone()))
    }

    pub fn log(&self) -> Self {
        Self::mk(
            self.shape().0,
            self.shape().1,
            self.data()
                .into_iter()
                .map(|x| x.max(1e-20).ln())
                .collect(),
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
            self.data()
                .into_iter()
                .map(|x| x / (1.0 + (-x).exp()))
                .collect(),
            Op::Silu(self.clone()),
        )
    }

    pub fn softmax(&self) -> Self {
        let (r, c) = self.shape();
        assert_eq!(r, 1, "softmax currently expects a row vector");
        let x = self.data();
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (*v - m).exp()).collect();
        let z = e.iter().sum::<f32>().max(1e-20);
        Self::mk(
            1,
            c,
            e.into_iter().map(|v| v / z).collect(),
            Op::Softmax(self.clone()),
        )
    }

    pub fn concat_rows(xs: &[Self]) -> Self {
        assert!(!xs.is_empty());
        let c = xs[0].shape().1;
        let mut d = Vec::new();
        for x in xs {
            assert_eq!(x.shape(), (1, c));
            d.extend(x.data());
        }
        Self::mk(xs.len(), c, d, Op::ConcatRows(xs.to_vec()))
    }

    pub fn concat_cols(xs: &[Self]) -> Self {
        assert!(!xs.is_empty());
        let r = xs[0].shape().0;
        let mut c = 0;
        for x in xs {
            assert_eq!(x.shape().0, r);
            c += x.shape().1;
        }
        let mut o = vec![0.0; r * c];
        let mut off = 0;
        for x in xs {
            let w = x.shape().1;
            let d = x.data();
            for i in 0..r {
                for j in 0..w {
                    o[i * c + off + j] = d[i * w + j];
                }
            }
            off += w;
        }
        Self::mk(r, c, o, Op::ConcatCols(xs.to_vec()))
    }

    pub fn slice_cols(&self, start: usize, len: usize) -> Self {
        let (r, c) = self.shape();
        assert!(start + len <= c);
        let d = self.data();
        let mut o = Vec::with_capacity(r * len);
        for i in 0..r {
            o.extend_from_slice(&d[i * c + start..i * c + start + len]);
        }
        Self::mk(r, len, o, Op::Slice(self.clone(), start, len))
    }

    pub fn gather(&self, i: usize) -> Self {
        let (r, c) = self.shape();
        assert_eq!(r, 1);
        assert!(i < c);
        Self::mk(1, 1, vec![self.data()[i]], Op::Gather(self.clone(), i))
    }

    pub fn row(&self, row: usize) -> Self {
        let (r, c) = self.shape();
        assert!(row < r);
        let d = self.data();
        Self::mk(1, c, d[row * c..(row + 1) * c].to_vec(), Op::RowGather(self.clone(), row))
    }

    pub fn backward(&self) {
        let mut topo_order = Vec::new();
        let mut seen = HashSet::new();
        topo(self, &mut seen, &mut topo_order);
        self.0.borrow_mut().g.fill(1.0);
        for v in topo_order.into_iter().rev() {
            back(&v);
        }
    }
}

fn topo(v: &Value, seen: &mut HashSet<usize>, order: &mut Vec<Value>) {
    if !seen.insert(v.id()) {
        return;
    }
    match v.0.borrow().op.clone() {
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
    order.push(v.clone());
}

fn ag(v: &Value, g: &[f32]) {
    let mut n = v.0.borrow_mut();
    for (dst, src) in n.g.iter_mut().zip(g) {
        *dst += *src;
    }
}

fn back(v: &Value) {
    let g = v.grad();
    match v.0.borrow().op.clone() {
        Op::Leaf => {}
        Op::Add(a, b) => {
            ag(&a, &g);
            ag(&b, &g);
        }
        Op::Mul(a, b) => {
            let x = a.data();
            let y = b.data();
            ag(&a, &g.iter().zip(&y).map(|(q, z)| q * z).collect::<Vec<_>>());
            ag(&b, &g.iter().zip(&x).map(|(q, z)| q * z).collect::<Vec<_>>());
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
                        let q = g[i * bc + j];
                        ga[i * ac + k] += q * y[k * bc + j];
                        gb[k * bc + j] += x[i * ac + k] * q;
                    }
                }
            }
            ag(&a, &ga);
            ag(&b, &gb);
        }
        Op::Transpose(a) => {
            let (out_r, out_c) = v.shape();
            let mut q = vec![0.0; g.len()];
            for i in 0..out_r {
                for j in 0..out_c {
                    q[j * out_r + i] = g[i * out_c + j];
                }
            }
            ag(&a, &q);
        }
        Op::Log(a) => ag(
            &a,
            &g.iter()
                .zip(a.data())
                .map(|(q, x)| q / x.max(1e-20))
                .collect::<Vec<_>>(),
        ),
        Op::Neg(a) => ag(&a, &g.iter().map(|q| -q).collect::<Vec<_>>()),
        Op::Relu(a) => ag(
            &a,
            &g.iter()
                .zip(a.data())
                .map(|(q, x)| if x > 0.0 { *q } else { 0.0 })
                .collect::<Vec<_>>(),
        ),
        Op::Silu(a) => {
            let x = a.data();
            let gx = x
                .iter()
                .zip(&g)
                .map(|(x, q)| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    q * s * (1.0 + x * (1.0 - s))
                })
                .collect::<Vec<_>>();
            ag(&a, &gx);
        }
        Op::Softmax(a) => {
            let y = v.data();
            let dot = g.iter().zip(&y).map(|(q, x)| q * x).sum::<f32>();
            ag(
                &a,
                &y.iter()
                    .zip(g)
                    .map(|(y, q)| y * (q - dot))
                    .collect::<Vec<_>>(),
            );
        }
        Op::ConcatRows(xs) => {
            let c = v.shape().1;
            for (i, x) in xs.iter().enumerate() {
                ag(x, &g[i * c..(i + 1) * c]);
            }
        }
        Op::ConcatCols(xs) => {
            let r = v.shape().0;
            let c = v.shape().1;
            let mut off = 0;
            for x in xs {
                let w = x.shape().1;
                let mut q = vec![0.0; r * w];
                for i in 0..r {
                    q[i * w..(i + 1) * w].copy_from_slice(&g[i * c + off..i * c + off + w]);
                }
                ag(&x, &q);
                off += w;
            }
        }
        Op::Slice(a, start, len) => {
            let (r, c) = a.shape();
            let mut q = vec![0.0; r * c];
            for i in 0..r {
                q[i * c + start..i * c + start + len]
                    .copy_from_slice(&g[i * len..(i + 1) * len]);
            }
            ag(&a, &q);
        }
        Op::Gather(a, i) => {
            let mut q = vec![0.0; a.data().len()];
            q[i] = g[0];
            ag(&a, &q);
        }
        Op::RowGather(a, row) => {
            let c = a.shape().1;
            let mut q = vec![0.0; a.data().len()];
            q[row * c..(row + 1) * c].copy_from_slice(g);
            ag(&a, &q);
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
        assert_eq!(a.grad(), vec![1., 1., 1., 1., 1., 1.]);
    }

    #[test]
    fn concat_cols_backward_for_matrix() {
        let a = Value::leaf(2, 1, vec![1., 2.]);
        let b = Value::leaf(2, 2, vec![3., 4., 5., 6.]);
        let y = Value::concat_cols(&[a.clone(), b.clone()]);
        y.backward();
        assert_eq!(a.grad(), vec![1., 1.]);
        assert_eq!(b.grad(), vec![1., 1., 1., 1.]);
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
        let p = y.data();
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silu_is_differentiable() {
        let x = Value::leaf(1, 1, vec![0.7]);
        let y = x.silu();
        y.backward();
        let s = 1.0 / (1.0 + (-0.7_f32).exp());
        let expected = s * (1.0 + 0.7 * (1.0 - s));
        assert!((x.grad()[0] - expected).abs() < 1e-6);
    }
}
