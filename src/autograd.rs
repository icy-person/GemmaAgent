use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

struct Node {
    data: Vec<f32>,
    grad: Vec<f32>,
    shape: Shape,
    parents: Vec<Value>,
    op: Op,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

enum Op {
    Leaf,
    Add,
    MatMul,
}

impl Value {
    pub fn leaf(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self(Rc::new(RefCell::new(Node {
            grad: vec![0.0; data.len()],
            data,
            shape: Shape { rows, cols },
            parents: Vec::new(),
            op: Op::Leaf,
        })))
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::leaf(rows, cols, vec![0.0; rows * cols])
    }

    pub fn data(&self) -> Vec<f32> { self.0.borrow().data.clone() }
    pub fn grad(&self) -> Vec<f32> { self.0.borrow().grad.clone() }
    pub fn shape(&self) -> Shape { self.0.borrow().shape.clone() }

    pub fn add(&self, rhs: &Self) -> Self {
        let a = self.0.borrow();
        let b = rhs.0.borrow();
        assert_eq!(a.data.len(), b.data.len());
        let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        drop(a); drop(b);
        Self(Rc::new(RefCell::new(Node {
            grad: vec![0.0; self.0.borrow().data.len()],
            data,
            shape: self.shape(),
            parents: vec![self.clone(), rhs.clone()],
            op: Op::Add,
        })))
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        let a = self.0.borrow();
        let b = rhs.0.borrow();
        assert_eq!(a.shape.cols, b.shape.rows);
        let (m, k, n) = (a.shape.rows, a.shape.cols, b.shape.cols);
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a.data[i * k + p];
                for j in 0..n { data[i * n + j] += av * b.data[p * n + j]; }
            }
        }
        drop(a); drop(b);
        Self(Rc::new(RefCell::new(Node {
            grad: vec![0.0; m * n],
            data,
            shape: Shape { rows: m, cols: n },
            parents: vec![self.clone(), rhs.clone()],
            op: Op::MatMul,
        })))
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut seen = HashSet::new();
        build_topo(self, &mut seen, &mut topo);
        {
            let mut root = self.0.borrow_mut();
            root.grad.fill(1.0);
        }
        for node in topo.into_iter().rev() {
            let grad = node.0.borrow().grad.clone();
            let op = node.0.borrow().op.clone();
            match op {
                Op::Leaf => {}
                Op::Add => {
                    for parent in &node.0.borrow().parents {
                        let mut p = parent.0.borrow_mut();
                        for (g, pg) in grad.iter().zip(&mut p.grad) { *pg += *g; }
                    }
                }
                Op::MatMul => {
                    let parents = node.0.borrow().parents.clone();
                    let a = parents[0].0.borrow();
                    let b = parents[1].0.borrow();
                    let (m, k, n) = (a.shape.rows, a.shape.cols, b.shape.cols);
                    let mut ga = vec![0.0; m * k];
                    let mut gb = vec![0.0; k * n];
                    for i in 0..m {
                        for p in 0..k {
                            for j in 0..n {
                                let g = grad[i * n + j];
                                ga[i * k + p] += g * b.data[p * n + j];
                                gb[p * n + j] += a.data[i * k + p] * g;
                            }
                        }
                    }
                    drop(a); drop(b);
                    let mut pa = parents[0].0.borrow_mut();
                    let mut pb = parents[1].0.borrow_mut();
                    for (dst, src) in pa.grad.iter_mut().zip(ga) { *dst += src; }
                    for (dst, src) in pb.grad.iter_mut().zip(gb) { *dst += src; }
                }
            }
        }
    }
}

impl Clone for Op {
    fn clone(&self) -> Self {
        match self { Op::Leaf => Op::Leaf, Op::Add => Op::Add, Op::MatMul => Op::MatMul }
    }
}

fn build_topo(v: &Value, seen: &mut HashSet<usize>, topo: &mut Vec<Value>) {
    let ptr = Rc::as_ptr(&v.0) as usize;
    if !seen.insert(ptr) { return; }
    let parents = v.0.borrow().parents.clone();
    for p in parents { build_topo(&p, seen, topo); }
    topo.push(v.clone());
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
        assert_eq!(y.data(), vec![31.0]);
        assert_eq!(a.grad(), vec![5.0, 7.0]);
        assert_eq!(b.grad(), vec![2.0, 3.0]);
    }
}
