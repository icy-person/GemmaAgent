use std::fmt;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> f32,
    {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, value: f32) {
        self.data[r * self.cols + c] = value;
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "matmul shape mismatch");
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 { continue; }
                for j in 0..rhs.cols {
                    let idx = i * out.cols + j;
                    out.data[idx] += a * rhs.get(k, j);
                }
            }
        }
        out
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mean_sq + eps).sqrt().recip();
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * inv * weight[i];
    }
}

pub fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-20);
    for v in values.iter_mut() { *v *= inv; }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
