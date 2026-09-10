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
    Silu(Value),
    RmsNorm(Value, f32),
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
        Self(Rc::new(RefCell::new(Node { r, c, g: vec![0.0; d.len()], d, op })))
    }
    pub fn leaf(r: usize, c: usize, d: Vec<f32>) -> Self { Self::mk(r, c, d, Op::Leaf) }
    /// Uniform init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]` (fan_in = number of columns), a
    /// LeCun-style bound. Unlike a fixed range, this keeps activation variance roughly constant
    /// as layer width (d_model, ffn) grows instead of shrinking relative to fan-in.
    pub fn parameter(r: usize, c: usize, seed: &mut u64) -> Self {
        let bound = (1.0 / c.max(1) as f32).sqrt();
        let mut d = Vec::with_capacity(r * c);
        for _ in 0..r * c {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((*seed >> 32) as u32) as f32 / u32::MAX as f32;
            d.push((u * 2.0 - 1.0) * bound);
        }
        Self::leaf(r, c, d)
    }
    pub fn id(&self) -> usize { Rc::as_ptr(&self.0) as usize }
    /// Returns an owned copy of the tensor's data. Prefer this only when you actually need an
    /// owned `Vec` (e.g. returning it past the borrow, or across an `await`/thread boundary);
    /// every op below borrows the underlying node directly instead of calling this, since
    /// `RefCell` allows any number of simultaneous immutable borrows and avoids the allocation.
    pub fn data(&self) -> Vec<f32> { self.0.borrow().d.clone() }
    pub fn grad(&self) -> Vec<f32> { self.0.borrow().g.clone() }
    pub fn shape(&self) -> (usize, usize) { let n = self.0.borrow(); (n.r, n.c) }
    pub fn zero_grad(&self) { self.0.borrow_mut().g.fill(0.0); }
    pub fn set_data(&self, data: Vec<f32>) { assert_eq!(data.len(), self.0.borrow().d.len(), "parameter length mismatch"); self.0.borrow_mut().d = data; }
    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "add shape mismatch");
        let (rows, cols) = self.shape();
        let out = { let a = self.0.borrow(); let b = rhs.0.borrow(); a.d.iter().zip(&b.d).map(|(x, y)| x + y).collect() };
        Self::mk(rows, cols, out, Op::Add(self.clone(), rhs.clone()))
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape(), rhs.shape(), "mul shape mismatch");
        let (rows, cols) = self.shape();
        let out = { let a = self.0.borrow(); let b = rhs.0.borrow(); a.d.iter().zip(&b.d).map(|(x, y)| x * y).collect() };
        Self::mk(rows, cols, out, Op::Mul(self.clone(), rhs.clone()))
    }
    pub fn scalar_mul(&self, scalar: f32) -> Self { assert!(scalar.is_finite()); let (r, c) = self.shape(); self.mul(&Self::leaf(r, c, vec![scalar; r * c])) }
    pub fn div_scalar(&self, scalar: f32) -> Self { assert!(scalar.is_finite() && scalar != 0.0); self.scalar_mul(1.0/scalar) }
    pub fn matmul(&self, rhs: &Self) -> Self {
        let (ar, ac) = self.shape(); let (br, bc) = rhs.shape(); assert_eq!(ac, br, "matmul shape mismatch: {ar}x{ac} · {br}x{bc}");
        let out = { let a = self.0.borrow(); let b = rhs.0.borrow(); matmul_raw(&a.d, &b.d, ar, ac, bc) };
        Self::mk(ar, bc, out, Op::MatMul(self.clone(), rhs.clone()))
    }
    pub fn transpose(&self) -> Self {
        let (rows, cols) = self.shape();
        let out = { let n = self.0.borrow(); let x = &n.d; let mut out = vec![0.0; rows * cols]; for i in 0..rows { for j in 0..cols { out[j * rows + i] = x[i * cols + j]; } } out };
        Self::mk(cols, rows, out, Op::Transpose(self.clone()))
    }
    pub fn log(&self) -> Self { let (rows, cols) = self.shape(); let out = { let n = self.0.borrow(); n.d.iter().map(|x| x.max(1e-20).ln()).collect() }; Self::mk(rows, cols, out, Op::Log(self.clone())) }
    pub fn neg(&self) -> Self { let (rows, cols) = self.shape(); let out = { let n = self.0.borrow(); n.d.iter().map(|x| -x).collect() }; Self::mk(rows, cols, out, Op::Neg(self.clone())) }
    pub fn silu(&self) -> Self { let (rows, cols) = self.shape(); let out = { let n = self.0.borrow(); n.d.iter().map(|x| x / (1.0 + (-x).exp())).collect() }; Self::mk(rows, cols, out, Op::Silu(self.clone())) }
    /// Row-wise RMSNorm: every row is normalized independently by its own root-mean-square, so
    /// this works the same whether `self` is a single `(1, cols)` state or a whole `(rows,
    /// cols)` sequence batched into one matrix.
    pub fn rms_norm(&self, eps: f32) -> Self {
        let (rows, cols) = self.shape(); assert!(eps.is_finite() && eps > 0.0);
        let out = {
            let n = self.0.borrow();
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row = &n.d[r * cols..(r + 1) * cols];
                let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
                let inv = (mean_sq + eps).sqrt().recip();
                let out_row = &mut out[r * cols..(r + 1) * cols];
                for (o, v) in out_row.iter_mut().zip(row) { *o = v * inv; }
            }
            out
        };
        Self::mk(rows, cols, out, Op::RmsNorm(self.clone(), eps))
    }
    /// Row-wise softmax: each row is normalized independently over its columns. A `(1, cols)`
    /// vector is the special case of a single row; a `(rows, cols)` matrix (e.g. a whole
    /// batched attention score matrix) gets one independent softmax per row in the same op,
    /// instead of needing a separate node per row.
    pub fn softmax(&self) -> Self {
        let (rows, cols) = self.shape();
        let out = {
            let n = self.0.borrow();
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row = &n.d[r * cols..(r + 1) * cols];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let out_row = &mut out[r * cols..(r + 1) * cols];
                let mut sum = 0.0f32;
                for (o, v) in out_row.iter_mut().zip(row) { *o = (*v - max).exp(); sum += *o; }
                let sum = sum.max(1e-20);
                for o in out_row.iter_mut() { *o /= sum; }
            }
            out
        };
        Self::mk(rows, cols, out, Op::Softmax(self.clone()))
    }
    pub fn concat_rows(xs: &[Self]) -> Self {
        assert!(!xs.is_empty());
        let cols = xs[0].shape().1;
        let refs: Vec<_> = xs.iter().map(|x| { assert_eq!(x.shape(), (1, cols), "concat_rows shape mismatch"); x.0.borrow() }).collect();
        let mut data = Vec::with_capacity(xs.len() * cols);
        for r in &refs { data.extend_from_slice(&r.d); }
        drop(refs);
        Self::mk(xs.len(), cols, data, Op::ConcatRows(xs.to_vec()))
    }
    pub fn concat_cols(xs: &[Self]) -> Self {
        assert!(!xs.is_empty());
        let rows = xs[0].shape().0;
        let refs: Vec<_> = xs.iter().map(|x| { assert_eq!(x.shape().0, rows, "concat_cols row mismatch"); x.0.borrow() }).collect();
        let cols: usize = refs.iter().map(|r| r.c).sum();
        let mut out = vec![0.0; rows * cols];
        let mut offset = 0;
        for r in &refs {
            let width = r.c;
            for i in 0..rows { out[i * cols + offset..i * cols + offset + width].copy_from_slice(&r.d[i * width..(i + 1) * width]); }
            offset += width;
        }
        drop(refs);
        Self::mk(rows, cols, out, Op::ConcatCols(xs.to_vec()))
    }
    pub fn slice_cols(&self, start: usize, len: usize) -> Self {
        let (rows, cols) = self.shape(); assert!(start <= cols && len <= cols - start);
        let out = { let n = self.0.borrow(); let mut out = Vec::with_capacity(rows * len); for i in 0..rows { out.extend_from_slice(&n.d[i * cols + start..i * cols + start + len]); } out };
        Self::mk(rows, len, out, Op::Slice(self.clone(), start, len))
    }
    pub fn gather(&self, index: usize) -> Self {
        let (rows, cols) = self.shape(); assert_eq!(rows, 1, "gather expects a row vector"); assert!(index < cols);
        let value = { let n = self.0.borrow(); n.d[index] };
        Self::mk(1, 1, vec![value], Op::Gather(self.clone(), index))
    }
    pub fn row(&self, row: usize) -> Self {
        let (rows, cols) = self.shape(); assert!(row < rows);
        let out = { let n = self.0.borrow(); n.d[row * cols..(row + 1) * cols].to_vec() };
        Self::mk(1, cols, out, Op::RowGather(self.clone(), row))
    }
    pub fn backward(&self) { let mut order=Vec::new(); let mut seen=HashSet::new(); topo(self,&mut seen,&mut order); self.0.borrow_mut().g.fill(1.0); for node in order.into_iter().rev(){back(&node);} }
}

fn topo(value:&Value,seen:&mut HashSet<usize>,order:&mut Vec<Value>){ if !seen.insert(value.id()){return;} match value.0.borrow().op.clone(){ Op::Leaf=>{}, Op::Add(a,b)|Op::Mul(a,b)|Op::MatMul(a,b)=>{topo(&a,seen,order);topo(&b,seen,order)}, Op::Transpose(a)|Op::Log(a)|Op::Neg(a)|Op::Softmax(a)|Op::Silu(a)|Op::RmsNorm(a,_)|Op::Slice(a,_,_)|Op::Gather(a,_)|Op::RowGather(a,_)=>topo(&a,seen,order), Op::ConcatRows(xs)|Op::ConcatCols(xs)=>{for x in xs{topo(&x,seen,order);}} } order.push(value.clone()); }
fn accumulate(value:&Value,grad:&[f32]){let mut node=value.0.borrow_mut(); assert_eq!(node.g.len(),grad.len()); for(dst,src) in node.g.iter_mut().zip(grad){*dst+=*src;}}

/// Every branch below reads parent tensors via a direct `RefCell` borrow instead of `.data()`,
/// so backward only allocates the gradient buffers it actually produces (`ga`/`gb`/...) rather
/// than an extra full clone of each parent's data first. This matters more once ops run on
/// whole-sequence matrices (batched attention/projections) instead of one row at a time, since
/// each avoided clone is now a much bigger allocation.
fn back(value:&Value){ let grad=value.grad(); match value.0.borrow().op.clone(){
    Op::Leaf=>{}
    Op::Add(a,b)=>{accumulate(&a,&grad);accumulate(&b,&grad)}
    Op::Mul(a,b)=>{
        let(ga,gb)={let na=a.0.borrow();let nb=b.0.borrow();(
            grad.iter().zip(nb.d.iter()).map(|(g,y)|g*y).collect::<Vec<_>>(),
            grad.iter().zip(na.d.iter()).map(|(g,x)|g*x).collect::<Vec<_>>(),
        )};
        accumulate(&a,&ga);accumulate(&b,&gb);
    }
    Op::MatMul(a,b)=>{
        let(ar,ac)=a.shape();let(_,bc)=b.shape();
        let(ga,gb)={
            let na=a.0.borrow();let nb=b.0.borrow();let x=&na.d;let y=&nb.d;
            let mut ga=vec![0.0;ar*ac];let mut gb=vec![0.0;ac*bc];
            for i in 0..ar{for k in 0..ac{let a_ik=x[i*ac+k];for j in 0..bc{let g=grad[i*bc+j];ga[i*ac+k]+=g*y[k*bc+j];gb[k*bc+j]+=a_ik*g;}}}
            (ga,gb)
        };
        accumulate(&a,&ga);accumulate(&b,&gb);
    }
    Op::Transpose(a)=>{let(out_rows,out_cols)=value.shape();let mut ga=vec![0.0;grad.len()];for i in 0..out_rows{for j in 0..out_cols{ga[j*out_rows+i]=grad[i*out_cols+j];}}accumulate(&a,&ga);}
    Op::Log(a)=>{let ga={let na=a.0.borrow();grad.iter().zip(na.d.iter()).map(|(g,x)|if *x>=1e-20{g/x}else{0.0}).collect::<Vec<_>>()};accumulate(&a,&ga);}
    Op::Neg(a)=>accumulate(&a,&grad.iter().map(|g|-g).collect::<Vec<_>>()),
    Op::Silu(a)=>{let ga={let na=a.0.borrow();na.d.iter().zip(&grad).map(|(x,g)|{let s=1.0/(1.0+(-x).exp());g*s*(1.0+x*(1.0-s))}).collect::<Vec<_>>()};accumulate(&a,&ga);}
    Op::RmsNorm(a,eps)=>{
        let(rows,cols)=a.shape();let n=cols as f32;
        let ga={
            let na=a.0.borrow();let mut ga=vec![0.0f32;rows*cols];
            for r in 0..rows{
                let x_row=&na.d[r*cols..(r+1)*cols];let g_row=&grad[r*cols..(r+1)*cols];
                let mean_sq=x_row.iter().map(|v|v*v).sum::<f32>()/n;let rr=(mean_sq+eps).sqrt();let inv=1.0/rr;
                let dot=g_row.iter().zip(x_row).map(|(g,x)|g*x).sum::<f32>();let coeff=dot/(n*rr*rr*rr);
                let ga_row=&mut ga[r*cols..(r+1)*cols];
                for j in 0..cols{ga_row[j]=g_row[j]*inv-x_row[j]*coeff;}
            }
            ga
        };
        accumulate(&a,&ga);
    }
    Op::Softmax(a)=>{
        let(rows,cols)=value.shape();
        let ga={
            let ny=value.0.borrow();let mut ga=vec![0.0f32;rows*cols];
            for r in 0..rows{
                let y_row=&ny.d[r*cols..(r+1)*cols];let g_row=&grad[r*cols..(r+1)*cols];
                let dot=g_row.iter().zip(y_row).map(|(g,y)|g*y).sum::<f32>();
                let ga_row=&mut ga[r*cols..(r+1)*cols];
                for j in 0..cols{ga_row[j]=y_row[j]*(g_row[j]-dot);}
            }
            ga
        };
        accumulate(&a,&ga);
    }
    Op::ConcatRows(xs)=>{let cols=value.shape().1;for(row,x)in xs.iter().enumerate(){accumulate(x,&grad[row*cols..(row+1)*cols]);}}
    Op::ConcatCols(xs)=>{let rows=value.shape().0;let cols=value.shape().1;let mut offset=0;for x in xs{let width=x.shape().1;let mut gx=vec![0.0;rows*width];for row in 0..rows{gx[row*width..(row+1)*width].copy_from_slice(&grad[row*cols+offset..row*cols+offset+width]);}accumulate(&x,&gx);offset+=width;}}
    Op::Slice(a,start,len)=>{let(rows,cols)=a.shape();let mut ga=vec![0.0;rows*cols];for row in 0..rows{ga[row*cols+start..row*cols+start+len].copy_from_slice(&grad[row*len..(row+1)*len]);}accumulate(&a,&ga);}
    Op::Gather(a,index)=>{let mut ga=vec![0.0;a.data().len()];ga[index]=grad[0];accumulate(&a,&ga);}
    Op::RowGather(a,row)=>{let cols=a.shape().1;let mut ga=vec![0.0;a.data().len()];ga[row*cols..(row+1)*cols].copy_from_slice(&grad);accumulate(&a,&ga);}
}}

/// Multiply-add count above which the `1 x ac` * `ac x bc` matmul shape (single-token row
/// vector times a weight matrix — still used for per-target logits/loss) is split across
/// threads by output column. Below this, or on a single-core host, thread-spawn overhead loses
/// to the plain sequential loop.
const PARALLEL_MATMUL_THRESHOLD: usize = 4096;
/// Multiply-add volume (`ar*ac*bc`) above which a multi-row matmul (`ar>1` — the shape used by
/// the batched per-layer projections and attention matrices, where every row is an independent
/// token/query) is split across threads by output row instead.
const PARALLEL_MATMUL_ROW_THRESHOLD: usize = 200_000;

fn matmul_raw(x: &[f32], y: &[f32], ar: usize, ac: usize, bc: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ar * bc];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if ar == 1 {
        if bc > 1 && threads > 1 && ac * bc >= PARALLEL_MATMUL_THRESHOLD {
            let chunk = bc.div_ceil(threads.min(bc));
            std::thread::scope(|scope| {
                for (idx, out_chunk) in out.chunks_mut(chunk).enumerate() {
                    let col_start = idx * chunk;
                    scope.spawn(move || vec_matmul_cols(x, y, ac, bc, col_start, out_chunk));
                }
            });
        } else {
            vec_matmul_cols(x, y, ac, bc, 0, &mut out);
        }
        return out;
    }
    if threads > 1 && ar * ac * bc >= PARALLEL_MATMUL_ROW_THRESHOLD {
        let rows_per_chunk = ar.div_ceil(threads.min(ar));
        std::thread::scope(|scope| {
            for (idx, out_chunk) in out.chunks_mut(rows_per_chunk * bc).enumerate() {
                let row_start = idx * rows_per_chunk;
                scope.spawn(move || mat_matmul_rows(x, y, ac, bc, row_start, out_chunk));
            }
        });
    } else {
        mat_matmul_rows(x, y, ac, bc, 0, &mut out);
    }
    out
}

/// Computes the `[col_start, col_start + out.len())` slice of `x (1 x ac) * y (ac x bc)`.
fn vec_matmul_cols(x: &[f32], y: &[f32], ac: usize, bc: usize, col_start: usize, out: &mut [f32]) {
    let width = out.len();
    for k in 0..ac {
        let a = x[k];
        if a == 0.0 { continue; }
        let y_row = &y[k * bc + col_start..k * bc + col_start + width];
        for (o, yv) in out.iter_mut().zip(y_row) { *o += a * yv; }
    }
}

/// Computes rows `[row_start, row_start + out.len()/bc)` of `x (ar x ac) * y (ac x bc)`.
fn mat_matmul_rows(x: &[f32], y: &[f32], ac: usize, bc: usize, row_start: usize, out: &mut [f32]) {
    let row_count = out.len() / bc;
    for local_i in 0..row_count {
        let i = row_start + local_i;
        let out_row = &mut out[local_i * bc..(local_i + 1) * bc];
        for k in 0..ac {
            let a = x[i * ac + k];
            if a == 0.0 { continue; }
            let y_row = &y[k * bc..(k + 1) * bc];
            for (o, yv) in out_row.iter_mut().zip(y_row) { *o += a * yv; }
        }
    }
}

#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn matmul_backward(){let a=Value::leaf(1,2,vec![2.,3.]);let b=Value::leaf(2,1,vec![5.,7.]);let y=a.matmul(&b);y.backward();assert_eq!(a.grad(),vec![5.,7.]);assert_eq!(b.grad(),vec![2.,3.]);}
 #[test]fn matmul_multi_row_backward_matches_finite_difference(){
    let a=Value::leaf(2,2,vec![1.,2.,3.,4.]);let b=Value::leaf(2,2,vec![5.,6.,7.,8.]);
    let y=a.matmul(&b);let loss=y.mul(&Value::leaf(2,2,vec![1.,1.,1.,1.]));
    loss.backward();let analytic=a.grad();let base=a.data();let eps=1e-3f32;
    for i in 0..4{
        let mut plus=base.clone();plus[i]+=eps;let mut minus=base.clone();minus[i]-=eps;
        let lp=Value::leaf(2,2,plus).matmul(&b).data().iter().sum::<f32>();
        let lm=Value::leaf(2,2,minus).matmul(&b).data().iter().sum::<f32>();
        let numeric=(lp-lm)/(2.*eps);
        assert!((analytic[i]-numeric).abs()<2e-3,"i={i} analytic={} numeric={}",analytic[i],numeric);
    }
 }
 #[test]fn transpose_backward(){let a=Value::leaf(2,3,vec![1.,2.,3.,4.,5.,6.]);let y=a.transpose();y.backward();assert_eq!(a.grad(),vec![1.;6]);}
 #[test]fn concat_cols_matrix_backward(){let a=Value::leaf(2,1,vec![1.,2.]);let b=Value::leaf(2,2,vec![3.,4.,5.,6.]);let y=Value::concat_cols(&[a.clone(),b.clone()]);y.backward();assert_eq!(a.grad(),vec![1.,1.]);assert_eq!(b.grad(),vec![1.,1.,1.,1.]);}
 #[test]fn slice_matrix_backward(){let a=Value::leaf(2,4,vec![1.,2.,3.,4.,5.,6.,7.,8.]);let y=a.slice_cols(1,2);y.backward();assert_eq!(a.grad(),vec![0.,1.,1.,0.,0.,1.,1.,0.]);}
 #[test]fn row_gather_backward(){let a=Value::leaf(3,2,vec![1.,2.,3.,4.,5.,6.]);let y=a.row(1);y.backward();assert_eq!(a.grad(),vec![0.,0.,1.,1.,0.,0.]);}
 #[test]fn softmax_sums_to_one(){let y=Value::leaf(1,3,vec![1.,2.,3.]).softmax();let sum:f32=y.data().iter().sum();assert!((sum-1.).abs()<1e-6);}
 #[test]fn softmax_is_row_wise_for_matrices(){
    let y=Value::leaf(2,3,vec![1.,2.,3., 10.,10.,10.]).softmax();
    let d=y.data();
    assert!((d[0..3].iter().sum::<f32>()-1.).abs()<1e-6);
    assert!((d[3..6].iter().sum::<f32>()-1.).abs()<1e-6);
    // uniform row (equal logits) softmaxes to a uniform distribution
    for v in &d[3..6]{assert!((v-1./3.).abs()<1e-6);}
 }
 #[test]fn rms_norm_is_row_wise_for_matrices(){
    let x=Value::leaf(2,3,vec![1.,2.,3., 2.,4.,6.]);
    let y=x.rms_norm(1e-5);let d=y.data();
    // both rows point in the same direction (row 2 = 2 * row 1), so RMSNorm must normalize
    // them to (nearly) the same unit-scale row independent of the other row's magnitude.
    for i in 0..3{assert!((d[i]-d[3+i]).abs()<1e-3,"row 0 and row 1 should normalize to the same values");}
 }
 #[test]fn silu_backward(){let x=Value::leaf(1,1,vec![0.7]);let y=x.silu();y.backward();let s=1./(1.+(-0.7_f32).exp());let expected=s*(1.+0.7*(1.-s));assert!((x.grad()[0]-expected).abs()<1e-6);}
 #[test]fn rms_norm_backward_finite_difference(){let x=Value::leaf(1,3,vec![0.4,-0.7,1.2]);let y=x.rms_norm(1e-5);let loss=y.mul(&Value::leaf(1,3,vec![0.3,-0.2,0.5]));loss.backward();let analytic=x.grad();let base=x.data();let eps=1e-3_f32;for i in 0..3{let mut plus=base.clone();plus[i]+=eps;let mut minus=base.clone();minus[i]-=eps;let lp=Value::leaf(1,3,plus).rms_norm(1e-5).mul(&Value::leaf(1,3,vec![0.3,-0.2,0.5])).data().iter().sum::<f32>();let lm=Value::leaf(1,3,minus).rms_norm(1e-5).mul(&Value::leaf(1,3,vec![0.3,-0.2,0.5])).data().iter().sum::<f32>();let numeric=(lp-lm)/(2.*eps);assert!((analytic[i]-numeric).abs()<2e-3,"i={i} analytic={} numeric={}",analytic[i],numeric);}}
 #[test]fn log_clamp_has_zero_gradient_below_floor(){let x=Value::leaf(1,1,vec![1e-25]);let y=x.log();y.backward();assert_eq!(x.grad(),vec![0.]);}
 #[test]fn parameter_init_scales_with_fan_in(){
    let mut seed_small=1; let small_fan_in=Value::parameter(50,4,&mut seed_small);
    let mut seed_large=1; let large_fan_in=Value::parameter(50,400,&mut seed_large);
    let max_abs=|v:&Value|v.data().iter().cloned().fold(0.0f32,|acc,x|acc.max(x.abs()));
    assert!(max_abs(&large_fan_in)<max_abs(&small_fan_in),"wider fan-in should yield smaller-magnitude weights");
 }
 #[test]fn matmul_parallel_path_matches_manual_dot_product(){
    let ac=130usize; let bc=130usize; // ac*bc is well above PARALLEL_MATMUL_THRESHOLD
    let mut seed=9; let x=Value::parameter(1,ac,&mut seed); let y=Value::parameter(ac,bc,&mut seed);
    let result=x.matmul(&y);
    let xd=x.data(); let yd=y.data();
    let mut expected=vec![0.0f32;bc];
    for k in 0..ac { for j in 0..bc { expected[j]+=xd[k]*yd[k*bc+j]; } }
    for(a,b) in result.data().iter().zip(expected.iter()) { assert!((a-b).abs()<1e-3,"parallel matmul diverged from manual dot product: {a} vs {b}"); }
 }
 #[test]fn matmul_multi_row_parallel_path_matches_manual_dot_product(){
    let ar=64usize; let ac=80usize; let bc=80usize; // ar*ac*bc is well above PARALLEL_MATMUL_ROW_THRESHOLD
    let mut seed=11; let x=Value::parameter(ar,ac,&mut seed); let y=Value::parameter(ac,bc,&mut seed);
    let result=x.matmul(&y);
    let xd=x.data(); let yd=y.data();
    let mut expected=vec![0.0f32;ar*bc];
    for i in 0..ar { for k in 0..ac { for j in 0..bc { expected[i*bc+j]+=xd[i*ac+k]*yd[k*bc+j]; } } }
    for(a,b) in result.data().iter().zip(expected.iter()) { assert!((a-b).abs()<1e-3,"parallel row-split matmul diverged from manual dot product: {a} vs {b}"); }
 }
 #[test]fn add_and_mul_on_self_do_not_panic_on_double_borrow(){
    // add(&self, &self) / mul(&self, &self) each take two immutable borrows of the *same*
    // RefCell at once. RefCell permits any number of simultaneous immutable borrows, so this
    // must not panic even though both arguments are the same node.
    let a=Value::leaf(1,3,vec![1.,2.,3.]);
    assert_eq!(a.add(&a).data(),vec![2.,4.,6.]);
    assert_eq!(a.mul(&a).data(),vec![1.,4.,9.]);
 }
 #[test]fn mul_self_backward_does_not_panic_and_sums_both_branches(){
    // y = a * a (elementwise); dy/da = 2a. Both operands of Op::Mul are the same node, so the
    // borrow-based backward above must borrow it twice immutably (fine) and accumulate into it
    // twice sequentially (also fine) rather than panicking on a double mutable borrow.
    let a=Value::leaf(1,3,vec![2.,3.,4.]);
    let y=a.mul(&a);
    y.backward();
    assert_eq!(a.grad(),vec![4.,6.,8.]);
 }
}
