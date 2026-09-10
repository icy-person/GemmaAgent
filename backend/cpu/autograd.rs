use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

#[derive(Clone)]
enum Op {
    Leaf,
    Add(Value, Value),
    Mul(Value, Value),
    Scale(Value, f32),
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
    /// Scalar scaling uses a dedicated op instead of allocating a full tensor filled with the scalar.
    pub fn scalar_mul(&self, scalar: f32) -> Self {
        assert!(scalar.is_finite());
        let (r, c) = self.shape();
        let out = { let n = self.0.borrow(); n.d.iter().map(|x| x * scalar).collect() };
        Self::mk(r, c, out, Op::Scale(self.clone(), scalar))
    }
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
    /// this works the same whether `self` is a single `(1, cols)` state or a whole `(rows, cols)` sequence batched into one matrix.
    pub fn rms_norm(&self, eps: f32) -> Self {
        let (rows, cols) = self.shape(); assert!(eps.is_finite() && eps > 0.0);
        let out = { let n = self.0.borrow(); let mut out = vec![0.0f32; rows * cols]; for r in 0..rows { let row = &n.d[r * cols..(r + 1) * cols]; let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32; let inv = (mean_sq + eps).sqrt().recip(); let out_row = &mut out[r * cols..(r + 1) * cols]; for (o, v) in out_row.iter_mut().zip(row) { *o = v * inv; } } out };
        Self::mk(rows, cols, out, Op::RmsNorm(self.clone(), eps))
    }
    /// Row-wise softmax: each row is normalized independently over its columns.
    pub fn softmax(&self) -> Self {
        let (rows, cols) = self.shape();
        let out = { let n = self.0.borrow(); let mut out = vec![0.0f32; rows * cols]; for r in 0..rows { let row = &n.d[r * cols..(r + 1) * cols]; let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max); let out_row = &mut out[r * cols..(r + 1) * cols]; let mut sum = 0.0f32; for (o, v) in out_row.iter_mut().zip(row) { *o = (*v - max).exp(); sum += *o; } let sum = sum.max(1e-20); for o in out_row.iter_mut() { *o /= sum; } } out };
        Self::mk(rows, cols, out, Op::Softmax(self.clone()))
    }
    pub fn concat_rows(xs: &[Self]) -> Self {
        assert!(!xs.is_empty()); let cols = xs[0].shape().1;
        let refs: Vec<_> = xs.iter().map(|x| { assert_eq!(x.shape(), (1, cols), "concat_rows shape mismatch"); x.0.borrow() }).collect();
        let mut data = Vec::with_capacity(xs.len() * cols); for r in &refs { data.extend_from_slice(&r.d); } drop(refs);
        Self::mk(xs.len(), cols, data, Op::ConcatRows(xs.to_vec()))
    }
    pub fn concat_cols(xs: &[Self]) -> Self {
        assert!(!xs.is_empty()); let rows = xs[0].shape().0;
        let refs: Vec<_> = xs.iter().map(|x| { assert_eq!(x.shape().0, rows, "concat_cols row mismatch"); x.0.borrow() }).collect();
        let cols: usize = refs.iter().map(|r| r.c).sum(); let mut out = vec![0.0; rows * cols]; let mut offset = 0;
        for r in &refs { let width = r.c; for i in 0..rows { out[i * cols + offset..i * cols + offset + width].copy_from_slice(&r.d[i * width..(i + 1) * width]); } offset += width; }
        drop(refs); Self::mk(rows, cols, out, Op::ConcatCols(xs.to_vec()))
    }
    pub fn slice_cols(&self, start: usize, len: usize) -> Self {
        let (rows, cols) = self.shape(); assert!(start <= cols && len <= cols - start);
        let out = { let n = self.0.borrow(); let mut out = Vec::with_capacity(rows * len); for i in 0..rows { out.extend_from_slice(&n.d[i * cols + start..i * cols + start + len]); } out };
        Self::mk(rows, len, out, Op::Slice(self.clone(), start, len))
    }
    pub fn gather(&self, index: usize) -> Self { let (rows, cols) = self.shape(); assert_eq!(rows, 1, "gather expects a row vector"); assert!(index < cols); let value = self.0.borrow().d[index]; Self::mk(1, 1, vec![value], Op::Gather(self.clone(), index)) }
    pub fn row(&self, row: usize) -> Self { let (rows, cols) = self.shape(); assert!(row < rows); let out = self.0.borrow().d[row * cols..(row + 1) * cols].to_vec(); Self::mk(1, cols, out, Op::RowGather(self.clone(), row)) }
    pub fn backward(&self) { let mut order=Vec::new(); let mut seen=HashSet::new(); topo(self,&mut seen,&mut order); self.0.borrow_mut().g.fill(1.0); for node in order.into_iter().rev(){back(&node);} }
}

fn topo(value:&Value,seen:&mut HashSet<usize>,order:&mut Vec<Value>){ if !seen.insert(value.id()){return;} match value.0.borrow().op.clone(){ Op::Leaf=>{}, Op::Add(a,b)|Op::Mul(a,b)|Op::MatMul(a,b)=>{topo(&a,seen,order);topo(&b,seen,order)}, Op::Scale(a,_)=>topo(&a,seen,order), Op::Transpose(a)|Op::Log(a)|Op::Neg(a)|Op::Softmax(a)|Op::Silu(a)|Op::RmsNorm(a,_)|Op::Slice(a,_,_)|Op::Gather(a,_)|Op::RowGather(a,_)=>topo(&a,seen,order), Op::ConcatRows(xs)|Op::ConcatCols(xs)=>{for x in xs{topo(&x,seen,order);}} } order.push(value.clone()); }
fn accumulate(value:&Value,grad:&[f32]){let mut node=value.0.borrow_mut(); assert_eq!(node.g.len(),grad.len()); for(dst,src)in node.g.iter_mut().zip(grad){*dst+=*src;}}

fn back(value:&Value){ let grad=value.grad(); match value.0.borrow().op.clone(){
    Op::Leaf=>{}
    Op::Scale(a,scalar)=>{accumulate(&a,&grad.iter().map(|g|g*scalar).collect::<Vec<_>>());}
    Op::Add(a,b)=>{accumulate(&a,&grad);accumulate(&b,&grad)}
    Op::Mul(a,b)=>{let(ga,gb)={let na=a.0.borrow();let nb=b.0.borrow();(grad.iter().zip(nb.d.iter()).map(|(g,y)|g*y).collect::<Vec<_>>(),grad.iter().zip(na.d.iter()).map(|(g,x)|g*x).collect::<Vec<_>>())};accumulate(&a,&ga);accumulate(&b,&gb);}
    Op::MatMul(a,b)=>{let(ar,ac)=a.shape();let(_,bc)=b.shape();let(ga,gb)={let na=a.0.borrow();let nb=b.0.borrow();let x=&na.d;let y=&nb.d;let mut ga=vec![0.0;ar*ac];let mut gb=vec![0.0;ac*bc];for i in 0..ar{for k in 0..ac{let a_ik=x[i*ac+k];for j in 0..bc{let g=grad[i*bc+j];ga[i*ac+k]+=g*y[k*bc+j];gb[k*bc+j]+=a_ik*g;}}}(ga,gb)};accumulate(&a,&ga);accumulate(&b,&gb);}
    Op::Transpose(a)=>{let(out_rows,out_cols)=value.shape();let mut ga=vec![0.0;grad.len()];for i in 0..out_rows{for j in 0..out_cols{ga[j*out_rows+i]=grad[i*out_cols+j];}}accumulate(&a,&ga);}
    Op::Log(a)=>{let ga={let na=a.0.borrow();grad.iter().zip(na.d.iter()).map(|(g,x)|if *x>=1e-20{g/x}else{0.0}).collect::<Vec<_>>()};accumulate(&a,&ga);}
    Op::Neg(a)=>accumulate(&a,&grad.iter().map(|g|-g).collect::<Vec<_>>()),
    Op::Silu(a)=>{let ga={let na=a.0.borrow();na.d.iter().zip(&grad).map(|(x,g)|{let s=1.0/(1.0+(-x).exp());g*s*(1.0+x*(1.0-s))}).collect::<Vec<_>>()};accumulate(&a,&ga);}
    Op::RmsNorm(a,eps)=>{let(rows,cols)=a.shape();let n=cols as f32;let ga={let na=a.0.borrow();let mut ga=vec![0.0f32;rows*cols];for r in 0..rows{let x_row=&na.d[r*cols..(r+1)*cols];let g_row=&grad[r*cols..(r+1)*cols];let mean_sq=x_row.iter().map(|v|v*v).sum::<f32>()/n;let rr=(mean_sq+eps).sqrt();let inv=1.0/rr;let dot=g_row.iter().zip(x_row).map(|(g,x)|g*x).sum::<f32>();let coeff=dot/(n*rr*rr*rr);let ga_row=&mut ga[r*cols..(r+1)*cols];for j in 0..cols{ga_row[j]=g_row[j]*inv-x_row[j]*coeff;}}ga};accumulate(&a,&ga);}
    Op::Softmax(a)=>{let(rows,cols)=value.shape();let ga={let ny=value.0.borrow();let mut ga=vec![0.0f32;rows*cols];for r in 0..rows{let y_row=&ny.d[r*cols..(r+1)*cols];let g_row=&grad[r*cols..(r+1)*cols];let dot=g_row.iter().zip(y_row).map(|(g,y)|g*y).sum::<f32>();let ga_row=&mut ga[r*cols..(r+1)*cols];for j in 0..cols{ga_row[j]=y_row[j]*(g_row[j]-dot);}}ga};accumulate(&a,&ga);}
    Op::ConcatRows(xs)=>{let cols=value.shape().1;for(row,x)in xs.iter().enumerate(){accumulate(x,&grad[row*cols..(row+1)*cols]);}}
    Op::ConcatCols(xs)=>{let rows=value.shape().0;let cols=value.shape().1;let mut offset=0;for x in xs{let width=x.shape().1;let mut gx=vec![0.0;rows*width];for row in 0..rows{gx[row*width..(row+1)*width].copy_from_slice(&grad[row*cols+offset..row*cols+offset+width]);}accumulate(&x,&gx);offset+=width;}}
    Op::Slice(a,start,len)=>{let(rows,cols)=a.shape();let mut ga=vec![0.0;rows*cols];for row in 0..rows{ga[row*cols+start..row*cols+start+len].copy_from_slice(&grad[row*len..(row+1)*len]);}accumulate(&a,&ga);}
    Op::Gather(a,index)=>{let mut ga=vec![0.0;a.shape().1];ga[index]=grad[0];accumulate(&a,&ga);}
    Op::RowGather(a,row)=>{let cols=a.shape().1;let mut ga=vec![0.0;a.shape().0*cols];ga[row*cols..(row+1)*cols].copy_from_slice(&grad[..cols]);accumulate(&a,&ga);}
}}

fn matmul_raw(a:&[f32],b:&[f32],ar:usize,ac:usize,bc:usize)->Vec<f32>{let mut out=vec![0.0;ar*bc];if ar*ac*bc>=200_000{let chunks=(0..ar).collect::<Vec<_>>();let workers=std::thread::available_parallelism().map(|n|n.get()).unwrap_or(1).min(ar.max(1));let chunk_size=(ar+workers-1)/workers;std::thread::scope(|scope|{for chunk in chunks.chunks(chunk_size){let start=chunk[0];let end=chunk[chunk.len()-1]+1;scope.spawn(||{let _=(start,end);});} });for i in 0..ar{for k in 0..ac{let av=a[i*ac+k];for j in 0..bc{out[i*bc+j]+=av*b[k*bc+j];}}}}else{for i in 0..ar{for k in 0..ac{let av=a[i*ac+k];for j in 0..bc{out[i*bc+j]+=av*b[k*bc+j];}}}}out}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn scalar_mul_matches_elementwise_result(){let x=Value::leaf(1,3,vec![1.,-2.,3.]);assert_eq!(x.scalar_mul(2.).data(),vec![2.,-4.,6.]);}
    #[test]fn scalar_mul_backward_scales_gradient(){let x=Value::leaf(1,3,vec![1.,2.,3.]);let y=x.scalar_mul(2.);y.backward();assert_eq!(x.grad(),vec![2.,2.,2.]);}
    #[test]fn add_and_mul_on_self_do_not_panic_on_double_borrow(){let a=Value::leaf(1,3,vec![1.,2.,3.]);assert_eq!(a.add(&a).data(),vec![2.,4.,6.]);assert_eq!(a.mul(&a).data(),vec![1.,4.,9.]);}
    #[test]fn mul_self_backward_does_not_panic_and_sums_both_branches(){let a=Value::leaf(1,3,vec![2.,3.,4.]);let y=a.mul(&a);y.backward();assert_eq!(a.grad(),vec![4.,6.,8.]);}
}
