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
    pub fn data(&self) -> Vec<f32> { self.0.borrow().d.clone() }
    pub fn grad(&self) -> Vec<f32> { self.0.borrow().g.clone() }
    pub fn shape(&self) -> (usize, usize) { let n = self.0.borrow(); (n.r, n.c) }
    pub fn zero_grad(&self) { self.0.borrow_mut().g.fill(0.0); }
    pub fn set_data(&self, data: Vec<f32>) { assert_eq!(data.len(), self.0.borrow().d.len(), "parameter length mismatch"); self.0.borrow_mut().d = data; }
    pub fn add(&self, rhs: &Self) -> Self { assert_eq!(self.shape(), rhs.shape(), "add shape mismatch"); let x=self.data(); let y=rhs.data(); Self::mk(self.shape().0,self.shape().1,x.iter().zip(y).map(|(a,b)|a+b).collect(),Op::Add(self.clone(),rhs.clone())) }
    pub fn mul(&self, rhs: &Self) -> Self { assert_eq!(self.shape(), rhs.shape(), "mul shape mismatch"); let x=self.data(); let y=rhs.data(); Self::mk(self.shape().0,self.shape().1,x.iter().zip(y).map(|(a,b)|a*b).collect(),Op::Mul(self.clone(),rhs.clone())) }
    pub fn scalar_mul(&self, scalar: f32) -> Self { assert!(scalar.is_finite()); self.mul(&Self::leaf(self.shape().0,self.shape().1,vec![scalar;self.data().len()])) }
    pub fn div_scalar(&self, scalar: f32) -> Self { assert!(scalar.is_finite() && scalar != 0.0); self.scalar_mul(1.0/scalar) }
    pub fn matmul(&self, rhs: &Self) -> Self {
        let (ar,ac)=self.shape(); let (br,bc)=rhs.shape(); assert_eq!(ac,br,"matmul shape mismatch: {ar}x{ac} · {br}x{bc}");
        let x=self.data(); let y=rhs.data();
        let out = matmul_raw(&x, &y, ar, ac, bc);
        Self::mk(ar,bc,out,Op::MatMul(self.clone(),rhs.clone()))
    }
    pub fn transpose(&self) -> Self { let (rows,cols)=self.shape(); let x=self.data(); let mut out=vec![0.0;rows*cols]; for i in 0..rows {for j in 0..cols {out[j*rows+i]=x[i*cols+j];}} Self::mk(cols,rows,out,Op::Transpose(self.clone())) }
    pub fn log(&self) -> Self { Self::mk(self.shape().0,self.shape().1,self.data().into_iter().map(|x|x.max(1e-20).ln()).collect(),Op::Log(self.clone())) }
    pub fn neg(&self) -> Self { Self::mk(self.shape().0,self.shape().1,self.data().into_iter().map(|x|-x).collect(),Op::Neg(self.clone())) }
    pub fn silu(&self) -> Self { Self::mk(self.shape().0,self.shape().1,self.data().into_iter().map(|x|x/(1.0+(-x).exp())).collect(),Op::Silu(self.clone())) }
    pub fn rms_norm(&self, eps: f32) -> Self {
        let (rows, cols)=self.shape(); assert_eq!(rows,1,"rms_norm expects a row vector"); assert!(eps.is_finite()&&eps>0.0);
        let x=self.data(); let mean_sq=x.iter().map(|v|v*v).sum::<f32>()/cols as f32; let inv=(mean_sq+eps).sqrt().recip();
        Self::mk(rows,cols,x.iter().map(|v|v*inv).collect(),Op::RmsNorm(self.clone(),eps))
    }
    pub fn softmax(&self) -> Self { let (rows,cols)=self.shape(); assert_eq!(rows,1,"softmax expects a row vector"); let x=self.data(); let max=x.iter().copied().fold(f32::NEG_INFINITY,f32::max); let exp:Vec<f32>=x.iter().map(|v|(*v-max).exp()).collect(); let sum=exp.iter().sum::<f32>().max(1e-20); Self::mk(1,cols,exp.into_iter().map(|v|v/sum).collect(),Op::Softmax(self.clone())) }
    pub fn concat_rows(xs:&[Self])->Self { assert!(!xs.is_empty()); let cols=xs[0].shape().1; let mut data=Vec::new(); for x in xs {assert_eq!(x.shape(),(1,cols),"concat_rows shape mismatch"); data.extend(x.data());} Self::mk(xs.len(),cols,data,Op::ConcatRows(xs.to_vec())) }
    pub fn concat_cols(xs:&[Self])->Self { assert!(!xs.is_empty()); let rows=xs[0].shape().0; let cols:usize=xs.iter().map(|x|{assert_eq!(x.shape().0,rows,"concat_cols row mismatch");x.shape().1}).sum(); let mut out=vec![0.0;rows*cols]; let mut offset=0; for x in xs {let width=x.shape().1; let data=x.data(); for i in 0..rows {out[i*cols+offset..i*cols+offset+width].copy_from_slice(&data[i*width..(i+1)*width]);} offset+=width;} Self::mk(rows,cols,out,Op::ConcatCols(xs.to_vec())) }
    pub fn slice_cols(&self,start:usize,len:usize)->Self { let (rows,cols)=self.shape(); assert!(start<=cols&&len<=cols-start); let data=self.data(); let mut out=Vec::with_capacity(rows*len); for i in 0..rows {out.extend_from_slice(&data[i*cols+start..i*cols+start+len]);} Self::mk(rows,len,out,Op::Slice(self.clone(),start,len)) }
    pub fn gather(&self,index:usize)->Self { let(rows,cols)=self.shape(); assert_eq!(rows,1,"gather expects a row vector"); assert!(index<cols); Self::mk(1,1,vec![self.data()[index]],Op::Gather(self.clone(),index)) }
    pub fn row(&self,row:usize)->Self { let(rows,cols)=self.shape(); assert!(row<rows); let data=self.data(); Self::mk(1,cols,data[row*cols..(row+1)*cols].to_vec(),Op::RowGather(self.clone(),row)) }
    pub fn backward(&self) { let mut order=Vec::new(); let mut seen=HashSet::new(); topo(self,&mut seen,&mut order); self.0.borrow_mut().g.fill(1.0); for node in order.into_iter().rev(){back(&node);} }
}

fn topo(value:&Value,seen:&mut HashSet<usize>,order:&mut Vec<Value>){ if !seen.insert(value.id()){return;} match value.0.borrow().op.clone(){ Op::Leaf=>{}, Op::Add(a,b)|Op::Mul(a,b)|Op::MatMul(a,b)=>{topo(&a,seen,order);topo(&b,seen,order)}, Op::Transpose(a)|Op::Log(a)|Op::Neg(a)|Op::Softmax(a)|Op::Silu(a)|Op::RmsNorm(a,_)|Op::Slice(a,_,_)|Op::Gather(a,_)|Op::RowGather(a,_)=>topo(&a,seen,order), Op::ConcatRows(xs)|Op::ConcatCols(xs)=>{for x in xs{topo(&x,seen,order);}} } order.push(value.clone()); }
fn accumulate(value:&Value,grad:&[f32]){let mut node=value.0.borrow_mut(); assert_eq!(node.g.len(),grad.len()); for(dst,src) in node.g.iter_mut().zip(grad){*dst+=*src;}}
fn back(value:&Value){ let grad=value.grad(); match value.0.borrow().op.clone(){
    Op::Leaf=>{}
    Op::Add(a,b)=>{accumulate(&a,&grad);accumulate(&b,&grad)}
    Op::Mul(a,b)=>{let x=a.data();let y=b.data();accumulate(&a,&grad.iter().zip(&y).map(|(g,y)|g*y).collect::<Vec<_>>());accumulate(&b,&grad.iter().zip(&x).map(|(g,x)|g*x).collect::<Vec<_>>());}
    Op::MatMul(a,b)=>{let(ar,ac)=a.shape();let(_,bc)=b.shape();let x=a.data();let y=b.data();let mut ga=vec![0.0;ar*ac];let mut gb=vec![0.0;ac*bc];for i in 0..ar{for k in 0..ac{for j in 0..bc{let g=grad[i*bc+j];ga[i*ac+k]+=g*y[k*bc+j];gb[k*bc+j]+=x[i*ac+k]*g;}}}accumulate(&a,&ga);accumulate(&b,&gb);}
    Op::Transpose(a)=>{let(out_rows,out_cols)=value.shape();let mut ga=vec![0.0;grad.len()];for i in 0..out_rows{for j in 0..out_cols{ga[j*out_rows+i]=grad[i*out_cols+j];}}accumulate(&a,&ga);}
    Op::Log(a)=>{let data=a.data();accumulate(&a,&grad.iter().zip(data).map(|(g,x)|if x>=1e-20{g/x}else{0.0}).collect::<Vec<_>>());}
    Op::Neg(a)=>accumulate(&a,&grad.iter().map(|g|-g).collect::<Vec<_>>()),
    Op::Silu(a)=>{let data=a.data();let ga=data.iter().zip(&grad).map(|(x,g)|{let s=1.0/(1.0+(-x).exp());g*s*(1.0+x*(1.0-s))}).collect::<Vec<_>>();accumulate(&a,&ga);}
    Op::RmsNorm(a,eps)=>{let x=a.data();let n=x.len() as f32;let mean_sq=x.iter().map(|v|v*v).sum::<f32>()/n;let r=(mean_sq+eps).sqrt();let inv=1.0/r;let dot=grad.iter().zip(&x).map(|(g,x)|g*x).sum::<f32>();let coeff=dot/(n*r*r*r);let ga=x.iter().zip(&grad).map(|(x,g)|g*inv-x*coeff).collect::<Vec<_>>();accumulate(&a,&ga);}
    Op::Softmax(a)=>{let y=value.data();let dot=grad.iter().zip(&y).map(|(g,y)|g*y).sum::<f32>();let ga=y.iter().zip(&grad).map(|(y,g)|y*(g-dot)).collect::<Vec<_>>();accumulate(&a,&ga);}
    Op::ConcatRows(xs)=>{let cols=value.shape().1;for(row,x)in xs.iter().enumerate(){accumulate(x,&grad[row*cols..(row+1)*cols]);}}
    Op::ConcatCols(xs)=>{let rows=value.shape().0;let cols=value.shape().1;let mut offset=0;for x in xs{let width=x.shape().1;let mut gx=vec![0.0;rows*width];for row in 0..rows{gx[row*width..(row+1)*width].copy_from_slice(&grad[row*cols+offset..row*cols+offset+width]);}accumulate(&x,&gx);offset+=width;}}
    Op::Slice(a,start,len)=>{let(rows,cols)=a.shape();let mut ga=vec![0.0;rows*cols];for row in 0..rows{ga[row*cols+start..row*cols+start+len].copy_from_slice(&grad[row*len..(row+1)*len]);}accumulate(&a,&ga);}
    Op::Gather(a,index)=>{let mut ga=vec![0.0;a.data().len()];ga[index]=grad[0];accumulate(&a,&ga);}
    Op::RowGather(a,row)=>{let cols=a.shape().1;let mut ga=vec![0.0;a.data().len()];ga[row*cols..(row+1)*cols].copy_from_slice(&grad);accumulate(&a,&ga);}
}}

/// Multiply-add count above which the dominant `1 x ac` * `ac x bc` shape in this model (every
/// Q/K/V/O/up/down/logits projection is a single-token row vector times a weight matrix) is
/// split across threads. Below this, or on a single-core host, the thread-spawn overhead loses
/// to the plain sequential loop.
const PARALLEL_MATMUL_THRESHOLD: usize = 4096;

fn matmul_raw(x: &[f32], y: &[f32], ar: usize, ac: usize, bc: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ar * bc];
    if ar == 1 {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
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
    for i in 0..ar {
        for k in 0..ac {
            let a = x[i * ac + k];
            if a == 0.0 { continue; }
            for j in 0..bc { out[i * bc + j] += a * y[k * bc + j]; }
        }
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

#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn matmul_backward(){let a=Value::leaf(1,2,vec![2.,3.]);let b=Value::leaf(2,1,vec![5.,7.]);let y=a.matmul(&b);y.backward();assert_eq!(a.grad(),vec![5.,7.]);assert_eq!(b.grad(),vec![2.,3.]);}
 #[test]fn transpose_backward(){let a=Value::leaf(2,3,vec![1.,2.,3.,4.,5.,6.]);let y=a.transpose();y.backward();assert_eq!(a.grad(),vec![1.;6]);}
 #[test]fn concat_cols_matrix_backward(){let a=Value::leaf(2,1,vec![1.,2.]);let b=Value::leaf(2,2,vec![3.,4.,5.,6.]);let y=Value::concat_cols(&[a.clone(),b.clone()]);y.backward();assert_eq!(a.grad(),vec![1.,1.]);assert_eq!(b.grad(),vec![1.,1.,1.,1.]);}
 #[test]fn slice_matrix_backward(){let a=Value::leaf(2,4,vec![1.,2.,3.,4.,5.,6.,7.,8.]);let y=a.slice_cols(1,2);y.backward();assert_eq!(a.grad(),vec![0.,1.,1.,0.,0.,1.,1.,0.]);}
 #[test]fn row_gather_backward(){let a=Value::leaf(3,2,vec![1.,2.,3.,4.,5.,6.]);let y=a.row(1);y.backward();assert_eq!(a.grad(),vec![0.,0.,1.,1.,0.,0.]);}
 #[test]fn softmax_sums_to_one(){let y=Value::leaf(1,3,vec![1.,2.,3.]).softmax();let sum:f32=y.data().iter().sum();assert!((sum-1.).abs()<1e-6);}
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
}