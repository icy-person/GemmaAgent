use crate::{autograd::Value,config::Config};
pub struct Linear{pub w:Value}
impl Linear{fn new(i:usize,o:usize,s:&mut u64)->Self{Self{w:Value::parameter(o,i,s)}}fn f(&self,x:&Value)->Value{x.matmul(&self.w.transpose())}}
pub struct Block{q:Linear,k:Linear,v:Linear,o:Linear,up:Linear,down:Linear}
impl Block{fn new(c:&Config,s:&mut u64)->Self{let d=c.d_model;let f=c.ffn;Self{q:Linear::new(d,d,s),k:Linear::new(d,d,s),v:Linear::new(d,d,s),o:Linear::new(d,d,s),up:Linear::new(d,f,s),down:Linear::new(f,d,s)}}}
pub struct Model{pub cfg:Config,pub emb:Vec<Value>,pub blocks:Vec<Block>}
impl Model{
 pub fn new(cfg:Config,seed:u64)->Self{let mut s=seed;let emb=(0..cfg.vocab).map(|_|Value::parameter(1,cfg.d_model,&mut s)).collect();let blocks=(0..cfg.layers).map(|_|Block::new(&cfg,&mut s)).collect();Self{cfg,emb,blocks}}
 fn attn(&self,b:&Block,st:&[Value],pos:usize)->Value{let q=b.q.f(&st[pos]);let k_all:Vec<Value>=st[..=pos].iter().map(|x|b.k.f(x)).collect();let v_all:Vec<Value>=st[..=pos].iter().map(|x|b.v.f(x)).collect();let hd=self.cfg.head_dim();let mut heads=Vec::new();for h in 0..self.cfg.heads{let qh=q.slice_cols(h*hd,hd);let mut scores=Vec::with_capacity(k_all.len());let mut vals=Vec::with_capacity(v_all.len());for j in 0..k_all.len(){let kh=k_all[j].slice_cols(h*hd,hd).transpose();scores.push(qh.matmul(&kh).div_scalar((hd as f32).sqrt()));vals.push(v_all[j].slice_cols(h*hd,hd));}let a=Value::concat_cols(&scores).softmax();heads.push(a.matmul(&Value::concat_rows(&vals)));}b.o.f(&Value::concat_cols(&heads))}
 pub fn forward_hidden(&self,tokens:&[usize])->Value{assert!(!tokens.is_empty()&&tokens.len()<=self.cfg.context);let mut st:Vec<Value>=tokens.iter().map(|&t|self.emb[t].clone()).collect();for b in &self.blocks{let mut next=Vec::with_capacity(st.len());for i in 0..st.len(){let r=st[i].add(&self.attn(b,&st,i));let h=b.up.f(&r).relu();next.push(r.add(&b.down.f(&h)));}st=next;}st.last().unwrap().clone()}
 pub fn logits(&self,h:&Value)->Value{let v:Vec<Value>=self.emb.iter().map(|e|h.matmul(&e.transpose())).collect();Value::concat_cols(&v)}
 pub fn parameters(&self)->Vec<Value>{let mut p=self.emb.clone();for b in &self.blocks{p.extend([b.q.w.clone(),b.k.w.clone(),b.v.w.clone(),b.o.w.clone(),b.up.w.clone(),b.down.w.clone()]);}p}
}
