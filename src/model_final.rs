use crate::autograd::Value;
use crate::config::Config;

pub struct Linear { pub w: Value }
impl Linear { pub fn new(i:usize,o:usize,s:&mut u64)->Self{Self{w:Value::parameter(o,i,s)}} pub fn f(&self,x:&Value)->Value{x.matmul(&self.w.transpose())} }

pub struct Block { pub q:Linear,pub k:Linear,pub v:Linear,pub o:Linear,pub up:Linear,pub down:Linear }
impl Block { fn new(c:&Config,s:&mut u64)->Self{let d=c.d_model;let f=c.ffn_dim;Self{q:Linear::new(d,d,s),k:Linear::new(d,d,s),v:Linear::new(d,d,s),o:Linear::new(d,d,s),up:Linear::new(d,f,s),down:Linear::new(f,d,s)}} }

pub struct Model { pub cfg:Config,pub emb:Vec<Value>,pub blocks:Vec<Block> }
impl Model {
 pub fn new(cfg:Config,seed:u64)->Self{let mut s=seed;let emb=(0..cfg.vocab_size).map(|_|Value::parameter(1,cfg.d_model,&mut s)).collect();let blocks=(0..cfg.n_layers).map(|_|Block::new(&cfg,&mut s)).collect();Self{cfg,emb,blocks}}
 fn attention(&self,b:&Block,states:&[Value],pos:usize)->Value{let q=b.q.f(&states[pos]);let mut scores=Vec::with_capacity(pos+1);let mut vals=Vec::with_capacity(pos+1);for j in 0..=pos{let k=b.k.f(&states[j]).transpose();scores.push(q.matmul(&k).div_scalar((self.cfg.head_dim()as f32).sqrt()));vals.push(b.v.f(&states[j]));}let a=Value::concat_cols(&scores).softmax();b.o.f(&a.matmul(&Value::concat_rows(&vals)))}
 pub fn forward_hidden(&self,tokens:&[usize])->Value{assert!(!tokens.is_empty());assert!(tokens.len()<=self.cfg.context);let mut st:Vec<Value>=tokens.iter().map(|&t|self.emb[t].clone()).collect();for b in &self.blocks{let mut next=Vec::with_capacity(st.len());for i in 0..st.len(){let r=st[i].add(&self.attention(b,&st,i));let h=b.up.f(&r).relu();let h=b.down.f(&h);next.push(r.add(&h));}st=next;}st.last().unwrap().clone()}
 pub fn logits(&self,h:&Value)->Value{let xs:Vec<Value>=self.emb.iter().map(|e|h.matmul(&e.transpose())).collect();Value::concat_cols(&xs)}
 pub fn parameters(&self)->Vec<Value>{let mut p=self.emb.clone();for b in &self.blocks{p.extend([b.q.w.clone(),b.k.w.clone(),b.v.w.clone(),b.o.w.clone(),b.up.w.clone(),b.down.w.clone()]);}p}
}
