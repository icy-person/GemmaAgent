use crate::{autograd::Value, config::Config};

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;

pub struct Linear { pub w: Value }
impl Linear { fn new(input: usize, output: usize, seed: &mut u64) -> Self { Self { w: Value::parameter(output, input, seed) } } fn forward(&self, x: &Value) -> Value { x.matmul(&self.w.transpose()) } }

pub struct Block { q: Linear, k: Linear, v: Linear, o: Linear, up: Linear, down: Linear }
impl Block { fn new(cfg: &Config, seed: &mut u64) -> Self { let d=cfg.d_model; let f=cfg.ffn; Self{q:Linear::new(d,d,seed),k:Linear::new(d,d,seed),v:Linear::new(d,d,seed),o:Linear::new(d,d,seed),up:Linear::new(d,f,seed),down:Linear::new(f,d,seed)} } }

pub struct Model { pub cfg: Config, pub emb: Value, pub blocks: Vec<Block> }
impl Model {
 pub fn new(cfg:Config,seed:u64)->Self{cfg.validate();let mut seed=seed;let emb=Value::parameter(cfg.vocab,cfg.d_model,&mut seed);let blocks=(0..cfg.layers).map(|_|Block::new(&cfg,&mut seed)).collect();Self{cfg,emb,blocks}}
 /// RoPE angle table for one position: `(cos, sin)`, each a `(1, head_dim/2)` row vector.
 /// Matches the AMD/Android Vulkan backend's `RotaryEncodingConfig` (theta=10000), so CPU and
 /// GPU rotate Q/K the same way instead of the two backends diverging in architecture.
 fn rope_angles(&self,pos:usize)->(Value,Value){let head_dim=self.cfg.head_dim();let half=head_dim/2;let mut cos=Vec::with_capacity(half);let mut sin=Vec::with_capacity(half);for i in 0..half{let exponent=(2*i) as f32/head_dim as f32;let freq=1.0/ROPE_THETA.powf(exponent);let angle=pos as f32*freq;cos.push(angle.cos());sin.push(angle.sin());}(Value::leaf(1,half,cos),Value::leaf(1,half,sin))}
 /// Applies RoPE independently to every head slice of a `(1, d_model)` projection, using the
 /// GPT-NeoX/LLaMA "rotate half" convention (contiguous halves instead of interleaved pairs) so
 /// it composes with the existing `slice_cols`/`concat_cols` ops without a new autograd Op.
 /// Each `(first[i], second[i])` pair is an independent 2D rotation, so this preserves the
 /// per-head vector norm exactly, same as any orthogonal rotation.
 fn apply_rope(&self,x:&Value,pos:usize)->Value{let head_dim=self.cfg.head_dim();let half=head_dim/2;let(cos,sin)=self.rope_angles(pos);let mut heads=Vec::with_capacity(self.cfg.heads);for head in 0..self.cfg.heads{let offset=head*head_dim;let first=x.slice_cols(offset,half);let second=x.slice_cols(offset+half,half);let rotated_first=first.mul(&cos).add(&second.mul(&sin).neg());let rotated_second=first.mul(&sin).add(&second.mul(&cos));heads.push(Value::concat_cols(&[rotated_first,rotated_second]));}Value::concat_cols(&heads)}
 fn attention(&self,block:&Block,query:&Value,keys:&[Value],values:&[Value],pos:usize)->Value{let head_dim=self.cfg.head_dim();let mut heads=Vec::with_capacity(self.cfg.heads);for head in 0..self.cfg.heads{let offset=head*head_dim;let qh=query.slice_cols(offset,head_dim);let mut scores=Vec::with_capacity(pos+1);let mut head_values=Vec::with_capacity(pos+1);for(key,value)in keys[..=pos].iter().zip(&values[..=pos]){scores.push(qh.matmul(&key.slice_cols(offset,head_dim).transpose()).div_scalar((head_dim as f32).sqrt()));head_values.push(value.slice_cols(offset,head_dim));}let weights=Value::concat_cols(&scores).softmax();heads.push(weights.matmul(&Value::concat_rows(&head_values)));}block.o.forward(&Value::concat_cols(&heads))}
 pub fn forward_all_hidden(&self,tokens:&[usize])->Vec<Value>{assert!(!tokens.is_empty()&&tokens.len()<=self.cfg.context);let mut states:Vec<Value>=tokens.iter().map(|&token|{assert!(token<self.cfg.vocab);self.emb.row(token)}).collect();for block in &self.blocks{let normed:Vec<Value>=states.iter().map(|x|x.rms_norm(RMS_EPS)).collect();let queries:Vec<Value>=normed.iter().enumerate().map(|(pos,x)|self.apply_rope(&block.q.forward(x),pos)).collect();let keys:Vec<Value>=normed.iter().enumerate().map(|(pos,x)|self.apply_rope(&block.k.forward(x),pos)).collect();let values:Vec<Value>=normed.iter().map(|x|block.v.forward(x)).collect();let mut next=Vec::with_capacity(states.len());for pos in 0..states.len(){let attention=self.attention(block,&queries[pos],&keys,&values,pos);let residual=states[pos].add(&attention);let hidden=block.up.forward(&residual.rms_norm(RMS_EPS)).silu();next.push(residual.add(&block.down.forward(&hidden)));}states=next;}states}
 pub fn forward_hidden(&self,tokens:&[usize])->Value{self.forward_all_hidden(tokens).pop().expect("non-empty token sequence")}
 pub fn logits(&self,hidden:&Value)->Value{hidden.rms_norm(RMS_EPS).matmul(&self.emb.transpose())}
 pub fn parameters(&self)->Vec<Value>{let mut parameters=vec![self.emb.clone()];for block in &self.blocks{parameters.extend([block.q.w.clone(),block.k.w.clone(),block.v.w.clone(),block.o.w.clone(),block.up.w.clone(),block.down.w.clone()]);}parameters}
}

#[cfg(test)]
mod tests{use super::*;
#[test]fn target_parameter_count_is_exact(){let cfg=Config::target();let model=Model::new(cfg,42);let total:usize=model.parameters().iter().map(|p|p.data().len()).sum();assert_eq!(total,19_275_776);assert_eq!(cfg.params(),total);}
#[test]fn forward_and_logits_have_expected_shapes(){let cfg=Config::debug();let model=Model::new(cfg,42);let hidden=model.forward_hidden(&[256,b'R' as usize,b'u' as usize]);assert_eq!(hidden.shape(),(1,cfg.d_model));assert_eq!(model.logits(&hidden).shape(),(1,cfg.vocab));}
#[test]fn forward_all_hidden_returns_every_position(){let cfg=Config::debug();let model=Model::new(cfg,42);let hidden=model.forward_all_hidden(&[256,65,66,67]);assert_eq!(hidden.len(),4);assert!(hidden.iter().all(|value|value.shape()==(1,cfg.d_model)));assert_eq!(hidden[3].data(),model.forward_hidden(&[256,65,66,67]).data());}
#[test]fn forward_is_deterministic_for_fixed_seed(){let cfg=Config::debug();let a=Model::new(cfg,123).forward_hidden(&[256,65,66,257]);let b=Model::new(cfg,123).forward_hidden(&[256,65,66,257]);assert_eq!(a.data(),b.data());}
#[test]fn rope_rotation_preserves_vector_norm(){let cfg=Config::debug();let model=Model::new(cfg,7);let mut seed=1;let x=Value::parameter(1,cfg.d_model,&mut seed);let rotated=model.apply_rope(&x,5);let norm=|v:&Value|v.data().iter().map(|d|d*d).sum::<f32>().sqrt();assert!((norm(&rotated)-norm(&x)).abs()<1e-4,"RoPE is a rotation and must preserve vector norm");}
#[test]fn rope_is_position_dependent(){let cfg=Config::debug();let model=Model::new(cfg,7);let mut seed=1;let x=Value::parameter(1,cfg.d_model,&mut seed);let at0=model.apply_rope(&x,0);let at5=model.apply_rope(&x,5);assert_ne!(at0.data(),at5.data());}
#[test]fn rope_at_position_zero_is_identity(){let cfg=Config::debug();let model=Model::new(cfg,7);let mut seed=1;let x=Value::parameter(1,cfg.d_model,&mut seed);let rotated=model.apply_rope(&x,0);for(a,b) in rotated.data().iter().zip(x.data().iter()){assert!((a-b).abs()<1e-5);}}
}