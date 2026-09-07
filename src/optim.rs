use crate::autograd::Value;
use std::collections::HashMap;

pub struct AdamW{step:usize,lr:f32,b1:f32,b2:f32,eps:f32,wd:f32,m:HashMap<usize,Vec<f32>>,v:HashMap<usize,Vec<f32>>}
impl AdamW{pub fn new(lr:f32)->Self{Self{step:0,lr,b1:.9,b2:.999,eps:1e-8,wd:.01,m:HashMap::new(),v:HashMap::new()}}pub fn step(&mut self,p:&[Value]){self.step+=1;let b1c=1.-self.b1.powi(self.step as i32);let b2c=1.-self.b2.powi(self.step as i32);for x in p{let id=std::rc::Rc::as_ptr(&x.0)as usize;let g=x.grad();let d=x.data();let mm=self.m.entry(id).or_insert_with(||vec![0.;d.len()]);let vv=self.v.entry(id).or_insert_with(||vec![0.;d.len()]);let mut nd=d.clone();for i in 0..d.len(){mm[i]=self.b1*mm[i]+(1.-self.b1)*g[i];vv[i]=self.b2*vv[i]+(1.-self.b2)*g[i]*g[i];let mh=mm[i]/b1c;let vh=vv[i]/b2c;nd[i]*=1.-self.lr*self.wd;nd[i]-=self.lr*mh/(vh.sqrt()+self.eps);}x.set_data(nd);x.zero_grad();}}}
