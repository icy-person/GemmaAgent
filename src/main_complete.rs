mod autograd;
mod checkpoint;
mod config;
mod model;
mod tokenizer;
mod optim;
use autograd::Value;use config::Config;use model::Model;use tokenizer::Tokenizer;use optim::AdamW;

fn argmax(x:&[f32])->usize{x.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).map(|(i,_)|i).unwrap_or(0)}
fn loss_for(model:&Model,input:&[usize],target:usize)->Value{let h=model.forward_hidden(input);let logits=model.logits(&h);let p=logits.softmax();p.gather(target).log().neg()}
fn train(steps:usize,ckpt:&str){let cfg=Config::debug();println!("training debug model: {} params",cfg.params());let tok=Tokenizer::new();let data=tok.encode(&tokenizer::tiny_corpus());let model=Model::new(cfg,42);let params=model.parameters();let mut opt=AdamW::new(0.003);for step in 1..=steps{let start=1+(step%(data.len()-cfg.context-2));let input=&data[start..start+cfg.context];let target=data[start+cfg.context];let loss=loss_for(&model,input,target);let lv=loss.data()[0];loss.backward();opt.step(&params);if step==1||step%50==0{println!("step {:5} loss {:.5}",step,lv);}}checkpoint::save(ckpt,&params).expect("checkpoint save failed");println!("saved {ckpt}");}
fn infer(ckpt:&str,prompt:&str){let cfg=Config::debug();let tok=Tokenizer::new();let model=Model::new(cfg,42);let params=model.parameters();if std::path::Path::new(ckpt).exists(){checkpoint::load(ckpt,&params).expect("checkpoint load failed");}else{println!("checkpoint not found; using random weights");}let mut ids=tok.encode(prompt);ids.pop();for _ in 0..80{let begin=ids.len().saturating_sub(cfg.context);let h=model.forward_hidden(&ids[begin..]);let next=argmax(&model.logits(&h).data());ids.push(next);if next==tokenizer::EOS{break;}}println!("{}",tok.decode(&ids));}
fn main(){let a:Vec<String>=std::env::args().collect();match a.get(1).map(String::as_str){Some("train")=>train(a.get(2).and_then(|x|x.parse().ok()).unwrap_or(500),a.get(3).map(String::as_str).unwrap_or("gemma-agent.ckpt")),Some("infer")=>infer(a.get(2).map(String::as_str).unwrap_or("gemma-agent.ckpt"),a.get(3).map(String::as_str).unwrap_or("Rust is")),_=>{println!("GemmaAgent Rust LLM");println!("  train [steps] [checkpoint]");println!("  infer [checkpoint] [prompt]");println!("  cargo test");}}}
