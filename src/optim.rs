use crate::autograd::Value;
use std::{collections::HashMap, fs::File, io::{self, Read, Write}, path::Path};

const MAGIC: &[u8; 4] = b"ADW1";

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub struct AdamW {
    t: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    max_grad_norm: f32,
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        assert!(lr.is_finite() && lr > 0.0);
        Self {
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        assert!(lr.is_finite() && lr > 0.0);
        self.lr = lr;
    }

    pub fn step(&mut self, parameters: &[Value]) {
        self.t += 1;
        let bias1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias2 = 1.0 - self.beta2.powi(self.t as i32);
        let mut global_norm_sq = 0.0f32;
        for parameter in parameters {
            global_norm_sq += parameter.grad().iter().map(|g| g * g).sum::<f32>();
        }
        let global_norm = global_norm_sq.sqrt();
        let clip_scale = if global_norm > self.max_grad_norm {
            self.max_grad_norm / global_norm
        } else {
            1.0
        };
        for parameter in parameters {
            let id = parameter.id();
            let data = parameter.data();
            let grad = parameter.grad();
            assert_eq!(data.len(), grad.len());
            let m = self.m.entry(id).or_insert_with(|| vec![0.0; data.len()]);
            let v = self.v.entry(id).or_insert_with(|| vec![0.0; data.len()]);
            let mut next = data.clone();
            for i in 0..data.len() {
                let clipped_grad = grad[i] * clip_scale;
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * clipped_grad;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * clipped_grad * clipped_grad;
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                next[i] *= 1.0 - self.lr * self.weight_decay;
                next[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            parameter.set_data(next);
            parameter.zero_grad();
        }
    }

    pub fn save_state<P: AsRef<Path>>(&self, path: P, parameters: &[Value]) -> io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(parameters.len() as u64).to_le_bytes())?;
        file.write_all(&(self.t as u64).to_le_bytes())?;
        for parameter in parameters {
            let id = parameter.id();
            let m = self.m.get(&id).cloned().unwrap_or_else(|| vec![0.0; parameter.data().len()]);
            let v = self.v.get(&id).cloned().unwrap_or_else(|| vec![0.0; parameter.data().len()]);
            if m.len() != parameter.data().len() || v.len() != parameter.data().len() {
                return Err(invalid("optimizer state tensor length mismatch"));
            }
            file.write_all(&(m.len() as u64).to_le_bytes())?;
            for value in m.iter().chain(v.iter()) {
                if !value.is_finite() {
                    return Err(invalid("refusing to save non-finite optimizer state"));
                }
                file.write_all(&value.to_le_bytes())?;
            }
        }
        file.flush()
    }

    pub fn load_state<P: AsRef<Path>>(&mut self, path: P, parameters: &[Value]) -> io::Result<()> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("unsupported or corrupt optimizer checkpoint"));
        }
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        let count = u64::from_le_bytes(buf);
        if count != parameters.len() as u64 {
            return Err(invalid("optimizer parameter count mismatch"));
        }
        file.read_exact(&mut buf)?;
        let step = u64::from_le_bytes(buf);
        if step > usize::MAX as u64 {
            return Err(invalid("optimizer step is too large"));
        }
        let mut next_m = HashMap::with_capacity(parameters.len());
        let mut next_v = HashMap::with_capacity(parameters.len());
        for parameter in parameters {
            file.read_exact(&mut buf)?;
            let len = u64::from_le_bytes(buf) as usize;
            if len != parameter.data().len() {
                return Err(invalid("optimizer tensor length mismatch"));
            }
            let mut m = vec![0.0f32; len];
            let mut v = vec![0.0f32; len];
            for value in &mut m {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes)?;
                *value = f32::from_le_bytes(bytes);
                if !value.is_finite() { return Err(invalid("optimizer checkpoint has non-finite values")); }
            }
            for value in &mut v {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes)?;
                *value = f32::from_le_bytes(bytes);
                if !value.is_finite() { return Err(invalid("optimizer checkpoint has non-finite values")); }
            }
            next_m.insert(parameter.id(), m);
            next_v.insert(parameter.id(), v);
        }
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(invalid("optimizer checkpoint has unexpected trailing data"));
        }
        self.t = step as usize;
        self.m = next_m;
        self.v = next_v;
        Ok(())
    }

    pub fn step_count(&self) -> usize { self.t }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};

    #[test]
    fn one_step_moves_parameter_and_clears_gradient() {
        let p = Value::leaf(1, 1, vec![1.0]);
        let loss = p.mul(&Value::leaf(1, 1, vec![2.0]));
        loss.backward();
        let mut opt = AdamW::new(0.01);
        opt.step(std::slice::from_ref(&p));
        assert!(p.data()[0] < 1.0);
        assert_eq!(p.grad(), vec![0.0]);
    }

    #[test]
    fn clipping_limits_a_large_gradient() {
        let p = Value::leaf(1, 1, vec![0.0]);
        let scale = Value::leaf(1, 1, vec![1000.0]);
        let loss = p.mul(&scale);
        loss.backward();
        let mut opt = AdamW::new(0.01);
        opt.step(std::slice::from_ref(&p));
        assert!(p.data()[0].abs() <= 0.011);
    }

    #[test]
    fn optimizer_state_round_trip_preserves_step() {
        let path = env::temp_dir().join(format!("gemma-agent-adam-{}.opt", std::process::id()));
        let p = Value::leaf(1, 1, vec![1.0]);
        let loss = p.mul(&Value::leaf(1, 1, vec![2.0]));
        loss.backward();
        let mut a = AdamW::new(0.01);
        a.step(std::slice::from_ref(&p));
        a.save_state(&path, std::slice::from_ref(&p)).unwrap();
        let mut b = AdamW::new(0.02);
        b.load_state(&path, std::slice::from_ref(&p)).unwrap();
        assert_eq!(b.step_count(), 1);
        fs::remove_file(path).ok();
    }
}
