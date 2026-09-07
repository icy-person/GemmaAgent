use crate::autograd::Value;
use std::collections::HashMap;

pub struct AdamW {
    t: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
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
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step(&mut self, parameters: &[Value]) {
        self.t += 1;
        let bias1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias2 = 1.0 - self.beta2.powi(self.t as i32);

        for parameter in parameters {
            let id = parameter.id();
            let data = parameter.data();
            let grad = parameter.grad();
            assert_eq!(data.len(), grad.len());

            let m = self
                .m
                .entry(id)
                .or_insert_with(|| vec![0.0; data.len()]);
            let v = self
                .v
                .entry(id)
                .or_insert_with(|| vec![0.0; data.len()]);

            let mut next = data.clone();
            for i in 0..data.len() {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                next[i] *= 1.0 - self.lr * self.weight_decay;
                next[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            parameter.set_data(next);
            parameter.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
