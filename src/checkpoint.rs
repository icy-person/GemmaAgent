use crate::autograd::Value;
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GRS2";
const MAX_ELEMENTS: u64 = 1 << 30;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub fn save<P: AsRef<Path>>(path: P, parameters: &[Value]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&(parameters.len() as u64).to_le_bytes())?;

    for parameter in parameters {
        let (rows, cols) = parameter.shape();
        let data = parameter.data();
        if data.iter().any(|value| !value.is_finite()) {
            return Err(invalid("refusing to save non-finite parameter values"));
        }
        file.write_all(&(rows as u64).to_le_bytes())?;
        file.write_all(&(cols as u64).to_le_bytes())?;
        for value in data {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()
}

pub fn load<P: AsRef<Path>>(path: P, parameters: &[Value]) -> io::Result<()> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("unsupported or corrupt checkpoint"));
    }

    let mut u64_buf = [0u8; 8];
    file.read_exact(&mut u64_buf)?;
    let count = u64::from_le_bytes(u64_buf);
    if count != parameters.len() as u64 {
        return Err(invalid("parameter count mismatch"));
    }

    let mut loaded = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        file.read_exact(&mut u64_buf)?;
        let rows = u64::from_le_bytes(u64_buf);
        file.read_exact(&mut u64_buf)?;
        let cols = u64::from_le_bytes(u64_buf);
        if rows == 0 || cols == 0 || rows.checked_mul(cols).is_none() || rows * cols > MAX_ELEMENTS {
            return Err(invalid("invalid checkpoint tensor shape"));
        }
        let rows = rows as usize;
        let cols = cols as usize;
        if parameter.shape() != (rows, cols) {
            return Err(invalid("parameter shape mismatch"));
        }

        let len = rows * cols;
        let mut data = vec![0.0f32; len];
        for value in &mut data {
            let mut bytes = [0u8; 4];
            file.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
            if !value.is_finite() {
                return Err(invalid("checkpoint contains non-finite parameter values"));
            }
        }
        loaded.push(data);
    }

    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        return Err(invalid("checkpoint has unexpected trailing data"));
    }

    for (parameter, data) in parameters.iter().zip(loaded) {
        parameter.set_data(data);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};

    fn temp_path(label: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!("gemma-agent-{label}-{}.ckpt", std::process::id()))
    }

    #[test]
    fn round_trip_preserves_values() {
        let mut seed = 7;
        let a = Value::parameter(2, 3, &mut seed);
        let b = Value::parameter(1, 4, &mut seed);
        let original_a = a.data();
        let original_b = b.data();
        let path = temp_path("round-trip");

        save(&path, &[a.clone(), b.clone()]).unwrap();
        a.set_data(vec![0.0; 6]);
        b.set_data(vec![0.0; 4]);
        load(&path, &[a.clone(), b.clone()]).unwrap();

        assert_eq!(a.data(), original_a);
        assert_eq!(b.data(), original_b);
        fs::remove_file(path).ok();
    }

    #[test]
    fn truncated_checkpoint_does_not_partially_modify_parameters() {
        let a = Value::leaf(1, 2, vec![3.0, 4.0]);
        let b = Value::leaf(1, 2, vec![5.0, 6.0]);
        let path = temp_path("truncated");
        save(&path, &[a.clone(), b.clone()]).unwrap();

        b.set_data(vec![9.0, 10.0]);
        let mut bytes = fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 2);
        fs::write(&path, bytes).unwrap();

        let before_a = a.data();
        let err = load(&path, &[a.clone(), b.clone()]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(a.data(), before_a);
        assert_eq!(b.data(), vec![9.0, 10.0]);
        fs::remove_file(path).ok();
    }

    #[test]
    fn trailing_data_is_rejected_without_modification() {
        let a = Value::leaf(1, 2, vec![1.0, 2.0]);
        let path = temp_path("trailing");
        save(&path, std::slice::from_ref(&a)).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.push(0xAA);
        fs::write(&path, bytes).unwrap();

        a.set_data(vec![7.0, 8.0]);
        let err = load(&path, std::slice::from_ref(&a)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(a.data(), vec![7.0, 8.0]);
        fs::remove_file(path).ok();
    }
}
