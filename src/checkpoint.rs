use crate::autograd::Value;
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GRS2";

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
    let count = u64::from_le_bytes(u64_buf) as usize;
    if count != parameters.len() {
        return Err(invalid("parameter count mismatch"));
    }

    for parameter in parameters {
        file.read_exact(&mut u64_buf)?;
        let rows = u64::from_le_bytes(u64_buf) as usize;
        file.read_exact(&mut u64_buf)?;
        let cols = u64::from_le_bytes(u64_buf) as usize;
        if parameter.shape() != (rows, cols) {
            return Err(invalid("parameter shape mismatch"));
        }

        let mut data = vec![0.0f32; rows * cols];
        for value in &mut data {
            let mut bytes = [0u8; 4];
            file.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
        }
        parameter.set_data(data);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, fs};

    #[test]
    fn round_trip_preserves_values() {
        let mut seed = 7;
        let a = Value::parameter(2, 3, &mut seed);
        let b = Value::parameter(1, 4, &mut seed);
        let original_a = a.data();
        let original_b = b.data();
        let path = env::temp_dir().join(format!("gemma-agent-{}.ckpt", std::process::id()));

        save(&path, &[a.clone(), b.clone()]).unwrap();
        a.set_data(vec![0.0; 6]);
        b.set_data(vec![0.0; 4]);
        load(&path, &[a.clone(), b.clone()]).unwrap();

        assert_eq!(a.data(), original_a);
        assert_eq!(b.data(), original_b);
        fs::remove_file(path).ok();
    }
}
