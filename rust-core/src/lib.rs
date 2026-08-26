use jni::objects::{JClass, JString};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: String,
    pub task: String,
    pub plan: String,
    pub actions: Vec<String>,
    pub result: String,
    pub success: bool,
    pub score: f64,
    pub created_at_epoch_ms: i64,
    pub project: Option<String>,
    pub tags: Vec<String>,
}

fn tokenize(s: &str) -> HashSet<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric())
        .filter(|x| x.len() > 2).map(ToOwned::to_owned).collect()
}

fn similarity(a: &str, b: &str) -> f64 {
    let aa = tokenize(a); let bb = tokenize(b);
    if aa.is_empty() || bb.is_empty() { return 0.0; }
    let inter = aa.intersection(&bb).count() as f64;
    let union = aa.union(&bb).count() as f64;
    inter / union
}

pub struct MemoryEngine { path: PathBuf }

impl MemoryEngine {
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into(); fs::create_dir_all(&dir)?;
        let path = dir.join("experiences.jsonl");
        if !path.exists() { fs::File::create(&path)?; }
        Ok(Self { path })
    }

    pub fn store(&self, mut e: Experience) -> std::io::Result<String> {
        if e.id.is_empty() { e.id = Uuid::new_v4().to_string(); }
        let line = serde_json::to_string(&e)?;
        let mut f = OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{line}")?;
        Ok(e.id)
    }

    pub fn search(&self, query: &str, limit: usize) -> std::io::Result<Vec<Experience>> {
        let f = fs::File::open(&self.path)?;
        let reader = BufReader::new(f); let mut scored = Vec::new();
        for line in reader.lines() {
            let line = line?; if line.trim().is_empty() { continue; }
            if let Ok(e) = serde_json::from_str::<Experience>(&line) {
                let sim = similarity(query, &format!("{} {} {}", e.task, e.plan, e.result));
                let success_bonus = if e.success { 0.20 } else { 0.0 };
                let score = sim * 0.65 + e.score * 0.25 + success_bonus;
                if score > 0.05 { scored.push((score, e)); }
            }
        }
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(_,e)| e).collect())
    }

    pub fn count(&self) -> std::io::Result<u64> {
        let f = fs::File::open(&self.path)?;
        Ok(BufReader::new(f).lines().count() as u64)
    }
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeCreate(mut env: JNIEnv, _class: JClass, path: JString) -> jlong {
    let path: String = match env.get_string(&path) { Ok(v) => v.into(), Err(_) => return 0 };
    match MemoryEngine::new(path) { Ok(e) => Box::into_raw(Box::new(e)) as jlong, Err(_) => 0 }
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeDestroy(_env: JNIEnv, _class: JClass, handle: jlong) {
    if handle != 0 { unsafe { drop(Box::from_raw(handle as *mut MemoryEngine)); } }
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeSearch(mut env: JNIEnv, _class: JClass, handle: jlong, query: JString, limit: i32) -> jstring {
    if handle == 0 { return env.new_string("[]").unwrap().into_raw(); }
    let query: String = match env.get_string(&query) { Ok(v) => v.into(), Err(_) => return env.new_string("[]").unwrap().into_raw() };
    let e = unsafe { &*(handle as *mut MemoryEngine) };
    let result = e.search(&query, limit.max(1) as usize).unwrap_or_default();
    env.new_string(serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string())).unwrap().into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeStore(mut env: JNIEnv, _class: JClass, handle: jlong, experience_json: JString) -> jstring {
    if handle == 0 { return env.new_string("").unwrap().into_raw(); }
    let raw: String = match env.get_string(&experience_json) { Ok(v) => v.into(), Err(_) => return env.new_string("").unwrap().into_raw() };
    let e: Experience = match serde_json::from_str(&raw) { Ok(v) => v, Err(_) => return env.new_string("").unwrap().into_raw() };
    let id = unsafe { &*(handle as *mut MemoryEngine) }.store(e).unwrap_or_default();
    env.new_string(id).unwrap().into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeCount(mut env: JNIEnv, _class: JClass, handle: jlong) -> jlong {
    if handle == 0 { return 0; }
    env.exception_clear();
    unsafe { &*(handle as *mut MemoryEngine) }.count().unwrap_or(0) as jlong
}
