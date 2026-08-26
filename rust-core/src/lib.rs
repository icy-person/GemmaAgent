use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

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
    pub duration_ms: i64,
    pub project: Option<String>,
    pub tags: Vec<String>,
    pub failure_reason: Option<String>,
}

#[derive(Debug)]
struct MemoryEngine {
    root: PathBuf,
    index: Vec<Experience>,
}

impl MemoryEngine {
    fn open(root: PathBuf) -> Self {
        let _ = fs::create_dir_all(&root);
        let file = root.join("experiences.json");
        let index = fs::read_to_string(file)
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<Experience>>(&s).ok())
            .unwrap_or_default();
        Self { root, index }
    }

    fn persist(&self) -> Result<(), String> {
        let file = self.root.join("experiences.json");
        serde_json::to_string_pretty(&self.index)
            .map_err(|e| e.to_string())
            .and_then(|s| fs::write(file, s).map_err(|e| e.to_string()))
    }

    fn store(&mut self, experience: Experience) -> Result<String, String> {
        let id = experience.id.clone();
        self.index.retain(|e| e.id != id);
        self.index.push(experience);
        self.persist()?;
        Ok(id)
    }

    fn count(&self) -> usize { self.index.len() }

    fn search(&self, query: &str, limit: usize) -> Vec<Experience> {
        let q = tokenize(query);
        let mut scored: Vec<(f64, &Experience)> = self.index.iter().map(|e| {
            let text = format!("{} {} {} {}", e.task, e.plan, e.result, e.tags.join(" "));
            let tokens = tokenize(&text);
            let overlap = tokens.intersection(&q).count() as f64;
            let success = if e.success { 0.2 } else { 0.0 };
            let score = overlap + e.score.clamp(0.0, 1.0) * 0.5 + success;
            (score, e)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit.max(1)).map(|(_, e)| e.clone()).collect()
    }
}

fn tokenize(s: &str) -> std::collections::HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2)
        .map(ToOwned::to_owned)
        .collect()
}

fn engines() -> &'static Mutex<HashMap<jlong, Box<MemoryEngine>>> {
    static ENGINES: OnceLock<Mutex<HashMap<jlong, Box<MemoryEngine>>>> = OnceLock::new();
    ENGINES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeCreate(mut env: JNIEnv, _class: JClass, path: JString) -> jlong {
    let raw: String = match env.get_string(&path) { Ok(v) => v.into(), Err(_) => return 0 };
    let engine = Box::new(MemoryEngine::open(PathBuf::from(raw)));
    let ptr = Box::into_raw(engine) as jlong;
    match engines().lock() {
        Ok(mut map) => {
            map.insert(ptr, unsafe { Box::from_raw(ptr as *mut MemoryEngine) });
            ptr
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeDestroy(_env: JNIEnv, _class: JClass, handle: jlong) {
    if handle == 0 { return; }
    if let Ok(mut map) = engines().lock() { map.remove(&handle); }
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeSearch(mut env: JNIEnv, _class: JClass, handle: jlong, query: JString, limit: jint) -> jstring {
    if handle == 0 { return env.new_string("[]").unwrap().into_raw(); }
    let q: String = match env.get_string(&query) { Ok(v) => v.into(), Err(_) => return env.new_string("[]").unwrap().into_raw() };
    let result = match engines().lock() {
        Ok(map) => map.get(&handle).map(|engine| engine.search(&q, limit.max(1) as usize)).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    env.new_string(serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string())).unwrap().into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeStore(mut env: JNIEnv, _class: JClass, handle: jlong, experience_json: JString) -> jstring {
    if handle == 0 { return env.new_string("").unwrap().into_raw(); }
    let raw: String = match env.get_string(&experience_json) { Ok(v) => v.into(), Err(_) => return env.new_string("").unwrap().into_raw() };
    let experience: Experience = match serde_json::from_str(&raw) { Ok(v) => v, Err(_) => return env.new_string("").unwrap().into_raw() };
    let id = match engines().lock() {
        Ok(mut map) => map.get_mut(&handle).and_then(|engine| engine.store(experience).ok()).unwrap_or_default(),
        Err(_) => String::new(),
    };
    env.new_string(id).unwrap().into_raw()
}

#[no_mangle]
pub extern "system" fn Java_com_example_gemmaagent_rust_RustMemory_nativeCount(env: JNIEnv, _class: JClass, handle: jlong) -> jlong {
    if handle == 0 { return 0; }
    let _ = env.exception_check();
    match engines().lock() {
        Ok(map) => map.get(&handle).map(|engine| engine.count() as jlong).unwrap_or(0),
        Err(_) => 0,
    }
}
