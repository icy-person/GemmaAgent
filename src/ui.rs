use eframe::egui::{self, Align, Color32, CornerRadius, FontId, Frame, Layout, Margin, RichText, Stroke, Vec2};
use std::{io::{BufRead, BufReader}, path::{Path, PathBuf}, process::{Child, Command, Stdio}, sync::{mpsc, Arc, Mutex}, thread, time::{Duration, Instant}};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend { Cpu, Gpu }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Page { Dashboard, Training, Inference, Hardware, Data, Settings, Logs }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision { F32, F16, Mixed }

struct JobControl { child: Arc<Mutex<Option<Child>>> }
impl JobControl {
    fn new() -> Self { Self { child: Arc::new(Mutex::new(None)) } }
    fn running(&self) -> bool { self.child.lock().map(|g| g.is_some()).unwrap_or(false) }
    fn stop(&self) { if let Ok(mut guard) = self.child.lock() { if let Some(mut child) = guard.take() { let _ = child.kill(); } } }
}

pub struct GemmaUi {
    page: Page, backend: Backend, precision: Precision, accent: Color32, dark: bool, compact: bool,
    steps: usize, batch: usize, grad_accum: usize, targets: usize, context: usize, lr: f64,
    checkpoint_every: usize, eval_every: usize, gpu: usize, gpu_kind: String, gpu_util: f64, cpu_threads: usize,
    data_path: String, val_path: String, checkpoint: String, resume: String, model: String,
    prep_script: String,
    temperature: f32, top_p: f32, top_k: usize, seed: u64, kv_cache: bool, flash_attention: bool,
    pinned_memory: bool, power_limit: bool, auto_save: bool, notifications: bool, logs: Vec<String>,
    rx: mpsc::Receiver<String>, tx: mpsc::Sender<String>, job: JobControl, started: Option<Instant>, toast: Option<(String, Instant)>,
}

impl GemmaUi {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();
        let app = Self {
            page: Page::Dashboard, backend: Backend::Gpu, precision: Precision::Mixed,
            accent: Color32::from_rgb(0, 105, 92), dark: true, compact: false,
            steps: 5000, batch: 2, grad_accum: 1, targets: 256, context: 1024,
            lr: 0.0003, checkpoint_every: 250, eval_every: 250, gpu: 0,
            gpu_kind: "integrated".into(), gpu_util: 50.0, cpu_threads: 8,
            data_path: "./data/train.txt".into(), val_path: "./data/val.txt".into(),
            checkpoint: "checkpoints/gemma-agent.bin".into(), resume: String::new(), prep_script: "Real corpus".into(),
            model: "Target · 19M".into(), temperature: 0.8, top_p: 0.95, top_k: 40,
            seed: 42, kv_cache: true, flash_attention: true, pinned_memory: true,
            power_limit: true, auto_save: true, notifications: true,
            logs: vec!["GemmaAgent UI initialized".into()], rx, tx, job: JobControl::new(), started: None, toast: None,
        };
        configure_theme(&cc.egui_ctx, app.accent, app.dark, app.compact); app
    }
    fn notify(&mut self, text: impl Into<String>) { self.toast = Some((text.into(), Instant::now())); }
    fn poll_logs(&mut self) {
        while let Ok(line) = self.rx.try_recv() {
            if let Some(output) = line.strip_prefix("__PREP_DONE__:") {
                match self.prep_script.as_str() {
                    "Real corpus" => { self.data_path = "data/train.txt".into(); self.val_path = "data/val.txt".into(); }
                    "Online corpus" => { self.data_path = "data/online/train.txt".into(); self.val_path = "data/online/val.txt".into(); }
                    "Curriculum" => { self.data_path = "data/curriculum/05-balanced.txt".into(); }
                    _ => {}
                }
                self.notify(format!("Dataset ready: {output}"));
            } else { self.logs.push(line); if self.logs.len() > 600 { self.logs.drain(..100); } }
        }
    }
    fn project_root() -> Option<PathBuf> {
        let starts = [std::env::current_dir().ok(), std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf))];
        for start in starts.into_iter().flatten() {
            let mut dir = start;
            loop { if dir.join("Cargo.toml").is_file() { return Some(dir); } if !dir.pop() { break; } }
        }
        None
    }
    fn find_script(name: &str) -> Option<PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(appdir) = std::env::var("APPDIR") { candidates.push(PathBuf::from(appdir).join("usr/share/gemma-agent/scripts").join(name)); }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join(name)); candidates.push(dir.join("scripts").join(name)); candidates.push(dir.join("../share/gemma-agent/scripts").join(name));
                if let Some(root) = dir.parent().and_then(|p| p.parent()) { candidates.push(root.join("scripts").join(name)); }
            }
        }
        if let Some(root) = Self::project_root() { candidates.push(root.join("scripts").join(name)); }
        candidates.into_iter().find(|path| path.is_file())
    }
    fn find_trainer(name: &str) -> Option<PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(exe) = std::env::current_exe() { if let Some(dir) = exe.parent() { candidates.push(dir.join(name)); candidates.push(dir.join("bin").join(name)); candidates.push(dir.join("trainers").join(name)); } }
        if let Some(root) = Self::project_root() { candidates.push(root.join("target").join("release").join(name)); candidates.push(root.join("target").join("debug").join(name)); candidates.push(root.join("bin").join(name)); candidates.push(root.join("trainers").join(name)); }
        if let Some(path_var) = std::env::var_os("PATH") { for dir in std::env::split_paths(&path_var) { let path = dir.join(name); if path.is_file() { return Some(path); } } }
        candidates.into_iter().find(|path| path.is_file())
    }
    fn start_prepare_job(&mut self) {
        if self.job.running() { self.notify("Another job is already running"); return; }
        let (script_name, output_dir) = match self.prep_script.as_str() { "Real corpus" => ("prepare_real_corpus.py", "data"), "Online corpus" => ("prepare_online_corpus.py", "data/online"), "Curriculum" => ("prepare_curriculum.py", "data/curriculum"), _ => return };
        let script = match Self::find_script(script_name) { Some(path) => path, None => { let message = format!("Preparation script not found: {script_name}"); self.logs.push(message.clone()); self.notify(message); return; } };
        let working_dir = Self::project_root().or_else(|| std::env::current_dir().ok()).unwrap_or_else(|| PathBuf::from("."));
        let script_display = script.display().to_string(); let tx = self.tx.clone(); let child_slot = self.job.child.clone(); let output_dir = output_dir.to_owned();
        thread::spawn(move || {
            let _ = tx.send(format!("Preparing dataset: {script_display}")); let _ = tx.send(format!("Output directory: {output_dir}"));
            let mut command = Command::new("python3"); command.arg(&script);
            if script_name == "prepare_real_corpus.py" || script_name == "prepare_online_corpus.py" { command.arg("--output-dir").arg(&output_dir); }
            else if script_name == "prepare_curriculum.py" { command.arg("--root-dir").arg(&working_dir).arg("--output-dir").arg(&output_dir); }
            command.current_dir(&working_dir).stdout(Stdio::piped()).stderr(Stdio::piped());
            match command.spawn() {
                Ok(mut child) => {
                    let stdout = child.stdout.take(); let stderr = child.stderr.take(); if let Ok(mut slot) = child_slot.lock() { *slot = Some(child); }
                    if let Some(stdout) = stdout { let tx_stdout = tx.clone(); thread::spawn(move || { let reader = BufReader::new(stdout); for line in reader.lines().flatten() { let _ = tx_stdout.send(line); } }); }
                    if let Some(stderr) = stderr { let tx_stderr = tx.clone(); thread::spawn(move || { let reader = BufReader::new(stderr); for line in reader.lines().flatten() { let _ = tx_stderr.send(line); } }); }
                    loop {
                        let done = match child_slot.lock() { Ok(mut slot) => match slot.as_mut() { Some(child) => match child.try_wait() { Ok(Some(status)) => { let _ = tx.send(format!("Dataset preparation exited: {status}")); if status.success() { let _ = tx.send(format!("__PREP_DONE__:{output_dir}")); } true }, Ok(None) => false, Err(e) => { let _ = tx.send(format!("Preparation process error: {e}")); true } }, None => true }, Err(_) => true };
                        if done { break; } thread::sleep(Duration::from_millis(250));
                    }
                    if let Ok(mut slot) = child_slot.lock() { *slot = None; }
                }
                Err(e) => { let _ = tx.send(format!("Failed to launch python3: {e}")); }
            }
        });
        self.started = Some(Instant::now()); self.notify("Dataset preparation started");
    }
    fn start_job(&mut self) {
        if self.job.running() { return; }
        let bin = match self.backend { Backend::Cpu => "cpu-train", Backend::Gpu => "amd-train" };
        let trainer = match Self::find_trainer(bin) { Some(path) => path, None => { let message = format!("Trainer not found: {bin}"); self.logs.push(message.clone()); self.notify(message); return; } };
        let trainer_display = trainer.display().to_string(); let working_dir = Self::project_root().or_else(|| std::env::current_dir().ok()).unwrap_or_else(|| PathBuf::from("."));
        let mut command = Command::new(&trainer); command.arg("--target").arg("--steps").arg(self.steps.to_string()).arg("--data").arg(&self.data_path).arg("--checkpoint").arg(&self.checkpoint);
        match self.backend {
            Backend::Cpu => { command.arg("--grad-accum").arg(self.grad_accum.to_string()).arg("--targets-per-step").arg(self.targets.to_string()).arg("--train-context").arg(self.context.to_string()).arg("--lr").arg(self.lr.to_string()); if !self.val_path.is_empty() { command.arg("--val-data").arg(&self.val_path); } if !self.resume.is_empty() { command.arg("--resume").arg(&self.resume); } }
            Backend::Gpu => { command.arg("--batch-size").arg(self.batch.to_string()).arg("--grad-accum").arg(self.grad_accum.to_string()).arg("--lr").arg(self.lr.to_string()).arg("--checkpoint-every").arg(self.checkpoint_every.to_string()).arg("--eval-every").arg(self.eval_every.to_string()).arg("--gpu").arg(self.gpu.to_string()).arg("--gpu-kind").arg(&self.gpu_kind).arg("--gpu-util").arg(self.gpu_util.to_string()); if !self.resume.is_empty() { command.arg("--resume").arg(&self.resume); } }
        }
        command.current_dir(&working_dir).stdout(Stdio::piped()).stderr(Stdio::piped()); let tx = self.tx.clone(); let child_slot = self.job.child.clone(); let working_dir_display = working_dir.display().to_string();
        thread::spawn(move || { let _ = tx.send(format!("Launching {bin}: {trainer_display}")); let _ = tx.send(format!("Working directory: {working_dir_display}")); match command.spawn() { Ok(mut child) => { let stdout = child.stdout.take(); let stderr = child.stderr.take(); if let Ok(mut slot) = child_slot.lock() { *slot = Some(child); } if let Some(stdout) = stdout { let tx_stdout = tx.clone(); thread::spawn(move || { let reader = BufReader::new(stdout); for line in reader.lines().flatten() { let _ = tx_stdout.send(line); } }); } if let Some(stderr) = stderr { let tx_stderr = tx.clone(); thread::spawn(move || { let reader = BufReader::new(stderr); for line in reader.lines().flatten() { let _ = tx_stderr.send(line); } }); } loop { let done = match child_slot.lock() { Ok(mut slot) => match slot.as_mut() { Some(child) => match child.try_wait() { Ok(Some(status)) => { let _ = tx.send(format!("Process exited: {status}")); true }, Ok(None) => false, Err(e) => { let _ = tx.send(format!("Process error: {e}")); true } }, None => true }, Err(_) => true }; if done { if let Ok(mut slot) = child_slot.lock() { *slot = None; } break; } thread::sleep(Duration::from_millis(250)); } }, Err(e) => { let _ = tx.send(format!("Failed to launch {bin}: {e}")); } } });
        self.started = Some(Instant::now()); self.notify(format!("Started {bin}"));
    }
    fn nav(&mut self, ui: &mut egui::Ui) { let items = [(Page::Dashboard,"⌂","Dashboard"),(Page::Training,"▶","Training"),(Page::Inference,"✦","Inference"),(Page::Hardware,"▣","Hardware"),(Page::Data,"▤","Data"),(Page::Settings,"⚙","Settings"),(Page::Logs,"≡","Logs")]; for (page, icon, label) in items { let selected = self.page == page; let fill = if selected { self.accent.gamma_multiply(0.18) } else { Color32::TRANSPARENT }; let button = egui::Button::new(RichText::new(format!("{icon}  {label}")).size(15.0).color(if selected { self.accent } else { ui.visuals().text_color() })).fill(fill).corner_radius(CornerRadius::same(18)).min_size(Vec2::new(190.0, 44.0)); if ui.add(button).clicked() { self.page = page; } ui.add_space(4.0); } }
    fn topbar(&mut self, ui: &mut egui::Ui) { ui.horizontal(|ui| { ui.label(RichText::new("GemmaAgent").font(FontId::proportional(23.0)).strong()); ui.label(RichText::new("Material You").color(self.accent).size(13.0)); ui.with_layout(Layout::right_to_left(Align::Center), |ui| { let status = if self.job.running() { "● Running" } else { "● Ready" }; ui.colored_label(if self.job.running() { Color32::from_rgb(30,160,90) } else { self.accent }, status); ui.separator(); if ui.button(if self.dark { "☾" } else { "☀" }).clicked() { self.dark = !self.dark; configure_theme(ui.ctx(), self.accent, self.dark, self.compact); } }); }); }
    fn card<R>(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui) -> R) { Frame::new().fill(ui.visuals().faint_bg_color).stroke(Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)).corner_radius(CornerRadius::same(22)).inner_margin(Margin::same(18)).show(ui, |ui| { ui.label(RichText::new(title).size(17.0).strong()); ui.add_space(10.0); add(ui); }); }
    fn dashboard(&mut self, ui: &mut egui::Ui) { ui.heading("Training workspace"); ui.label("One control surface for CPU, AMD Vulkan, datasets, checkpoints and inference."); ui.add_space(14.0); ui.columns(3, |cols| { Self::card(&mut cols[0], "Compute", |ui| { ui.label(if self.backend==Backend::Cpu { "CPU · Rust autograd" } else { "GPU · AMD Vulkan / WGPU" }); ui.add_space(8.0); ui.horizontal(|ui| { if ui.selectable_label(self.backend==Backend::Cpu,"CPU").clicked(){self.backend=Backend::Cpu; self.notify("Backend switched to CPU");} if ui.selectable_label(self.backend==Backend::Gpu,"GPU").clicked(){self.backend=Backend::Gpu; self.notify("Backend switched to GPU");} }); }); Self::card(&mut cols[1], "Model", |ui| { ui.label(&self.model); ui.label(format!("Context {} · LR {:.5}", self.context, self.lr)); }); Self::card(&mut cols[2], "Process", |ui| { ui.label(if self.job.running(){"Training is active"}else{"No active job"}); if self.job.running(){ if ui.button("Stop training").clicked(){self.job.stop(); self.notify("Training stopped");} } else if ui.button("Start training").clicked(){self.start_job();} }); }); ui.add_space(14.0); ui.columns(2, |cols| { Self::card(&mut cols[0], "Training profile", |ui| { metric(ui,"Steps",&self.steps.to_string()); metric(ui,"Batch",&self.batch.to_string()); metric(ui,"Grad accumulation",&self.grad_accum.to_string()); metric(ui,"Precision",match self.precision {Precision::F32=>"F32",Precision::F16=>"F16",Precision::Mixed=>"Mixed"}); }); Self::card(&mut cols[1], "Performance", |ui| { let elapsed=self.started.map(|t|t.elapsed().as_secs()).unwrap_or(0); metric(ui,"Elapsed",&format!("{}m {:02}s",elapsed/60,elapsed%60)); metric(ui,"GPU target",&format!("{:.0}%",self.gpu_util)); metric(ui,"KV cache",if self.kv_cache{"Enabled"}else{"Disabled"}); metric(ui,"Flash attention",if self.flash_attention{"Enabled"}else{"Disabled"}); }); }); }
    fn training(&mut self, ui: &mut egui::Ui) { ui.heading("Training"); ui.add_space(12.0); Self::card(ui,"Compute backend",|ui|{ ui.horizontal(|ui|{ if ui.selectable_label(self.backend==Backend::Cpu,"CPU").clicked(){self.backend=Backend::Cpu;} if ui.selectable_label(self.backend==Backend::Gpu,"GPU · AMD Vulkan").clicked(){self.backend=Backend::Gpu;} }); ui.small("The selector changes the actual trainer binary used by Start training."); }); ui.add_space(12.0); ui.columns(2,|cols|{ Self::card(&mut cols[0],"Optimization",|ui|{ slider(ui,"Steps",&mut self.steps,100,100000); slider(ui,"Batch size",&mut self.batch,1,64); slider(ui,"Grad accumulation",&mut self.grad_accum,1,64); slider_f64(ui,"Learning rate",&mut self.lr,0.000001,0.01); slider(ui,"Checkpoint every",&mut self.checkpoint_every,0,10000); slider(ui,"Eval every",&mut self.eval_every,0,10000); }); Self::card(&mut cols[1],"Architecture / memory",|ui|{ slider(ui,"Context",&mut self.context,128,4096); slider(ui,"Targets per step",&mut self.targets,1,1024); ui.checkbox(&mut self.kv_cache,"KV cache"); ui.checkbox(&mut self.flash_attention,"Flash/tiled attention"); ui.checkbox(&mut self.pinned_memory,"Pinned/reused buffers"); ui.horizontal(|ui|{ui.label("Precision"); for (p,n) in [(Precision::F32,"F32"),(Precision::Mixed,"Mixed"),(Precision::F16,"F16")]{ui.selectable_value(&mut self.precision,p,n);}}); }); }); ui.add_space(12.0); Self::card(ui,"Paths",|ui|{ text(ui,"Training data",&mut self.data_path); text(ui,"Validation data",&mut self.val_path); text(ui,"Checkpoint",&mut self.checkpoint); text(ui,"Resume",&mut self.resume); }); ui.add_space(12.0); ui.horizontal(|ui|{if self.job.running(){if ui.button("Stop").clicked(){self.job.stop();}}else if ui.button(RichText::new("Start training").color(Color32::WHITE)).clicked(){self.start_job();} if ui.button("Open logs").clicked(){self.page=Page::Logs;}}); }
    fn inference(&mut self, ui: &mut egui::Ui) { ui.heading("Inference"); ui.add_space(12.0); Self::card(ui,"Generation",|ui|{ text(ui,"Model",&mut self.model); slider_f32(ui,"Temperature",&mut self.temperature,0.0,2.0); slider_f32(ui,"Top-p",&mut self.top_p,0.05,1.0); slider(ui,"Top-k",&mut self.top_k,0,500); ui.add(egui::DragValue::new(&mut self.seed).prefix("Seed ")); ui.checkbox(&mut self.kv_cache,"Use KV cache"); }); ui.add_space(12.0); Self::card(ui,"Runtime",|ui|{ui.label(format!("Backend: {}",if self.backend==Backend::Cpu{"CPU"}else{"AMD Vulkan GPU"})); ui.label(format!("Precision: {}",match self.precision{Precision::F32=>"F32",Precision::F16=>"F16",Precision::Mixed=>"Mixed"}));}); }
    fn hardware(&mut self, ui: &mut egui::Ui) { ui.heading("Hardware"); ui.add_space(12.0); Self::card(ui,"GPU runtime",|ui|{ui.horizontal(|ui|{ui.label("GPU kind"); egui::ComboBox::from_id_salt("gpu-kind").selected_text(&self.gpu_kind).show_ui(ui, |ui| { for k in ["integrated","discrete","best"] { ui.selectable_value(&mut self.gpu_kind,k.to_owned(),k); } });}); slider(ui,"GPU index",&mut self.gpu,0,8); slider_f64(ui,"GPU utilization target",&mut self.gpu_util,1.0,100.0); ui.checkbox(&mut self.power_limit,"Respect GPU utilization target"); ui.label("AMD trainer arguments are generated from these values."); }); ui.add_space(12.0); Self::card(ui,"CPU runtime",|ui|{slider(ui,"Worker threads",&mut self.cpu_threads,1,32); ui.label("CPU trainer remains available even when the UI itself is GPU-rendered.");}); }
    fn data(&mut self, ui: &mut egui::Ui) { ui.heading("Data & checkpoints"); ui.label("Prepare a corpus from the bundled scripts, then send the generated files directly to the trainer."); ui.add_space(12.0); Self::card(ui, "Dataset preparation", |ui| { ui.horizontal(|ui| { ui.label("Source"); egui::ComboBox::from_id_salt("prep-script").selected_text(&self.prep_script).show_ui(ui, |ui| { for source in ["Real corpus", "Online corpus", "Curriculum"] { ui.selectable_value(&mut self.prep_script, source.to_owned(), source); } }); }); ui.add_space(8.0); match self.prep_script.as_str() { "Real corpus" => ui.small("Public-domain books + Tiny Shakespeare → data/train.txt and data/val.txt"), "Online corpus" => ui.small("Wikimedia topics → data/online/train.txt and data/online/val.txt"), "Curriculum" => ui.small("Mix literature, code, math and reasoning → data/curriculum/*.txt"), _ => ui.small("Unknown preparation source"), }; ui.add_space(8.0); let disabled = self.job.running(); ui.add_enabled_ui(!disabled, |ui| { if ui.button("Prepare dataset").clicked() { self.start_prepare_job(); } }); if disabled { ui.small("A job is already running; wait for it to finish."); } }); ui.add_space(12.0); Self::card(ui, "Dataset used by training", |ui| { text(ui, "Train corpus", &mut self.data_path); text(ui, "Validation corpus", &mut self.val_path); ui.small("After preparation, these paths are updated automatically. You can still edit them manually."); }); ui.add_space(12.0); Self::card(ui, "Checkpoint policy", |ui| { text(ui, "Checkpoint path", &mut self.checkpoint); text(ui, "Resume from", &mut self.resume); ui.checkbox(&mut self.auto_save, "Auto-save final checkpoint"); }); }
    fn settings(&mut self, ui: &mut egui::Ui) { ui.heading("Settings"); ui.add_space(12.0); Self::card(ui,"Material You",|ui|{ui.horizontal(|ui|{ui.label("Accent"); let mut rgb=[self.accent.r(),self.accent.g(),self.accent.b()]; if ui.color_edit_button_srgb(&mut rgb).changed(){self.accent=Color32::from_rgb(rgb[0],rgb[1],rgb[2]); configure_theme(ui.ctx(),self.accent,self.dark,self.compact);}}); ui.checkbox(&mut self.dark,"Dark theme"); ui.checkbox(&mut self.compact,"Compact density");}); ui.add_space(12.0); Self::card(ui,"Application",|ui|{ui.checkbox(&mut self.notifications,"Desktop notifications"); ui.label("Settings are applied immediately to this session.");}); }
    fn logs(&mut self, ui: &mut egui::Ui) { ui.heading("Logs"); ui.add_space(10.0); Frame::new().fill(Color32::from_black_alpha(if self.dark{70}else{20})).corner_radius(CornerRadius::same(18)).inner_margin(Margin::same(12)).show(ui, |ui| { egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| { for line in &self.logs { ui.monospace(line); } }); }); }
}
impl eframe::App for GemmaUi { fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) { self.poll_logs(); let ctx=ui.ctx().clone(); if self.job.running(){ctx.request_repaint_after(Duration::from_millis(200));} let panel_fill=ui.visuals().panel_fill; egui::Panel::top("top").frame(Frame::new().fill(panel_fill).inner_margin(Margin::symmetric(20,14))).show(ui,|ui|self.topbar(ui)); let panel_fill=ui.visuals().panel_fill; egui::Panel::left("nav").resizable(false).exact_size(220.0).frame(Frame::new().fill(panel_fill).inner_margin(Margin::symmetric(14,20))).show(ui,|ui|{ui.label(RichText::new("CONTROL CENTER").size(11.0).color(self.accent));ui.add_space(14.0);self.nav(ui);ui.with_layout(Layout::bottom_up(Align::Center),|ui|{ui.separator();ui.small("GemmaAgent · Rust");});}); egui::CentralPanel::default().frame(Frame::new().inner_margin(Margin::symmetric(28,24))).show(ui,|ui|{egui::ScrollArea::vertical().auto_shrink([false,false]).show(ui,|ui|match self.page{Page::Dashboard=>self.dashboard(ui),Page::Training=>self.training(ui),Page::Inference=>self.inference(ui),Page::Hardware=>self.hardware(ui),Page::Data=>self.data(ui),Page::Settings=>self.settings(ui),Page::Logs=>self.logs(ui)});}); if let Some((msg,t))=&self.toast{if t.elapsed()<Duration::from_secs(3){egui::Area::new("toast".into()).anchor(egui::Align2::RIGHT_BOTTOM,[-24.0,-24.0]).show(&ctx,|ui|{Frame::new().fill(self.accent).corner_radius(CornerRadius::same(18)).inner_margin(Margin::symmetric(16,11)).show(ui,|ui|ui.label(RichText::new(msg).color(Color32::WHITE)));});}else{self.toast=None;}}} }
fn configure_theme(ctx:&egui::Context,accent:Color32,dark:bool,compact:bool){let theme=if dark{egui::Theme::Dark}else{egui::Theme::Light};let mut style=(*ctx.style_of(theme)).clone();style.spacing.item_spacing=if compact{Vec2::new(8.0,6.0)}else{Vec2::new(10.0,9.0)};style.visuals=if dark{egui::Visuals::dark()}else{egui::Visuals::light()};style.visuals.selection.bg_fill=accent;style.visuals.selection.stroke=Stroke::new(1.0,accent);style.visuals.hyperlink_color=accent;style.visuals.widgets.active.bg_fill=accent;style.visuals.widgets.hovered.bg_fill=accent.gamma_multiply(0.18);style.visuals.widgets.hovered.fg_stroke.color=accent;style.visuals.widgets.active.fg_stroke.color=Color32::WHITE;style.visuals.window_corner_radius=CornerRadius::same(24);ctx.set_style_of(theme,style);}
fn metric(ui:&mut egui::Ui,k:&str,v:&str){ui.horizontal(|ui|{ui.label(RichText::new(k).color(ui.visuals().weak_text_color()));ui.with_layout(Layout::right_to_left(Align::Center),|ui|ui.label(RichText::new(v).strong()));});}
fn slider(ui:&mut egui::Ui,label:&str,v:&mut usize,min:usize,max:usize){ui.horizontal(|ui|{ui.label(label);ui.add(egui::Slider::new(v,min..=max).show_value(true));});}
fn slider_f64(ui:&mut egui::Ui,label:&str,v:&mut f64,min:f64,max:f64){ui.horizontal(|ui|{ui.label(label);ui.add(egui::Slider::new(v,min..=max).logarithmic(true));});}
fn slider_f32(ui:&mut egui::Ui,label:&str,v:&mut f32,min:f32,max:f32){ui.horizontal(|ui|{ui.label(label);ui.add(egui::Slider::new(v,min..=max));});}
fn text(ui:&mut egui::Ui,label:&str,v:&mut String){ui.horizontal(|ui|{ui.label(label);ui.add(egui::TextEdit::singleline(v).desired_width(330.0));});}
pub fn run()->eframe::Result{let options=eframe::NativeOptions{viewport:egui::ViewportBuilder::default().with_inner_size([1440.0,900.0]).with_min_inner_size([1100.0,700.0]),..Default::default()};eframe::run_native("GemmaAgent",options,Box::new(|cc|Ok(Box::new(GemmaUi::new(cc)))))}
