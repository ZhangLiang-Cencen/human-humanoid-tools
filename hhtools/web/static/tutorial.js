/** Interactive first-run guide for the hhtools web UI. */

const STORAGE_KEY = "hhtools.web.tutorial.v1.done";

const STEPS = [
  {
    id: "welcome",
    panel: "motion",
    anchor: "#sidebar-body",
    title: "欢迎使用 hhtools Web",
    body:
      "本教程按推荐工作流介绍界面：<b>动作数据 → 机器人 → 标定 → Retarget → 3D 可视化 → 导出</b>。" +
      "点击「知道了」进入下一步；可随时「跳过教程」，之后可在左侧栏「操作教程」重新打开。",
    placement: "right",
  },
  {
    id: "motion",
    panel: "motion",
    anchor: "#tour-motion-import",
    title: "① 人体动作数据来源",
    body:
      "在右侧 <b>动作 Motion</b> 面板加载 clip：<br>" +
      "• <b>intermimic</b>：OMOMO 等人体+物体（<code>.pkl</code> + 物体 <code>.obj</code>）<br>" +
      "• <b>meshmimic</b>：跑酷地形（<code>.pkl/.npz</code> + <code>_terrain.obj</code>）<br>" +
      "• <b>mimic 通用</b>：AMASS / BVH / GLB / NPZ 等<br>" +
      "也可直接拖到中间舞台预览。下一步介绍资源库。",
    placement: "left",
  },
  {
    id: "motion-library",
    panel: "motion",
    anchor: "#tour-motion-library",
    title: "② 资源库直接打开",
    body:
      "下方 <b>资源库</b> 列出 <code>assets/motions/</code> 里已有的 clip，无需重复上传：<br>" +
      "• <b>目录下拉</b>：按 intermimic / meshmimic / mimic 等子目录筛选<br>" +
      "• <b>搜索框</b>：按名称过滤（如 kick、kungfu）<br>" +
      "• <b>点击一行</b>：加载到中间舞台；行末 <b>＋</b> 可加入批量篮子",
    placement: "left",
  },
  {
    id: "robot",
    panel: "robot",
    anchor: "#tour-robot-import",
    title: "③ 机器人来源",
    body:
      "切换到 <b>机器人 · Retarget</b>：拖入 <code>.urdf</code>，再拖入 <code>meshes/</code> 网格文件夹。" +
      "系统自动识别并生成配置，无需手调 <code>robot.yaml</code>。" +
      "也可在「已注册机器人」下拉框选择此前注册的机器人，点 <b>加载选中机器人</b>。" +
      "加载后中间舞台会显示机器人模型。",
    placement: "left",
  },
  {
    id: "calibration",
    panel: "robot",
    anchor: "#tour-calibration",
    title: "④ 标定 Calibration",
    body:
      "首次 Retarget 前需标定：把<b>灰色机器人</b>对齐到画面中的<b>蓝色参考骨架</b>（当前参考格式的标准姿态，不是动作播放帧）。" +
      "在右栏滑块或 3D 画面中点击关节并拖动旋转；<b>归零</b> 为 URDF 零位，<b>重置</b> 恢复上次保存的标定。" +
      "调整完成后点 <b>保存标定</b>。",
    placement: "left",
  },
  {
    id: "retarget",
    panel: "robot",
    anchor: "#tour-retarget",
    title: "⑤ Retarget 重映射",
    body:
      "动作与机器人就绪且已标定后，选择后端：<br>" +
      "• <b>Newton IK</b>：纯骨架（跳舞、行走等 mimic）<br>" +
      "• <b>Interaction-Mesh</b>：含交互物体或地形（OMOMO / meshmimic）<br>" +
      "可设置 Retarget FPS 加速求解；点击 <b>开始 Retarget</b>，完成后舞台会播放机器人动画。",
    placement: "left",
  },
  {
    id: "view",
    panel: "motion",
    anchor: "#view-hud",
    title: "⑥ 3D 可视化开关",
    body:
      "舞台左上角可切换显示层：<br>" +
      "<b>骨架 / 身体</b>：原始人体动作<br>" +
      "<b>物体/地形</b>：交互物体与跑酷地形<br>" +
      "<b>缩放骨架 / 缩放场景</b>：按机器人标定缩放后的预览<br>" +
      "<b>机器人</b>：Retarget 结果。可多层同时打开对比。",
    placement: "bottom",
    beforeShow: ({ revealViewHud }) => revealViewHud(true),
    afterLeave: ({ revealViewHud }) => revealViewHud(false),
  },
  {
    id: "export",
    panel: "robot",
    anchor: "#rt-export-card",
    title: "⑦ 导出保存",
    body:
      "Retarget 完成后在 <b>4 · 导出</b> 区域下载结果：默认 <b>CSV</b>（机器人 + 交互物体轨迹）；" +
      "含地形/物体时可打 ZIP 包（机器人 CSV/PKL、物体轨迹、缩放后的 OBJ）。" +
      "导出 FPS 仅对轨迹插值，不会重新求解。文件保存到浏览器默认下载目录。",
    placement: "left",
    beforeShow: ({ revealExportCard }) => revealExportCard(true),
    afterLeave: ({ revealExportCard }) => revealExportCard(false),
  },
  {
    id: "done",
    panel: "motion",
    anchor: "#nav-tour",
    title: "教程完成",
    body:
      "之后可随时点击左侧 <b>操作教程</b> 重新查看。" +
      "批量处理请用 <b>批量 Batch</b> 栏：把多个 clip 加入篮子后一次性 Retarget 并导出 ZIP。",
    placement: "right",
    last: true,
    beforeShow: ({ revealNavTour }) => revealNavTour(true),
  },
];

function switchPanel(panelId) {
  if (!panelId) return;
  const btn = document.querySelector(`.nav-item[data-panel="${panelId}"]`);
  const panel = document.querySelector(`#inspector .panel[data-panel="${panelId}"]`);
  if (!btn || !panel) return;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll("#inspector .panel").forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  panel.classList.add("active");
}

export class GuidedTour {
  /** @param {(msg: string, isErr?: boolean) => void} toastFn */
  constructor(toastFn) {
    this._toast = toastFn;
    this.idx = 0;
    this.active = false;
    this.root = document.getElementById("tour-root");
    this.highlight = document.getElementById("tour-highlight");
    this.popover = document.getElementById("tour-popover");
    this.titleEl = document.getElementById("tour-title");
    this.bodyEl = document.getElementById("tour-body");
    this.stepEl = document.getElementById("tour-step");
    this.nextBtn = document.getElementById("tour-next");
    this.skipBtn = document.getElementById("tour-skip");
    this.navBtn = document.getElementById("nav-tour");
    this._onResize = () => { if (this.active) this._positionCurrent(); };
    window.addEventListener("resize", this._onResize);
    this.skipBtn?.addEventListener("click", () => this.finish(true));
    this.nextBtn?.addEventListener("click", () => this.next());
    this.navBtn?.addEventListener("click", () => this.start(0));
  }

  isDone() {
    try {
      return localStorage.getItem(STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  }

  markDone() {
    try {
      localStorage.setItem(STORAGE_KEY, "1");
    } catch { /* private mode */ }
    this.navBtn?.classList.remove("hidden");
  }

  revealViewHud(on) {
    const hud = document.getElementById("view-hud");
    if (!hud) return;
    if (on) {
      hud.classList.remove("hidden");
      hud.dataset.tourForced = "1";
      return;
    }
    if (!hud.dataset.tourForced) return;
    const motionLoaded = window.__hh?.player?.active;
    if (!motionLoaded) hud.classList.add("hidden");
    delete hud.dataset.tourForced;
  }

  revealExportCard(on) {
    const card = document.getElementById("rt-export-card");
    if (!card) return;
    if (on) {
      if (!card.dataset.tourForced) {
        card.dataset.tourPrevDisplay = card.style.display || "none";
      }
      card.style.display = "block";
      card.dataset.tourForced = "1";
      return;
    }
    if (!card.dataset.tourForced) return;
    card.style.display = card.dataset.tourPrevDisplay || "none";
    delete card.dataset.tourForced;
    delete card.dataset.tourPrevDisplay;
  }

  revealNavTour(on) {
    const btn = this.navBtn;
    if (!btn) return;
    if (on) {
      btn.classList.remove("hidden");
      btn.dataset.tourPreview = "1";
    }
  }

  _stepCtx() {
    return {
      revealViewHud: (v) => this.revealViewHud(v),
      revealExportCard: (v) => this.revealExportCard(v),
      revealNavTour: (v) => this.revealNavTour(v),
    };
  }

  maybeAutoStart() {
    this.navBtn?.classList.toggle("hidden", !this.isDone());
    if (!this.isDone()) {
      requestAnimationFrame(() => {
        setTimeout(() => this.start(0), 400);
      });
    }
  }

  start(fromIdx = 0) {
    this.idx = fromIdx;
    this.active = true;
    this.root?.classList.add("active");
    document.body.classList.add("tour-active");
    const app = document.getElementById("app");
    if (app?.classList.contains("inspector-hidden")) {
      app.classList.remove("inspector-hidden");
      this._restoredInspector = true;
    }
    if (app?.classList.contains("sidebar-hidden")) {
      app.classList.remove("sidebar-hidden");
      this._restoredSidebar = true;
    }
    this._showStep();
  }

  finish(skipped = false) {
    this.active = false;
    const step = STEPS[this.idx];
    step?.afterLeave?.(this._stepCtx());
    this.root?.classList.remove("active");
    document.body.classList.remove("tour-active");
    this.highlight?.classList.remove("visible");
    this.popover?.classList.remove("visible");
    this.markDone();
    if (!skipped) this._toast?.("教程已完成，祝使用愉快！");
    else this._toast?.("已跳过教程，可随时从左侧栏重新打开");
  }

  next() {
    const step = STEPS[this.idx];
    step?.afterLeave?.(this._stepCtx());
    if (step?.last) {
      this.finish(false);
      return;
    }
    this.idx += 1;
    this._showStep();
  }

  _showStep() {
    const step = STEPS[this.idx];
    if (!step) {
      this.finish(false);
      return;
    }
    switchPanel(step.panel);
    step.beforeShow?.(this._stepCtx());
    requestAnimationFrame(() => {
      requestAnimationFrame(() => this._positionCurrent());
    });
    this.titleEl.textContent = step.title;
    this.bodyEl.innerHTML = step.body;
    this.stepEl.textContent = `${this.idx + 1} / ${STEPS.length}`;
    this.nextBtn.textContent = step.last ? "完成" : "知道了";
  }

  _positionCurrent() {
    const step = STEPS[this.idx];
    if (!step) return;
    const el = document.querySelector(step.anchor);
    if (!el) {
      this._centerPopover();
      return;
    }
    el.scrollIntoView({ block: "nearest", behavior: "auto" });
    const rect = el.getBoundingClientRect();
    const pad = 8;
    const h = this.highlight;
    h.style.left = `${Math.max(0, rect.left - pad)}px`;
    h.style.top = `${Math.max(0, rect.top - pad)}px`;
    h.style.width = `${rect.width + pad * 2}px`;
    h.style.height = `${rect.height + pad * 2}px`;
    h.classList.add("visible");

    const pop = this.popover;
    const margin = 14;
    const pw = pop.offsetWidth || 300;
    const ph = pop.offsetHeight || 160;
    let left = 0;
    let top = 0;
    const place = step.placement || "bottom";
    if (place === "left") {
      left = rect.left - pw - margin;
      top = rect.top + rect.height / 2 - ph / 2;
    } else if (place === "right") {
      left = rect.right + margin;
      top = rect.top + rect.height / 2 - ph / 2;
    } else if (place === "top") {
      left = rect.left + rect.width / 2 - pw / 2;
      top = rect.top - ph - margin;
    } else {
      left = rect.left + rect.width / 2 - pw / 2;
      top = rect.bottom + margin;
    }
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    left = Math.min(vw - pw - 12, Math.max(12, left));
    top = Math.min(vh - ph - 12, Math.max(64, top));
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.classList.add("visible");
  }

  _centerPopover() {
    this.highlight?.classList.remove("visible");
    const pop = this.popover;
    const pw = pop.offsetWidth || 300;
    const ph = pop.offsetHeight || 160;
    pop.style.left = `${(window.innerWidth - pw) / 2}px`;
    pop.style.top = `${(window.innerHeight - ph) / 2}px`;
    pop.classList.add("visible");
  }
}

/** @param {(msg: string, isErr?: boolean) => void} toastFn */
export function initTutorial(toastFn) {
  const tour = new GuidedTour(toastFn);
  window.__hhTour = tour;
  return tour;
}
