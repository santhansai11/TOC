/**
 * EmotionAI – script.js
 * Handles: API polling, emotion bars,
 * timeline graph, dominant card, face SVG overlay (with bbox),
 * session report, FPS counter, MediaPipe status badge
 */

"use strict";

// ══════════════════════════════════════════
//  CONSTANTS
// ══════════════════════════════════════════
const EMOTIONS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"];
const EMOTION_EMOJIS = {
  Angry: "😠", Disgust: "🤢", Fear: "😨",
  Happy: "😄", Neutral: "😐", Sad: "😢", Surprise: "😲"
};
const EMOTION_DESCRIPTIONS = {
  Angry:    "Furrowed brows and tightened lips detected → Anger response",
  Disgust:  "Nose wrinkle and raised upper lip detected → Disgust response",
  Fear:     "Widened eyes and raised brows detected → Fear response",
  Happy:    "Smile and cheek raise detected → Happiness response",
  Neutral:  "Relaxed facial muscles → Neutral expression",
  Sad:      "Dropped corners and inner brow raise → Sadness response",
  Surprise: "Raised brows and open mouth detected → Surprise response",
};
const EMOTION_COLORS = {
  Angry:    "#ff4040", Disgust: "#aa44ff", Fear:    "#ff9000",
  Happy:    "#00ffb0", Neutral: "#00cfff", Sad:     "#5588ff",
  Surprise: "#ffe000"
};
const POLL_MS    = 90;   // emotion API poll interval
const SESSION_POLL_MS = 3000; // session report refresh
const RING_CIRC  = 213.6; // 2πr where r=34

// ══════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════
let overlayOn       = true;
let timelineOn      = false;
let sessionOn       = false;
let selectedFaceId  = 0;
let lastTimestamp   = 0;
let fpsFrames       = [];
let currentFaces    = [];
let alertActive     = false;
let isDetecting     = true;   // tracks whether webcam capture is running

// Timeline data: per-emotion arrays of {t, v}
const tlData = {};
EMOTIONS.forEach(e => tlData[e] = []);
const TL_DURATION = 60;   // seconds shown

// ══════════════════════════════════════════
//  DOM REFS
// ══════════════════════════════════════════
const $ = id => document.getElementById(id);
const statusBadge    = $("status-badge");
const hdrFaceCount   = $("hdr-face-count");
const hdrFps         = $("hdr-fps");
const hdrSessionTime = $("hdr-session-time");
const overlayToggle  = $("overlay-toggle");
const timelineToggle = $("timeline-toggle");
const sessionToggle  = $("session-toggle");
const stopBtn        = $("stop-btn");
const sessionResetBtn= $("session-reset-btn");
const domEmoji       = $("dom-emoji");
const domLabel       = $("dom-label");
const domConfVal     = $("dom-conf-val");
const domDesc        = $("dom-desc");
const ringFill       = $("ring-fill");
const emotionBars    = $("emotion-bars");
const faceSelector   = $("face-selector");
const explanationTxt = $("explanation-text");
const timelineSec    = $("timeline-section");
const timelineLegend = $("timeline-legend");
const tlCanvas       = $("timeline-canvas");
const tlCtx          = tlCanvas.getContext("2d");
const sessionSec     = $("session-section");
const sessionElapsed = $("session-elapsed");
const sessDom        = $("sess-dom");
const sessionBars    = $("session-bars");
const modelToast     = $("model-toast");
const scanText       = $("scan-text");
const videoWrapper   = $("video-wrapper");
const faceSvg        = $("face-svg");
const latencyText    = $("latency-text");
const emotionAlert   = $("emotion-alert");
const pausedOverlay  = $("paused-overlay");
const scanLine       = $("scan-line");

// ══════════════════════════════════════════
//  INIT EMOTION BARS
// ══════════════════════════════════════════
function buildEmotionBars() {
  const barsContainer = $("emotion-bars");
  const title = barsContainer.querySelector(".bars-title");
  barsContainer.innerHTML = "";
  barsContainer.appendChild(title);

  EMOTIONS.forEach(em => {
    const color = EMOTION_COLORS[em];
    const row = document.createElement("div");
    row.className = "ebar";
    row.dataset.emotion = em;
    row.id = `ebar-${em}`;
    row.innerHTML = `
      <span class="ebar-label" id="elabel-${em}">${em}</span>
      <div class="ebar-track">
        <div class="ebar-fill" id="efill-${em}" style="width:0%;background:${color};box-shadow:0 0 6px ${color};"></div>
      </div>
      <span class="ebar-pct" id="epct-${em}">0%</span>
    `;
    barsContainer.appendChild(row);
  });
}
buildEmotionBars();

// Build timeline legend
function buildTimelineLegend() {
  timelineLegend.innerHTML = "";
  EMOTIONS.forEach(em => {
    const item = document.createElement("div");
    item.className = "tl-legend-item";
    item.innerHTML = `
      <div class="tl-legend-dot" style="background:${EMOTION_COLORS[em]};box-shadow:0 0 4px ${EMOTION_COLORS[em]};"></div>
      <span>${em}</span>
    `;
    timelineLegend.appendChild(item);
  });
}
buildTimelineLegend();

// Build session bars
function buildSessionBars() {
  sessionBars.innerHTML = "";
  EMOTIONS.forEach(em => {
    const color = EMOTION_COLORS[em];
    const row = document.createElement("div");
    row.className = "sbar";
    row.innerHTML = `
      <span class="sbar-label">${em}</span>
      <div class="sbar-track">
        <div class="sbar-fill" id="sfill-${em}" style="width:0%;background:${color};"></div>
      </div>
      <span class="sbar-pct" id="spct-${em}">0%</span>
    `;
    sessionBars.appendChild(row);
  });
}
buildSessionBars();

// ══════════════════════════════════════════
//  UPDATE EMOTION UI
// ══════════════════════════════════════════
function updateEmotionUI(face) {
  if (!face) {
    domLabel.textContent = "NO FACE";
    domLabel.style.color = "var(--text-dim)";
    domEmoji.textContent = "👤";
    domConfVal.textContent = "0";
    ringFill.style.strokeDashoffset = RING_CIRC;
    ringFill.style.stroke = "var(--text-dim)";
    explanationTxt.textContent = "No face detected. Please position yourself in frame.";
    scanText.textContent = "SEARCHING…";
    EMOTIONS.forEach(em => {
      $(`efill-${em}`).style.width = "0%";
      $(`epct-${em}`).textContent = "0%";
      $(`elabel-${em}`).classList.remove("dominant");
      $(`epct-${em}`).classList.remove("dominant");
    });
    return;
  }

  const dom  = face.dominant;
  const conf = face.confidence;
  const probs = face.probs;
  const col  = EMOTION_COLORS[dom] || "var(--neon-green)";

  // dominant card
  domEmoji.textContent   = EMOTION_EMOJIS[dom] || "😐";
  domEmoji.style.filter  = `drop-shadow(0 0 14px ${col}88)`;
  domLabel.textContent   = dom.toUpperCase();
  domLabel.style.color   = col;
  domLabel.style.textShadow = `0 0 18px ${col}88`;
  domConfVal.textContent = conf.toFixed(0);
  ringFill.style.stroke  = col;
  ringFill.style.strokeDashoffset = RING_CIRC - (conf / 100) * RING_CIRC;
  ringFill.style.filter  = `drop-shadow(0 0 4px ${col})`;
  domDesc.textContent    = EMOTION_DESCRIPTIONS[dom] || "";
  explanationTxt.textContent = `${EMOTION_DESCRIPTIONS[dom]}. Confidence: ${conf.toFixed(1)}%`;
  scanText.textContent   = "FACE DETECTED ✓";

  // Alert flash on very high confidence emotions (not neutral)
  if (conf > 80 && dom !== "Neutral" && !alertActive) {
    triggerEmotionAlert(col);
  }

  // bars
  EMOTIONS.forEach(em => {
    const pct  = (probs[em] || 0).toFixed(1);
    const fill  = $(`efill-${em}`);
    const label = $(`elabel-${em}`);
    const pctEl = $(`epct-${em}`);
    fill.style.width = pct + "%";
    pctEl.textContent = pct + "%";
    const isDom = em === dom;
    label.classList.toggle("dominant", isDom);
    pctEl.classList.toggle("dominant", isDom);
  });

  // timeline
  const now = Date.now() / 1000;
  EMOTIONS.forEach(em => {
    tlData[em].push({ t: now, v: probs[em] || 0 });
    while (tlData[em].length > 0 && now - tlData[em][0].t > TL_DURATION + 2) {
      tlData[em].shift();
    }
  });
  if (timelineOn) drawTimeline();
}

// ══════════════════════════════════════════
//  EMOTION ALERT FLASH
// ══════════════════════════════════════════
function triggerEmotionAlert(color) {
  alertActive = true;
  emotionAlert.classList.remove("hidden");
  emotionAlert.style.background = `radial-gradient(ellipse at center, ${color}22 0%, transparent 70%)`;
  emotionAlert.style.border = `2px solid ${color}44`;
  setTimeout(() => {
    emotionAlert.classList.add("hidden");
    alertActive = false;
  }, 500);
}

// ══════════════════════════════════════════
//  TIMELINE GRAPH
// ══════════════════════════════════════════
function drawTimeline() {
  const W = tlCanvas.width, H = tlCanvas.height;
  tlCtx.clearRect(0, 0, W, H);

  const now  = Date.now() / 1000;
  const tMin = now - TL_DURATION;

  // grid lines
  tlCtx.strokeStyle = "rgba(255,255,255,0.05)";
  tlCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * H;
    tlCtx.beginPath(); tlCtx.moveTo(0, y); tlCtx.lineTo(W, y); tlCtx.stroke();
  }
  // vertical time ticks every 10s
  for (let s = 0; s <= TL_DURATION; s += 10) {
    const x = (s / TL_DURATION) * W;
    tlCtx.strokeStyle = "rgba(255,255,255,0.04)";
    tlCtx.beginPath(); tlCtx.moveTo(x, 0); tlCtx.lineTo(x, H); tlCtx.stroke();
  }

  EMOTIONS.forEach(em => {
    const pts = tlData[em].filter(d => d.t >= tMin);
    if (pts.length < 2) return;
    tlCtx.beginPath();
    tlCtx.strokeStyle = EMOTION_COLORS[em];
    tlCtx.lineWidth = 1.5;
    tlCtx.shadowColor = EMOTION_COLORS[em];
    tlCtx.shadowBlur = 4;
    pts.forEach((d, i) => {
      const x = ((d.t - tMin) / TL_DURATION) * W;
      const y = H - (d.v / 100) * H;
      i === 0 ? tlCtx.moveTo(x, y) : tlCtx.lineTo(x, y);
    });
    tlCtx.stroke();
    tlCtx.shadowBlur = 0;
  });
}

// ══════════════════════════════════════════
//  FACE SVG OVERLAY (bounding boxes on top of video)
// ══════════════════════════════════════════
const SVG_NS = "http://www.w3.org/2000/svg";

function updateFaceSvg(faces) {
  faceSvg.innerHTML = "";
  if (!overlayOn) return;

  faces.forEach((face, idx) => {
    const [fx, fy, fw, fh] = face.bbox;
    const col  = EMOTION_COLORS[face.dominant] || "#00ffb0";
    const conf = face.confidence;
    const isSelected = idx === selectedFaceId;
    const opacity = isSelected ? 1 : 0.5;

    // ── bounding rect ──
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", fx);
    rect.setAttribute("y", fy);
    rect.setAttribute("width", fw);
    rect.setAttribute("height", fh);
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", col);
    rect.setAttribute("stroke-width", isSelected ? "1.5" : "1");
    rect.setAttribute("stroke-opacity", opacity);
    rect.setAttribute("stroke-dasharray", "4 2");
    faceSvg.appendChild(rect);

    // ── corner brackets ──
    const tl = 16;
    const corners = [
      // TL
      `M${fx},${fy + tl} L${fx},${fy} L${fx + tl},${fy}`,
      // TR
      `M${fx + fw - tl},${fy} L${fx + fw},${fy} L${fx + fw},${fy + tl}`,
      // BL
      `M${fx},${fy + fh - tl} L${fx},${fy + fh} L${fx + tl},${fy + fh}`,
      // BR
      `M${fx + fw - tl},${fy + fh} L${fx + fw},${fy + fh} L${fx + fw},${fy + fh - tl}`,
    ];
    corners.forEach(d => {
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", col);
      path.setAttribute("stroke-width", "2.5");
      path.setAttribute("stroke-opacity", opacity);
      path.setAttribute("stroke-linecap", "square");
      faceSvg.appendChild(path);
    });

    // ── label pill ──
    const label = `${face.dominant}  ${conf.toFixed(0)}%`;
    const CHAR_W = 6.5, PAD = 10;
    const lw = label.length * CHAR_W + PAD * 2;
    const lh = 18;
    const lx = fx;
    const ly = fy > lh + 4 ? fy - lh - 4 : fy + fh + 4;

    const pill = document.createElementNS(SVG_NS, "rect");
    pill.setAttribute("x", lx); pill.setAttribute("y", ly);
    pill.setAttribute("width", lw); pill.setAttribute("height", lh);
    pill.setAttribute("rx", "4"); pill.setAttribute("ry", "4");
    pill.setAttribute("fill", `${col}26`);
    pill.setAttribute("stroke", col);
    pill.setAttribute("stroke-width", "1");
    pill.setAttribute("fill-opacity", opacity);
    faceSvg.appendChild(pill);

    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", lx + PAD);
    text.setAttribute("y", ly + 13);
    text.setAttribute("fill", col);
    text.setAttribute("font-family", "JetBrains Mono, monospace");
    text.setAttribute("font-size", "10");
    text.setAttribute("font-weight", "500");
    text.setAttribute("opacity", opacity);
    text.textContent = label;
    faceSvg.appendChild(text);

    // ── face center dot ──
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", fx + fw / 2);
    dot.setAttribute("cy", fy + fh / 2);
    dot.setAttribute("r", "3");
    dot.setAttribute("fill", col);
    dot.setAttribute("opacity", "0.5");
    faceSvg.appendChild(dot);
    // cross-hair lines
    [
      `M${fx + fw/2 - 8},${fy + fh/2} L${fx + fw/2 + 8},${fy + fh/2}`,
      `M${fx + fw/2},${fy + fh/2 - 8} L${fx + fw/2},${fy + fh/2 + 8}`,
    ].forEach(d => {
      const line = document.createElementNS(SVG_NS, "path");
      line.setAttribute("d", d);
      line.setAttribute("stroke", col);
      line.setAttribute("stroke-width", "1");
      line.setAttribute("opacity", "0.4");
      faceSvg.appendChild(line);
    });
  });
}

function resizeCanvas() {
  // reserved for future responsive overlay coordinate transforms
}

// ══════════════════════════════════════════
//  FACE SELECTOR
// ══════════════════════════════════════════
function updateFaceSelector(faces) {
  hdrFaceCount.textContent = faces.length;
  if (faces.length <= 1) {
    faceSelector.textContent = faces.length === 1 ? "FACE #1" : "—";
    selectedFaceId = 0;
    return;
  }
  faceSelector.innerHTML = faces.map((f, i) =>
    `<button onclick="selectFace(${i})" style="
      background:${i===selectedFaceId?'rgba(0,255,176,0.15)':'transparent'};
      border:1px solid ${i===selectedFaceId?'#00ffb0':'rgba(255,255,255,0.15)'};
      color:${i===selectedFaceId?'#00ffb0':'#6a7a9a'};
      border-radius:100px;padding:2px 9px;cursor:pointer;font-family:var(--mono);font-size:0.58rem;margin:0 2px;">
      FACE ${i+1}</button>`
  ).join("");
}
window.selectFace = idx => {
  if (idx < 0 || idx >= currentFaces.length) return;
  selectedFaceId = idx;
  updateFaceSelector(currentFaces);
  updateFaceSvg(currentFaces);
};

// ══════════════════════════════════════════
//  FPS COUNTER
// ══════════════════════════════════════════
function recordFps() {
  const now = performance.now();
  fpsFrames.push(now);
  fpsFrames = fpsFrames.filter(t => now - t < 2000);
  hdrFps.textContent = Math.round(fpsFrames.length / 2);
}

// ══════════════════════════════════════════
//  STATUS CHECK
// ══════════════════════════════════════════
async function checkStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    if (d.model_ready) {
      statusBadge.textContent = "READY";
      statusBadge.classList.add("ready");
    } else {
      statusBadge.textContent = "NO MODEL";
      statusBadge.classList.add("error");
      modelToast.classList.remove("hidden");
    }
  } catch { /* server not yet ready */ }
}

// ══════════════════════════════════════════
//  STOP / START DETECTION
// ══════════════════════════════════════════
function setDetectionState(running) {
  isDetecting = running;

  if (running) {
    stopBtn.textContent = "⏹ STOP";
    stopBtn.classList.remove("stopped");
    stopBtn.title = "Stop face detection";
    pausedOverlay.classList.remove("visible");
    scanLine.style.display = "";
    statusBadge.textContent = "READY";
    statusBadge.classList.remove("error");
    statusBadge.classList.add("ready");
    pollEmotions();
  } else {
    stopBtn.textContent = "▶ RESUME";
    stopBtn.classList.add("stopped");
    stopBtn.title = "Resume face detection";
    pausedOverlay.classList.add("visible");
    scanLine.style.display = "none";
    statusBadge.textContent = "PAUSED";
    statusBadge.classList.remove("ready");
    statusBadge.classList.add("error");
    // Reset displayed values
    hdrFaceCount.textContent = "0";
    hdrFps.textContent = "—";
    faceSvg.innerHTML = "";
    updateEmotionUI(null);
  }
}

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  try {
    if (isDetecting) {
      await fetch("/api/capture/stop", { method: "POST" });
      setDetectionState(false);
    } else {
      await fetch("/api/capture/start", { method: "POST" });
      setDetectionState(true);
    }
  } catch (e) {
    console.warn("Stop/start error:", e);
  } finally {
    stopBtn.disabled = false;
  }
});

// ══════════════════════════════════════════
//  EMOTION API POLL
// ══════════════════════════════════════════
async function pollEmotions() {
  if (!isDetecting) return;   // don't poll when paused
  const t0 = performance.now();
  try {
    const r = await fetch("/api/emotions");
    const d = await r.json();
    const latMS = Math.round(performance.now() - t0);
    latencyText.textContent = `LATENCY: ${latMS} ms`;

    currentFaces = d.faces || [];
    if (selectedFaceId >= currentFaces.length) selectedFaceId = 0;
    updateFaceSelector(currentFaces);
    updateFaceSvg(currentFaces);

    const face = currentFaces[selectedFaceId] || currentFaces[0] || null;
    updateEmotionUI(face);
    recordFps();
  } catch (e) {
    console.warn("Poll error:", e);
  }
  if (isDetecting) setTimeout(pollEmotions, POLL_MS);
}

// ══════════════════════════════════════════
//  SESSION REPORT POLL
// ══════════════════════════════════════════
let _sessionStartLocal = Date.now();

async function pollSession() {
  if (!sessionOn) { setTimeout(pollSession, SESSION_POLL_MS); return; }
  try {
    const r = await fetch("/api/session");
    const d = await r.json();

    // elapsed timer
    const elapsed = d.elapsed_s;
    const mm = Math.floor(elapsed / 60);
    const ss = elapsed % 60;
    const elStr = mm > 0 ? `${mm}m ${ss}s` : `${ss}s`;
    sessionElapsed.textContent = elStr;
    hdrSessionTime.textContent = elStr;

    // session dominant
    const dom = d.dominant || "—";
    sessDom.textContent = dom;
    sessDom.style.color = EMOTION_COLORS[dom] || "var(--neon-green)";

    // session bars (percentages)
    EMOTIONS.forEach(em => {
      const pct = d.percentages[em] || 0;
      const fill = $(`sfill-${em}`);
      const pctEl = $(`spct-${em}`);
      if (fill) fill.style.width  = pct + "%";
      if (pctEl) pctEl.textContent = pct.toFixed(1) + "%";
    });
  } catch (e) {
    console.warn("Session poll error:", e);
  }
  setTimeout(pollSession, SESSION_POLL_MS);
}

// ══════════════════════════════════════════
//  TOGGLE HANDLERS
// ══════════════════════════════════════════
overlayToggle.addEventListener("click", () => {
  overlayOn = !overlayOn;
  overlayToggle.classList.toggle("active", overlayOn);
});
overlayToggle.classList.add("active");

timelineToggle.addEventListener("click", () => {
  timelineOn = !timelineOn;
  timelineToggle.classList.toggle("active", timelineOn);
  timelineSec.classList.toggle("hidden", !timelineOn);
});

sessionToggle.addEventListener("click", () => {
  sessionOn = !sessionOn;
  sessionToggle.classList.toggle("active", sessionOn);
  sessionSec.classList.toggle("hidden", !sessionOn);
  if (sessionOn) pollSession();
});

sessionResetBtn.addEventListener("click", async () => {
  try {
    await fetch("/api/session/reset", { method: "POST" });
    _sessionStartLocal = Date.now();
    EMOTIONS.forEach(em => {
      const f = $(`sfill-${em}`); if (f) f.style.width = "0%";
      const p = $(`spct-${em}`);  if (p) p.textContent = "0%";
    });
    sessDom.textContent = "—";
    sessionElapsed.textContent = "0s";
    hdrSessionTime.textContent = "0s";
    console.log("[INFO] Session reset.");
  } catch(e) { console.warn("Reset error:", e); }
});

$("toast-close").addEventListener("click", () => modelToast.classList.add("hidden"));

// ══════════════════════════════════════════
//  BOOT
// ══════════════════════════════════════════
window.addEventListener("load", async () => {
  resizeCanvas();
  await checkStatus();
  // Sync stop button state with backend
  try {
    const r = await fetch("/api/capture/status");
    const d = await r.json();
    isDetecting = d.running;
    setDetectionState(isDetecting);
  } catch {
    // Assume running if endpoint unreachable
    setDetectionState(true);
  }
  // start session polling silently in background
  setTimeout(pollSession, SESSION_POLL_MS);
});

window.addEventListener("resize", resizeCanvas);
