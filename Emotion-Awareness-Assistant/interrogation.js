// ─── DOM ────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const video        = $('video');
const canvas       = $('canvas');
const ctx          = canvas.getContext('2d');
const loadingEl    = $('loading-overlay');
const progressEl   = $('loader-progress');
const loadStatus   = $('loading-status');
const btnStart     = $('btn-start');
const btnStop      = $('btn-stop');
const btnSnap      = $('btn-snap');
const idleState    = $('idle-state');
const videoFrame   = $('video-frame');
const statusDot    = document.querySelector('.dot');
const statusText   = $('status-text');
const alertCard    = $('alert-card');
const alertIconWrap= $('alert-icon-wrap');
const alertIcon    = $('alert-icon');
const alertTitle   = $('alert-title');
const alertSubtitle= $('alert-subtitle');
const threatFill   = $('threat-fill');
const threatMarker = $('threat-marker');
const threatLabel  = $('threat-label');
const threatPct    = $('threat-pct');
const observationEl= $('observation-text');
const timelineList = $('timeline-list');
const flaggedGrid  = $('flagged-grid');
const alertFlash   = $('alert-flash');
const sirenBars    = $('siren-bars');
const camTimestamp = $('cam-timestamp');
const statDuration = $('stat-duration');
const statAlerts   = $('stat-alerts');
const statDominant = $('stat-dominant');
const statScore    = $('stat-score');

// ─── Config ──────────────────────────────────────────────────────────────────
// Suspicion weights per emotion (higher = more concerning in interrogation context)
const SUSPICION_WEIGHTS = {
    angry:     0.9,
    fearful:   0.85,
    disgusted: 0.8,
    surprised: 0.5,
    sad:       0.4,
    neutral:   0.05,
    happy:     0.0
};

const OBSERVATIONS = {
    low: [
        "Subject appears cooperative. Emotional responses within normal parameters.",
        "No significant behavioral anomalies detected. Calm demeanor observed.",
        "Emotional baseline stable. Subject's responses appear voluntary and natural."
    ],
    medium: [
        "⚠️ Elevated emotional response detected. Subject may be experiencing anxiety.",
        "⚠️ Unusual emotional pattern emerging. Heightened vigilance recommended.",
        "⚠️ Micro-expressions indicate potential discomfort. Consider follow-up questioning."
    ],
    high: [
        "🚨 HIGH ALERT: Significant stress markers detected. Subject may be concealing information.",
        "🚨 SUSPICIOUS BEHAVIOR: Elevated fear and anxiety indicators. Intensify interrogation protocol.",
        "🚨 ANOMALY DETECTED: Emotional signature deviates significantly from baseline. Review footage."
    ],
    critical: [
        "🔴 CRITICAL: Extreme emotional distress detected. Possible deception or knowledge of incident.",
        "🔴 IMMEDIATE ACTION REQUIRED: Subject's expressions show severe duress. Evidence of concealment.",
        "🔴 RED ALERT: Composite suspicion score exceeds safe threshold. Escalate protocol immediately."
    ]
};

const EMOJIS_MAP = { angry:'😠', fearful:'😨', disgusted:'🤢', surprised:'😲', sad:'😔', neutral:'😐', happy:'😊' };
const COLORS_MAP = {
    angry:'#f43f5e', fearful:'#a855f7', disgusted:'#ec4899',
    surprised:'#f59e0b', sad:'#38bdf8', neutral:'#6b7fa3', happy:'#10b981'
};

// ─── State ────────────────────────────────────────────────────────────────────
let stream         = null;
let detectionTimer = null;
let clockTimer     = null;
let timestampTimer = null;
let modelsLoaded   = false;
let usingFaceApi   = false;
let sessionStart   = null;
let currentThreat  = 0;       // 0-100
let alertCount     = 0;
let threatHistory  = [];      // rolling window for smoothing
let lastAlertLevel = 'low';
let sirenOn        = false;
let suspicionScore = 0;       // cumulative

// ─── Model Loading ────────────────────────────────────────────────────────────
const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';
const FACE_DETECTOR_OPTIONS = new faceapi.TinyFaceDetectorOptions({
    inputSize: 416,
    scoreThreshold: 0.2
});

async function loadModels() {
    try {
        setProgress(10, 'Loading face detector...');
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        setProgress(60, 'Loading expression recognition...');
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        setProgress(100, 'Systems online.');
        modelsLoaded  = true;
        usingFaceApi  = true;
    } catch (e) {
        console.warn('[face-api] Simulation mode.', e);
        modelsLoaded  = true;
        usingFaceApi  = false;
        setProgress(100, 'Ready (simulation mode)');
    }
    setTimeout(() => loadingEl.classList.add('fade-out'), 700);
}

function setProgress(pct, msg) {
    progressEl.style.width = pct + '%';
    if (msg) loadStatus.textContent = msg;
}

// ─── Camera ──────────────────────────────────────────────────────────────────
async function startCamera() {
    if (!modelsLoaded) { alert('Please wait for models to load.'); return; }
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
        video.srcObject   = stream;
        video.style.display = 'block';
        idleState.style.display = 'none';
        videoFrame.classList.add('active');
        btnStart.classList.add('hidden');
        btnStop.classList.remove('hidden');
        btnSnap.disabled = false;
        setStatus(true);
        sessionStart = Date.now();
        startClock();
        video.addEventListener('loadeddata', () => {
            canvas.width  = video.videoWidth;
            canvas.height = video.videoHeight;
            startDetection();
            startTimestampUpdate();
        }, { once: true });
    } catch (err) {
        alert('Camera access denied. Please grant permission.');
        console.error(err);
    }
}

function stopCamera() {
    clearTimeout(detectionTimer);
    clearInterval(clockTimer);
    clearInterval(timestampTimer);
    stream?.getTracks().forEach(t => t.stop());
    stream = null;
    video.style.display = 'none';
    idleState.style.display = '';
    videoFrame.classList.remove('active');
    btnStart.classList.remove('hidden');
    btnStop.classList.add('hidden');
    btnSnap.disabled = true;
    setStatus(false);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stopSiren();
    document.body.classList.remove('danger-active');
}

// ─── Detection Loop ───────────────────────────────────────────────────────────
function startDetection() {
    async function detect() {
        if (!stream) return;
        try {
            let scores;
            if (usingFaceApi) {
                const result = await faceapi
                    .detectSingleFace(video, FACE_DETECTOR_OPTIONS)
                    .withFaceExpressions();
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (result) {
                    drawBox(result.detection.box);
                    scores = buildScores(result.expressions);
                } else {
                    scores = null;
                }
            } else {
                scores = simulateScores();
            }

            if (scores) {
                const threat = computeThreat(scores);
                updateBars(scores);
                updateThreatMeter(threat);
                updateAlertCard(threat);
                addSuspicionScore(threat);
                updateStatDominant(scores);
            }
        } catch (e) { console.error(e); }
        detectionTimer = setTimeout(detect, 600);
    }
    detect();
}

// ─── Score building ────────────────────────────────────────────────────────────
function buildScores(expressions) {
    const s = {};
    for (const k of Object.keys(SUSPICION_WEIGHTS)) {
        s[k] = Math.round((expressions[k] || 0) * 100);
    }
    return s;
}

function computeThreat(scores) {
    let raw = 0;
    for (const [emotion, weight] of Object.entries(SUSPICION_WEIGHTS)) {
        raw += (scores[emotion] || 0) * weight;
    }
    const threat = Math.min(100, Math.round(raw));

    // Rolling average (last 6 readings) for smooth meter
    threatHistory.push(threat);
    if (threatHistory.length > 6) threatHistory.shift();
    const smooth = Math.round(threatHistory.reduce((a,b) => a+b, 0) / threatHistory.length);
    return smooth;
}

function simulateScores() {
    const shift = Math.random();
    let scores;
    if (shift > 0.9) {
        // simulate high fear/anger spike
        scores = { angry: 35 + Math.random()*30, fearful: 25 + Math.random()*25, disgusted: 10, surprised: 10, sad: 10, neutral: 5, happy: 5 };
    } else if (shift > 0.7) {
        scores = { angry: 15, fearful: 10, disgusted: 8, surprised: 20 + Math.random()*20, sad: 12, neutral: 25, happy: 10 };
    } else {
        scores = { angry: 5, fearful: 5, disgusted: 3, surprised: 7, sad: 8, neutral: 45 + Math.random()*20, happy: 17 };
    }
    // Normalize
    const total = Object.values(scores).reduce((a,b) => a+b, 0);
    for (const k of Object.keys(scores)) scores[k] = Math.round(scores[k] / total * 100);
    return scores;
}

// ─── Bar Updates ──────────────────────────────────────────────────────────────
function updateBars(scores) {
    const suspicionKeys = ['angry', 'fearful', 'disgusted', 'surprised', 'sad'];
    let composite = 0;

    suspicionKeys.forEach(key => {
        const fill = $('fill-' + key);
        const pct  = $('pct-'  + key);
        const val  = scores[key] || 0;
        if (fill) { fill.style.width = val + '%'; fill.classList.add('active-bar'); }
        if (pct)  pct.textContent = val + '%';
        composite += val * SUSPICION_WEIGHTS[key];
    });

    const compRounded = Math.min(100, Math.round(composite));
    const fillComp = $('fill-composite');
    const pctComp  = $('pct-composite');
    if (fillComp) fillComp.style.width = compRounded + '%';
    if (pctComp)  pctComp.textContent  = compRounded + '%';
}

// ─── Threat Meter ─────────────────────────────────────────────────────────────
function updateThreatMeter(threat) {
    currentThreat = threat;
    threatFill.style.width   = threat + '%';
    threatMarker.style.left  = threat + '%';
    threatPct.textContent    = threat + '%';

    let level, color;
    if (threat < 25) {
        level = 'LOW'; color = '#22c55e';
    } else if (threat < 50) {
        level = 'MEDIUM'; color = '#eab308';
    } else if (threat < 75) {
        level = 'HIGH'; color = '#f97316';
    } else {
        level = 'CRITICAL'; color = '#ef4444';
    }

    threatLabel.textContent = level;
    threatLabel.style.color  = color;
    threatPct.style.color    = color;

    if (threat >= 75) {
        threatLabel.classList.add('critical-text');
    } else {
        threatLabel.classList.remove('critical-text');
    }
}

// ─── Alert Card Update ───────────────────────────────────────────────────────
function updateAlertCard(threat) {
    let newLevel;
    if (threat < 25)      newLevel = 'low';
    else if (threat < 50) newLevel = 'medium';
    else if (threat < 75) newLevel = 'high';
    else                  newLevel = 'critical';

    if (newLevel !== lastAlertLevel) {
        lastAlertLevel = newLevel;
        applyAlertLevel(newLevel, threat);
        logTimeline(newLevel, threat);
        if (newLevel === 'critical') {
            triggerRedAlert();
        } else {
            stopSiren();
            document.body.classList.remove('danger-active');
        }
    }

    // Always update observation text
    const obs = OBSERVATIONS[newLevel];
    observationEl.textContent = obs[Math.floor(Math.random() * obs.length)];
}

function applyAlertLevel(level, threat) {
    alertCard.className = 'card alert-card glass-red animate-up delay-1';
    alertIconWrap.className = 'alert-icon-wrap';

    if (level === 'low') {
        alertIconWrap.innerHTML = `<i class='bx bx-check-shield'></i>`;
        alertIconWrap.style.cssText = 'background:rgba(34,197,94,0.15);border-color:rgba(34,197,94,0.3);';
        alertTitle.textContent = 'System Clear';
        alertTitle.style.color = '#86efac';
        alertSubtitle.textContent = 'No suspicious behavior detected. Subject appears calm.';
    } else if (level === 'medium') {
        alertIconWrap.innerHTML = `<i class='bx bx-error'></i>`;
        alertIconWrap.style.cssText = '';
        alertIconWrap.classList.add('warning-state');
        alertCard.classList.add('warning-card');
        alertTitle.textContent = '⚠️ Caution – Elevated Response';
        alertTitle.style.color = '#fde68a';
        alertSubtitle.textContent = 'Elevated emotional response detected. Monitor closely.';
    } else if (level === 'high') {
        alertIconWrap.innerHTML = `<i class='bx bx-shield-x'></i>`;
        alertIconWrap.style.cssText = '';
        alertIconWrap.classList.add('warning-state');
        alertCard.classList.add('warning-card');
        alertTitle.textContent = '🚨 High Alert – Suspicious Behavior';
        alertTitle.style.color = '#fdba74';
        alertSubtitle.textContent = 'Significant stress markers detected. Heighten vigilance.';
    } else {
        alertIconWrap.innerHTML = `<i class='bx bx-error-circle'></i>`;
        alertIconWrap.style.cssText = '';
        alertIconWrap.classList.add('danger-state');
        alertCard.classList.add('danger-card');
        alertTitle.textContent = '🔴 CRITICAL ALERT — RED FLAG';
        alertTitle.style.color = '#fca5a5';
        alertSubtitle.textContent = 'EXTREME suspicious behavior detected! Immediate action required.';
    }
}

function triggerRedAlert() {
    alertCount++;
    $('stat-alerts').textContent = alertCount;
    document.body.classList.add('danger-active');
    startSiren();
    // Flash
    alertFlash.classList.remove('hidden');
    void alertFlash.offsetWidth;
    alertFlash.style.animation = 'none';
    void alertFlash.offsetWidth;
    alertFlash.style.animation = '';
    setTimeout(() => alertFlash.classList.add('hidden'), 700);
    // Auto-snap flag
    flagSnapshot('Auto-flagged (Critical)');
}

// ─── Siren ─────────────────────────────────────────────────────────────────────
function startSiren() {
    if (sirenOn) return;
    sirenOn = true;
    sirenBars.classList.remove('hidden');
}

function stopSiren() {
    sirenOn = false;
    sirenBars.classList.add('hidden');
}

// ─── Timeline ────────────────────────────────────────────────────────────────
function logTimeline(level, threat) {
    if (timelineList.querySelector('.timeline-empty')) timelineList.innerHTML = '';
    const el = document.createElement('div');
    el.className = `timeline-item level-${level}`;
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });
    const labels = { low:'Low', medium:'Medium', high:'High', critical:'⚠ Critical' };
    const badges = { low:'badge-low', medium:'badge-medium', high:'badge-high', critical:'badge-critical' };
    const events = {
        low:      'Emotional state normalized',
        medium:   'Elevated emotional response detected',
        high:     'Suspicious behavioral pattern flagged',
        critical: 'CRITICAL: Red alert triggered'
    };
    el.innerHTML = `
        <span class="timeline-time">${time}</span>
        <span class="timeline-event">${events[level]}</span>
        <span class="timeline-badge ${badges[level]}">${labels[level]}</span>`;
    timelineList.insertBefore(el, timelineList.firstChild);

    // Keep only last 20 events
    while (timelineList.children.length > 20) {
        timelineList.removeChild(timelineList.lastChild);
    }
}

// ─── Snapshots ───────────────────────────────────────────────────────────────
function flagSnapshot(label) {
    if (!stream) return;
    const off = document.createElement('canvas');
    off.width = video.videoWidth; off.height = video.videoHeight;
    const oc = off.getContext('2d');
    oc.drawImage(video, 0, 0);
    oc.drawImage(canvas, 0, 0);
    const dataUrl = off.toDataURL('image/png');

    if (flaggedGrid.querySelector('.empty-msg')) flaggedGrid.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'flagged-snap';
    const img = document.createElement('img');
    img.src = dataUrl;
    const badge = document.createElement('div');
    badge.className = 'snap-badge';
    badge.textContent = label || new Date().toLocaleTimeString();
    card.appendChild(img);
    card.appendChild(badge);
    card.title = 'Click to open';
    card.addEventListener('click', () => window.open(dataUrl));
    flaggedGrid.appendChild(card);
}

// ─── Session Stats ────────────────────────────────────────────────────────────
function addSuspicionScore(threat) {
    suspicionScore = Math.min(9999, suspicionScore + (threat > 50 ? Math.round(threat * 0.1) : 0));
    $('stat-score').textContent = suspicionScore;
}

function updateStatDominant(scores) {
    let best = { name: 'neutral', val: 0 };
    for (const [k, v] of Object.entries(scores)) {
        if (v > best.val) best = { name: k, val: v };
    }
    $('stat-dominant').textContent = EMOJIS_MAP[best.name] + ' ' + capitalize(best.name);
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function startClock() {
    clockTimer = setInterval(() => {
        if (!sessionStart) return;
        const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
        const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const s = String(elapsed % 60).padStart(2, '0');
        statDuration.textContent = `${m}:${s}`;
    }, 1000);
}

function startTimestampUpdate() {
    clearInterval(timestampTimer);
    timestampTimer = setInterval(() => {
        const now = new Date();
        camTimestamp.innerHTML = `${now.toLocaleTimeString('en-US', { hour12:false })} &nbsp; REC ●`;
    }, 1000);
}

// ─── Face Box ─────────────────────────────────────────────────────────────────
function drawBox(box) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const cw = canvas.width, ch = canvas.height;
    const sx = (box.x / vw) * cw,    sy = (box.y / vh) * ch;
    const sw = (box.width / vw) * cw, sh = (box.height / vh) * ch;
    const cs = 20;
    const col = currentThreat > 75 ? '#ef4444' : currentThreat > 50 ? '#f97316' : currentThreat > 25 ? '#eab308' : '#22c55e';

    ctx.save();
    ctx.shadowColor = col; ctx.shadowBlur = 20;
    ctx.strokeStyle = col + 'bb'; ctx.lineWidth = 1.5;
    ctx.strokeRect(sx, sy, sw, sh);
    ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.shadowBlur = 28;
    ctx.beginPath(); ctx.moveTo(sx, sy+cs); ctx.lineTo(sx,sy); ctx.lineTo(sx+cs,sy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx+sw-cs,sy); ctx.lineTo(sx+sw,sy); ctx.lineTo(sx+sw,sy+cs); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx,sy+sh-cs); ctx.lineTo(sx,sy+sh); ctx.lineTo(sx+cs,sy+sh); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx+sw-cs,sy+sh); ctx.lineTo(sx+sw,sy+sh); ctx.lineTo(sx+sw,sy+sh-cs); ctx.stroke();
    ctx.restore();
}

// ─── Utilities ─────────────────────────────────────────────────────────────────
function setStatus(active) {
    if (active) {
        statusDot.className = 'dot dot-active';
        statusText.textContent = 'Monitoring';
    } else {
        statusDot.className = 'dot dot-offline';
        statusText.textContent = 'Standby';
    }
}

function capitalize(str) { return str ? str.charAt(0).toUpperCase() + str.slice(1) : str; }

// ─── Button Ripple ─────────────────────────────────────────────────────────────
function attachRipple(btn) {
    btn.addEventListener('click', function(e) {
        const rect = btn.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const r = document.createElement('span');
        r.className = 'ripple-effect';
        r.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - rect.left - size/2}px;top:${e.clientY - rect.top - size/2}px;`;
        btn.appendChild(r);
        setTimeout(() => r.remove(), 600);
    });
}

// ─── Event Listeners ─────────────────────────────────────────────────────────
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click',  stopCamera);
btnSnap.addEventListener('click',  () => flagSnapshot('Manual flag – ' + new Date().toLocaleTimeString()));
[btnStart, btnStop, btnSnap].forEach(attachRipple);

// ─── Boot ─────────────────────────────────────────────────────────────────────
loadModels();
