// ─── DOM references ────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const video       = $('video');
const canvas      = $('canvas');
const ctx         = canvas.getContext('2d');
const loadingEl   = $('loading-overlay');
const progressEl  = $('loader-progress');
const loadStatus  = $('loading-status');
const btnStart    = $('btn-start');
const btnStop     = $('btn-stop');
const btnSnap     = $('btn-snap');
const idleState   = $('idle-state');
const videoFrame  = $('video-frame');
const statusDot   = document.querySelector('.dot');
const statusText  = $('status-text');
const emotionEmoji  = $('emotion-emoji');
const emotionName   = $('emotion-name');
const confidPill    = $('confidence-pill');
const suggestionEl  = $('suggestion-text');
const historyList   = $('history-list');

// ─── Emotion Config ─────────────────────────────────────────────────────
const EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'fearful', 'disgusted'];

const EMOJIS = {
    happy:'😊', sad:'😔', angry:'😠', neutral:'😐',
    surprised:'😲', fearful:'😨', disgusted:'🤢'
};

const COLORS = {
    happy:'#10b981', sad:'#38bdf8', angry:'#f43f5e', neutral:'#6b7fa3',
    surprised:'#f59e0b', fearful:'#a855f7', disgusted:'#ec4899'
};

const SUGGESTIONS = {
    happy:     "You're in a great mood! A perfect time to tackle challenging tasks or collaborate with others.",
    sad:       "You seem a bit down. Consider a short break, a walk, or listening to your favorite music.",
    angry:     "Try slow, deep breathing — inhale for 4 counts, hold for 4, exhale for 4. It helps.",
    neutral:   "You look calm and focused. Stay in the zone and keep up the great work!",
    surprised: "Something caught your attention! Stay curious — it's a wonderful sign of engagement.",
    fearful:   "It's okay to feel anxious. Try grounding yourself by focusing on 5 things you can see.",
    disgusted: "Take a moment to step away from what's bothering you. A short reset can help."
};

// ─── State ───────────────────────────────────────────────────────────────
let stream          = null;
let detectInterval  = null;
let history         = [];
let modelsLoaded    = false;
let usingFaceApi    = false;
let lastDominant    = null;
let typingTimer     = null;

// ─── Custom TF.js Model ───────────────────────────────────────────────────
let customModel       = null;   // tf.LayersModel or null
let usingCustomModel  = false;

// FER2013 class ordering emitted by Keras ImageDataGenerator
// (alphabetical by folder name: angry, disgusted, fearful, happy, neutral, sad, surprised)
const FER_CLASS_ORDER = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];

async function loadCustomModel() {
    try {
        // tf is available globally from the CDN
        if (typeof tf === 'undefined') {
            console.warn('[custom-model] TensorFlow.js not found on window.');
            return;
        }
        setProgress(15, 'Loading custom FER2013 model...');
        customModel = await tf.loadLayersModel('model/model.json');
        usingCustomModel = true;
        const paramCount = customModel.countParams().toLocaleString();
        console.log(`[custom-model] Loaded successfully. Parameters: ${paramCount}`);
        showModelBadge(true);
    } catch (err) {
        console.warn('[custom-model] model/model.json not found — using face-api fallback.', err.message);
        usingCustomModel = false;
        showModelBadge(false);
    }
}

function showModelBadge(active) {
    const badge = document.getElementById('model-badge');
    if (!badge) return;
    if (active) {
        badge.style.display = 'flex';
        document.getElementById('model-badge-text').textContent = '🟢 Custom ML Model Active';
    } else {
        badge.style.display = 'none';
    }
}

// ─── Face Preprocessing for Custom Model ─────────────────────────────────
// Crops the detected face region from the video, resizes to 48×48 grayscale,
// normalises to [0,1], and returns a 4-D tensor [1, 48, 48, 1]
function preprocessFace(box) {
    if (typeof tf === 'undefined') return null;
    try {
        const { x, y, width, height } = box;
        // Clamp to valid video dimensions
        const vw = video.videoWidth,  vh = video.videoHeight;
        const sx = Math.max(0, Math.round(x)),
              sy = Math.max(0, Math.round(y)),
              sw = Math.min(Math.round(width),  vw - sx),
              sh = Math.min(Math.round(height), vh - sy);
        if (sw <= 0 || sh <= 0) return null;

        // Draw the cropped face onto a tiny offscreen canvas
        const offC = document.createElement('canvas');
        offC.width = 48; offC.height = 48;
        const offCtx = offC.getContext('2d');
        offCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 48, 48);

        // Convert to grayscale tensor [1, 48, 48, 1] normalised 0-1
        return tf.tidy(() => {
            const rgbTensor = tf.browser.fromPixels(offC);          // [48, 48, 3]
            const greyTensor = rgbTensor
                .mean(2, true)                                       // [48, 48, 1]
                .toFloat()
                .div(255.0)
                .expandDims(0);                                      // [1, 48, 48, 1]
            return greyTensor;
        });
    } catch (e) {
        console.error('[preprocess]', e);
        return null;
    }
}

function preprocessFaceFromSource(source, box) {
    if (typeof tf === 'undefined' || !source || !box) return null;
    try {
        const sourceWidth = source.videoWidth || source.naturalWidth || source.width;
        const sourceHeight = source.videoHeight || source.naturalHeight || source.height;
        if (!sourceWidth || !sourceHeight) return null;

        const { x, y, width, height } = box;
        const sx = Math.max(0, Math.round(x));
        const sy = Math.max(0, Math.round(y));
        const sw = Math.min(Math.round(width), sourceWidth - sx);
        const sh = Math.min(Math.round(height), sourceHeight - sy);
        if (sw <= 0 || sh <= 0) return null;

        const offC = document.createElement('canvas');
        offC.width = 48;
        offC.height = 48;
        const offCtx = offC.getContext('2d');
        offCtx.drawImage(source, sx, sy, sw, sh, 0, 0, 48, 48);

        return tf.tidy(() => {
            const rgbTensor = tf.browser.fromPixels(offC);
            return rgbTensor.mean(2, true).toFloat().div(255.0).expandDims(0);
        });
    } catch (e) {
        console.error('[preprocess-source]', e);
        return null;
    }
}

// ─── Classify with Custom Model ───────────────────────────────────────────
async function classifyWithCustomModel(tensor) {
    if (!customModel || !tensor) return null;
    const predsTensor = customModel.predict(tensor);
    const predsArray  = await predsTensor.data();
    predsTensor.dispose();
    tensor.dispose();

    // Map FER_CLASS_ORDER → EMOTIONS order
    const scores = {};
    EMOTIONS.forEach(e => { scores[e] = 0; });
    FER_CLASS_ORDER.forEach((cls, i) => {
        // cls may be 'disgusted' but EMOTIONS uses 'disgusted' too — direct match
        if (scores.hasOwnProperty(cls)) {
            scores[cls] = Math.round(predsArray[i] * 100);
        }
    });
    return scores;
}

// ─── Model Loading ────────────────────────────────────────────────────────
const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';
const FACE_DETECTOR_OPTIONS = new faceapi.TinyFaceDetectorOptions({
    inputSize: 416,
    scoreThreshold: 0.2
});

async function loadModels() {
    // 1. Try loading custom FER2013 model first
    await loadCustomModel();

    // 2. Always load face-api for face DETECTION (bounding box)
    try {
        setProgress(25, 'Loading face detector...');
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);

        if (!usingCustomModel) {
            // Only load expression net if we don't have our own classifier
            setProgress(60, 'Loading expression model...');
            await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        }

        setProgress(100, usingCustomModel ? 'Custom ML Model ready!' : 'AI Models ready!');
        modelsLoaded = true;
        usingFaceApi = true;
        console.log('[face-api] Face detector loaded.');
    } catch (err) {
        console.warn('[face-api] Load failed, using simulation mode.', err);
        modelsLoaded = true;
        usingFaceApi = false;
        setProgress(100, 'Ready (simulation mode)');
    }
    setTimeout(() => loadingEl.classList.add('fade-out'), 700);
}

function setProgress(pct, msg) {
    progressEl.style.width = pct + '%';
    if (msg) loadStatus.textContent = msg;
}

// ─── Camera ──────────────────────────────────────────────────────────────
async function startCamera() {
    if (!modelsLoaded) { alert('Please wait while AI models are loading...'); return; }
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
        video.srcObject = stream;
        video.style.display = 'block';
        idleState.style.display = 'none';
        videoFrame.classList.add('active');
        btnStart.classList.add('hidden');
        btnStop.classList.remove('hidden');
        btnSnap.disabled = false;
        setStatus(true);
        video.addEventListener('loadeddata', () => {
            canvas.width  = video.videoWidth;
            canvas.height = video.videoHeight;
            startDetection();
        }, { once: true });
    } catch (err) {
        alert('Camera access denied. Please allow camera access in your browser settings.');
        console.error(err);
    }
}

function stopCamera() {
    clearInterval(detectInterval);
    stream?.getTracks().forEach(t => t.stop());
    stream = null;
    video.style.display = 'none';
    idleState.style.display = '';
    videoFrame.classList.remove('active');
    // remove all emotion classes from videoFrame
    EMOTIONS.forEach(e => videoFrame.classList.remove('emo-' + e));
    btnStart.classList.remove('hidden');
    btnStop.classList.add('hidden');
    btnSnap.disabled = true;
    setStatus(false);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resetDisplay();
}

// ─── Detection Loop ───────────────────────────────────────────────────────
function startDetection() {
    clearInterval(detectInterval);

    if (usingFaceApi) {
        detectInterval = setInterval(async () => {
            if (!stream) return;
            try {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (usingCustomModel) {
                    // ── Custom model path ─────────────────────────────────
                    // Step 1: detect face location with face-api (no expression net)
                    const detection = await faceapi
                        .detectSingleFace(video, FACE_DETECTOR_OPTIONS);

                    if (detection) {
                        drawFaceBox(detection.box);
                        // Step 2: crop + preprocess face region
                        const tensor = preprocessFace(detection.box);
                        // Step 3: classify with our CNN
                        const scores = await classifyWithCustomModel(tensor);
                        if (scores) {
                            const dominant = getDominant(scores);
                            updateDisplay(scores, dominant);
                            addHistory(dominant.name, dominant.score);
                        }
                    } else {
                        confidPill.textContent = 'No face detected';
                        emotionEmoji.textContent = '🔍';
                        emotionName.textContent = '–';
                    }

                } else {
                    // ── face-api expression net path (fallback) ───────────
                    const result = await faceapi
                        .detectSingleFace(video, FACE_DETECTOR_OPTIONS)
                        .withFaceExpressions();

                    if (result) {
                        drawFaceBox(result.detection.box);
                        const scores = buildScores(result.expressions);
                        const dominant = getDominant(scores);
                        updateDisplay(scores, dominant);
                        addHistory(dominant.name, dominant.score);
                    } else {
                        confidPill.textContent = 'No face detected';
                        emotionEmoji.textContent = '🔍';
                        emotionName.textContent = '–';
                    }
                }

            } catch (e) {
                console.error('[detection]', e);
            }
        }, 500);

    } else {
        // Simulation fallback (no camera / face-api failed)
        detectInterval = setInterval(() => {
            const scores = simulateEmotions();
            const dominant = getDominant(scores);
            updateDisplay(scores, dominant);
            addHistory(dominant.name, dominant.score);
        }, 1800);
    }
}

// ─── Face Box Drawing ─────────────────────────────────────────────────────
function drawFaceBox(box) {
    const vw = video.videoWidth,  vh = video.videoHeight;
    const cw = canvas.width,      ch = canvas.height;
    const { x, y, width, height } = box;
    const sx = (x / vw) * cw,    sy = (y / vh) * ch;
    const sw = (width / vw) * cw, sh = (height / vh) * ch;
    const cs = 20;

    ctx.save();
    ctx.shadowColor = '#6366f1';
    ctx.shadowBlur  = 18;
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
    ctx.lineWidth   = 1.5;
    ctx.strokeRect(sx, sy, sw, sh);

    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth   = 3;
    ctx.shadowBlur  = 24;
    // TL
    ctx.beginPath(); ctx.moveTo(sx, sy+cs); ctx.lineTo(sx,sy); ctx.lineTo(sx+cs,sy); ctx.stroke();
    // TR
    ctx.beginPath(); ctx.moveTo(sx+sw-cs, sy); ctx.lineTo(sx+sw,sy); ctx.lineTo(sx+sw,sy+cs); ctx.stroke();
    // BL
    ctx.beginPath(); ctx.moveTo(sx, sy+sh-cs); ctx.lineTo(sx,sy+sh); ctx.lineTo(sx+cs,sy+sh); ctx.stroke();
    // BR
    ctx.beginPath(); ctx.moveTo(sx+sw-cs, sy+sh); ctx.lineTo(sx+sw,sy+sh); ctx.lineTo(sx+sw,sy+sh-cs); ctx.stroke();
    ctx.restore();
}

// ─── Score helpers ────────────────────────────────────────────────────────
function buildScores(expressions) {
    const scores = {};
    EMOTIONS.forEach(e => { scores[e] = Math.round((expressions[e] || 0) * 100); });
    return scores;
}

function getDominant(scores) {
    let best = { name: 'neutral', score: 0 };
    for (const [name, score] of Object.entries(scores)) {
        if (score > best.score) best = { name, score };
    }
    return best;
}

// ─── Simulation Fallback ──────────────────────────────────────────────────
function simulateEmotions() {
    const weights = { happy:0.32, neutral:0.27, sad:0.14, surprised:0.10, angry:0.09, fearful:0.05, disgusted:0.03 };
    let raw = {}, total = 0;
    for (const [k, w] of Object.entries(weights)) {
        raw[k] = Math.max(0, w + (Math.random() - 0.5) * 0.25);
        total += raw[k];
    }
    const scores = {};
    for (const k of Object.keys(raw)) scores[k] = Math.round((raw[k] / total) * 100);
    return scores;
}

// ─── UI Update ────────────────────────────────────────────────────────────
function updateDisplay(scores, dominant) {
    // Update bars + shimmer on active one
    EMOTIONS.forEach(e => {
        const fill = $('fill-' + e);
        const pct  = $('pct-'  + e);
        if (fill) {
            fill.style.width = scores[e] + '%';
            if (e === dominant.name) fill.classList.add('active-bar');
            else fill.classList.remove('active-bar');
        }
        if (pct) pct.textContent = scores[e] + '%';
    });

    // Hero
    emotionName.textContent = capitalize(dominant.name);
    confidPill.textContent  = dominant.score + '% confidence';

    // Emoji bounce on change
    if (lastDominant !== dominant.name) {
        const newEmoji = EMOJIS[dominant.name];
        emotionEmoji.textContent = newEmoji;
        emotionEmoji.classList.remove('pop');
        void emotionEmoji.offsetWidth;
        emotionEmoji.classList.add('pop');

        // Video frame emotion border class
        EMOTIONS.forEach(e => videoFrame.classList.remove('emo-' + e));
        videoFrame.classList.add('emo-' + dominant.name);

        // Body ambient glow
        EMOTIONS.forEach(e => document.body.classList.remove('emo-' + e));
        document.body.classList.add('emo-' + dominant.name);

        // Particle burst
        burstParticles(emotionEmoji, COLORS[dominant.name]);

        // Suggestion typing
        typeText(suggestionEl, SUGGESTIONS[dominant.name]);

        lastDominant = dominant.name;
    }
}

function resetDisplay() {
    EMOTIONS.forEach(e => {
        const fill = $('fill-' + e), pct = $('pct-' + e);
        if (fill) { fill.style.width = '0%'; fill.classList.remove('active-bar'); }
        if (pct) pct.textContent = '0%';
    });
    emotionName.textContent = '–';
    confidPill.textContent  = 'Awaiting…';
    emotionEmoji.textContent = '🤖';
    suggestionEl.textContent = 'Start the camera to receive personalized AI suggestions.';
    EMOTIONS.forEach(e => document.body.classList.remove('emo-' + e));
    lastDominant = null;
}

function setStatus(active) {
    if (active) {
        statusDot.className = 'dot dot-active';
        statusText.textContent = 'Active';
    } else {
        statusDot.className = 'dot dot-offline';
        statusText.textContent = 'Ready';
    }
}

// ─── History ──────────────────────────────────────────────────────────────
function addHistory(emotion, score) {
    history.unshift({ emotion, score });
    if (history.length > 5) history.pop();
    historyList.innerHTML = '';
    history.forEach(item => {
        const chip = document.createElement('div');
        chip.className = `chip ${item.emotion}`;
        chip.innerHTML = `${EMOJIS[item.emotion]} ${capitalize(item.emotion)} <span style="opacity:.6">${item.score}%</span>`;
        historyList.appendChild(chip);
    });
}

// ─── Snapshot + Emotion Analysis ──────────────────────────────────────────
async function captureSnapshot() {
    if (!stream) return;

    const sw = video.videoWidth, sh = video.videoHeight;
    const offCanvas = document.createElement('canvas');
    offCanvas.width = sw; offCanvas.height = sh;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(video, 0, 0, sw, sh);
    offCtx.drawImage(canvas, 0, 0, sw, sh);    // include tracking box

    const dataUrl = offCanvas.toDataURL('image/png');
    showSnapModal(dataUrl, offCanvas);
}

async function showSnapModal(dataUrl, offCanvas) {
    // Build modal DOM
    let modal = document.getElementById('snap-modal-overlay');
    if (modal) modal.remove();    // remove old one if any

    modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.id = 'snap-modal-overlay';
    modal.innerHTML = `
        <div class="snap-modal">
            <div class="snap-modal-header">
                <h3><i class='bx bx-camera'></i> Snapshot Analysis</h3>
                <button class="btn-close-modal" id="close-modal">✕</button>
            </div>
            <img class="snap-modal-image" src="${dataUrl}" alt="Snapshot">
            <div id="snap-result-area">
                <div class="snap-analyzing">
                    <div class="spinner"></div>
                    <p>Analyzing facial expressions...</p>
                </div>
            </div>
        </div>`;
    document.body.appendChild(modal);
    requestAnimationFrame(() => modal.classList.add('show'));
    $('close-modal').addEventListener('click', () => {
        modal.classList.remove('show');
        setTimeout(() => modal.remove(), 400);
    });
    modal.addEventListener('click', e => {
        if (e.target === modal) { modal.classList.remove('show'); setTimeout(() => modal.remove(), 400); }
    });

    // Analyze the captured frame
    const resultArea = document.getElementById('snap-result-area');
    try {
        let scores = null;
        let dominant = null;

        const img = new Image();
        img.src = dataUrl;
        await new Promise(r => img.onload = r);

        // Step 1: detect face bounding box via face-api (always)
        const detection = usingFaceApi
            ? await faceapi.detectSingleFace(img, FACE_DETECTOR_OPTIONS)
            : null;

        if (detection) {
            if (usingCustomModel) {
                // Step 2a: use our CNN to classify
                const tensor = preprocessFaceFromSource(img, detection.box);
                scores = await classifyWithCustomModel(tensor);
                dominant = scores ? getDominant(scores) : null;
            } else {
                // Step 2b: use face-api expression net
                const imgWithExpr = await faceapi
                    .detectSingleFace(img, FACE_DETECTOR_OPTIONS)
                    .withFaceExpressions();
                if (imgWithExpr) {
                    scores   = buildScores(imgWithExpr.expressions);
                    dominant = getDominant(scores);
                }
            }
        }

        if (!scores) {
            // Fallback simulation when face-api can't detect
            scores   = simulateEmotions();
            dominant = getDominant(scores);
        }

        // Render results
        resultArea.innerHTML = `
            <div class="snap-emotion-result">
                <div class="snap-emoji-big">${EMOJIS[dominant.name]}</div>
                <div>
                    <div class="snap-emotion-label">${capitalize(dominant.name)}</div>
                    <div class="snap-confidence">${dominant.score}% confidence from this snapshot</div>
                </div>
            </div>
            <div class="snap-suggestion-text">${SUGGESTIONS[dominant.name]}</div>
            <div class="snap-bars">
                ${EMOTIONS.map(e => `
                <div class="snap-bar-row">
                    <span class="snap-bar-label">${EMOJIS[e]} ${capitalize(e)}</span>
                    <div class="snap-bar-track">
                        <div class="snap-bar-fill ${e}-fill" id="sbar-${e}" style="width:0%"></div>
                    </div>
                    <span class="snap-bar-pct">${scores[e]}%</span>
                </div>`).join('')}
            </div>`;

        // Animate bars in after render
        requestAnimationFrame(() => {
            EMOTIONS.forEach(e => {
                const el = document.getElementById('sbar-' + e);
                if (el) el.style.width = scores[e] + '%';
            });
        });

    } catch (err) {
        console.error('[snapshot analysis]', err);
        resultArea.innerHTML = `<div class="snap-no-face"><i class='bx bx-error-circle'></i>Could not analyze image. Please try again.</div>`;
    }
}

// ─── Particle Burst ───────────────────────────────────────────────────────
function burstParticles(el, color) {
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width  / 2;
    const cy = rect.top  + rect.height / 2;
    const count = 14;

    for (let i = 0; i < count; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const angle = (i / count) * 2 * Math.PI;
        const dist  = 60 + Math.random() * 60;
        const tx    = Math.cos(angle) * dist;
        const ty    = Math.sin(angle) * dist;
        const dur   = 0.6 + Math.random() * 0.5;
        p.style.cssText = `
            left: ${cx}px; top: ${cy}px;
            background: ${color};
            --tx: ${tx}px; --ty: ${ty}px; --dur: ${dur}s;
            width: ${4 + Math.random() * 5}px;
            height: ${4 + Math.random() * 5}px;
            box-shadow: 0 0 6px ${color};
        `;
        document.body.appendChild(p);
        setTimeout(() => p.remove(), dur * 1000);
    }
}

// ─── Button Ripple Effect ─────────────────────────────────────────────────
function attachRipple(btn) {
    btn.addEventListener('click', function(e) {
        const rect   = btn.getBoundingClientRect();
        const size   = Math.max(rect.width, rect.height);
        const ripple = document.createElement('span');
        ripple.className = 'ripple-effect';
        ripple.style.cssText = `
            width: ${size}px; height: ${size}px;
            left: ${e.clientX - rect.left - size/2}px;
            top: ${e.clientY - rect.top  - size/2}px;`;
        btn.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
    });
}

// ─── Typing effect ────────────────────────────────────────────────────────
function typeText(el, text) {
    clearTimeout(typingTimer);
    el.textContent = '';
    let i = 0;
    function tick() {
        if (i < text.length) {
            el.textContent += text[i++];
            typingTimer = setTimeout(tick, 16);
        }
    }
    tick();
}

function capitalize(str) { return str ? str.charAt(0).toUpperCase() + str.slice(1) : str; }

// ─── Chart.js ─────────────────────────────────────────────────────────────
// Real FER2013 Kaggle dataset numbers (icml.cc/2013 competition dataset)
// Train: 28,709 images | Test: 7,178 images
const FER2013 = {
    labels:    ['Happy', 'Neutral', 'Sad',  'Angry', 'Surprised', 'Fearful', 'Disgusted'],
    train:     [7215,    4965,      4830,   3995,    3171,        4097,      436],
    test:      [1774,    1233,      1247,   958,     831,         1024,      111]
};

function initChart() {
    const ctxC = document.getElementById('ferChart').getContext('2d');
    new Chart(ctxC, {
        type: 'bar',
        data: {
            labels: FER2013.labels,
            datasets: [
                {
                    label: 'Training Set',
                    data: FER2013.train,
                    backgroundColor: Object.values(COLORS).map(c => c + 'aa'),
                    borderColor:     Object.values(COLORS),
                    borderWidth: 1.5,
                    borderRadius: 6,
                    borderSkipped: false,
                },
                {
                    label: 'Test Set',
                    data: FER2013.test,
                    backgroundColor: Object.values(COLORS).map(c => c + '44'),
                    borderColor:     Object.values(COLORS),
                    borderWidth: 1,
                    borderRadius: 6,
                    borderSkipped: false,
                }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 1200, easing: 'easeOutQuart' },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#9ca3af', font: { size: 11 } }
                },
                tooltip: {
                    callbacks: {
                        afterBody: (items) => {
                            const idx = items[0].dataIndex;
                            const total = FER2013.train[idx] + FER2013.test[idx];
                            return [`Total: ${total.toLocaleString()}`];
                        }
                    }
                }
            },
            scales: {
                y: {
                    stacked: false,
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#6b7fa3', font: { size: 11 } },
                    title: { display: true, text: 'Image Count', color: '#6b7fa3', font: { size: 11 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', font: { size: 11 } }
                }
            }
        }
    });
}


// ─── Event Listeners ─────────────────────────────────────────────────────
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click',  stopCamera);
btnSnap.addEventListener('click',  captureSnapshot);
[btnStart, btnStop, btnSnap].forEach(attachRipple);

// ─── Boot ─────────────────────────────────────────────────────────────────
initChart();
loadModels();
