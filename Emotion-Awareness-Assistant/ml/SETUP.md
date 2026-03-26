# 🧠 ML Setup Guide — FER2013 Custom Model

This guide walks you through training your own CNN on the FER2013 Kaggle dataset and loading it in the browser.

---

## Prerequisites

- Python 3.9 or 3.10 installed
- A Kaggle account (free)

---

## Step 1 — Get Your Kaggle API Key

1. Go to [https://www.kaggle.com](https://www.kaggle.com) and sign in
2. Click your profile picture → **Settings**
3. Scroll to **API** section → click **Create New Token**
4. This downloads a file called `kaggle.json`
5. Place it here:
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

---

## Step 2 — Install Python Dependencies

Open a terminal in your project folder and run:

```bash
pip install -r ml/requirements.txt
```

This installs TensorFlow, TensorFlow.js converter, Kaggle API, NumPy, Pillow, and more.

> ⚠️ Make sure you have at least **3 GB of free disk space** for the dataset and model files.

---

## Step 3 — Run a Quick Smoke Test

Before full training, verify everything is set up correctly:

```bash
python ml/train_model.py --smoke-test
```

Expected output:
```
✅  Smoke test passed — model builds and forward-pass works
    Output shape : (1, 7)
    Probabilities: ['0.143', '0.142', '0.144', ...]
```

---

## Step 4 — Train the Model

```bash
python ml/train_model.py
```

What happens:
1. Downloads FER2013 dataset from Kaggle (~60 MB)
2. Builds a 4-block CNN (~1.2M parameters)
3. Trains for up to 50 epochs (early stopping kicks in ~30–40)
4. Evaluates on the test set
5. Exports the model to `model/model.json` + weight files

**Training time:**
| Hardware | Time |
|---|---|
| GPU (NVIDIA) | ~15–25 minutes |
| CPU only | ~90–150 minutes |

**Expected accuracy:** ~63–67% on FER2013 test set (research-grade baseline)

---

## Step 5 — Load in the Browser

1. Open `index.html` using **Live Server** (VS Code extension) or any local HTTP server
2. The app will automatically find `model/model.json`
3. You'll see **🟢 Custom ML Model Active** in the UI
4. Open DevTools Console to confirm:
   ```
   [custom-model] Loaded successfully. Parameters: ~1.2M
   ```

> ⚠️ The model files MUST be served over HTTP (not `file://`) for the browser to load them.  
> Use VS Code **Live Server** extension — right-click `index.html` → Open with Live Server.

---

## File Structure After Training

```
Emotion-Awareness-Assistant/
├── ml/
│   ├── train_model.py      ← Training script
│   ├── requirements.txt    ← Python deps
│   ├── SETUP.md            ← This guide
│   └── fer2013_cnn.h5      ← Keras checkpoint (generated)
├── model/
│   ├── model.json          ← TF.js model architecture (generated)
│   └── group1-shard1of1.bin ← Model weights binary (generated)
├── index.html
└── app.js
```

---

## Dataset Info — FER2013

| Class | Train | Test |
|---|---|---|
| Happy | 7,215 | 1,774 |
| Neutral | 4,965 | 1,233 |
| Sad | 4,830 | 1,247 |
| Angry | 3,995 | 958 |
| Fearful | 4,097 | 1,024 |
| Surprised | 3,171 | 831 |
| Disgusted | 436 | 111 |
| **Total** | **28,709** | **7,178** |

Source: [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
