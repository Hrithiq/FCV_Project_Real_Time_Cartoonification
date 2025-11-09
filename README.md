## FCV Project – Real-time Cartoonification and Offline Stylization

### Overview
- Script: `main.py`
- Modes:
  - Real-time webcam stylization (optimized for ≥24 FPS with adjustable quality)
  - Offline multi-pass processing for maximum quality on video files

### Requirements
- Python 3.8+
- Packages: `opencv-python`, `numpy`

```bash
pip install opencv-python numpy
```

---

## Quick Start

### Real-time (webcam)
```bash
python main.py
# or
python main.py realtime
```

- Press `q` to quit.

### Offline (file → file)
```bash
python main.py offline --input input.mp4 --output output.mp4
```

---

## Real-time Mode

### Basic usage
```bash
python main.py realtime
```

### With quality/performance controls
```bash
python main.py realtime \
  --camera_id 0 --width 1280 --height 720 \
  --process_scale 0.8 --bilateral_iters 3 \
  --quant_levels 10 --edge_strength 0.7 --sharpness 0.5 \
  --auto_scale --target_fps 24
```

### Processed-only view
```bash
python main.py realtime --processed_only
```

### Parameters (real-time)
- **--camera_id [int]**
  - What: OS camera index to open.
  - Typical: 0, 1, 2…
  - Impact: Device selection only.

- **--width [int], --height [int]**
  - What: Requested capture resolution; 0 keeps camera default.
  - Range: 0 or supported modes (e.g., 640×480, 1280×720, 1920×1080).
  - Impact: Higher resolution ⇒ more detail, lower FPS.

- **--process_scale [float]**
  - What: Internal processing scale (rendering stays at capture size).
  - Range: 0.5–1.0 (internally clamped to 0.25–1.0); default 1.0.
  - Impact: Lower = faster but softer. Sweet spot: 0.75–0.9.

- **--bilateral_iters [int]**
  - What: Iterations of fast edge-preserving smoothing.
  - Range: 1–6; default 5.
  - Impact: Higher = smoother/painterly but can blur and reduce FPS.

- **--quant_levels [int]**
  - What: Fast per-channel uniform quantization (posterization).
  - Range: 4–32; default 8.
  - Impact: Higher = more color detail, less banding; small CPU cost.

- **--edge_strength [float]**
  - What: Edge darkening blend strength.
  - Range: 0.0–1.0; default 0.6.
  - Impact: Higher emphasizes line art and perceived detail; too high looks harsh.

- **--sharpness [float]**
  - What: Unsharp mask amount.
  - Range: 0.0–1.5; default 0.4.
  - Impact: Restores micro-contrast; too high causes halos/noise.

- **--processed_only [flag]**
  - What: Show only the processed frame (not side-by-side).
  - Impact: Slight FPS gain and larger stylized view.

- **--no_stabilize [flag]**
  - What: Disable optical-flow stabilization.
  - Impact: Big FPS boost; may see camera jitter. Recommended ON for ≥24 FPS.

- **--auto_scale [flag]**
  - What: Dynamically adjusts `--process_scale` to meet FPS target.

- **--target_fps [float]**
  - What: Target FPS when `--auto_scale` is enabled.
  - Range: 12–60; default 24.0.
  - Impact: Higher target may downscale more aggressively.

---

## Offline Mode

### Standard pipeline
```bash
python main.py offline \
  --input input.mp4 --output output.mp4 \
  --target_fps 60 --bilateral_iters 7 --palette 24 \
  --stabilization_radius 25
```

### Parameters (offline)
- **--input [path]**
  - Source video file.

- **--output [path]**
  - Destination mp4 file (mp4v).

- **--target_fps [int/float]**
  - Output FPS after interpolation.
  - Range: ≥ source FPS; typical 30/60.
  - Impact: Higher ⇒ smoother motion and larger file.

- **--bilateral_iters [int]**
  - Iterations for offline smoothing (slower but higher quality than real-time).
  - Range: 3–10; default 7.

- **--palette [int]**
  - K-means color palette size.
  - Range: 8–48; default 24.
  - Impact: Higher ⇒ more color fidelity; slower.

- **--stabilization_radius [int]**
  - Temporal smoothing radius for global stabilization (frames).
  - Range: 10–60; default 25.
  - Impact: Higher reduces jitter but may lag on fast motion.

- **--workdir [path or None]**
  - Optional directory to save intermediate outputs (for debugging/inspection).

---

## Recommended Presets

- Real-time 24 FPS balanced:
```bash
python main.py realtime \
  --no_stabilize --process_scale 0.8 \
  --bilateral_iters 3 --quant_levels 10 \
  --edge_strength 0.7 --sharpness 0.5 \
  --auto_scale --target_fps 24
```

- Real-time higher detail (maintain ~24 FPS):
```bash
python main.py realtime \
  --no_stabilize --process_scale 0.85 \
  --bilateral_iters 4 --quant_levels 12 \
  --edge_strength 0.75 --sharpness 0.7 \
  --auto_scale --target_fps 24
```

- Real-time with stabilization (good lighting, mid-FPS):
```bash
python main.py realtime \
  --process_scale 0.75 \
  --bilateral_iters 3 --quant_levels 10 \
  --edge_strength 0.6 --sharpness 0.5
```

- Offline maximum quality:
```bash
python main.py offline \
  --input input.mp4 --output output.mp4 \
  --target_fps 60 --bilateral_iters 8 --palette 32 \
  --stabilization_radius 30
```

---

## Tuning Guide

- **Increase detail (reduce blur):**
  - Raise `--quant_levels` (12–16), raise `--sharpness` (0.6–0.8),
  - Reduce `--bilateral_iters` slightly if too smooth,
  - Keep `--process_scale` ≥ 0.75.

- **Achieve ≥24 FPS:**
  - Use `--no_stabilize`,
  - Lower `--process_scale` to 0.75–0.8,
  - Reduce `--bilateral_iters` to 2–3,
  - Enable `--auto_scale --target_fps 24`.

- **Edge aesthetics:**
  - Stronger lines: `--edge_strength 0.7–0.9`,
  - Softer lines: `--edge_strength 0.3–0.5`.

---

## Troubleshooting

- **Webcam not opening:**
  - Try another device index: `--camera_id 1`, ensure no other app is using the camera.

- **FPS below 24:**
  - Add `--no_stabilize`,
  - Lower `--process_scale` to 0.75–0.8,
  - Reduce `--bilateral_iters` to 2–3,
  - Use `--auto_scale --target_fps 24`.

- **Image looks too blurred:**
  - Increase `--quant_levels` (12–16),
  - Increase `--sharpness` (0.6–0.8),
  - Decrease `--bilateral_iters` by 1.

- **Harsh edges or halos:**
  - Reduce `--edge_strength` to 0.5–0.6,
  - Reduce `--sharpness` to 0.3–0.5.

---

## File Reference
- Entry point: `main.py`
- Real-time class: `RealTimeCartoonifier`
  - `run_webcam(...)`: camera capture + display loop
  - `process_frame(...)`: optional stabilization + stylization
  - `apply_realtime_style(...)`: bilateral → quantization → edge darken → unsharp mask


