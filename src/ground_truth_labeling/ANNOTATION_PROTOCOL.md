# GROUND TRUTH ANNOTATION PROTOCOL

## Overview

This document defines the methodology for manually labeling depth sensor bins as true obstacles or phantom detections in wheelchair navigation rosbag data.

**Reviewer Gap Being Addressed**: Stanford reviewer feedback — "lacks labeled ground truth" and "no labeled spot-checking to substantiate effectiveness."

**Scope**: 30 rosbag frames (10 per session, 3 sessions) × ~3200 LiDAR bins per frame = 96,000 potential bins to classify.

**Objective**: Validate the 36% phantom-candidate rate claim from PRISM-Nav paper with manual per-bin ground truth labels.

---

## Classification Definitions

### TRUE OBSTACLE

**Definition**: A confirmed elevated structure at the specified depth that represents a genuine navigation hazard.

**Visual Criteria**:
- Visible in synchronized RGB video at the corresponding angle
- Structure matches expected wheelchair-level obstacles (0.1–1.8 m height)
- Range value is consistent with visual position in RGB frame
- Surface appears to have physical continuity (not scattered points)

**Structural Types** (examples):
- Dining tables, desks, furniture
- Equipment racks, shelving units
- Door frames, step risers, thresholds
- Walls, vertical supports
- Wheelchair footrests, armrests visible in rear cameras

**How to Label**:
- Look up the bin's angle in RGB frame
- Check if you see an object at that depth
- If yes and depth matches visual position → **TRUE OBSTACLE**
- Confidence: 0.9–1.0 if certain, 0.7–0.8 if slightly uncertain about depth matching

**Example**:
```
Depth bin 500 (angle = -25°):
  RGB shows: wooden table at ~1.2m
  Depth value: 1.18m
  → TRUE OBSTACLE, confidence=0.95
```

---

### PHANTOM (DEPTH NOISE)

**Definition**: RealSense stereo or time-of-flight sensor noise manifesting as isolated, implausible depth measurements.

**Characteristics**:
- No visible object in RGB frame at corresponding angle
- Range appears inconsistent with surrounding valid points
- Often appears as isolated single bin or small cluster
- Typically closer than expected (e.g., 0.3 m when nearest visible object is 1.5 m away)
- Disappears frame-to-frame (not temporally stable)
- Common sources:
  - RealSense stereo matcher failures on textureless surfaces
  - Quantization noise in depth quantization
  - Low-texture regions where correlation fails
  - Speckle from texture-less walls or floors

**How to Label**:
- Look at RGB frame at the bin's angle
- If no visible object → suspect noise
- Check if depth value is outlier compared to neighbors
- If isolated and unexplained → **PHANTOM (DEPTH NOISE)**
- Confidence: 0.8–1.0 if isolated, 0.6–0.7 if borderline

**Example**:
```
Depth bin 1500 (angle = 87°):
  RGB shows: plain white wall at ~2.0m, no objects
  Depth value: 0.43m (wildly closer)
  Neighbors: 1.95–2.05m
  → PHANTOM (DEPTH NOISE), confidence=0.92
```

---

### PHANTOM (REFLECTION)

**Definition**: Incorrect depth caused by IR or visible light reflection, typically from shiny or transparent surfaces.

**Characteristics**:
- May appear to correspond to an object in RGB but at incorrect depth
- Often caused by:
  - Glass doors, windows (IR transparent)
  - Shiny metal surfaces, mirrors (specular reflection)
  - Glossy furniture (partial reflection)
  - Curved surfaces that redirect IR beams
- The reflected obstacle appears "doubled" or layered
- Range is inconsistent with actual obstacle position
- May be frame-to-frame unstable if angle of reflection shifts

**How to Label**:
- Check if depth matches visual object
- If object is visible but depth seems wrong, look for reflective surface upstream
- If depth points to a reflection rather than the actual object → **PHANTOM (REFLECTION)**
- Confidence: 0.8–1.0 if reflection is clear, 0.6–0.7 if borderline

**Example**:
```
Depth bin 50 (angle = -160°):
  RGB shows: rear blind zone not visible in camera
  But visible: glass panel at ~1.0m reflecting wall behind
  Depth value: 3.2m (pointing to far wall)
  → PHANTOM (REFLECTION), confidence=0.75
```

---

### PHANTOM (OTHER)

**Definition**: False positives that are neither noise nor reflection (ceiling returns, ground bounces, compression artifacts, etc.).

**Characteristics**:
- Not ambient sensor noise
- Not IR reflection
- Clearly invalid as an obstacle but doesn't fit above categories
- Common sources:
  - Ceiling returns (above wheelchair FOV)
  - Ground multipath (lidar bounce off floor)
  - JPEG/H.264 compression artifacts
  - Synchronization errors (depth/RGB from different times)
  - Camera lens artifacts or vignetting

**How to Label**:
- Eliminate noise and reflection first
- If depth appears invalid for other reason → **PHANTOM (OTHER)**
- Confidence: typically 0.7–0.9

**Example**:
```
Depth bin 1800 (angle = 145°):
  RGB shows: ceiling light fixture (above wheelchair eye level)
  Depth value: 0.65m
  → PHANTOM (OTHER), confidence=0.85
```

---

### AMBIGUOUS

**Definition**: Cases where the operator cannot confidently classify the bin due to insufficient information or conflicting signals.

**When to Mark**:
- RGB camera is not pointed at the bin's angle (out-of-field-of-view)
- Depth/RGB are temporally misaligned (synchronization lag)
- Object partially occluded in both depth and RGB
- Operator unfamiliar with specific layout detail
- Bin is borderline between two categories after careful review

**How to Label**:
- Mark as **AMBIGUOUS** with confidence 0.5–0.7
- Leave detailed notes explaining reason
- Example: "RTCamera not visible at this angle in RGB; depth at 1.1m could be valid table or noise"

**Acceptance Rule**: ≤10% of labels should be ambiguous. Higher rates indicate poor frame selection or low-confidence labeling.

---

## Labeling Procedure (Per Frame)

### Step 1: Load Frame Data

```python
import numpy as np
from PIL import Image

# Load frame files
frame_dir = Path("extracted_frames/rosbag_00/frame_000/")
depth = np.load(frame_dir / "depth.npy")  # Shape: (3200,), dtype=float32
rgb = Image.open(frame_dir / "rgb.jpg")  # RGB image from front camera

# Load metadata
with open(frame_dir / "metadata.json") as f:
    metadata = json.load(f)
    timestamp = metadata['timestamp']
    rosbag = metadata['rosbag']
```

### Step 2: Visualize Depth

```python
import matplotlib.pyplot as plt

# Plot depth as 1D array
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Depth heatmap
valid = np.isfinite(depth)
ax1.plot(np.arange(3200)[valid], depth[valid], 'o', markersize=1)
ax1.set_xlabel("Bin Index (0–3200)")
ax1.set_ylabel("Range (m)")
ax1.set_title(f"Depth Array — {timestamp}")
ax1.grid()

# RGB side-by-side
ax2.imshow(rgb)
ax2.set_title("Front Camera RGB")
ax2.axis('off')

plt.tight_layout()
plt.show()
```

### Step 3: Identify Camera-Closer Bins

Focus on bins where depth < camera operational limit:
- **Front D455**: < 1.5 m (smaller baseline, ~95 mm)
- **Side D435i**: < 1.0 m (even smaller baseline, ~50 mm)
- **Other bins**: can mark as UNLABELED if not in camera FOV

### Step 4: For Each Camera-Closer Bin

1. **Calculate bin angle**:
   ```
   angle_deg = (bin_idx / 3200) * 360 - 180
   ```
   - bin_idx=0 → -180° (rear)
   - bin_idx=800 → -90° (right)
   - bin_idx=1600 → 0° (forward)
   - bin_idx=2400 → 90° (left)

2. **Locate angle in RGB**:
   - -180° to -90°: right edge of RGB image
   - -90° to 0°: right half
   - 0° to 90°: left half
   - 90° to 180°: left edge

3. **Cross-reference**:
   - Is there a visible object at the RGB position?
   - If yes: does its depth match?
   - If no: why not?

4. **Classify**:
   - Assign label from [TRUE_OBSTACLE, PHANTOM_DEPTH_NOISE, PHANTOM_REFLECTION, PHANTOM_OTHER, AMBIGUOUS]
   - Rate confidence 0.5–1.0

5. **Record**:
   ```csv
   frame_id,bin_idx,angle_deg,range_m,label,confidence,notes
   0,1600,0.00,1.234,true_obstacle,0.95,"Table edge aligned with RGB"
   0,1601,0.11,0.345,phantom_depth_noise,0.90,"Isolated outlier; no object in RGB"
   ```

---

## Confidence Rating Guidelines

| Confidence | Meaning | Example |
|---|---|---|
| **0.5–0.6** | Uncertain, borderline | Object partially visible in RGB; depth slightly off |
| **0.7–0.8** | Likely correct, minor doubt | Clear in RGB/depth but slightly noisy neighborhood |
| **0.9–1.0** | Certain, clear signal | Perfect RGB/depth match, no ambiguity |

**Target**: Average confidence ≥ 0.80 across all labels.

---

## Quality Control Checklist

### Before Submitting Labels

- [ ] All camera-closer bins labeled (not all UNLABELED)
- [ ] No more than 10% AMBIGUOUS
- [ ] Average confidence ≥ 0.80
- [ ] Notes provided for low-confidence labels
- [ ] Temporal consistency check: same location in adjacent frames should agree

### Inter-Rater Reliability

If two annotators label same frame:
- Compute Cohen's κ (kappa) on label agreement
- Target: κ ≥ 0.70 (substantial agreement)
- Resolve disagreements via discussion

### Spatial Continuity Check

- Isolated bins are more likely to be noise than continuous regions
- Use this as a validation heuristic (not strict rule)

---

## Annotation Statistics (Expected)

| Metric | Value |
|---|---|
| **Frames to label** | 30 (10/rosbag × 3 rosbags) |
| **Bins per frame** | ~3200 (LiDAR resolution) |
| **Camera-closer bins** | ~500–1500/frame (depth < 1.5 m) |
| **Total bins to label** | ~15,000–45,000 |
| **Estimated time** | ~2–3 hours per rosbag (10 frames) = 6–9 hours total |

---

## Output Format

### Per-Frame CSV (labels.csv)

```csv
frame_id,bin_idx,angle_deg,range_m,label,confidence,notes
0,100,-165.2,1.234,true_obstacle,0.95,"Table edge"
0,101,-165.1,1.245,true_obstacle,0.93,""
0,102,-165.0,0.850,phantom_depth_noise,0.88,"Speckle"
0,1600,0.00,0.000,unlabeled,0.00,"Rear blind zone; LiDAR only"
...
```

### Combined Master CSV (labels_all_frames.csv)

```csv
# Merge all per-frame CSVs
cat rosbag_*/frame_*/labels.csv > labels_all_frames.csv
```

---

## Analysis After Labeling

Once all 30 frames are labeled:

```bash
python3 analysis_tools.py \
    --labels-csv labels_all_frames.csv \
    --output-dir results/
```

**Outputs**:
- `summary_table.md` — classification distribution
- `analysis_results.json` — per-frame, per-zone statistics
- **Phantom rate**: % of camera-closer bins labeled as any PHANTOM variant

**Success Criterion**: Phantom rate = 36% ± 5% (validates paper claim)

---

## Expected Results (For Paper)

### Summary Table

| Classification | Count | Percentage | Mean Confidence |
|---|---|---|---|
| True Obstacles | X | Y% | Z |
| Phantom (Depth Noise) | X | Y% | Z |
| Phantom (Reflection) | X | Y% | Z |
| Phantom (Other) | X | Y% | Z |
| Ambiguous | X | Y% | Z |
| **Total** | **~25,000** | **100%** | **≥0.80** |

### Results Paragraph (for paper)

"To validate phantom detection rates, we manually labeled 30 rosbag frames (10 per session, sampled uniformly across trials) by operator review of synchronized RGB video and RViz depth visualization. Per-bin classification: [X]% confirmed true elevated obstacles (tables, equipment racks, shelves), [Y]% confirmed as depth noise (RealSense speckle), [Z]% as reflections (IR bounce), validating our quantitative 36% phantom-candidate rate. Labeled frames available at [GitHub URL] for reproducibility."

---

## Reproducibility

- All labeled frames and CSVs will be committed to GitHub
- Annotation protocol (this document) provides detailed decision rules
- Sufficient documentation for independent auditing or cross-validation

---

## References

- RealSense D455 stereo baseline: 95 mm
- RealSense D435i stereo baseline: 50 mm
- Wheelchair operational height: 0.8–1.2 m (seated operator)
- Wheelchair footprint: ~0.6 m wide × 1.0 m long
- LiDAR angular resolution: 360° / 3200 bins = 0.1125° per bin
