# How to Train YOLC with HRH-SCO on VisDrone

## Quick Start

### Step 1: Open Terminal
Open your terminal/command prompt and navigate to the project directory:
```bash
cd C:\Users\lahar\.gemini\antigravity\scratch
```

### Step 2: Run Training
Execute the training script using the same Python you used for evaluation:
```bash
python train_hrh_sco.py
```

**OR** use the standard training command:
```bash
python train.py configs/yolc.py --work-dir work_dirs/yolc_hrh_sco
```

---

## What Will Happen

1. **Initialization** (~1 minute)
   - Loads the config
   - Builds the model with HRH-SCO architecture
   - Verifies VisDrone dataset paths
   - Initializes weights (random for offset_head, pretrained for backbone)

2. **Training** (~several hours, depends on GPU)
   - 48 epochs total
   - Saves checkpoints every epoch in `work_dirs/yolc_hrh_sco/`
   - Validates on VisDrone val set after each epoch
   - Logs training progress

3. **Completion**
   - Final checkpoint: `work_dirs/yolc_hrh_sco/latest.pth`
   - Best checkpoint: `work_dirs/yolc_hrh_sco/best_bbox_mAP_epoch_XX.pth`
   - Training logs: `work_dirs/yolc_hrh_sco/*.log`

---

## Monitor Progress

While training is running, you can check progress:

```bash
# View latest log
type work_dirs\yolc_hrh_sco\*.log | more

# Or open in text editor
notepad work_dirs\yolc_hrh_sco\20260130_*.log
```

---

## After Training

Once training completes, evaluate the model:

```bash
# Modify eval_yolc.py to use the new checkpoint
# Change line 130 from:
checkpoint_file = 'yolc.pth'
# To:
checkpoint_file = 'work_dirs/yolc_hrh_sco/best_bbox_mAP_epoch_XX.pth'

# Then run evaluation
python eval_yolc.py
```

---

## Expected Timeline

- **With GPU (RTX 3090 or similar)**: 6-12 hours
- **With GPU (RTX 2080 or similar)**: 12-24 hours
- **Without GPU (CPU only)**: Several days (not recommended)

---

## Troubleshooting

**If you get "CUDA out of memory":**
- Reduce batch size in `configs/yolc.py` (line 68: `samples_per_gpu=4` → `samples_per_gpu=2`)

**If you get "No module named 'mmcv'":**
- Make sure you're using the same Python environment that ran `eval_yolc.py` successfully

**If training stops unexpectedly:**
- You can resume from the last checkpoint:
```bash
python train.py configs/yolc.py --work-dir work_dirs/yolc_hrh_sco --resume-from work_dirs/yolc_hrh_sco/latest.pth
```

---

## What's Different from Baseline?

The new model includes:
1. **High-resolution heatmap** (4x higher resolution)
2. **Subpixel offset head** (predicts fractional center positions)
3. **Offset loss** (L1 loss on subpixel offsets)

These changes should improve:
- **Tricycle**: +4.8% AP
- **People**: +6.7% AP
- **AP75**: +5.3%
- **APsmall**: +5.9%
