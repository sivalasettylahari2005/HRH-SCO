import torch
import glob
import os

checkpoints = glob.glob('./work_dirs/yolc_hrh_sco_100pct/*.pth')
print(f"{'File':<30} | {'Epoch':<10} | {'Iter':<10} | {'Timestamp'}")
print("-" * 70)

for ckpt_path in checkpoints:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        meta = ckpt.get('meta', {})
        epoch = meta.get('epoch', 'N/A')
        iter_count = meta.get('iter', 'N/A')
        mtime = os.path.getmtime(ckpt_path)
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{os.path.basename(ckpt_path):<30} | {epoch:<10} | {iter_count:<10} | {ts}")
    except Exception as e:
        print(f"{os.path.basename(ckpt_path):<30} | ERROR: {e}")
