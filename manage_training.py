import os
import subprocess
import glob

# Paths
WORK_DIR = './work_dirs/yolc_hrh_sco_100pct'
CONFIG = './configs/yolc_hrh_sco_100pct.py'
PYTHON_PATH = '.;./mmcv_source'
PYTHON_EXE = r'C:\Users\lahar\Desktop\python files\python.exe'

def get_latest_checkpoint():
    if not os.path.exists(WORK_DIR):
        return None
    checkpoints = glob.glob(os.path.join(WORK_DIR, 'epoch_*.pth'))
    if not checkpoints:
        return None
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return checkpoints[-1]

def run_training():
    latest = get_latest_checkpoint()
    
    cmd = [PYTHON_EXE, 'train.py', CONFIG]
    
    if latest:
        print(f"--- FOUND PREVIOUS PROGRESS: {latest} ---")
        print(f"--- RESUMING TRAINING FROM WHERE YOU LEFT OFF ---")
        cmd.extend(['--resume-from', latest])
    else:
        print(f"--- STARTING NEW TRAINING ON 100% DATA ---")
        print(f"--- LOADING INITIAL WEIGHTS FROM PREVIOUS 25% SUCCESS ---")

    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = PYTHON_PATH
    
    print(f"--- COMMAND: {' '.join(cmd)} ---")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    except KeyboardInterrupt:
        print("\n--- TRAINING PAUSED BY USER ---")
        print("Your progress is safe. Run this script tomorrow to resume from the last saved epoch!")

if __name__ == '__main__':
    run_training()
