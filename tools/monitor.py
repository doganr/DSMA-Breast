import subprocess
import time
import os
import sys
import shutil
import argparse

def get_tmux_output(session_name):
    try:
        subprocess.run(['tmux', 'resize-window', '-t', session_name, '-x', '500', '-y', '50'], 
                      capture_output=True, timeout=2)
        subprocess.run(['tmux', 'resize-pane', '-t', session_name, '-x', '500', '-y', '50'], 
                      capture_output=True, timeout=2)
        result = subprocess.run(['tmux', 'capture-pane', '-t', session_name, '-p', '-J', '-S', '-3000'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            raw = result.stdout
            cleaned_lines = []
            for line in raw.split('\n'):
                parts = line.split('\r')
                for p in parts:
                    p = p.strip()
                    if p:
                        cleaned_lines.append(p)
            
            if not cleaned_lines:
                return ["No output yet or session is booting up..."]
            
            # For each "Fold X|Y" collect all versions and keep the longest (most complete)
            import re
            best_fold_lines = {}  # key: "Fold X|Y" -> longest line
            active_lines = []

            for line in cleaned_lines:
                fold_match = re.search(r'(Fold\s+\d+\|\d+)', line)
                if fold_match and '└─' not in line:
                    fold_key = fold_match.group(1)
                    if fold_key not in best_fold_lines or len(line) > len(best_fold_lines[fold_key]):
                        best_fold_lines[fold_key] = line
                elif re.search(r'Fold\s+\d+ best:', line):
                    # "Fold X best:" summary lines are independent of tqdm and always complete
                    active_lines.append(line)
                elif '└─' in line:
                    active_lines.append(line)
                elif 'Reporting' in line or 'Done!' in line or 'Error' in line:
                    active_lines.append(line)

            # Sort fold rows in numerical order (Fold 1, 2, 3, ...)
            sorted_folds = sorted(best_fold_lines.items(), key=lambda x: x[0])
            fold_display = [v for _, v in sorted_folds]

            # Final display: fold summaries followed by the most recent active lines
            result_lines = fold_display + active_lines[-3:]
            return result_lines if result_lines else cleaned_lines[-5:]
        else:
            return [f"Session '{session_name}' is not currently running."]
    except Exception as e:
        return [f"Error reading tmux: {e}"]

def get_gpu_status():
    try:
        # Query nvidia-smi for specific GPU stats without units to parse them easily
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 6:
                    idx, name, temp, util, mem_used, mem_total = parts
                    gpu_stats.append(f"GPU {idx}: {name} | Temp: {temp}°C | Util: {util}% | VRAM: {mem_used}MB / {mem_total}MB")
            return gpu_stats if gpu_stats else ["No GPU stats found."]
        return ["Nvidia driver or nvidia-smi not available."]
    except Exception as e:
        return [f"Error getting GPU info: {e}"]

def get_all_tmux_sessions():
    """Fetches all active tmux session names."""
    try:
        result = subprocess.run(['tmux', 'list-sessions', '-F', '#S'], capture_output=True, text=True)
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return []
    except Exception:
        return []

def main():
    parser = argparse.ArgumentParser(description="Live Training Monitor for Tmux Sessions")
    parser.add_argument('sessions', nargs='*', default=None,
                        help="Names of the tmux sessions to monitor (space separated). Leave empty to monitor ALL active sessions.")
    args = parser.parse_args()
    
    print("Starting Training Monitor...")
    try:
        while True:
            term_width = shutil.get_terminal_size().columns
            os.system('clear')
            print("=" * term_width)
            title = "DSMA-Breast Training Live Monitor"
            print(title.center(term_width))
            print(f"Latest update: {time.strftime('%H:%M:%S')}".center(term_width))
            if args.sessions:
                sessions_to_monitor = args.sessions
            else:
                sessions_to_monitor = get_all_tmux_sessions()
                if not sessions_to_monitor:
                    print("\nNo active tmux sessions found. Waiting... (Press Ctrl+C to exit)\n")
                    time.sleep(5)
                    continue
            print("=" * term_width)
            
            for session in sessions_to_monitor:
                print(f"\n>>> TMUX SESSION: [ {session.upper()} ]")
                print("-" * term_width)
                lines = get_tmux_output(session)
                for line in lines:
                    if line:
                        print(f"  {line}")
                print("=" * term_width)
                
            print(f"\n>>> SYSTEM STATUS: [ GPU ]")
            print("-" * term_width)
            gpu_lines = get_gpu_status()
            for g_line in gpu_lines:
                print(f"  {g_line}")
            print("=" * term_width)
                
            print("\n(Updating every 5 seconds. Press Ctrl+C to exit monitor.)")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nExiting monitor. The training is still running in the background!")

if __name__ == "__main__":
    main()
