import glob, subprocess
print(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
                     "pgrep -af '[p]wjax.ppo' | cut -c1-150", shell=True, capture_output=True, text=True).stdout)
for f in sorted(glob.glob("/content/runs/*/stdout.log")):
    print("==", f)
    print("".join(open(f).readlines()[-4:])[-2500:])
