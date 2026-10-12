"""Print a script for `colab exec` that starts a training run in the background on the VM."""
import os

run = os.environ["RUN"]
args = os.environ.get("ARGS", "")
print(f'''import subprocess, os
os.makedirs("/content/runs/{run}", exist_ok=True)
subprocess.Popen("cd /content/pw && nohup python -m pwjax.ppo --out /content/runs/{run} {args} "
                 "> /content/runs/{run}/stdout.log 2>&1 &", shell=True, start_new_session=True)
print("started {run}")
''')
