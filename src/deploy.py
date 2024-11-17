import sys
import subprocess
from pathlib import Path

from utils.common import readEnv

_,_,_,_,deploymentType,_,_,port= readEnv()

def run_api(port=8000):
    print("Starting FastAPI server...")
    subprocess.run(["uvicorn", "src.api:app", "--reload", "--port", str(port), "--host", "0.0.0.0"])

def run_batch():
    print("Starting batch prediction scheduler...")
    subprocess.run(["python", str(Path("src/batchPredict.py"))])

if __name__ == "__main__":
    mode = deploymentType.lower()
    if mode == "api":
        run_api(port)
    elif mode == "batch":
        run_batch()
    else:
        print("Error: Unrecognized mode. Use 'api' or 'batch'.")
        sys.exit(1)
