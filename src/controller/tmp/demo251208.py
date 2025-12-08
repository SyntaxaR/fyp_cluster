# Used for 2025/12/08 demo only, controller side
# Send folder to worker remote directory /user/pi/fyp/model with scp
# Load model remotely and send inference requests
# Evaluate performance and log results
# Requires sshpass to be installed on controller machine
import subprocess

def send_directory(worker_ip: str, local_dir, remote_dir:str="/home/pi/fyp/model/",username: str="pi", password: str="raspberry"):
    # Transfer with subprocess
    subprocess.run(["sshpass", "-p", password, "scp", "-r", local_dir, f"{username}@{worker_ip}:{remote_dir}"])
    print(f"Sent directory {local_dir} to worker at {worker_ip}:{remote_dir}")
