#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
API_BASE = "https://cloud.lambda.ai/api/v1"
TOKEN = os.environ.get("LAMBDA_API_TOKEN")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
SSH_USER = "ubuntu"  # fixed

if not TOKEN:
    print("Error: Please set LAMBDA_API_TOKEN in your environment.", file=sys.stderr)
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def list_instances():
    """Fetch all instances and return a list of those not terminated/terminating."""
    resp = requests.get(f"{API_BASE}/instances", headers=HEADERS)
    resp.raise_for_status()
    items = resp.json().get("data", [])
    alive = []
    for inst in items:
        status = inst.get("status")
        if status in ("terminated", "terminating"):
            continue
        alive.append(inst)
    return alive


def show_and_select(instances):
    """Print a numbered list of instances and prompt the user to pick one."""
    print("Available instances:")
    for idx, inst in enumerate(instances, start=1):
        iid = inst.get("id")
        name = inst.get("name", "<no-name>")
        status = inst.get("status")
        print(f"  [{idx}] {name}  (ID: {iid}, status: {status})")
    choice = input(f"Select an instance [1–{len(instances)}]: ").strip()
    try:
        i = int(choice) - 1
        if i < 0 or i >= len(instances):
            raise ValueError
    except ValueError:
        print("Invalid selection. Exiting.", file=sys.stderr)
        sys.exit(1)
    return instances[i]


def wait_for_active(instance_id):
    """Poll GET /instances/{id} until status == 'active'."""
    while True:
        resp = requests.get(f"{API_BASE}/instances/{instance_id}", headers=HEADERS)
        resp.raise_for_status()
        inst = resp.json().get("data", {})
        status = inst.get("status")
        print(f"  → Instance {instance_id} status: {status}")
        if status == "active":
            return inst  # contains ip, etc.
        if status in ("terminated", "terminating"):
            print(f"Instance {instance_id} is {status}. Cannot proceed.", file=sys.stderr)
            sys.exit(1)
        time.sleep(5)


def scp_copy(local_dir, remote_ip, remote_dir_name):
    """Use scp to copy local_dir into ~/<remote_dir_name> on the instance."""
    target = f"{SSH_USER}@{remote_ip}:~/{remote_dir_name}"
    print(f"Copying `{local_dir}` → `{target}` …")
    try:
        subprocess.run(
            ["scp", "-r", local_dir, target],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"scp failed: {e}", file=sys.stderr)
        sys.exit(1)


def ssh_and_install(remote_ip, remote_dir_name):
    """SSH into the instance, run installation, then open interactive session."""
    install_cmd = f"cd ~/{remote_dir_name} && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh && pip install --no-deps -e ."
    
    # Add wandb login if API key is available
    if WANDB_API_KEY:
        install_cmd += f" && wandb login {WANDB_API_KEY}"
    
    ssh_install_cmd = ["ssh", f"{SSH_USER}@{remote_ip}", install_cmd]
    print(f"Running on remote: {install_cmd}")
    try:
        subprocess.run(ssh_install_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"SSH/install failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Open interactive SSH session
    print(f"Opening interactive SSH session to {remote_ip}...")
    interactive_cmd = ["ssh", f"{SSH_USER}@{remote_ip}"]
    try:
        subprocess.run(interactive_cmd)
    except subprocess.CalledProcessError as e:
        print(f"SSH session ended: {e}", file=sys.stderr)


def main():
    instances = list_instances()
    if not instances:
        print("No non-terminated instances found. Exiting.", file=sys.stderr)
        sys.exit(1)

    chosen = show_and_select(instances)
    iid = chosen.get("id")
    print(f"Waiting for instance `{iid}` to become active…")
    inst_info = wait_for_active(iid)

    ip = inst_info.get("ip")
    if not ip:
        print(f"Error: No public IP found for instance {iid}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Copy current directory
    local_dir = os.getcwd()
    remote_dir_name = os.path.basename(local_dir.rstrip("/"))
    scp_copy(local_dir, ip, remote_dir_name)

    # SSH and install
    ssh_and_install(ip, remote_dir_name)
    print("SSH session ended.")


if __name__ == "__main__":
    main()
