#!/usr/bin/env python3
import os
import sys
import json
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
API_BASE = "https://cloud.lambda.ai/api/v1"
TOKEN = os.environ.get("LAMBDA_API_TOKEN")
if not TOKEN:
    print("Error: Please set LAMBDA_API_TOKEN in your environment.", file=sys.stderr)
    sys.exit(1)
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def list_alive_instances():
    """Return a list of instances with status not 'terminated' or 'terminating'."""
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


def confirm_and_terminate(ids_to_kill):
    """Prompt for confirmation and POST to terminate endpoints."""
    print("The following instances will be terminated:")
    for iid in ids_to_kill:
        print(f"  • {iid}")
    ans = input("Proceed with termination? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborting. No instances were terminated.")
        sys.exit(0)

    payload = {"instance_ids": ids_to_kill}
    resp = requests.post(
        f"{API_BASE}/instance-operations/terminate",
        headers=HEADERS,
        data=json.dumps(payload),
    )
    if resp.status_code != 200:
        print(f"Error from API: {resp.status_code} - {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json().get("data", {})
    terminated = data.get("terminated_instances", [])
    print("Termination request sent. Results:")
    for t in terminated:
        print(f"  • ID: {t.get('id')}, status now: {t.get('status')}")
    print("Done.")


def main():
    alive = list_alive_instances()
    if not alive:
        print("No alive instances to terminate.")
        sys.exit(0)

    # Show names + IDs
    for inst in alive:
        name = inst.get("name", "<no-name>")
        iid = inst.get("id")
        status = inst.get("status")
        print(f"{name} (ID: {iid}, status: {status})")
    ids = [inst.get("id") for inst in alive]
    confirm_and_terminate(ids)


if __name__ == "__main__":
    main()
