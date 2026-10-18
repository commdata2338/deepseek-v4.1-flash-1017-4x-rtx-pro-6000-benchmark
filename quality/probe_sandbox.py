"""Run capability and negative-connectivity checks in the actual task sandbox."""
import argparse
from pathlib import Path
import subprocess

from common import ROOT, container_flags, unique_name
from sandbox import NETWORK, GATEWAY, PORT, verify

PROBE = r'''
import json, os, socket, urllib.request
from pathlib import Path
results = {}
url = "http://GATEWAY:PORT/v1/models"
with urllib.request.urlopen(url, timeout=8) as r:
    results["relay_models_status"] = r.status
    results["model_present"] = any(m["id"] == "glm-5.3-flash" for m in json.load(r)["data"])
assert results["model_present"]
for name, host, port in [
    ("internet_https", "1.1.1.1", 443),
    ("internet_dns", "8.8.8.8", 53),
    ("host_engine_direct", "GATEWAY", 8801),
    ("host_other_engine_direct", "GATEWAY", 8020),
    ("host_ssh", "GATEWAY", 22),
    ("container_loopback_engine", "127.0.0.1", 8801),
    ("other_bridge_gateway", "172.17.0.1", 8801),
]:
    try:
        with socket.create_connection((host, port), timeout=2): pass
    except OSError:
        results[name] = "blocked"
    else:
        raise AssertionError(name + " unexpectedly reachable")
try:
    socket.getaddrinfo("example.com", 443)
except OSError:
    results["external_dns_resolution"] = "blocked"
else:
    raise AssertionError("external DNS resolution unexpectedly available")
try:
    urllib.request.urlopen("http://GATEWAY:PORT/not-an-api", timeout=3)
except urllib.error.HTTPError as e:
    assert e.code == 403
    results["relay_non_api"] = "403"
assert os.getuid() != 0
results["uid_nonroot"] = True
for path in ["/probe-forbidden", "/opt/pi-config/probe-forbidden"]:
    try:
        Path(path).write_text("bad")
    except OSError:
        pass
    else:
        raise AssertionError("root filesystem writable")
results["root_read_only"] = True
results["memory_max"] = Path("/sys/fs/cgroup/memory.max").read_text().strip()
results["cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
results["pids_max"] = Path("/sys/fs/cgroup/pids.max").read_text().strip()
print(json.dumps(results, indent=2))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True, help="empty disposable folder kept after probe")
    args = p.parse_args()
    verify()
    args.work.mkdir(parents=True, exist_ok=True)
    if list(args.work.iterdir()):
        p.error("probe work folder must be empty")
    script = PROBE.replace("GATEWAY", GATEWAY).replace("PORT", str(PORT))
    cmd = container_flags(unique_name("probe"), NETWORK, args.work)
    subprocess.run(cmd + ["--entrypoint", "python", "bench-pi-agent:0.86.1", "-c", script],
                   check=True, timeout=45)


if __name__ == "__main__":
    main()
