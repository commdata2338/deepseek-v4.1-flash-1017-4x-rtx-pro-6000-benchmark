"""Create one isolated bridge and host rules restricted to that bridge."""
import argparse
import json
import subprocess

NETWORK = "quality-sandbox"
BRIDGE = "qbench0"
SUBNET = "192.168.254.0/24"
GATEWAY = "192.168.254.1"
PORT = 18080
CHAIN = "QUALITY_SANDBOX"


def command(*args, check=True):
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=30)


def rules():
    return [
        (CHAIN, "-d", GATEWAY, "-p", "tcp", "--dport", str(PORT), "-j", "ACCEPT"),
        (CHAIN, "-j", "REJECT"),
        ("INPUT", "-i", BRIDGE, "-j", CHAIN),
        ("FORWARD", "-i", BRIDGE, "-j", "DROP"),
    ]


def inspect_network():
    result = command("docker", "network", "inspect", NETWORK, check=False)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def validate(network):
    if not network or not (network["Internal"] and not network["EnableIPv6"]
        and network["Labels"].get("quality.harness") == "pi-0.86.1"
        and network["Options"].get("com.docker.network.bridge.name") == BRIDGE
        and network["Options"].get("com.docker.network.bridge.enable_icc") == "false"
        and network["IPAM"]["Config"] == [{"Subnet": SUBNET, "Gateway": GATEWAY}]):
        raise RuntimeError("sandbox network missing or differs from required isolation")


def verify():
    validate(inspect_network())
    for rule in rules():
        command("sudo", "-n", "iptables", "-w", "5", "-C", *rule)
    expected = ["-N " + CHAIN,
                f"-A {CHAIN} -d {GATEWAY}/32 -p tcp -m tcp --dport {PORT} -j ACCEPT",
                f"-A {CHAIN} -j REJECT --reject-with icmp-port-unreachable"]
    if command("sudo", "-n", "iptables", "-w", "5", "-S", CHAIN).stdout.splitlines() != expected:
        raise RuntimeError("sandbox chain contains unexpected rules or ordering")
    for chain, target in (("INPUT", CHAIN), ("FORWARD", "DROP")):
        lines = command("sudo", "-n", "iptables", "-w", "5", "-S", chain).stdout.splitlines()
        first = next((line for line in lines if line.startswith("-A ")), "")
        if first != f"-A {chain} -i {BRIDGE} -j {target}":
            raise RuntimeError("sandbox hook must precede other host rules; inspect firewall ordering")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["setup", "check"])
    args = p.parse_args()
    if args.action == "setup":
        if inspect_network() is None:
            command("docker", "network", "create", "--internal", "--driver", "bridge",
                    "--subnet", SUBNET, "--gateway", GATEWAY,
                    "--opt", "com.docker.network.bridge.name=" + BRIDGE,
                    "--opt", "com.docker.network.bridge.enable_icc=false",
                    "--label", "quality.harness=pi-0.86.1", NETWORK)
        validate(inspect_network())
        if command("sudo", "-n", "iptables", "-w", "5", "-S", CHAIN, check=False).returncode:
            command("sudo", "-n", "iptables", "-w", "5", "-N", CHAIN)
        for rule in rules():
            if command("sudo", "-n", "iptables", "-w", "5", "-C", *rule, check=False).returncode:
                if rule[0] in ("INPUT", "FORWARD"):
                    command("sudo", "-n", "iptables", "-w", "5", "-I", rule[0], "1", *rule[1:])
                else:
                    command("sudo", "-n", "iptables", "-w", "5", "-A", *rule)
    verify()
    print(json.dumps({"network": NETWORK, "internal": True, "gateway": GATEWAY,
                      "only_host_port": PORT, "peer_forwarding": False, "firewall": "verified"}))


if __name__ == "__main__":
    main()
