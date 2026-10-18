"""Prepare an ephemeral pi directory, then exec the unmodified upstream CLI."""
import json
import os
from pathlib import Path
import sys

target = Path(os.environ["PI_CODING_AGENT_DIR"])
target.mkdir(parents=True, exist_ok=True)
models = json.loads(Path("/opt/pi-config/models.json").read_text())
for provider in models["providers"].values():
    provider["baseUrl"] = os.environ["QUALITY_RELAY_URL"]
(target / "models.json").write_text(json.dumps(models))
(target / "settings.json").write_bytes(Path("/opt/pi-config/settings.json").read_bytes())
os.execvp("pi", ["pi", *sys.argv[1:]])
