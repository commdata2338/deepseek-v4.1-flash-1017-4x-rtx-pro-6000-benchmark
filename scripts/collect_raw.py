"""Copy the raw benchmark output into raw/, without private details of the test host.

  python3 scripts/collect_raw.py --results DIR --state DIR --kk-results DIR --kk-dir DIR --glm-paper DIR \
      [--only deepseek-v4.1-flash-1017]

The script copies only result files (tool output, probe output, logs). It replaces host paths with placeholders,
then it examines each copied file. A file that still contains a private pattern is not kept: the script writes its
name to raw/NOT_INCLUDED.txt and stops with an error, so that a person can decide. The GLM-5.3-Flash files come from
raw/ of the GLM paper, which is public already, and are copied with no change.

Roots: --results and --state are the result and state folders of the SGLang test server (they also hold the
paper-suite folders of the Karmic Kraken runs, which the same test tools wrote); --kk-results and --kk-dir are the
result folder and the folder of the Karmic Kraken test server. --only copies the entries under one raw/ folder.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
KK, SG = "deepseek-v4.1-flash-1017", "deepseek-v4.1-flash-1017-sglang"
# (folder in raw/, root, folder under the root, summaries only)
FOLDERS = [
    # The main run: DeepSeek-V4.1-Flash-1017 on Karmic Kraken (vLLM fork), 17 October 2026, 19:58 to 21:23
    (f"{KK}/8-slots", "results", "paper-suite-20261017/deepseek-1017-kk", False),
    (f"{KK}/8-slots-round2", "results", "paper-suite-20261017/deepseek-1017-kk-r2", False),
    (f"{KK}/16-slots", "results", "paper-suite-20261017/deepseek-1017-kk-16", False),
    (f"quality-pi/{KK}", "results", "quality-pi-20261017/deepseek-1017-kk-full-c8", False),
    (f"{KK}/more-tests/qualification-features", "kk-results", "20261018T005941Z-e0b7f9e4", True),
    (f"{KK}/more-tests/qualification-integrity", "kk-results", "20261018T010014Z-cba4b1ad", True),
    (f"{KK}/more-tests/qualification-greedy", "kk-results", "20261018T010039Z-3079aca5", True),
    (f"{KK}/more-tests/qualification-image", "kk-results", "20261018T010040Z-7d90ca6f", True),
    (f"{KK}/more-tests/qualification-speed", "kk-results", "20261018T010224Z-10f41710", True),
    (f"{KK}/more-tests/agent-shapes", "results", "shapes-deepseek-20261017T204606", True),
    # The SGLang run of the same weights, 17 October 2026, 15:13 to 17:34 (Appendix F)
    (SG, "results", "paper-suite-20261017/deepseek-1017-ram", False),
    (f"quality-pi/{SG}", "results", "quality-pi-20261017/deepseek-1017-full-c8", False),
    (f"{SG}/more-tests/qualification-features", "results", "20261017T201906Z-6e4145de", True),
    (f"{SG}/more-tests/qualification-integrity", "results", "20261017T201912Z-a492c9a3", True),
    (f"{SG}/more-tests/qualification-greedy", "results", "20261017T202000Z-daf80205", True),
    (f"{SG}/more-tests/qualification-image", "results", "20261017T202002Z-7b69e27f", True),
    (f"{SG}/more-tests/qualification-speed", "results", "20261017T202013Z-a1e0da2d", True),
    (f"{SG}/more-tests/soak-60-minutes", "results", "20261017T203406Z-3355f245", True),
    (f"{SG}/more-tests/agent-shapes-glm-5.3-flash", "results", "shapes-glm-20261017T150840", True),
    (f"{SG}/more-tests/agent-shapes-deepseek-v4.1-flash-1017", "results", "shapes-deepseek-20261017T152917", True),
    (f"{SG}/more-tests/single-request-recipe-matrix", "state", "matrix-20261017T213540", True),
]
# (file in raw/, root, file under the root)
SINGLE = [
    (f"{KK}/cold-requests/burst-deepseek-1017-kk.jsonl", "results", "paper-suite-20261017/burst/burst-deepseek-1017-kk.jsonl"),
    (f"{KK}/cold-requests/burst-deepseek-1017-kk-16.jsonl", "results",
     "paper-suite-20261017/burst/burst-deepseek-1017-kk-16.jsonl"),
    (f"{KK}/cold-requests/burst-requests-deepseek-1017.json", "results",
     "paper-suite-20261017/burst/burst-requests-deepseek-1017.json"),
    (f"{KK}/server-log-facts-8-slots.json", "kk-results", "server-log-facts-kk8-20261017.json"),
    (f"{KK}/server-log-facts-16-slots.json", "kk-results", "server-log-facts-kk16-final-20261017.json"),
    (f"{KK}/server-log-facts-tuning.json", "kk-results", "server-log-facts-kk-tuning-start2-20261017.json"),
    (f"{KK}/window-monitor.csv", "kk-results", "monitor-window-kk-20261017.csv"),
    # One check of the Engram tables at 20:15 (Appendix A): the command ran in the test session and wrote no file;
    # this file keeps its command and output from the record of that session (version 1.1)
    (f"{KK}/engram-residency-check.json", "kk-results", "engram-residency-check-20261017.json"),
    (f"{KK}/batch-schema-check.json", "kk-results", "batch-schema-20261017T200045.json"),
    (f"{KK}/smoke.log", "kk-results", "smoke-20261017.log"),
    (f"{KK}/launcher-config-8-slots.json", "kk-dir", "checks/print-config-2.txt"),
    (f"{KK}/launcher-config-16-slots.json", "kk-dir", "checks/print-config-16.txt"),
    (f"{KK}/weights-verification.json", "results", "verification-1017.json"),
    (f"{KK}/more-tests/api-feature-probes.json", "results", "probe-lab-features-20261017T200216.json"),
    (f"{SG}/cold-requests/burst-deepseek-1017-ram.jsonl", "results", "paper-suite-20261017/burst/burst-deepseek-1017-ram.jsonl"),
    (f"{SG}/cold-requests/burst-deepseek-1017-ram-d1017.jsonl", "results",
     "paper-suite-20261017/burst/burst-deepseek-1017-ram-d1017.jsonl"),
    (f"{SG}/cold-requests/burst-requests-deepseek-1017.json", "results",
     "paper-suite-20261017/burst/burst-requests-deepseek-1017.json"),
    (f"{SG}/engram-residency.jsonl", "results", "engram-residency.jsonl"),
    (f"{SG}/server-log-facts.json", "results", "server-log-facts-20261017T173400.json"),
    (f"{SG}/window-monitor.csv", "results", "monitor-window-20261017.csv"),
    (f"{SG}/weights-verification.json", "results", "verification-1017.json"),
    (f"{SG}/server-start.log", "results", "up.log"),
    (f"{SG}/more-tests/api-feature-probes.json", "results", "probe-lab-features-20261017T163432.json"),
]
GLM = {  # folder in raw/: folder under raw/ of the GLM paper (version 1.5)
    "glm-5.3-flash-r38/16-slots-policy-aligned": "session6/jovian-judgement-r38-16-slots-policy-aligned",
    "glm-5.3-flash-r38/8-slots": "session6/jovian-judgement-r38-8-slots",
    "glm-5.3-flash-r38/cold-requests": "session6/cold-requests",
    "quality-pi/glm-5.3-flash-r38": "quality-pi/jovian-judgement-r38",
}
KEEP = re.compile(r"\.(json|jsonl|log|txt|csv|md)$")
SKIP = re.compile(r"^(load\.log|warmup.*|wave-check.*|run\.lock)$")  # helper output that holds no result
REPLACE = [  # longest first
    (re.compile(r"/home/[A-Za-z0-9_.-]+/models/[A-Za-z0-9_.-]+/results"), "<results-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/models/DeepSeek-V4\.1-Flash-1017"), "<weights-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/models/[A-Za-z0-9_.-]+"), "<serving-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/\.cache/[A-Za-z0-9_.-]+"), "<cache-dir>"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "<home>"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*"), "<data-dir>"),
    (re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d+\.\d+(:\d+)?"), "<container-address>"),
    # SGLang's server information gives the API key of the server, in its fields and in its raw answer text
    (re.compile(r'(api_key\\?"\s*:\s*\\?")[^"\\]+'), r"\1<removed>"),
    # nvidia-smi samples give the serial UUID of each GPU
    (re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<gpu-uuid>"),
]
PRIVATE = re.compile(r"/home/|/media/|@[A-Za-z0-9-]+\.(com|net|org)|\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b"
                     r"|GPU-[0-9a-f]{8}-|api_key\\?\"\s*:\s*\\?\"(?!<removed>)[^\"\\]", re.I)
SUMMARY = re.compile(r"^(summary\.json|.*-summary\.json|server-info\.json|models\.json|resources\.jsonl|TABLE\.md|SPEED\.md|.*\.log)$")


def clean(text):
    for pattern, placeholder in REPLACE:
        text = pattern.sub(placeholder, text)
    return text


def files(folder, summaries_only=False):
    return [p for p in sorted(folder.iterdir()) if p.is_file() and KEEP.search(p.name) and not SKIP.match(p.name)
            and (not summaries_only or SUMMARY.match(p.name))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("results", "state", "kk-results", "kk-dir", "glm-paper"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--only", help="copy only the entries under this folder of raw/")
    parser.add_argument("--private-words", default="", help="comma-separated words that must not occur (host, user)")
    parser.add_argument("--drop-lines", default="", help="a regular expression: lines of the start log to leave out")
    parser.add_argument("--replace-phrase", action="append", default=[], help="OLD::NEW, a private phrase to replace")
    args = parser.parse_args()
    roots = {"results": args.results, "state": args.state, "kk-results": args.kk_results, "kk-dir": args.kk_dir}
    words = [w for w in args.private_words.split(",") if w]
    drop = re.compile(args.drop_lines, re.I) if args.drop_lines else None
    phrases = [tuple(item.split("::", 1)) for item in args.replace_phrase]
    wanted = (lambda name: args.only is None or name == args.only or name.startswith(args.only + "/")
              or name == f"quality-pi/{args.only}")
    pairs = [(RAW / name, roots[root] / source) for name, root, source in SINGLE if wanted(name)]
    for name, root, folder, summaries in FOLDERS:
        if not wanted(name):
            continue
        if not (roots[root] / folder).is_dir():
            print(f"no folder for {name}: {folder}")
            continue
        pairs += [(RAW / name / path.name, path) for path in files(roots[root] / folder, summaries)]
    bad, count = [], 0
    for target, source in pairs:
        if not source.is_file():
            print(f"no file for {target.relative_to(ROOT)}")
            continue
        text = clean(source.read_text(encoding="utf-8", errors="replace"))
        for old, new in phrases:
            text = text.replace(old, new)
        if drop and target.name == "server-start.log":   # keep the lines of the server, not those of other host jobs
            text = "".join(line for line in text.splitlines(keepends=True) if not drop.search(line))
        hits = [w for w in words if re.search(re.escape(w), text, re.I)]
        if PRIVATE.search(text) or hits:
            bad.append(str(target.relative_to(ROOT)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        count += 1
    if args.only is None:
        for name, folder in GLM.items():
            source = args.glm_paper / "raw" / folder
            for path in files(source):
                target = RAW / name / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                count += 1
    print(f"copied {count} files into raw/")
    (RAW / "NOT_INCLUDED.txt").write_text(("\n".join(bad) if bad else "none") + "\n")
    if bad:
        print("NOT INCLUDED (private pattern found):", *bad, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
