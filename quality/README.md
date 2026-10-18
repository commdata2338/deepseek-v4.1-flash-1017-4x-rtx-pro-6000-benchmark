# Task accuracy through the pi agent

This directory contains the harness for the quality run of the paper (section 7, Appendix B.3, and Appendix D). The harness gives the same Python tasks to each
engine through an agent. Then it grades the files that the agent wrote.

**The agent is pi 0.86.1.** It is the npm package `@earendil-works/pi-coding-agent` with no changes and no
extensions. The tasks are HumanEval+ (164 tasks) and MBPP+ (378 tasks) from EvalPlus 0.3.1. The grader uses the
tests of EvalPlus.

**Do not compare these scores with scores from single requests.** The agent can run Python, read its errors, and
write the file again. Each task gets one attempt of the agent. One attempt can contain many model calls.

## Settings

| Setting | Value |
|---|---|
| API | OpenAI chat completions with a stream |
| Model name | `glm-5.3-flash` |
| Thinking | on, with `reasoning_effort: "max"` in each request |
| Temperature | 0 |
| Maximum output for each model call | 32,768 tokens, thoughts included |
| Tools of the agent | `read`, `bash`, `edit`, `write` (the tools of pi) |
| Time limit for each task | 1,200 s |
| Concurrent tasks | 8 |
| Sessions, extensions, skills, context files | none |
| Compaction and retry in pi | off |

The system prompt and the tool descriptions are those of pi. Each task gets the same instruction:

    Read PROBLEM.md. Implement the requested function in solution.py. You may run Python to check your work.
    Stop when solution.py is complete.

`PROBLEM.md` contains only the prompt of the task. The agent does not get the tests.

## The sandbox

- Each task has its own container, with 2 GiB of memory, 1 CPU, and 128 processes.
- The container has a read-only root file system and no capabilities. Its user is not root.
- The container can write only to the directory of its task.
- The container is on a Docker network with no route to other networks (`quality-sandbox`).
- A relay on the host listens on the gateway address of that network. It sends requests only to the port of the
  engine on the loopback address.
- The relay permits `GET /v1/models` and `POST /v1/chat/completions` only. It does not change the request.
- `sandbox.py setup` makes the network and one firewall chain. The chain applies only to the bridge of that network.
  The setup uses `sudo` for `iptables`.
- The grader containers have no network.

## Commands

Build the two images and make the network:

    docker build -f Dockerfile.agent -t bench-pi-agent:0.86.1 .
    docker build -f Dockerfile.grader -t bench-evalplus:0.3.1 .
    python3 -B sandbox.py setup
    python3 -B -m unittest test_harness

Start the relay for the port of the engine, and keep it in operation during the run:

    python3 -B relay.py --engine-port 8020 --log runs/window/relay.jsonl

Make sure that a container gets only the relay:

    python3 -B probe_sandbox.py --work runs/window/probe-work

Run the tasks. Use `--provider tensorfold` or `--provider vllm`. The two values send the same requests:

    python3 -B run_tasks.py --endpoint http://192.168.254.1:18080/v1 --provider tensorfold --dataset both \
      --concurrency 8 --wall-limit 1200 --output runs/tensorfold-full-c8

Grade the run, and compare two runs:

    python3 -B grade.py --samples runs/tensorfold-full-c8/samples.jsonl --output runs/tensorfold-full-c8-grade
    python3 -B compare.py --a runs/tensorfold-full-c8-grade/graded.json --b runs/vllm-full-c8-grade/graded.json

If you run the same `run_tasks.py` command again, it continues a run that stopped. It does not do a completed task
again.

Put the run into one directory for `scripts/build_data.py` of the paper:

    python3 -B collect_run.py --run runs/tensorfold-full-c8 --grade runs/tensorfold-full-c8-grade \
      --relay runs/window/relay.jsonl --out collected/tensorfold-full-c8

`collect_run.py` copies the records, the solutions, the grades, and the relay log. It also makes three files from
the task directories:

- `timing.json` gives the start and the end of the run. The builder uses them to find the model calls of the run in
  the relay log.
- `first_calls.jsonl` gives the prompt tokens of the first model call of each task.
- `exceptions.jsonl` gives the model calls and the last tool call of each task that did not complete.

The builder reads the directory names `tensorfold-release-full-c8`, `tensorfold-fast-full-c8`, and `vllm-full-c8`
below `quality-pi-20261004/`.

## What the files contain

| File | Contents |
|---|---|
| `records.jsonl` | For each task: the time, the number of model calls, the tokens that pi counted, the status, and if the time limit stopped the task |
| `samples.jsonl` | For each task: the contents of `solution.py`, in the format of EvalPlus |
| `graded.json` | For each task: pass or fail for the base tests and for the added tests of EvalPlus |
| `tasks/<task>/<attempt>/events.jsonl` | The event stream of pi for the task |

A task with no `solution.py` gets a fail. A task that the time limit stopped gets a grade for the file that it wrote
before the stop. The grader does not change or repair a file.

`plus_pass` in `graded.json` is true only if the base tests and the added tests pass. The paper gives this value as
"all tests".

`compare.py` gives the 95% Wilson interval for each pass rate. It also gives the paired counts and the exact McNemar
test for the tasks where only one of the two runs passes.

## Data sets and licenses

`datasets/` contains the task id, the prompt, and the entry point of each task. EvalPlus 0.3.1 supplied them:
HumanEval+ v0.1.10 and MBPP+ v0.2.0. `datasets/manifest.json` contains their hashes, and the runner stops if a file
is different. The tests stay in the grader image.

- HumanEval is from OpenAI (MIT license).
- MBPP is from Google Research (CC BY 4.0).
- EvalPlus is from the EvalPlus team (Apache License 2.0).

## Limits

- One attempt for each task, and one run for each engine. We did not measure the variation between runs of the same
  engine on Jovian Judgement.
- The result contains the effect of the weights, the engine, and its parser for tool calls. It does not show the
  effect of the quantization alone.
- The two data sets are small Python tasks. They are old. Thus the model possibly saw them before.
- The sandbox was not the subject of a security audit.
