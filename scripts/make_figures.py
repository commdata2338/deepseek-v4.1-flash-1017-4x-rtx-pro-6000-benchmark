"""Draw the paper's figures from data/*.csv into figures/ (PNG at 300 dpi and SVG).

  python3 scripts/make_figures.py

The style of the GLM-5.3-Flash paper: 7.0 inches wide and not tall, IBM Plex Sans (scripts/fonts/, SIL Open Font
License), no text below 9.5 points, one line below the plot and none at the left, light horizontal lines for the
values, one legend above the panels, panel letters, units in the axis labels, a comma in numbers of four digits.
Two series: DeepSeek-V4.1-Flash-1017 on Karmic Kraken (blue, circles; 16 slots, and 8 slots for the coding tasks)
and GLM-5.3-Flash on Jovian Judgement r38 (orange, squares; the hue of Jovian Judgement in the GLM paper). Markers
have a thin white edge, so that a point stays visible where the two lines meet.

The speed figures 2 to 4 use the style of the speed charts on our results page (the owner, 2026-10-17: "I love the
charts that are in the artifact for speed, can we recreate those in the paper?"): rows with a dot for each
inference stack on one axis from 0, a light bar between the two dots, and each value beside its dot; the prefill
lines on a speed axis from 0 with each value at its point. Figure 2 has four panels in two rows (the owner, the same
day), so it is taller than the other figures. Every plotted value is in a table of the README or in data/*.csv;
where a run has two passes of a test, the mean.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "figures"
FONTS = Path(__file__).resolve().parent / "fonts"
for font in sorted(FONTS.glob("*.ttf")):
    font_manager.fontManager.addfont(str(font))
FAMILY = "IBM Plex Sans" if any(FONTS.glob("IBMPlexSans-*.ttf")) else "DejaVu Sans"
INK, MUTED, RULE, GRID = "#1a1a1a", "#5d5b55", "#8f8d86", "#e7e5df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT = {BLUE: "#1b5aa6", ORANGE: "#b4441a"}   # darker shades of the two colors, for the values that a figure prints
LINK = "#d9d6ce"                               # the light bar between the two dots of a row
WIDTH, SMALL = 7.0, 9.5
DS, DSQ, GLM16, GLM8 = "deepseek_1017_kk_16", "deepseek_1017_kk_8", "glm_r38_16_aligned", "glm_r38_8"
STYLE = {DS: ("DeepSeek-V4.1-Flash-1017 (Karmic Kraken)", BLUE, "o", "-"),
         DSQ: ("DeepSeek-V4.1-Flash-1017, 8 slots", BLUE, "o", "-"),
         GLM16: ("GLM-5.3-Flash r38 (Jovian Judgement)", ORANGE, "s", "-"),
         GLM8: ("GLM-5.3-Flash r38, 8 slots", ORANGE, "D", (0, (4, 2)))}
NOTE = "GLM-5.3-Flash: the GLM-5.3-Flash paper, version 1.5 (5 October 2026). DeepSeek: 17 October 2026."
plt.rcParams.update({"font.family": FAMILY, "font.size": 10.5, "axes.edgecolor": RULE, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5, "axes.titleweight": "medium",
                     "svg.fonttype": "none", "figure.dpi": 100})
COMMA = FuncFormatter(lambda v, _: f"{v:,.0f}")


def rows(name):
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def value(x):
    return None if x in (None, "", "None") else float(x)


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=SMALL)


def legend(fig, runs, y=1.0, kind="line", marker=None):
    """One legend above the panels: a line with its marker, a dot for a figure of dots, or a square for bars."""
    if kind == "bars":
        handles = [Patch(facecolor=STYLE[r][1]) for r in runs]
    elif kind == "dots":
        handles = [plt.Line2D([], [], color=STYLE[r][1], marker="o", linestyle="none", markersize=8,
                              markeredgecolor="white", markeredgewidth=1.0) for r in runs]
    else:
        handles = [plt.Line2D([], [], color=STYLE[r][1], marker=marker or STYLE[r][2], linestyle=STYLE[r][3],
                              linewidth=2.5, markersize=7, markeredgecolor="white", markeredgewidth=1.0)
                   for r in runs]
    fig.legend(handles, [STYLE[r][0] for r in runs], loc="upper left", bbox_to_anchor=(0.075, y), ncol=len(runs),
               frameon=False, fontsize=SMALL, handlelength=2.4 if kind == "line" else 1.0,
               handleheight=1.0 if kind == "bars" else 0.7, columnspacing=1.6)


def inches(ax, text, **text_style):
    """The width of a text in inches, as the figure draws it."""
    probe = ax.text(0, 0, text, fontsize=SMALL, transform=None, **text_style)
    width = probe.get_window_extent(ax.figure.canvas.get_renderer()).width / ax.figure.dpi
    probe.remove()
    return width


def dumbbell(ax, items, runs, fmt, xmax):
    """The rows of the results page: one row for each item from the top, a dot for each run on one axis from 0, a
    light bar between the dots, and each value beside its dot. The higher value is at the right of its dot. The lower
    value is at the left of its dot, or between the dots if the left has no room, or above its dot if neither has
    room. An item is (label, {run: value}), or (heading, None) for a group heading. Call it after subplots_adjust:
    the room for a value comes from the size of the axes."""
    rows_y = list(range(len(items) - 1, -1, -1))
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(items) - 0.1)
    per_unit = ax.get_position().width * ax.figure.get_figwidth() / xmax   # inches for one unit of the axis
    gap = 5 / 72                                                           # inches between a dot and its value
    for y, (label, values) in zip(rows_y, items):
        if values is None:
            continue
        points = sorted((values[r], r) for r in runs if values.get(r) is not None)
        if len(points) > 1:
            ax.plot([points[0][0], points[-1][0]], [y, y], color=LINK, linewidth=5, solid_capstyle="round", zorder=1)
        for j, (v, run) in enumerate(points):
            color = STYLE[run][1]
            ax.plot([v], [y], marker="o", markersize=8, color=color, markeredgecolor="white", markeredgewidth=1.0,
                    linestyle="none", zorder=3)
            text = fmt(v)
            width = inches(ax, text, fontweight="medium")
            box = None
            if j == len(points) - 1:
                offset, ha, va = (5, 0), "left", "center"
            elif v * per_unit >= width + gap + 0.03:
                offset, ha, va = (-5, 0), "right", "center"
            elif (points[-1][0] - v) * per_unit >= width + 2 * gap + 0.04:   # on the bar: a white ground
                offset, ha, va = (5, 0), "left", "center"
                box = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none")
            else:
                offset, ha, va = (0, 6), "center", "bottom"
            ax.annotate(text, (v, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=SMALL,
                        fontweight="medium", color=TEXT[color], zorder=4, bbox=box)
    ax.set_yticks(rows_y, [label for label, _ in items])
    for tick, (label, values) in zip(ax.get_yticklabels(), items):
        if values is None:
            tick.set_fontweight("medium")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=SMALL)
    ax.tick_params(axis="y", labelcolor=INK)


def note(fig, text=NOTE):
    fig.text(0.075, 0.012, text, fontsize=SMALL, color=MUTED, ha="left", va="bottom")


def letter(ax, text):
    ax.set_title(text, loc="left", fontsize=10.5, color=INK, pad=6)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"figures/{name}.png and .svg")


def figure1():
    """(a) sparkDash decode at 8 requests, by output type; (b) HumanEval+ and MBPP+, all tests."""
    decode = {(r["run"], r["output_type"], int(r["concurrency"])): value(r["total_tok_s"]) for r in rows("decode_sparkdash.csv")}
    quality = {(r["run"], r["data_set"]): r for r in rows("quality_pi_summary.csv")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    types = ["prose", "code", "structured", "json"]
    x = range(len(types))
    for offset, run in ((-0.2, DS), (0.2, GLM16)):
        vals = [decode[(run, t, 8)] for t in types]
        a.bar([i + offset for i in x], vals, width=0.38, color=STYLE[run][1])
        for i, v in zip(x, vals):
            a.text(i + offset, v + 25, f"{v:,.0f}", ha="center", va="bottom", fontsize=SMALL, color=INK)
    a.set_ylim(0, max(decode[(r, t, 8)] for r in (DS, GLM16) for t in types) * 1.16)
    a.set_xticks(list(x), ["Prose", "Code", "Count task", "JSON"])
    a.set_ylabel("Total tokens/s")
    a.yaxis.set_major_formatter(COMMA)
    letter(a, "a  Decode at 8 requests (sparkDash, 16 slots)")
    sets = ["HumanEval+", "MBPP+", "both"]
    for offset, run in ((-0.2, DSQ), (0.2, GLM8)):
        vals = [float(quality[(run, s)]["all_tests_percent"]) for s in sets]
        low = [v - float(quality[(run, s)]["all_tests_ci95_low"]) for v, s in zip(vals, sets)]
        high = [float(quality[(run, s)]["all_tests_ci95_high"]) - v for v, s in zip(vals, sets)]
        b.bar([i + offset for i in range(3)], vals, width=0.38, color=STYLE[run][1])
        b.errorbar([i + offset for i in range(3)], vals, yerr=[low, high], fmt="none", ecolor=INK, elinewidth=1,
                   capsize=3)
    b.set_xticks(range(3), sets)
    b.set_ylim(0, 100)
    b.set_ylabel("All tests pass, %")
    letter(b, "b  Coding tasks (8 slots)")
    for ax in (a, b):
        style(ax)
    legend(fig, [DS, GLM16], kind="bars")
    note(fig)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.80, bottom=0.17, wspace=0.3)
    save(fig, "figure1-overview")


def figure2():
    """sparkDash decode as the rows of the results page: one panel for each output type in two rows, one row for each
    number of requests, the same axis from 0 in all four panels."""
    data = defaultdict(dict)
    for r in rows("decode_sparkdash.csv"):
        data[(r["run"], r["output_type"])][int(r["concurrency"])] = value(r["total_tok_s"])
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.14, hspace=0.36, wspace=0.05)
    titles = {"prose": "Prose", "code": "Code", "structured": "Count task", "json": "JSON"}
    top = max(v for (run, kind), found in data.items() if run in (DS, GLM16) for v in found.values() if v)
    xmax = 500 * (int(top * 1.16) // 500 + 1)   # room at the right for the value of the fastest row
    for ax, kind, tag in zip(axes.flat, titles, "abcd"):
        items = [(str(c), {run: data[(run, kind)].get(c) for run in (DS, GLM16)}) for c in (1, 2, 4, 8, 16)]
        dumbbell(ax, items, (DS, GLM16), lambda v: f"{v:,.0f}", xmax)
        ax.set_xticks(range(0, xmax, 500))
        ax.xaxis.set_major_formatter(COMMA)
        letter(ax, f"{tag}  {titles[kind]}")
    for ax in axes[:, 0]:
        ax.set_ylabel("Requests")
    for ax in axes[1]:
        ax.set_xlabel("Total tokens per second")
    legend(fig, [DS, GLM16], kind="dots")
    note(fig)
    save(fig, "figure2-decode")


def figure3():
    """Prefill of one cold prompt, 8K to 256K tokens, as on the results page: the speed axis from 0, and each value
    at its point, above the higher of the two points and below the lower one."""
    data = defaultdict(dict)
    for r in rows("prefill_sparkdash.csv"):
        data[r["run"]][int(r["target_tokens"])] = value(r["prefill_tok_s"])
    sizes = sorted(data[DS])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.83, bottom=0.23)
    for run, other in ((GLM16, DS), (DS, GLM16)):
        ax.plot(range(len(sizes)), [data[run].get(s) for s in sizes], color=STYLE[run][1], linewidth=2.5, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        for i, size in enumerate(sizes):
            v, o = data[run].get(size), data[other].get(size)
            if v is None:
                continue
            above = o is None or v >= o
            ax.annotate(f"{v:,.0f}", (i, v), xytext=(0, 7 if above else -7), textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=SMALL, color=INK, zorder=4)
    ax.set_xticks(range(len(sizes)), [f"{s // 1024}K" for s in sizes])
    ax.set_xlim(-0.45, len(sizes) - 0.55)
    top = max(v for run in (DS, GLM16) for v in data[run].values() if v)
    ymax = 2500 * (int(top * 1.12) // 2500 + 1)   # room above the highest point for its value
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, ymax, 5000))
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_xlabel("Prompt, tokens")
    ax.set_ylabel("Prompt tokens/s")
    style(ax)
    legend(fig, [DS, GLM16], marker="o")
    note(fig)
    save(fig, "figure3-prefill")


def figure4():
    """The long shared context (56K tokens, thinking at max effort) as on the results page: (a) the total speed and
    (b) the median time to the first token, one row for each sampler setting and number of requests. Where a run has
    two passes of a setting, the mean of the passes, as in the table longctx_body."""
    passes = defaultdict(list)
    for r in rows("concurrent_waves.csv"):
        if r["probe"] == "long_context_sampled" and value(r["total_tok_s"]) is not None:
            passes[(r["run"], r["top_p"], r["top_k"], int(r["concurrency"]))].append(r)

    def mean_of(run, top_p, top_k, c, field):
        found = [value(r[field]) for r in passes.get((run, top_p, top_k, c), []) if value(r[field]) is not None]
        return sum(found) / len(found) if found else None

    speed, first = [], []
    for top_p, top_k in (("1.0", "off"), ("0.95", "off"), ("0.95", "20")):
        heading = f"top_p {top_p}, top_k {top_k}"
        speed.append((heading, None))
        first.append((heading, None))
        for c in sorted({k[3] for k in passes if k[0] in (DS, GLM16) and k[1:3] == (top_p, top_k)}):
            speed.append((f"{c} requests", {run: mean_of(run, top_p, top_k, c, "total_tok_s") for run in (DS, GLM16)}))
            first.append((f"{c} requests", {run: mean_of(run, top_p, top_k, c, "median_ttft_s") for run in (DS, GLM16)}))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 4.1), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.86, bottom=0.15, wspace=0.1)
    top = max(v for _, found in speed if found for v in found.values() if v)
    xmax = 250 * (int(top * 1.25) // 250 + 1)   # room at the right for the value of the fastest row
    dumbbell(a, speed, (DS, GLM16), lambda v: f"{v:,.0f}", xmax)
    a.set_xticks(range(0, xmax, 500))   # no tick at the right edge, where panel b starts
    a.xaxis.set_major_formatter(COMMA)
    a.set_xlabel("Total tokens per second")
    letter(a, "a  Total speed")
    late = max(v for _, found in first if found for v in found.values() if v)
    dumbbell(b, first, (DS, GLM16), lambda v: f"{v:.2f} s", int(late * 1.3) + 1)
    b.set_xlabel("Seconds to the first token, median")
    letter(b, "b  Time to the first token")
    legend(fig, [DS, GLM16], kind="dots")
    note(fig)
    save(fig, "figure4-long-context")


def figure5():
    """(a) Time to the first token of a conversation, cold and reused; (b) four cold requests with a shared prefix."""
    rounds = defaultdict(list)
    for r in rows("chat_reuse_ttft.csv"):
        rounds[(r["run"], r["nominal_size"], r["case"])].append(value(r["ttft_s"]))
    chat = defaultdict(dict)
    for (run, size, case), vals in rounds.items():   # the mean of the two rounds, as in the table warm_chat
        chat[(run, size)][case] = sum(vals) / len(vals)
    burst = defaultdict(list)
    for r in rows("cold_burst.csv"):
        burst[r["run"]].append(value(r["ttft_s"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
    cases = ["cold", "identical resend", "resend + new turn", "own reply + new turn"]
    labels = ["cold", "resend", "new turn", "own reply"]
    x = range(len(cases))
    for offset, run in ((-0.2, DS), (0.2, GLM16)):
        vals = [chat[(run, "56000")].get(c) for c in cases]
        a.bar([i + offset for i in x], vals, width=0.38, color=STYLE[run][1])
        for i, v in zip(x, vals):
            a.text(i + offset, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=SMALL, color=INK)
    a.set_ylim(0, max(v for run in (DS, GLM16) for v in chat[(run, "56000")].values() if v) * 1.2)
    a.set_xticks(list(x), labels)
    a.set_ylabel("First token, s")
    letter(a, "a  A conversation with a 56K context")
    for i, run in enumerate((DS, GLM8, GLM16)):   # a light bar from the first to the last of the four requests
        vals = sorted(burst[run])
        b.plot([i, i], [vals[0], vals[-1]], color=STYLE[run][1], alpha=0.3, linewidth=7, solid_capstyle="round",
               zorder=2)
        b.scatter([i] * len(vals), vals, color=STYLE[run][1], marker=STYLE[run][2], s=46, zorder=3, linewidths=0)
    b.annotate("all four", (0, sorted(burst[DS])[0]), xytext=(12, 0), textcoords="offset points", fontsize=SMALL,
               color=MUTED, ha="left", va="center")
    b.set_xticks(range(3), ["DeepSeek\n16 slots", "GLM r38\n8 slots", "GLM r38\n16 slots"])
    b.set_xlim(-0.6, 2.6)
    b.set_ylim(0, 19)
    b.set_ylabel("First token, s")
    letter(b, "b  Four cold requests, shared prefix")
    for ax in (a, b):
        style(ax)
    legend(fig, [DS, GLM16], kind="bars")
    note(fig)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.24, wspace=0.3)
    save(fig, "figure5-first-token")


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()


if __name__ == "__main__":
    main()
