"""Count the sentences of the paper that obey the ASD-STE100 writing rules that a script can examine.

  python3 scripts/ste_check.py README.md recipes/*.md          summary for each file
  python3 scripts/ste_check.py --show README.md                also print each sentence that does not obey a rule

What it examines in each sentence (the rule numbers are those of ASD-STE100):
  - length: 25 words maximum in a description, 20 words maximum in an instruction (rules 5.1, 6.3)
  - no semicolon and no contraction (rules 4.2, 8)
  - no "-ing" form of a verb, unless the word is in the list of technical names below (rule 3.5)
  - no verb structure with has, have or had and a past participle (rule 3.4)
  - no passive voice in an instruction (rule 3.6); in a description the script only counts it
  - none of the frequent words that the STE dictionary does not approve (rules 1.1 to 1.3), from the list below
  - American English spelling (rule 1.14), from the list below

What it does not examine: the full STE dictionary (it is not in this repository), the part of speech of each word,
noun clusters, and the structure of paragraphs. Thus a result of 100% does not show full agreement with STE.
Text in code blocks, tables, headings and links is not text of the paper and the script ignores it.
"""
import re
import sys

TECHNICAL_ING = {  # technical names and nouns that end in -ing
    "thinking", "reasoning", "during", "nothing", "something", "thing", "things", "ring", "string", "strings",
    "setting", "settings", "warning", "king",
}
NOT_APPROVED = {  # word or pattern: the approved alternative
    r"about (?=\d|half|the same)": "approximately", r"however": "but", r"whether": "if", r"ensur\w*": "make sure",
    r"utili[sz]\w*": "use", r"perform\w*": "do", r"needs?|needed": "necessary, must", r"requir\w*": "necessary, must",
    r"provid\w*": "give, supply", r"allow\w*": "let, permit", r"prior": "before", r"just": "only", r"big\w*": "large",
    r"begin\w*|began|begun": "start", r"obtain\w*": "get", r"should|would|could|might|may": "can, must",
    r"via": "through", r"per": "for each", r"since": "because", r"e\.g\.|i\.e\.|etc\.": "for example",
    r"very": "(remove)", r"roughly|nearly": "approximately, almost", r"twice": "two times",
    r"fix(?:ed|es)?": "correct, correction", r"works|worked": "operate", r"therefore": "thus",
    r"although|though": "but", r"within": "in", r"upon": "on", r"verif\w*": "make sure", r"confirm\w*": "make sure",
    r"happen\w*": "occur", r"look(?:s|ed)?": "see, find", r"want\w*": "(rewrite)", r"tr(?:y|ies|ied)": "(rewrite)",
    r"wait\w*": "(rewrite)", r"refus\w*": "not permit", r"fail(?:s|ed)": "not operate", r"hang|hangs|hung": "stop",
    r"additional\w*": "more, also", r"numerous|several": "(give the number)", r"enabl\w*|disabl\w*": "set to on, off",
    r"switch(?:ed|es)": "set", r"turn(?:ed|s) (?:on|off)": "set to on, off", r"identical": "the same, equal",
    r"indicat\w*": "show", r"commenc\w*|terminat\w*": "start, stop", r"approx\.": "approximately",
}
BRITISH = r"quantis\w*|initialis\w*|utilis\w*|licence\w*|colour\w*|neighbour\w*|behaviour\w*|centre\w*|optimis\w*|" \
          r"organis\w*|recognis\w*|synchronis\w*|summaris\w*|normalis\w*|labelled|modelled|travelled|programme\w*|grey"
PARTICIPLE = r"(?:\w+ed|been|done|made|seen|shown|given|taken|run|set|put|found|kept|sent|built|written|read|got|known)"
IMPERATIVE = {"run", "make", "do", "use", "set", "start", "stop", "add", "build", "clone", "send", "keep", "poll", "put",
              "refer", "apply", "examine", "compare", "install", "mount", "write", "read", "download", "get", "open",
              "list", "give", "tell", "let", "record", "replace", "copy", "change"}


def plain_text(markdown):
    """The prose of a Markdown file: one string for each paragraph or list item."""
    blocks, current, in_code = [], [], False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        skip = (in_code or line.startswith("    ") or line.lstrip().startswith(("|", "#", "![", "<")))
        item = re.match(r"\s*(?:[-*]|\d+\.)\s+", line)
        if skip or not line.strip() or item:
            if current:
                blocks.append(" ".join(current))
                current = []
            if item and not skip:
                current = [line[item.end():].strip()]
            continue
        current.append(line.strip())
    if current:
        blocks.append(" ".join(current))
    cleaned = []
    for block in blocks:
        block = re.sub(r"`[^`]*`", "CODE", block)
        block = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", block)
        block = re.sub(r"[*_]{1,2}", "", block)
        cleaned.append(block)
    return cleaned


def sentences(block):
    """Sentences of a block. Text in parentheses is a sentence of its own (STE word-count rule)."""
    inner = re.findall(r"\(([^()]*)\)", block)
    outer = re.sub(r"\s*\([^()]*\)", "", block)
    parts = re.split(r"(?<=[.?!:])\s+(?=[A-Z0-9\"'“(])", outer)
    return [p.strip() for p in parts + inner if len(re.findall(r"[A-Za-z]{2,}", p)) >= 2]


def words(sentence):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_./%:+×÷−–'-]*", sentence)


def problems(sentence):
    found, notes = [], []
    tokens = words(sentence)
    first = tokens[0].lower() if tokens else ""
    instruction = first in IMPERATIVE
    limit = 20 if instruction else 25
    if len(tokens) > limit:
        found.append(f"{len(tokens)} words (maximum {limit})")
    if ";" in sentence:
        found.append("semicolon")
    if re.search(r"\b\w+(?:n't|'re|'ll|'ve|'d)\b|\b(?:it|that|there|what|here)'s\b", sentence, re.I):
        found.append("contraction")
    for word in re.findall(r"\b[A-Za-z-]+ing\b", sentence):
        if word.lower().split("-")[-1] not in TECHNICAL_ING:
            found.append(f"-ing form: {word}")
    if re.search(rf"\b(?:has|have|had)\s+(?:not\s+)?{PARTICIPLE}\b", sentence, re.I):
        found.append("verb structure with has, have or had")
    if re.search(rf"\b(?:is|are|was|were|be|been)\s+(?:not\s+)?{PARTICIPLE}\b", sentence, re.I):
        (found if instruction else notes).append("passive voice")
    for pattern, approved in NOT_APPROVED.items():
        match = re.search(rf"\b(?:{pattern})(?!\w)", sentence, re.I)
        if match:
            found.append(f"not approved: {match.group(0).strip()} (use: {approved})")
    match = re.search(rf"\b(?:{BRITISH})\b", sentence, re.I)
    if match:
        found.append(f"British spelling: {match.group(0)}")
    return found, notes


def main():
    show = "--show" in sys.argv
    total = good = passive = 0
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        file_total = file_good = 0
        with open(name, encoding="utf-8") as handle:
            blocks = plain_text(handle.read())
        for block in blocks:
            for sentence in sentences(block):
                found, notes = problems(sentence)
                file_total += 1
                file_good += not found
                passive += bool(notes)
                if show and found:
                    print(f"{name}: {sentence}\n    -> " + "; ".join(found))
        total += file_total
        good += file_good
        print(f"{name}: {file_good} of {file_total} sentences obey the examined rules ({100 * file_good / max(1, file_total):.0f}%)")
    print(f"all files: {good} of {total} ({100 * good / max(1, total):.0f}%); passive voice in a description: {passive}")


if __name__ == "__main__":
    main()
