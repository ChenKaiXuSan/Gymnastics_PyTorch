#!/usr/bin/env python3
"""Check Sports Engineering manuscript constraints and evidence anchors."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "manuscript.tex"
BIB = ROOT / "references.bib"
LOG = ROOT / "build" / "manuscript.log"
ONLINE_LOG = ROOT / "build" / "online_resource_1.log"
ONLINE_RESOURCE = ROOT / "online_resource_1.tex"
MAKEFILE = ROOT / "Makefile"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def strip_latex(text: str) -> str:
    text = re.sub(
        r"\\begin\{(?:figure|table|equation)\}.*?\\end\{(?:figure|table|equation)\}",
        " ",
        text,
        flags=re.S,
    )
    text = re.sub(r"\\cite\{[^}]*\}|\\(?:ref|label)\{[^}]*\}", " ", text)
    text = re.sub(
        r"\\(?:section|subsection|subsubsection)\{([^}]*)\}", r" \1 ", text
    )
    text = re.sub(r"\\[A-Za-z@]+(?:\[[^]]*\])?", " ", text)
    return re.sub(r"[$\\{}~^_&%#]", " ", text)


def extract_braced_command(text: str, command: str) -> str:
    marker = f"\\{command}{{"
    start = text.index(marker) + len(marker)
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    raise ValueError(f"unterminated \\{command} command")


def count_display_items(text: str) -> tuple[int, int]:
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", text))
    table_count = len(re.findall(r"\\begin\{table\*?\}", text))
    return figure_count, table_count


def load_input_sources(text: str, root: Path) -> str:
    """Load direct LaTeX inputs referenced by one document."""
    sources: list[str] = []
    for relative in re.findall(r"\\input\{([^}]+)\}", text):
        path = root / relative
        if path.suffix == "":
            path = path.with_suffix(".tex")
        if path.exists():
            sources.append(path.read_text(encoding="utf-8"))
    return "\n".join(sources)


def build_log_failures(log_paths: tuple[Path, ...]) -> list[str]:
    """Return fatal, unresolved-reference, and overflow findings for all builds."""
    failures: list[str] = []
    patterns = (
        r"undefined citations",
        r"undefined references",
        r"Fatal error",
        r"Emergency stop",
        r"Overfull \\hbox",
        r"Overfull \\vbox",
    )
    for log_path in log_paths:
        if not log_path.exists():
            failures.append(f"{log_path.name} is missing; compile before checking")
            continue
        log = log_path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if re.search(pattern, log, flags=re.I):
                failures.append(f"{log_path.name} contains: {pattern}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission-ready",
        action="store_true",
        help="fail if author-only submission placeholders remain",
    )
    args = parser.parse_args()

    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    online_resource = ONLINE_RESOURCE.read_text(encoding="utf-8")
    generated_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "generated").glob("*.tex"))
    )
    main_input_sources = load_input_sources(tex, ROOT)
    makefile = MAKEFILE.read_text(encoding="utf-8")
    failures: list[str] = []

    abstract_count = len(words(strip_latex(extract_braced_command(tex, "abstract"))))
    if not 150 <= abstract_count <= 250:
        failures.append(f"abstract has {abstract_count} words; expected 150--250")

    body = tex[tex.index(r"\section{Introduction}") : tex.index(r"\bibliography")]
    body_count = len(words(strip_latex(body)))
    if body_count > 4000:
        failures.append(f"main body has approximately {body_count} words; maximum is 4000")

    figure_count, table_count = count_display_items(body + "\n" + main_input_sources)
    if figure_count + table_count > 10:
        failures.append(
            f"main article has {figure_count} figures and {table_count} tables; maximum combined is 10"
        )

    for section in ("Introduction", "Methods", "Results", "Discussion", "Conclusions"):
        if f"\\section{{{section}}}" not in tex:
            failures.append(f"missing required section: {section}")

    conclusion = tex[
        tex.index(r"\section{Conclusions}") : tex.index(r"\bibliography")
    ]
    conclusion_text = conclusion.split("\n\n", 1)[1].strip()
    if "\n\n" in conclusion_text:
        failures.append("Conclusions must be one paragraph")

    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        citation_keys.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_keys = sorted(citation_keys - bib_keys)
    if missing_keys:
        failures.append("missing bibliography keys: " + ", ".join(missing_keys))

    required_anchors = (
        "47.25\\%",
        "60.78",
        "166.537",
        "178.506",
        "30.259",
        "928 eligible",
        "0.3006",
        "0.4233",
        "0.0377",
        "representation-dependent",
        "65.25",
        "63.07",
        "109 of 137",
        "40.35",
        "11 of 20",
    )
    for anchor in required_anchors:
        if anchor not in tex:
            failures.append(f"missing evidence anchor: {anchor}")

    combined_sources = "\n".join((tex, online_resource, generated_sources))
    for label in (
        "tab:extrinsic-comparison",
        "tab:joint-accuracy-main",
        "tab:joint-accuracy-all70",
    ):
        if label not in combined_sources:
            failures.append(f"missing comparison table label: {label}")
    for phrase in (
        "camera-assisted comparator",
        "same-video evidence",
        "framewise hip centring",
    ):
        if phrase not in tex:
            failures.append(f"missing evidence-boundary phrase: {phrase}")
    if "generated/*.tex" not in makefile:
        failures.append("source package does not include generated/*.tex")

    failures.extend(build_log_failures((LOG, ONLINE_LOG)))

    blockers = (
        "author verification required before submission",
        r"\affil*[1]{\orgname{CCS}}",
    )
    active_blockers = [blocker for blocker in blockers if blocker.lower() in tex.lower()]
    cover = (ROOT / "cover_letter.md").read_text(encoding="utf-8")
    if "AUTHOR ACTION BEFORE USE" in cover:
        active_blockers.append("cover-letter author action note")

    print(f"Abstract: {abstract_count} words")
    print(f"Main body: approximately {body_count} words")
    print(f"Main display items: {figure_count} figures + {table_count} tables")
    print(f"Citations: {len(citation_keys)} keys, all resolved")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nTechnical manuscript checks passed.")
    if active_blockers:
        print("Submission blockers still present:")
        for blocker in active_blockers:
            print(f"- {blocker}")
        if args.submission_ready:
            return 2
    elif args.submission_ready:
        print("No encoded submission blockers remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
