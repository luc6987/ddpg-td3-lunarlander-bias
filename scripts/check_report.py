#!/usr/bin/env python
"""CLI: compile the LaTeX report and audit it against the submission checklist.

    uv run python scripts/check_report.py          # or `./run.sh report`

It compiles `report/main.tex` with `latexmk` (fallback: `pdflatex` twice, with a
`bibtex` pass, so that cross-references and the bibliography resolve) inside the
report directory, prints the page count, and then checks
`docs/report_outline.md`'s last section:

* **warnings** — every remaining `TODO(author)`, every figure/table label the
  text never references, every `figures/*.pdf` no float includes, every overfull
  box;
* **errors** — a missing include, a LaTeX error, an undefined `\\ref` or
  citation, no PDF, and more than `PAGE_LIMIT` pages.

TODOs and unreferenced labels are warnings on purpose: the milestone-5 skeleton
has no prose yet, so the check must not fail because of them.  It exits 1 only
for the errors above, i.e. for a report that could not be submitted as it is.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: `docs/report_outline.md`: 4 pages, figures, tables and references included.
PAGE_LIMIT = 4
#: The literal placeholder the author replaces; only code writes it.
TODO = "TODO(author)"

_COMMENT = re.compile(r"(?<!\\)%.*")
_INCLUDEGRAPHICS = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_INPUT = re.compile(r"\\input\s*\{([^}]*)\}")
_LABEL = re.compile(r"\\label\s*\{([^}]*)\}")
_REF = re.compile(r"\\(?:ref|eqref|autoref|Cref|cref|pageref|nameref)\s*\{([^}]*)\}")
_PAGES = re.compile(r"Output written on .*?\((\d+) pages?")
_UNDEFINED = re.compile(r"LaTeX Warning: (Reference|Citation) [`']([^']*)' .*undefined")
_OVERFULL = re.compile(r"^(Overfull \\hbox \([0-9.]+pt too wide\)[^\n]*)", re.MULTILINE)
#: With `-file-line-error`, LaTeX errors carry a `file:line: ` prefix.
_FILE_LINE_ERROR = re.compile(r"^[^\s].*?\.(?:tex|sty|cls|def|cfg):\d+: ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report-dir", default="report",
                   help="directory that holds the report (default: report)")
    p.add_argument("--main", default="main.tex",
                   help="LaTeX file, relative to --report-dir (default: main.tex)")
    p.add_argument("--engine", choices=("auto", "latexmk", "pdflatex"), default="auto",
                   help="compiler; auto prefers latexmk (default)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="seconds allowed per compiler call (default: 600)")
    return p.parse_args(argv)


# --- reading the source -----------------------------------------------------------

def strip_comments(text: str) -> str:
    """Drop `%` comments: what is left is what LaTeX actually typesets.

    Labels and references inside a comment (e.g. the optional Figure 4) are not
    part of the document and must not count as defined or as referenced.
    """
    return _COMMENT.sub("", text)


def todo_lines(path: Path, text: str) -> list[str]:
    """Every `TODO(author)` in `text`, as `path:line` (comments included: the
    placeholders of the milestone-5 skeleton are themselves comments)."""
    return [f"{path}:{line_no}: {line.strip().lstrip('% ').strip()}"
            for line_no, line in enumerate(text.splitlines(), start=1) if TODO in line]


def targets(pattern: re.Pattern[str], text: str) -> list[str]:
    return [m.group(1).strip() for m in pattern.finditer(text)]


# --- compiling --------------------------------------------------------------------

def run(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    """Run one compiler call; never raise on a compiler error."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timed out after {timeout:.0f} s"
    return proc.returncode, proc.stdout + proc.stderr


def compile_pdf(report_dir: Path, main: str, engine: str,
                timeout: float) -> tuple[str, float, list[str]]:
    """Compile `main` inside `report_dir`.

    Returns `(engine description, seconds, compiler output)`; the compiler's exit
    codes are deliberately ignored, since the log and the PDF say more.
    """
    t0 = time.time()
    out: list[str] = []
    stem = Path(main).stem
    latexmk = shutil.which("latexmk")
    if engine in ("auto", "latexmk") and latexmk is not None:
        _, text = run([latexmk, "-pdf", "-interaction=nonstopmode", "-file-line-error",
                       main], report_dir, timeout)
        return "latexmk -pdf (pdflatex + bibtex)", time.time() - t0, [text]

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("error: neither latexmk nor pdflatex is on PATH", file=sys.stderr)
        sys.exit(1)
    if engine == "latexmk":
        print("error: --engine latexmk, but latexmk is not on PATH", file=sys.stderr)
        sys.exit(1)
    flags = ["-interaction=nonstopmode", "-file-line-error"]
    pdflatex_cmd = [pdflatex, *flags, main]
    # pdflatex writes the .aux, bibtex turns it into the .bbl, and two more
    # passes resolve the cross-references the bibliography introduces.
    cmds = [pdflatex_cmd]
    bibtex = shutil.which("bibtex")
    if bibtex is not None:
        cmds.append([bibtex, stem])
    cmds += [pdflatex_cmd, pdflatex_cmd]
    for cmd in cmds:
        _, text = run(cmd, report_dir, timeout)
        out.append(text)
    return "pdflatex x3 + bibtex", time.time() - t0, out


# --- auditing ---------------------------------------------------------------------

def latex_problems(log: str) -> tuple[list[str], list[str]]:
    """Split the compile log into `(errors, warnings)` a human should see."""
    errors = [line.strip() for line in log.splitlines()
              if line.startswith("!") or _FILE_LINE_ERROR.match(line)]
    errors += [f"undefined {kind.lower()} `{name}'"
               for kind, name in _UNDEFINED.findall(log)]
    warnings = [m.group(1).strip() for m in _OVERFULL.finditer(log)]
    return errors, warnings


def report_lines(report_dir: Path, main: str) -> tuple[list[str], list[str], list[str]]:
    """`(active text, include targets, input targets)` of main.tex plus its inputs.

    The audit reads the file LaTeX reads: comments are stripped, and the
    `\\input` files (the generated tables) are read too.
    """
    active = strip_comments((report_dir / main).read_text(encoding="utf-8"))
    inputs = targets(_INPUT, active)
    for name in inputs:
        path = report_dir / name
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        if path.is_file():
            active += "\n" + strip_comments(path.read_text(encoding="utf-8"))
    return active, targets(_INCLUDEGRAPHICS, active), inputs


def missing_files(report_dir: Path, names: list[str], suffix: str) -> list[str]:
    return [name for name in names
            if not (report_dir / (name if Path(name).suffix else name + suffix)).is_file()]


def unused_assets(report_dir: Path, subdir: str, used: list[str]) -> list[str]:
    """Files of `report/<subdir>/` that no float includes (compared by stem)."""
    used_stems = {Path(name).stem for name in used}
    folder = report_dir / subdir
    if not folder.is_dir():
        return []
    return sorted(f"{subdir}/{path.name}" for path in folder.glob("*.pdf")
                  if path.stem not in used_stems)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    main_path = report_dir / args.main
    if not main_path.is_file():
        print(f"error: {main_path} does not exist", file=sys.stderr)
        return 1

    source = main_path.read_text(encoding="utf-8")
    active, graphics, inputs = report_lines(report_dir, args.main)

    engine, seconds, compiler_out = compile_pdf(report_dir, args.main, args.engine,
                                                args.timeout)
    stem = Path(args.main).stem
    log_path, pdf_path = report_dir / f"{stem}.log", report_dir / f"{stem}.pdf"
    log = (log_path.read_text(encoding="utf-8", errors="replace")
           if log_path.is_file() else "")

    errors: list[str] = []
    warnings: list[str] = []
    matching = _PAGES.search(log)
    pages = int(matching.group(1)) if matching else None

    errors += [f"missing file: {name}" for name in
               missing_files(report_dir, graphics, ".pdf")
               + missing_files(report_dir, inputs, ".tex")]
    latex_errors, latex_warnings = latex_problems(log)
    errors += [f"LaTeX: {line}" for line in latex_errors]
    warnings += [f"layout: {line}" for line in latex_warnings]
    if pages is None:
        errors.append("the compile log has no 'Output written on ... (N pages)': "
                      "the document did not compile")
    elif pages > PAGE_LIMIT:
        errors.append(f"{pages} pages > the {PAGE_LIMIT}-page limit of "
                      f"docs/report_outline.md")

    labels, refs = targets(_LABEL, active), targets(_REF, active)
    unreferenced = [label for label in labels if label not in refs]
    warnings += [f"label `{label}' is defined but never referenced in the text"
                 for label in unreferenced]
    warnings += [f"{name} is not included by {args.main}"
                 for name in unused_assets(report_dir, "figures", graphics)]
    todos = todo_lines(main_path, source)
    for name in inputs:
        path = report_dir / name
        if path.is_file():
            todos += todo_lines(path, path.read_text(encoding="utf-8"))
    if pages is not None and pages >= PAGE_LIMIT and todos:
        warnings.append(f"the document already uses {pages} of {PAGE_LIMIT} pages "
                        f"while {len(todos)} {TODO} remain: the text has to displace "
                        f"floats, not add pages")

    # --- the report ---------------------------------------------------------------
    print(f"# check_report — {main_path}")
    print(f"compiled : {engine}, {seconds:.1f} s")
    if pdf_path.is_file():
        print(f"pdf      : {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} kB)")
    else:
        print(f"pdf      : {pdf_path} (none)")
    page_text = "could not be read" if pages is None else str(pages)
    verdict = "" if pages is None or pages <= PAGE_LIMIT else "  <- OVER THE LIMIT"
    print(f"pages    : {page_text} (limit {PAGE_LIMIT}){verdict}")
    print(f"floats   : {len(graphics)} figure file(s) and {len(inputs)} table file(s) "
          f"included")
    if labels:
        print("labels   :")
        for label in labels:
            state = "referenced" if label in refs else "NOT referenced"
            print(f"  {label:<28} {state}")
    print()
    print(f"TODO(author) still to write ({len(todos)})")
    for line in todos:
        print(f"  {line}")
    print()
    print(f"warnings ({len(warnings)})")
    for line in warnings:
        print(f"  {line}")
    print()
    print(f"errors ({len(errors)})")
    for line in errors:
        print(f"  {line}")

    if errors and compiler_out:
        print()
        print("last lines of the compile output:")
        for line in compiler_out[-1].splitlines()[-15:]:
            print(f"  {line}")
    print()
    if errors:
        print(f"FAIL — {len(errors)} error(s), {len(warnings)} warning(s). "
              f"{len(todos)} TODO(author) left.")
        return 1
    print(f"PASS — compiled, {pages} page(s) <= {PAGE_LIMIT}, 0 errors, "
          f"{len(warnings)} warning(s), {len(todos)} TODO(author) left to write.")
    print(f"       the submission is `<name>.pdf`: `{report_dir}/{stem}.pdf` is the draft.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
