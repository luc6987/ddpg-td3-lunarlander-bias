"""`check_report.py`: the source audit that runs around the LaTeX compile.

The compile itself is exercised by `./run.sh report`; what is tested here is the
audit, whose silent failure mode is the dangerous one — a check that finds
nothing because a pattern broke would still print "PASS". Comments must not
count (the skeleton's placeholders are comments, but so are the commented-out
optional Figure 4 and the `\\clearpage` markers), `\\%` must not start a
comment, and an undefined `\\ref` must be an error while a leftover TODO is only
a warning.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SENTINEL = "TODO" + "(author)"


def load_script(name: str) -> ModuleType:
    """Import a CLI script by path: `scripts/` is not an importable package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_report = load_script("check_report")


def test_strip_comments_keeps_escaped_percent_and_line_count() -> None:
    text = "a % \\ref{fig:x}\nb \\% stays\n% \\label{fig:y}\n"
    stripped = check_report.strip_comments(text)
    assert "ref" not in stripped and "label" not in stripped
    assert "\\% stays" in stripped
    assert len(stripped.splitlines()) == len(text.splitlines())


def test_todo_lines_include_comment_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "main.tex"
    path.write_text(f"\\section{{Introduction}}\n% {SENTINEL}\n"
                    f"\\caption{{{SENTINEL}}}\n", encoding="utf-8")
    lines = check_report.todo_lines(path, path.read_text(encoding="utf-8"))
    assert len(lines) == 2  # the section body (a comment) and the caption
    assert lines[0].startswith(f"{path}:2:")
    assert lines[1].startswith(f"{path}:3:")


def test_report_lines_reads_inputs_and_ignores_commented_floats(tmp_path: Path) -> None:
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "table1_hparams.tex").write_text(
        "\\label{tab:hparams}\n", encoding="utf-8")
    (tmp_path / "main.tex").write_text(
        "\\input{tables/table1_hparams.tex}\n"
        "\\includegraphics[width=\\linewidth]{figures/fig1.pdf}\n"
        "\\label{fig:one}\n"
        "% \\includegraphics{figures/fig4.pdf}\n"
        "% \\label{fig:four}\n"
        "\\ref{fig:one}\n", encoding="utf-8")
    active, graphics, inputs = check_report.report_lines(tmp_path, "main.tex")
    assert graphics == ["figures/fig1.pdf"]
    assert inputs == ["tables/table1_hparams.tex"]
    labels = check_report.targets(check_report._LABEL, active)
    refs = check_report.targets(check_report._REF, active)
    # main.tex first, then its \input files; fig:four is commented out.
    assert labels == ["fig:one", "tab:hparams"]
    assert refs == ["fig:one"]


def test_includegraphics_without_options() -> None:
    text = "\\includegraphics{figures/fig2_bias}\n"
    assert check_report.targets(check_report._INCLUDEGRAPHICS, text) == [
        "figures/fig2_bias"]


def test_latex_problems_splits_errors_from_layout_warnings() -> None:
    log = (
        "./tables/table2_results.tex:22: LaTeX Error: Unicode character not set up.\n"
        "! Undefined control sequence.\n"
        "LaTeX Warning: Reference `fig:x' on page 1 undefined on input line 12.\n"
        "Overfull \\hbox (28.7pt too wide) in paragraph at lines 18--30\n"
        "Underfull \\hbox (badness 10000) in paragraph at lines 2--18\n"
    )
    errors, warnings = check_report.latex_problems(log)
    assert len(errors) == 3
    assert any("table2_results.tex:22" in line for line in errors)
    assert "undefined reference `fig:x'" in errors
    assert warnings == ["Overfull \\hbox (28.7pt too wide) in paragraph at lines 18--30"]


def test_page_limit_is_the_outline_budget() -> None:
    assert check_report.PAGE_LIMIT == 4


def test_missing_and_unused_assets(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    for name in ("fig1_learning_curves", "fig4_exploratory"):
        (tmp_path / "figures" / f"{name}.pdf").write_bytes(b"%PDF-1.4\n")
    missing = check_report.missing_files(
        tmp_path, ["figures/fig1_learning_curves.pdf", "figures/fig9.pdf"], ".pdf")
    assert missing == ["figures/fig9.pdf"]
    unused = check_report.unused_assets(
        tmp_path, "figures", ["figures/fig1_learning_curves.pdf"])
    assert unused == ["figures/fig4_exploratory.pdf"]


def test_page_count_is_read_from_the_log() -> None:
    log = "Output written on main.pdf (4 pages, 366278 bytes).\n"
    assert check_report._PAGES.search(log).group(1) == "4"
    with pytest.raises(AttributeError):  # no 'Output written': the compile failed
        check_report._PAGES.search("! Emergency stop.\n").group(1)
