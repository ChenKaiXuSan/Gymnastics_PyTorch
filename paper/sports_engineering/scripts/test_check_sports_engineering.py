from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_sports_engineering import (
    build_log_failures,
    count_display_items,
    load_input_sources,
)


def test_count_display_items_includes_starred_tables() -> None:
    source = r"""
    \begin{figure}x\end{figure}
    \begin{table}x\end{table}
    \begin{table*}x\end{table*}
    """

    assert count_display_items(source) == (1, 2)


def test_build_log_failures_scans_manuscript_and_online_resource(tmp_path: Path) -> None:
    manuscript_log = tmp_path / "manuscript.log"
    supplement_log = tmp_path / "online_resource_1.log"
    manuscript_log.write_text("clean build", encoding="utf-8")
    supplement_log.write_text(r"Overfull \hbox (2.0pt too wide)", encoding="utf-8")

    failures = build_log_failures((manuscript_log, supplement_log))

    assert failures == [
        r"online_resource_1.log contains: Overfull \\hbox"
    ]


def test_load_input_sources_reads_only_files_referenced_by_given_document(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "main_table.tex").write_text(
        r"\begin{table}main\end{table}", encoding="utf-8"
    )
    (generated / "supplement_table.tex").write_text(
        r"\begin{table}supplement\end{table}", encoding="utf-8"
    )

    loaded = load_input_sources(
        r"\input{generated/main_table}", root=tmp_path
    )

    assert "main" in loaded
    assert "supplement" not in loaded
