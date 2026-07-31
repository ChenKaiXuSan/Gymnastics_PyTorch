from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_sports_engineering import build_log_failures, count_display_items


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
