"""Partial index diagnostics must be visible and must not claim clean success."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from codegraphcontext.cli import cli_helpers, config_manager


@pytest.fixture
def output(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(cli_helpers, "console", Console(file=stream, width=220, color_system=None))
    return stream


def test_summary_prints_bounded_literal_failure_details(output):
    failures = [
        {"path": f"/repo/[red]file_{index:02}.py[/red]", "error": "cache [bold]denied[/bold]"}
        for index in range(12)
    ]
    builder = SimpleNamespace(last_index_summary={
        "total_scanned_files": 12,
        "failed_files": 12,
        "failed_file_details": failures,
    })

    cli_helpers._print_index_execution_summary(builder)

    text = output.getvalue()
    assert "Files failed to parse" in text
    assert text.count("Failed: ") == 10
    for failure in failures[:10]:
        assert f"Failed: {failure['path']}: {failure['error']}" in text
    assert "file_10.py" not in text
    assert "file_11.py" not in text
    assert "... and 2 more failed file(s)." in text
    assert "cgc reindex <path>" in text


def test_clean_summary_keeps_metrics_without_failure_output(output):
    builder = SimpleNamespace(last_index_summary={
        "total_scanned_files": 2,
        "files_by_extension": {".py (python)": 2},
        "function_nodes": 3,
        "class_nodes": 1,
        "call_edges": 2,
        "serialization_seconds": 1.25,
        "failed_files": 0,
        "failed_file_details": [],
    })

    cli_helpers._print_index_execution_summary(builder)

    text = output.getvalue()
    for label in ("CGC Index Execution Summary", "Total scanned files", ".py (python): 2",
                  "Function nodes", "Class nodes", "CALLS edges", "Serialization seconds", "1.25"):
        assert label in text
    assert "failed" not in text.lower()
    assert "cgc reindex" not in text


@pytest.mark.parametrize("command, success_text", [
    ("index", "Successfully finished indexing:"),
    ("reindex", "Successfully re-indexed:"),
    ("package", "Successfully finished indexing package:"),
])
@pytest.mark.parametrize("failed_files", [0, 1])
def test_commands_distinguish_partial_completion_from_clean_success(
    monkeypatch, tmp_path, output, command, success_text, failed_files
):
    db_manager = Mock()
    builder = SimpleNamespace(
        last_call_resolution_diagnostics=[],
        last_index_summary={
            "total_scanned_files": 1,
            "failed_files": failed_files,
            "failed_file_details": ([{"path": "/repo/broken.py", "error": "cache denied"}] if failed_files else []),
        },
    )
    finder = Mock()
    finder.list_indexed_repositories.return_value = []
    ctx = SimpleNamespace(cgcignore_path=None, mode="global")
    monkeypatch.setattr(cli_helpers, "_initialize_services", Mock(return_value=(db_manager, builder, finder, ctx)))
    run_index = AsyncMock()
    monkeypatch.setattr(cli_helpers, "_run_index_with_progress", run_index)
    monkeypatch.setattr(config_manager, "get_config_value", lambda name: "false")
    monkeypatch.setattr(cli_helpers, "get_local_package_path", lambda *args: str(tmp_path))

    if command == "package":
        cli_helpers.add_package_helper("example", "python")
    else:
        getattr(cli_helpers, f"{command}_helper")(str(tmp_path), no_progress=True)

    text = output.getvalue()
    if failed_files:
        assert "Indexing finished with 1 failed file(s); see the errors above." in text
        assert "Failed: /repo/broken.py: cache denied" in text
        assert "Successfully" not in text
    else:
        assert success_text in text
        assert "Indexing finished with" not in text
        assert "Failed: " not in text
    run_index.assert_awaited_once()
    db_manager.close_driver.assert_called_once()
