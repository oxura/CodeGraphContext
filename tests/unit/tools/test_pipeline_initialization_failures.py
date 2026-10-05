"""A broken supported grammar is reported without hiding other files (#1746)."""

import asyncio
from unittest.mock import Mock

import pytest

from codegraphcontext.cli import config_manager
from codegraphcontext.core.jobs import JobManager, JobStatus
from codegraphcontext.tools.handlers.management_handlers import check_job_status
from codegraphcontext.tools.indexing import pipeline
from codegraphcontext.tools.indexing.persistence.writer import GraphWriter
from codegraphcontext.tools.indexing.resolution import calls


@pytest.fixture
def run_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(config_manager, "get_config_value", lambda name: "false")
    monkeypatch.setattr(calls, "get_config_value", lambda name: "false")
    monkeypatch.setattr(pipeline, "get_parallel_workers", lambda: 2)
    monkeypatch.setattr(pipeline, "pre_scan_for_imports", lambda *args: {})

    def run(payloads, previous_errors):
        files = [tmp_path / name for name in payloads]
        for file in files:
            file.write_text("placeholder\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "discover_files_to_index", lambda *args, **kwargs: (files, tmp_path))

        writer = Mock(spec=GraphWriter)
        writer.repair_missing_contains_links.return_value = 0
        minimal = Mock()
        manager = JobManager()
        job_id = manager.create_job(str(tmp_path))
        manager.update_job(job_id, errors=previous_errors)
        summary = {}

        def parse_file(repo, file, is_dependency):
            return {"path": str(file), **payloads[file.name]}

        asyncio.run(pipeline.run_tree_sitter_index_async(
            path=tmp_path,
            is_dependency=True,
            job_id=job_id,
            cgcignore_path=None,
            writer=writer,
            job_manager=manager,
            parsers={".py": "python"},
            get_parser=lambda extension: None,
            parse_file=parse_file,
            add_minimal_file_node=minimal,
            index_summary=summary,
        ))
        return summary, manager.get_job(job_id), writer, minimal, check_job_status(manager, job_id=job_id)

    return run


@pytest.mark.parametrize("previous_errors", [[], ["an earlier job diagnostic"]])
def test_mixed_files_report_failures_and_preserve_job_errors(run_pipeline, tmp_path, previous_errors):
    original_errors = list(previous_errors)
    payloads = {
        "broken.py": {
            "error": "Failed to initialize parser for python: cache denied",
            "parse_failed": True,
            "parser_initialization_failed": True,
        },
        "decode.py": {"error": "invalid source encoding", "parse_failed": True},
        "README.md": {"error": "Generic file type .md", "unsupported": False},
        "data.unknown": {"error": "No parser for .unknown", "unsupported": True},
        "working.py": {"functions": [], "classes": [], "function_calls": [], "imports": []},
    }

    summary, job, writer, minimal, status = run_pipeline(payloads, previous_errors)

    expected_failures = {
        str(tmp_path / name): payloads[name]["error"] for name in ("broken.py", "decode.py")
    }
    assert summary["failed_files"] == 2
    assert {entry["path"]: entry["error"] for entry in summary["failed_file_details"]} == expected_failures
    assert summary["total_scanned_files"] == 5
    assert job.status is JobStatus.COMPLETED
    assert job.end_time is not None
    assert job.total_files == job.processed_files == 5
    assert job.errors[:len(original_errors)] == original_errors
    assert set(job.errors[len(original_errors):]) == {
        f"{path}: {error}" for path, error in expected_failures.items()
    }
    assert previous_errors == original_errors
    assert status["success"] is True
    assert status["job"]["status"] == "completed"
    assert status["job"]["errors"] == job.errors

    writer.add_file_to_graph.assert_called_once()
    assert writer.add_file_to_graph.call_args.args[0]["path"] == str(tmp_path / "working.py")
    assert {call.args[0] for call in minimal.call_args_list} == {
        tmp_path / "README.md", tmp_path / "decode.py"
    }
    assert minimal.call_count == 2
    # In particular, the broken grammar must leave no misleading minimal File
    # node that could make a subsequent ordinary index incorrectly skip retry.
    assert all(call.args[0] != tmp_path / "broken.py" for call in minimal.call_args_list)
    writer.write_function_call_groups.assert_called_once()


def test_clean_pipeline_keeps_previous_errors_without_new_failures(run_pipeline):
    previous_errors = ["an earlier job diagnostic"]

    summary, job, writer, minimal, status = run_pipeline({"working.py": {}}, previous_errors)

    assert summary["failed_files"] == 0
    assert summary["failed_file_details"] == []
    assert job.status is JobStatus.COMPLETED
    assert job.total_files == job.processed_files == 1
    assert job.errors == previous_errors
    assert status["job"]["errors"] == previous_errors
    writer.add_file_to_graph.assert_called_once()
    minimal.assert_not_called()
