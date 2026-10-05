"""Supported files must not disappear when a parser cannot initialize (#1746)."""

import asyncio
import sys
import threading
from types import ModuleType
from unittest.mock import MagicMock, call

import pytest

from codegraphcontext.core.jobs import JobManager, JobStatus
from codegraphcontext.tools import graph_builder
from codegraphcontext.tools.indexing import pipeline
from codegraphcontext.utils import tree_sitter_manager


@pytest.fixture
def builder():
    instance = graph_builder.GraphBuilder.__new__(graph_builder.GraphBuilder)
    instance.parsers = {".py": "python", ".js": "javascript"}
    instance.generic_extensions = {".md"}
    instance.generic_filenames = set()
    instance._parsed_cache = threading.local()
    return instance


def fail_python(language):
    if language == "python":
        raise RuntimeError("grammar cache is owned by uid 0")
    parser = MagicMock(language_name=language)
    parser.parse.side_effect = lambda path, *args, **kwargs: {
        "path": str(path),
        "functions": [],
        "classes": [],
        "imports": [],
    }
    return parser


def test_supported_parser_init_failure_is_not_unsupported(builder, monkeypatch, tmp_path):
    monkeypatch.setattr(graph_builder, "TreeSitterParser", fail_python)
    result = builder.parse_file(tmp_path, tmp_path / "broken.py", False)
    assert result.get("parse_failed") is True
    assert not result.get("unsupported")
    assert "grammar cache is owned by uid 0" in result["error"]


def test_unsupported_and_generic_files_are_not_parse_failures(builder, monkeypatch, tmp_path):
    parser_factory = MagicMock(side_effect=AssertionError("must not initialize"))
    monkeypatch.setattr(graph_builder, "TreeSitterParser", parser_factory)
    unsupported = builder.parse_file(tmp_path, tmp_path / "unknown.xyz", False)
    generic = builder.parse_file(tmp_path, tmp_path / "README.md", False)
    assert unsupported["unsupported"] is True
    assert not unsupported.get("parse_failed")
    assert not generic["unsupported"]
    assert not generic.get("parse_failed")
    parser_factory.assert_not_called()


@pytest.mark.parametrize(
    "names, failed_count",
    [
        (["broken.py"], 1),
        (["broken.py", "empty.js", "README.md"], 1),
        (["empty.js", "README.md"], 0),
    ],
)
def test_pipeline_reports_init_failure_without_losing_other_files(builder, monkeypatch, tmp_path, names, failed_count):
    files = [tmp_path / name for name in names]
    for path in files:
        path.touch()
    monkeypatch.setattr(graph_builder, "TreeSitterParser", fail_python)
    monkeypatch.setattr(pipeline, "discover_files_to_index", lambda *a, **k: (files, tmp_path))
    monkeypatch.setattr(pipeline, "get_parallel_workers", lambda: 2)
    # Keep the actual pre-scan, parse_file, failure collection, and completion path.
    monkeypatch.setattr("codegraphcontext.cli.config_manager.get_config_value", lambda key: "false")
    monkeypatch.setattr(graph_builder, "get_config_value", lambda key: "false")
    monkeypatch.setattr("codegraphcontext.tools.indexing.resolution.calls.get_config_value", lambda key: "false")
    manager = JobManager()
    job_id = manager.create_job(str(tmp_path))
    manager.update_job(job_id, errors=["existing diagnostic"])
    writer = MagicMock()
    writer.repair_missing_contains_links.return_value = 0
    minimal_node = MagicMock()
    summary = {}
    asyncio.run(
        pipeline.run_tree_sitter_index_async(
            tmp_path,
            True,
            job_id,
            None,
            writer,
            manager,
            builder.parsers,
            builder.get_parser,
            builder.parse_file,
            minimal_node,
            index_summary=summary,
        )
    )
    job = manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED  # Existing partial-success policy.
    assert job.processed_files == len(files)
    assert summary["failed_files"] == failed_count
    assert job.errors[0] == "existing diagnostic"
    assert len(job.errors) == failed_count + 1
    if failed_count:
        assert summary["failed_file_details"][0]["path"] == str(files[0])
        assert "grammar cache is owned by uid 0" in summary["failed_file_details"][0]["error"]
        assert str(files[0]) in job.errors[1]
        assert "grammar cache is owned by uid 0" in job.errors[1]
    else:
        assert summary["failed_file_details"] == []
    # Initialization failures leave no placeholder File node, so a normal
    # subsequent index can retry; benign generic files still get their node.
    assert {call.args[0] for call in minimal_node.call_args_list} == {path for path in files if path.suffix == ".md"}
    assert writer.add_file_to_graph.call_count == int("empty.js" in names)
    if "empty.js" in names:
        assert writer.add_file_to_graph.call_args.args[0]["path"] == str(tmp_path / "empty.js")


@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
def test_language_pack_runtime_error_is_preserved(monkeypatch, error_type):
    for name in ("_Language", "_Parser", "_get_language"):
        monkeypatch.setattr(tree_sitter_manager, name, None)
    tree_sitter = ModuleType("tree_sitter")
    tree_sitter.Language = MagicMock()
    tree_sitter.Parser = MagicMock()
    pack = ModuleType("tree_sitter_language_pack")
    original = error_type("grammar download failed: connection refused")
    pack.get_language = MagicMock(side_effect=original)
    monkeypatch.setitem(sys.modules, "tree_sitter", tree_sitter)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", pack)
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", None)
    with pytest.raises(error_type) as caught:
        tree_sitter_manager._load_tree_sitter_dependencies()
    assert caught.value is original


def test_parser_initialization_can_recover(builder, monkeypatch, tmp_path):
    path = tmp_path / "recovered.py"
    parser = MagicMock(language_name="python")
    parser.parse.return_value = {"path": str(path), "functions": [{"name": "recovered"}]}
    monkeypatch.setattr(graph_builder, "get_config_value", lambda key: "false")
    factory = MagicMock(side_effect=[RuntimeError("temporary download failure"), parser])
    monkeypatch.setattr(graph_builder, "TreeSitterParser", factory)
    assert builder.get_parser(".py") is None
    assert builder.get_parser(".py") is parser
    assert builder.get_parser(".py") is parser
    assert factory.call_count == 2
    # Verify recovery through the public parsing behavior, not an error cache.
    result = builder.parse_file(tmp_path, path)
    assert "error" not in result
    assert result["functions"] == [{"name": "recovered"}]
    parser.parse.assert_called_once_with(path, False, is_notebook=False, index_source=False)
    assert factory.call_count == 2


def test_missing_language_pack_still_uses_legacy_fallback(monkeypatch):
    for name in ("_Language", "_Parser", "_get_language"):
        monkeypatch.setattr(tree_sitter_manager, name, None)
    tree_sitter = ModuleType("tree_sitter")
    tree_sitter.Language = MagicMock()
    tree_sitter.Parser = MagicMock()
    legacy = ModuleType("tree_sitter_languages")
    legacy.get_language = MagicMock()
    monkeypatch.setitem(sys.modules, "tree_sitter", tree_sitter)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", legacy)
    assert tree_sitter_manager._load_tree_sitter_dependencies() == (
        tree_sitter.Language,
        tree_sitter.Parser,
        legacy.get_language,
    )


@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
def test_broken_language_pack_still_uses_working_legacy(monkeypatch, error_type):
    for name in ("_Language", "_Parser", "_get_language"):
        monkeypatch.setattr(tree_sitter_manager, name, None)
    tree_sitter = ModuleType("tree_sitter")
    tree_sitter.Language = MagicMock()
    tree_sitter.Parser = MagicMock()
    modern = ModuleType("tree_sitter_language_pack")
    modern.get_language = MagicMock(side_effect=error_type("modern grammar unavailable"))
    legacy = ModuleType("tree_sitter_languages")
    legacy.get_language = MagicMock(return_value="legacy grammar")
    monkeypatch.setitem(sys.modules, "tree_sitter", tree_sitter)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", modern)
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", legacy)
    _, _, get_language = tree_sitter_manager._load_tree_sitter_dependencies()
    assert get_language("python") == "legacy grammar"
    # The selected fallback is now validated once before its caller uses it.
    assert legacy.get_language.call_args_list == [call("python"), call("python")]
    tree_sitter.Parser.assert_called_once_with("legacy grammar")
