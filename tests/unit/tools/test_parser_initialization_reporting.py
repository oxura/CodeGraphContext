"""Supported parser initialization failures are not unsupported files (#1746)."""

from unittest.mock import Mock

import pytest

from codegraphcontext.core.jobs import JobManager
from codegraphcontext.tools import graph_builder


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.setattr(graph_builder.GraphBuilder, "create_schema", lambda self: None)
    monkeypatch.setattr(graph_builder, "get_config_value", lambda name: "false")
    return graph_builder.GraphBuilder(Mock(), JobManager(), None)


def test_unknown_extension_is_none_even_in_strict_mode(builder, monkeypatch):
    factory = Mock(side_effect=AssertionError("unknown extensions must not initialize a parser"))
    monkeypatch.setattr(graph_builder, "TreeSitterParser", factory)

    assert builder.get_parser(".unknown", raise_on_error=True) is None
    factory.assert_not_called()


def test_best_effort_prescan_and_strict_parse_can_retry_after_failure(builder, monkeypatch):
    error = RuntimeError("grammar cache unavailable")
    parser = Mock()
    factory = Mock(side_effect=[error, error, parser])
    monkeypatch.setattr(graph_builder, "TreeSitterParser", factory)

    assert builder.get_parser(".py") is None
    with pytest.raises(RuntimeError) as raised:
        builder.get_parser(".py", raise_on_error=True)
    assert raised.value is error

    assert builder.get_parser(".py", raise_on_error=True) is parser
    assert builder.get_parser(".py") is parser
    assert factory.call_count == 3


@pytest.mark.parametrize("returns_none", [False, True])
def test_supported_initialization_failure_is_counted(builder, monkeypatch, tmp_path, returns_none):
    path = tmp_path / "broken.py"
    get_parser = Mock(return_value=None) if returns_none else Mock(side_effect=RuntimeError("cache denied"))
    monkeypatch.setattr(builder, "get_parser", get_parser)

    result = builder.parse_file(tmp_path, path)

    get_parser.assert_called_once_with(".py", raise_on_error=True)
    assert result["path"] == str(path)
    assert result["parse_failed"] is True
    assert result["parser_initialization_failed"] is True
    assert not result.get("unsupported")
    assert "python" in result["error"]
    assert ("no parser returned" if returns_none else "cache denied") in result["error"]


@pytest.mark.parametrize("filename", ["README.md", ".gitignore"])
def test_generic_files_remain_benign(builder, monkeypatch, tmp_path, filename):
    get_parser = Mock(side_effect=AssertionError("generic files do not need a parser"))
    monkeypatch.setattr(builder, "get_parser", get_parser)

    result = builder.parse_file(tmp_path, tmp_path / filename)

    assert result["unsupported"] is False
    assert "Generic file type" in result["error"]
    assert not result.get("parse_failed")
    assert not result.get("parser_initialization_failed")
    get_parser.assert_not_called()


def test_unknown_file_remains_unsupported(builder, tmp_path):
    result = builder.parse_file(tmp_path, tmp_path / "data.unknown")

    assert result["unsupported"] is True
    assert not result.get("parse_failed")
    assert not result.get("parser_initialization_failed")


@pytest.mark.parametrize("raises", [False, True])
def test_normal_parse_errors_remain_distinct_from_initialization(builder, monkeypatch, tmp_path, raises):
    path = tmp_path / "broken.py"
    parser = Mock(language_name="python")
    if raises:
        parser.parse.side_effect = ValueError("source could not be decoded")
    else:
        parser.parse.return_value = {"path": str(path), "error": "source could not be decoded"}
    monkeypatch.setattr(builder, "get_parser", Mock(return_value=parser))

    result = builder.parse_file(tmp_path, path)

    assert result["parse_failed"] is True
    assert result["error"] == "source could not be decoded"
    assert not result.get("parser_initialization_failed")
    assert not result.get("unsupported")


def test_successful_parse_retains_metadata(builder, monkeypatch, tmp_path):
    path = tmp_path / "ok.py"
    parser = Mock(language_name="python")
    parser.parse.return_value = {"path": str(path), "functions": [{"name": "ok"}]}
    monkeypatch.setattr(builder, "get_parser", Mock(return_value=parser))

    result = builder.parse_file(tmp_path, path)

    assert result == {"path": str(path), "repo_path": str(tmp_path), "functions": [{"name": "ok"}]}
    parser.parse.assert_called_once_with(path, False, is_notebook=False, index_source=False)


def test_file_update_preserves_existing_graph_until_parser_recovers(builder, monkeypatch, tmp_path):
    path = tmp_path / "changed.py"
    path.write_text("def changed(): pass\n", encoding="utf-8")
    parser = Mock(language_name="python")
    parser.parse.return_value = {"path": str(path), "functions": [{"name": "changed"}]}
    monkeypatch.setattr(builder, "get_parser", Mock(side_effect=[RuntimeError("cache denied"), parser]))
    delete = Mock()
    add = Mock()
    minimal = Mock()
    monkeypatch.setattr(builder, "delete_file_from_graph", delete)
    monkeypatch.setattr(builder, "add_file_to_graph", add)
    monkeypatch.setattr(builder, "add_minimal_file_node", minimal)

    assert builder.update_file_in_graph(path, tmp_path, {}) is None
    delete.assert_not_called()
    add.assert_not_called()
    minimal.assert_not_called()

    result = builder.update_file_in_graph(path, tmp_path, {})

    assert result["functions"] == [{"name": "changed"}]
    delete.assert_called_once_with(path.resolve().as_posix())
    add.assert_called_once_with(result, tmp_path.name, {})
    minimal.assert_not_called()


def test_update_of_deleted_file_still_removes_its_graph(builder, monkeypatch, tmp_path):
    path = tmp_path / "deleted.py"
    delete = Mock()
    parse_file = Mock()
    monkeypatch.setattr(builder, "delete_file_from_graph", delete)
    monkeypatch.setattr(builder, "parse_file", parse_file)

    assert builder.update_file_in_graph(path, tmp_path, {}) == {
        "deleted": True, "path": path.resolve().as_posix()
    }
    delete.assert_called_once_with(path.resolve().as_posix())
    parse_file.assert_not_called()
