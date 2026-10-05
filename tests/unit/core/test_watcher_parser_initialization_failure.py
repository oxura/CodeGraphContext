"""A failed grammar must not erase live watcher state or neighbors (#1746)."""

from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

from codegraphcontext.cli import config_manager
from codegraphcontext.core import watcher
from codegraphcontext.tools.graph_builder import GraphBuilder


def test_failed_update_preserves_imports_and_neighbors_before_queued_success(monkeypatch, tmp_path):
    broken = tmp_path / "broken.py"
    working = tmp_path / "working.js"
    neighbor = tmp_path / "neighbor.js"
    for path in (broken, working, neighbor):
        path.write_text("placeholder\n", encoding="utf-8")
    broken_str, working_str, neighbor_str = (str(path.resolve()) for path in (broken, working, neighbor))
    monkeypatch.setattr(watcher.RepositoryEventHandler, "_load_ignore_spec", lambda *args: None)
    monkeypatch.setattr(watcher.RepositoryEventHandler, "_should_ignore", lambda *args: False)
    monkeypatch.setattr(config_manager, "get_config_value", lambda name: "false")

    graph = Mock(spec=GraphBuilder)
    graph.parsers = {".py": "python", ".js": "javascript"}
    graph.get_caller_file_paths.side_effect = lambda path: [neighbor_str] if path == broken_str else []
    graph.get_inheritance_neighbor_paths.side_effect = lambda path: [neighbor_str] if path == broken_str else []
    graph.get_repo_class_lookup.return_value = {}
    # Pre-scan may succeed before the actual parser initialization fails. Its
    # merge extends an existing value list, so rollback needs copies of lists.
    graph.pre_scan_imports.side_effect = lambda paths: (
        {"NewSymbol": [working_str]} if working in paths else {
            "NeighborSymbol": [broken_str], "TransientSymbol": [broken_str]
        }
    )
    initialization_failure = {
        "path": broken_str,
        "error": "Failed to initialize parser for python: cache denied",
        "parse_failed": True,
        "parser_initialization_failed": True,
    }
    broken_update_entered = threading.Event()
    release_broken_update = threading.Event()
    second_queued = threading.Event()

    def wait_for_queued_event():
        broken_update_entered.set()
        assert release_broken_update.wait(timeout=5), "test did not release the failed update"

    def parse_file(repo, path):
        if path == broken:
            wait_for_queued_event()
            return dict(initialization_failure)
        return {"path": str(path), "functions": [], "classes": []}

    graph.parse_file.side_effect = parse_file
    # This is the existing GraphBuilder contract: initialization failure leaves
    # the old graph untouched and returns None; successful updates return data.
    def update_file(path, *args):
        if path == broken:
            wait_for_queued_event()
            return None
        return {"path": str(path), "functions": [], "classes": []}

    graph.update_file_in_graph.side_effect = update_file
    handler = watcher.RepositoryEventHandler(graph, tmp_path, perform_initial_scan=False)
    handler.imports_map = {
        "OldSymbol": [broken_str],
        "NeighborSymbol": [neighbor_str],
        "OldWorkingSymbol": [working_str],
        "SharedSymbol": [broken_str, working_str],
    }

    def handle_second_event():
        second_queued.set()
        handler._handle_modification(working_str)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handler._handle_modification, broken_str)
        try:
            assert broken_update_entered.wait(timeout=5), "failed file update never started"
            second = pool.submit(handle_second_event)
            assert second_queued.wait(timeout=5), "second event was never queued"
            assert not second.done(), "the second update bypassed the existing update lock"
        finally:
            release_broken_update.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert handler.imports_map == {
        "OldSymbol": [broken_str],
        "NeighborSymbol": [neighbor_str],
        "SharedSymbol": [broken_str],
        "NewSymbol": [working_str],
    }
    graph.delete_outgoing_calls_from_files.assert_not_called()
    graph.delete_inherits_for_files.assert_not_called()
    graph.link_function_calls.assert_called_once()
    graph.link_inheritance.assert_called_once()
    assert graph.link_function_calls.call_args.args[0] == [
        {"path": working_str, "functions": [], "classes": []}
    ]


def test_deleted_file_result_still_updates_neighbors(monkeypatch, tmp_path):
    deleted = tmp_path / "deleted.py"
    neighbor = tmp_path / "neighbor.py"
    neighbor.write_text("pass\n", encoding="utf-8")
    deleted_str, neighbor_str = str(deleted.resolve()), str(neighbor.resolve())
    monkeypatch.setattr(watcher.RepositoryEventHandler, "_load_ignore_spec", lambda *args: None)
    monkeypatch.setattr(watcher.RepositoryEventHandler, "_should_ignore", lambda *args: False)
    monkeypatch.setattr(config_manager, "get_config_value", lambda name: "false")
    graph = Mock(spec=GraphBuilder)
    graph.parsers = {".py": "python"}
    graph.get_caller_file_paths.return_value = [neighbor_str]
    graph.get_inheritance_neighbor_paths.return_value = [neighbor_str]
    graph.get_repo_class_lookup.return_value = {}
    graph.update_file_in_graph.return_value = {"deleted": True, "path": deleted_str}
    neighbor_data = {"path": neighbor_str, "functions": [], "classes": []}
    graph.parse_file.return_value = neighbor_data
    handler = watcher.RepositoryEventHandler(graph, tmp_path, perform_initial_scan=False)
    handler.imports_map = {"DeletedSymbol": [deleted_str], "NeighborSymbol": [neighbor_str]}

    handler._handle_modification(deleted_str)

    assert handler.imports_map == {"NeighborSymbol": [neighbor_str]}
    graph.pre_scan_imports.assert_not_called()
    graph.delete_outgoing_calls_from_files.assert_called_once_with([neighbor_str])
    graph.delete_inherits_for_files.assert_called_once_with([neighbor_str])
    graph.link_function_calls.assert_called_once_with([neighbor_data], handler.imports_map, {})
    graph.link_inheritance.assert_called_once_with([neighbor_data], handler.imports_map)
