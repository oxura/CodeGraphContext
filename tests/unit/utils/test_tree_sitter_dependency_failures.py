"""Provider absence and runtime grammar failures must remain distinguishable."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from codegraphcontext.utils import tree_sitter_manager as manager


class DownloadError(Exception):
    """Synthetic provider failure; native error behavior has integration coverage."""


@pytest.fixture
def dependencies(monkeypatch):
    for name in ("_Language", "_Parser", "_get_language", "_tree_sitter_import_error"):
        monkeypatch.setattr(manager, name, None)

    language = object()
    tree_sitter = ModuleType("tree_sitter")
    tree_sitter.Language = type("Language", (), {})
    tree_sitter.Parser = Mock(name="Parser")
    modern = ModuleType("tree_sitter_language_pack")
    modern.get_language = Mock(return_value=language)
    legacy = ModuleType("tree_sitter_languages")
    legacy.get_language = Mock(return_value=language)
    for module in (tree_sitter, modern, legacy):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return tree_sitter, modern, legacy, language


def test_missing_modern_provider_selects_installed_legacy(monkeypatch, dependencies):
    tree_sitter, _modern, legacy, _language = dependencies
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)

    assert manager._load_tree_sitter_dependencies() == (
        tree_sitter.Language, tree_sitter.Parser, legacy.get_language
    )


@pytest.mark.parametrize("missing", ["providers", "tree_sitter"])
def test_missing_dependencies_keep_actionable_install_error(monkeypatch, dependencies, missing):
    monkeypatch.setattr(manager.sys, "version_info", (3, 12, 0))
    names = ("tree_sitter_language_pack", "tree_sitter_languages") if missing == "providers" else ("tree_sitter",)
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ImportError, match=r"pip install codegraphcontext\[parsing\]") as raised:
        manager._load_tree_sitter_dependencies()

    assert isinstance(raised.value.__cause__, ImportError)
    assert manager._Language is manager._Parser is manager._get_language is None


@pytest.mark.parametrize("error_type", [DownloadError, ImportError])
def test_runtime_language_error_is_preserved_and_can_be_retried(monkeypatch, dependencies, error_type):
    tree_sitter, modern, legacy, language = dependencies
    error = error_type("native cache manifest could not be read")
    modern.get_language.side_effect = [error, language]
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", None)

    with pytest.raises(error_type) as raised:
        manager._load_tree_sitter_dependencies()

    assert raised.value is error
    legacy.get_language.assert_not_called()
    tree_sitter.Parser.assert_not_called()
    assert manager._Language is manager._Parser is manager._get_language is None
    assert manager._tree_sitter_import_error is None

    expected = (tree_sitter.Language, tree_sitter.Parser, modern.get_language)
    assert manager._load_tree_sitter_dependencies() == expected
    assert manager._load_tree_sitter_dependencies() == expected
    assert modern.get_language.call_count == 2
    tree_sitter.Parser.assert_called_once_with(language)
    legacy.get_language.assert_not_called()


def test_native_parser_constructor_error_is_not_replaced_by_missing_package(monkeypatch, dependencies):
    tree_sitter, _modern, legacy, _language = dependencies
    error = RuntimeError("native parser ABI could not be initialized")
    tree_sitter.Parser.side_effect = error
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", None)

    with pytest.raises(RuntimeError) as raised:
        manager._load_tree_sitter_dependencies()

    assert raised.value is error
    legacy.get_language.assert_not_called()
    assert manager._Language is manager._Parser is manager._get_language is None


def test_older_parser_api_remains_supported(dependencies):
    tree_sitter, modern, legacy, language = dependencies
    parser = Mock()
    tree_sitter.Parser.side_effect = [TypeError("constructor takes no arguments"), parser]

    assert manager._load_tree_sitter_dependencies() == (
        tree_sitter.Language, tree_sitter.Parser, modern.get_language
    )
    parser.set_language.assert_called_once_with(language)
    legacy.get_language.assert_not_called()


def test_older_parser_initialization_error_is_preserved(monkeypatch, dependencies):
    tree_sitter, _modern, legacy, _language = dependencies
    error = RuntimeError("native language initialization failed")
    parser = Mock()
    parser.set_language.side_effect = error
    tree_sitter.Parser.side_effect = [TypeError("old constructor"), parser]
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", None)

    with pytest.raises(RuntimeError) as raised:
        manager._load_tree_sitter_dependencies()

    assert raised.value is error
    legacy.get_language.assert_not_called()
    assert manager._Language is manager._Parser is manager._get_language is None


@pytest.mark.parametrize("module_name", ["tree_sitter", "tree_sitter_language_pack", "tree_sitter_languages"])
@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
def test_installed_module_import_failure_keeps_original_diagnostic(
    monkeypatch, dependencies, module_name, error_type
):
    import builtins

    _tree_sitter, _modern, legacy, _language = dependencies
    if module_name == "tree_sitter_languages":
        monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    elif module_name == "tree_sitter_language_pack":
        monkeypatch.setitem(sys.modules, "tree_sitter_languages", None)
    error = error_type("installed native extension is broken", name=f"{module_name}._native")
    original_import = builtins.__import__

    def import_with_broken_extension(name, *args, **kwargs):
        if name == module_name:
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_extension)

    with pytest.raises(ImportError) as raised:
        manager._load_tree_sitter_dependencies()

    if module_name == "tree_sitter":
        assert raised.value.__cause__ is error
        assert str(error) in str(raised.value)
        assert "Install" in str(raised.value)
    else:
        assert raised.value is error
    legacy.get_language.assert_not_called()
    assert manager._Language is manager._Parser is manager._get_language is None


@pytest.mark.parametrize("failure_stage", ["language", "constructor"])
def test_failed_legacy_probe_retains_original_primary_error(dependencies, failure_stage):
    tree_sitter, modern, legacy, language = dependencies
    primary_error = DownloadError("modern grammar cache is unavailable")
    legacy_error = RuntimeError(f"legacy {failure_stage} initialization failed")
    modern.get_language.side_effect = primary_error
    if failure_stage == "language":
        legacy.get_language.side_effect = legacy_error
    else:
        tree_sitter.Parser.side_effect = legacy_error

    with pytest.raises(DownloadError) as raised:
        manager._load_tree_sitter_dependencies()

    assert raised.value is primary_error
    assert raised.value.__cause__ is legacy_error
    # A fallback raised inside the primary except scope would form a cycle
    # when the original primary exception is raised with the fallback as cause.
    assert legacy_error.__context__ is not primary_error
    assert legacy_error.__cause__ is not primary_error
    legacy.get_language.assert_called_once_with("python")
    if failure_stage == "constructor":
        tree_sitter.Parser.assert_called_once_with(language)
    else:
        tree_sitter.Parser.assert_not_called()
    assert manager._Language is manager._Parser is manager._get_language is None


@pytest.mark.parametrize("failure_stage", ["language", "constructor"])
def test_missing_modern_provider_exposes_broken_legacy(monkeypatch, dependencies, failure_stage):
    tree_sitter, _modern, legacy, language = dependencies
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    error = RuntimeError(f"legacy {failure_stage} initialization failed")
    if failure_stage == "language":
        legacy.get_language.side_effect = error
    else:
        tree_sitter.Parser.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        manager._load_tree_sitter_dependencies()

    assert raised.value is error
    legacy.get_language.assert_called_once_with("python")
    if failure_stage == "constructor":
        tree_sitter.Parser.assert_called_once_with(language)
    else:
        tree_sitter.Parser.assert_not_called()
    assert manager._Language is manager._Parser is manager._get_language is None


def test_identical_provider_errors_do_not_create_self_causes(dependencies):
    _tree_sitter, modern, legacy, _language = dependencies
    error = DownloadError("shared grammar-cache failure")
    modern.get_language.side_effect = error
    legacy.get_language.side_effect = error

    with pytest.raises(DownloadError) as raised:
        manager._load_tree_sitter_dependencies()

    assert raised.value is error
    assert error.__cause__ is not error
    assert error.__context__ is not error
    assert manager._Language is manager._Parser is manager._get_language is None
