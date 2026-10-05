"""Real embedded-graph regressions for a native grammar-cache outage.

Each scenario runs in a fresh process because the grammar provider, parser
manager, and embedded backend cache process-global state. No production module,
parser, database, schema, writer, or job handler is replaced. The configurable
native cache API is optional across the supported language-pack versions.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.integration
_RESULT_PREFIX = "CGC_NATIVE_RESULT="


def _run_scenario(tmp_path, scenario, backend):
    pytest.importorskip({"kuzudb": "kuzu", "ladybugdb": "ladybug"}[backend])
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_language_pack")
    package = importlib.util.find_spec("codegraphcontext")
    source_root = str(Path(package.origin).resolve().parent.parent)
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("NEO4J_", "NORNIC_", "FALKORDB_", "CGC_RUNTIME_")) or key in {
            "DEFAULT_DATABASE", "DATABASE_TYPE", "KUZUDB_PATH", "LADYBUGDB_PATH",
            "LOG_FILE_PATH", "DEBUG_LOG_PATH",
        }:
            del env[key]
    env.update(
        PYTHONPATH=source_root,
        PYTHONDONTWRITEBYTECODE="1",
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        ENABLE_APP_LOGS="CRITICAL",
        DEBUG_LOGS="false",
        LOG_FILE_PATH=str(home / "integration.log"),
        DEBUG_LOG_PATH=str(home / "debug.log"),
        CGC_EMBEDDED_BUFFER_POOL_MB="256",
        SCIP_INDEXER="false",
        CGC_IGNORE_PROJECT_ENV="true",
        ENABLE_VECTOR_RESOLVE="false",
        ENABLE_INHERIT_RESOLVE="false",
        ENABLE_AUTO_WATCH="false",
        PARALLEL_WORKERS="2",
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario, str(tmp_path), backend],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = [line for line in completed.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    assert len(results) == 1, completed.stdout + completed.stderr
    result = json.loads(results[0][len(_RESULT_PREFIX):])
    if "skip" in result:
        pytest.skip(result["skip"])
    return result


@pytest.mark.parametrize("backend", ["kuzudb", "ladybugdb"])
def test_native_cache_failure_reaches_completed_job_and_summary(tmp_path, backend):
    result = _run_scenario(tmp_path, "pipeline", backend)
    summary = result["summary"]
    assert summary, result["status"]
    assert summary["total_scanned_files"] == 2
    assert summary["failed_files"] == 1
    assert summary["function_nodes"] == summary["class_nodes"] == 0
    assert len(summary["failed_file_details"]) == 1
    failure = summary["failed_file_details"][0]
    assert failure["path"] == result["source_path"]
    assert result["native_error"] in failure["error"]
    assert "pip install" not in failure["error"]

    status = result["status"]
    assert status["success"] is True
    job = status["job"]
    assert job["status"] == "completed"
    assert job["total_files"] == job["processed_files"] == 2
    assert job["end_time"] is not None
    assert job["errors"] == ["Earlier diagnostic", f"{failure['path']}: {failure['error']}"]
    # Only the generic file persists. Writing a placeholder for the failed
    # source would incorrectly satisfy the normal-index resume census.
    assert result["files"] == [result["generic_path"]]


@pytest.mark.parametrize("backend", ["kuzudb", "ladybugdb"])
def test_native_cache_failure_preserves_existing_file_graph(tmp_path, backend):
    result = _run_scenario(tmp_path, "update", backend)
    assert result["before"] == {"Repository": 1, "File": 1, "Function": 1}
    assert result["after"] == result["before"]
    assert result["function_names"] == ["preserved"]
    assert result["update_result"] is None
    parsed = result["parsed"]
    assert parsed["parse_failed"] is True
    assert parsed["parser_initialization_failed"] is True
    assert result["native_error"] in parsed["error"]
    assert "unsupported" not in parsed


def _native_cache_failure(root):
    """Create a genuine provider filesystem error, before any download."""
    import tree_sitter_language_pack as pack

    if not all(hasattr(pack, name) for name in ("configure", "PackConfig")):
        return None, "Installed language-pack has no configurable native cache API"
    cache_file = root / "grammar-cache-file"
    cache_file.write_text("The configured cache is deliberately an ordinary file.", encoding="utf-8")
    try:
        configuration = pack.PackConfig(cache_dir=str(cache_file))
    except TypeError:
        return None, "Installed language-pack does not accept PackConfig(cache_dir=...)"
    pack.configure(configuration)
    try:
        pack.get_language("python")
    except Exception as error:
        # Ensure this is the local setup error being exercised, rather than a
        # missing wheel or an unrelated network/download-service failure.
        if str(cache_file) not in str(error):
            raise AssertionError(f"Unexpected native failure: {error}") from error
        return str(error), None
    return None, "Installed language-pack bypasses the configurable cache for Python"


def _scenario(scenario, root, backend):
    import asyncio

    from codegraphcontext.core import get_database_manager
    from codegraphcontext.core.jobs import JobManager
    from codegraphcontext.tools.graph_builder import GraphBuilder

    native_error, skip = _native_cache_failure(root)
    if skip:
        return {"skip": skip}

    repo = root / "repo"
    repo.mkdir()
    source = repo / "sample.py"
    source.write_text("def preserved():\n    return 42\n", encoding="utf-8")
    jobs = JobManager()
    os.environ["CGC_RUNTIME_DB_TYPE"] = backend
    db = get_database_manager(db_path=str(root / "graph"))
    try:
        builder = GraphBuilder(db, jobs, None)
        result = {"source_path": str(source), "native_error": native_error}
        if scenario == "pipeline":
            from codegraphcontext.tools.handlers.management_handlers import check_job_status

            generic = repo / "README.md"
            generic.write_text("A generic file remains indexable.\n", encoding="utf-8")
            job_id = jobs.create_job(str(repo))
            jobs.update_job(job_id, errors=["Earlier diagnostic"])
            asyncio.run(builder.build_graph_from_path_async(repo, job_id=job_id))
            with db.get_driver().session() as session:
                files = sorted(row["path"] for row in session.run("MATCH (f:File) RETURN f.path AS path"))
            result.update(
                generic_path=generic.resolve().as_posix(),
                summary=builder.last_index_summary,
                status=check_job_status(jobs, job_id=job_id),
                files=files,
            )
        elif scenario == "update":
            builder.add_repository_to_graph(repo)
            builder.add_file_to_graph(
                {
                    "path": str(source), "repo_path": str(repo), "lang": "python",
                    "functions": [{
                        "name": "preserved", "line_number": 1, "end_line": 2,
                        "args": [], "docstring": "", "source": source.read_text(encoding="utf-8"),
                    }],
                    "classes": [], "imports": [], "function_calls": [],
                },
                repo.name,
                {},
                repo_path_str=str(repo),
            )

            def census():
                with db.get_driver().session() as session:
                    return {
                        label: session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()["count"]
                        for label in ("Repository", "File", "Function")
                    }

            result["before"] = census()
            result["parsed"] = builder.parse_file(repo, source)
            result["update_result"] = builder.update_file_in_graph(source, repo, {})
            result["after"] = census()
            with db.get_driver().session() as session:
                result["function_names"] = sorted(
                    row["name"] for row in session.run("MATCH (f:Function) RETURN f.name AS name")
                )
        else:
            raise ValueError(scenario)
        return result
    finally:
        db.close_driver()


if __name__ == "__main__":
    print(_RESULT_PREFIX + json.dumps(_scenario(sys.argv[1], Path(sys.argv[2]), sys.argv[3])))
