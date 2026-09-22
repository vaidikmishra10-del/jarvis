"""Tests for the user-memory store and @function_tool wrappers."""

import inspect

import pytest

import memory_store
from memory_store import MemoryStore, add_fact, list_facts

# ---------------------------------------------------------------------------
# MemoryStore core
# ---------------------------------------------------------------------------


def _store(tmp_path: object) -> MemoryStore:
    return MemoryStore(tmp_path / "mem.db")  # type: ignore[arg-type]


class TestRememberFact:
    def test_returns_true_on_first_insert(self, tmp_path: object) -> None:
        assert _store(tmp_path).remember_fact("The user is Satvik") is True

    def test_returns_false_on_duplicate(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        store.remember_fact("Favourite language is Dart")
        assert store.remember_fact("Favourite language is Dart") is False

    def test_whitespace_is_normalized(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        store.remember_fact("   The   user   likes   chai   ")
        assert store.recall_facts() == ["The user likes chai"]

    @pytest.mark.parametrize("fact", ["", "   ", "\n"])
    def test_blank_raises_value_error(self, tmp_path: object, fact: str) -> None:
        with pytest.raises(ValueError):
            _store(tmp_path).remember_fact(fact)


class TestRecallFacts:
    def test_empty_when_new(self, tmp_path: object) -> None:
        assert _store(tmp_path).recall_facts() == []

    def test_returns_newest_first(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        store.remember_fact("one")
        store.remember_fact("two")
        store.remember_fact("three")
        assert store.recall_facts() == ["three", "two", "one"]


class TestPersistence:
    def test_facts_survive_new_instance_on_same_db(self, tmp_path: object) -> None:
        db = tmp_path / "facts.db"  # type: ignore[operator]
        MemoryStore(db).remember_fact("Persistence check")
        assert MemoryStore(db).recall_facts() == ["Persistence check"]

    def test_second_independent_db_is_isolated(self, tmp_path: object) -> None:
        a = MemoryStore(tmp_path / "a.db")  # type: ignore[arg-type]
        b = MemoryStore(tmp_path / "b.db")  # type: ignore[arg-type]
        a.remember_fact("only in A")
        assert b.recall_facts() == []


class TestClear:
    def test_removes_all_facts(self, tmp_path: object) -> None:
        store = _store(tmp_path)
        store.remember_fact("temp")
        store.clear()
        assert store.recall_facts() == []


# ---------------------------------------------------------------------------
# Module-level helpers (JARVIS_MEMORY_DB env-driven)
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_add_and_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("JARVIS_MEMORY_DB", str(tmp_path / "helper.db"))
        assert add_fact("Stored via helper") is True
        assert list_facts() == ["Stored via helper"]

    def test_list_facts_returns_empty_for_fresh_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("JARVIS_MEMORY_DB", str(tmp_path / "fresh.db"))
        assert list_facts() == []


# ---------------------------------------------------------------------------
# @function_tool wrappers - schema / basic wiring
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_tool_names(self) -> None:
        assert memory_store.remember_fact.id == "remember_fact"  # type: ignore[union-attr]
        assert memory_store.recall_facts.id == "recall_facts"  # type: ignore[union-attr]

    def test_remember_fact_accepts_string_fact(self) -> None:
        sig = inspect.signature(memory_store.remember_fact._func)  # type: ignore[union-attr]
        params = [p for p in sig.parameters.values() if p.annotation != "RunContext"]
        assert len(params) == 1
        assert params[0].name == "fact"

    @pytest.mark.asyncio
    async def test_remember_fact_tool_persists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("JARVIS_MEMORY_DB", str(tmp_path / "tool.db"))
        await memory_store.remember_fact(None, "Tool wiring check")  # type: ignore[union-attr]
        assert list_facts() == ["Tool wiring check"]

    @pytest.mark.asyncio
    async def test_recall_facts_tool_returns_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("JARVIS_MEMORY_DB", str(tmp_path / "tool2.db"))
        add_fact("known fact")
        result = await memory_store.recall_facts(None)  # type: ignore[union-attr]
        assert "known fact" in result
