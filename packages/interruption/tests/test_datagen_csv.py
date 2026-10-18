import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "multiprocess_datagen.py"


@pytest.fixture
def datagen(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("multiprocess_datagen_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_procthor_10k_dir", lambda: tmp_path)
    return module


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_csv_path_includes_scene_key(datagen, tmp_path):
    assert datagen._csv_path("procthor_data", 64, 123) == tmp_path / "procthor_data_64_123.csv"


def test_different_keys_write_to_different_csvs(datagen, tmp_path):
    datagen.write_datum_to_file(64, 111, ("graph", 1.0), 0)
    datagen.write_datum_to_file(64, 222, ("graph", 2.0), 0)

    assert _lines(tmp_path / "procthor_data_64_111.csv") == [str(datagen.datum_pickle_path(64, 111, 0))]
    assert _lines(tmp_path / "procthor_data_64_222.csv") == [str(datagen.datum_pickle_path(64, 222, 0))]


def test_datums_are_listed_in_the_order_written(datagen, tmp_path):
    for counter in range(3):
        datagen.write_datum_to_file(64, 111, ("graph", float(counter)), counter)

    assert _lines(tmp_path / "procthor_data_64_111.csv") == [
        str(datagen.datum_pickle_path(64, 111, c)) for c in range(3)
    ]


def test_individual_task_costs_have_their_own_key_named_csv(datagen, tmp_path):
    datagen.write_out_individual_task_costs(64, 111, "graph", (["taskA", "taskB"], [1.0, 2.0]), 0)
    datagen.write_datum_to_file(64, 111, ("graph", 1.0), 0)

    assert _lines(tmp_path / "procthor_individual_task_data_64_111.csv") == [
        str(datagen._task_datum_pickle_path(64, 111, 0, idx)) for idx in range(2)
    ]
    assert _lines(tmp_path / "procthor_data_64_111.csv") == [
        str(datagen.datum_pickle_path(64, 111, 0))
    ]
