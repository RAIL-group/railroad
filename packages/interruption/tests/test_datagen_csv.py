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


def test_csv_path_includes_scene_key_and_shard(datagen, tmp_path):
    assert datagen._csv_path("procthor_data", 64, 123) == tmp_path / "procthor_data_64_123.csv"
    assert datagen._csv_path("procthor_data", 64, 123, 2) == tmp_path / "procthor_data_64_123_2.csv"


def test_different_keys_write_to_different_csvs(datagen, tmp_path):
    datagen.write_datum_to_file(64, 111, ("graph", 1.0), 0, csv_suffix=0)
    datagen.write_datum_to_file(64, 222, ("graph", 2.0), 0, csv_suffix=0)
    datagen._merge_csv_shards(64, 111, num_workers=1)
    datagen._merge_csv_shards(64, 222, num_workers=1)

    first = _lines(tmp_path / "procthor_data_64_111.csv")
    second = _lines(tmp_path / "procthor_data_64_222.csv")
    assert first == [str(datagen.datum_pickle_path(64, 111, 0))]
    assert second == [str(datagen.datum_pickle_path(64, 222, 0))]


def test_merge_combines_only_the_matching_keys_shards(datagen, tmp_path):
    for worker_id in range(2):
        datagen.write_datum_to_file(64, 111, ("graph", 1.0), worker_id, csv_suffix=worker_id)
    datagen.write_datum_to_file(64, 222, ("graph", 2.0), 0, csv_suffix=0)

    datagen._merge_csv_shards(64, 111, num_workers=2)

    combined = _lines(tmp_path / "procthor_data_64_111.csv")
    assert sorted(combined) == sorted(
        str(datagen.datum_pickle_path(64, 111, c)) for c in range(2)
    )
    assert not (tmp_path / "procthor_data_64_111_0.csv").exists()
    assert not (tmp_path / "procthor_data_64_111_1.csv").exists()
    # the other key's shard is untouched until that key is merged
    assert (tmp_path / "procthor_data_64_222_0.csv").exists()
    assert not (tmp_path / "procthor_data_64_222.csv").exists()


def test_individual_task_csvs_are_key_named_and_merge_separately(datagen, tmp_path):
    datagen.write_out_individual_task_costs(
        64, 111, "graph", (["taskA", "taskB"], [1.0, 2.0]), 0, csv_suffix=0
    )
    datagen.write_datum_to_file(64, 111, ("graph", 1.0), 0, csv_suffix=0)

    datagen._merge_csv_shards(64, 111, num_workers=1, individual_tasks=True)
    datagen._merge_csv_shards(64, 111, num_workers=1)

    task_lines = _lines(tmp_path / "procthor_individual_task_data_64_111.csv")
    assert task_lines == [
        str(datagen._task_datum_pickle_path(64, 111, 0, idx)) for idx in range(2)
    ]
    assert _lines(tmp_path / "procthor_data_64_111.csv") == [
        str(datagen.datum_pickle_path(64, 111, 0))
    ]
