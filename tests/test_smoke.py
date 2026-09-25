"""Bootstrap smoke test: package imports and config paths resolve."""

import rag_lab
from rag_lab import config


def test_package_imports() -> None:
    assert rag_lab.__version__ == "0.1.0"


def test_raw_data_dir_exists() -> None:
    assert config.RAW_DATA_DIR.is_dir()
