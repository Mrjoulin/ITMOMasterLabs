import pytest
from pathlib import Path

from security import keys as keys_mod


def test__read_file_success(tmp_path):
    p = tmp_path / "k.pem"
    p.write_text("secret123")
    assert keys_mod._read_file(str(p)) == "secret123"


def test__read_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        keys_mod._read_file(str(tmp_path / "nope.pem"))
