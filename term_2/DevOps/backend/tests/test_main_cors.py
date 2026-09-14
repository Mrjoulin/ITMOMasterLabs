
import unittest
from src.core import constants

def test_constants_strings():
    assert isinstance(constants.AREA_NOT_FOUND, str)
    assert "Area" in constants.AREA_NOT_FOUND
    assert isinstance(constants.NOTE_NOT_FOUND, str)
    assert isinstance(constants.TASK_NOT_FOUND, str)
