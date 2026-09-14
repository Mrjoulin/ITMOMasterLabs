import unittest
from src.core.database import get_session
from sqlmodel import Session


def test_get_session_yields_session_and_closes():
    gen = get_session()
    sess = next(gen)
    assert isinstance(sess, Session)
    # Close generator to trigger context manager cleanup
    gen.close()
