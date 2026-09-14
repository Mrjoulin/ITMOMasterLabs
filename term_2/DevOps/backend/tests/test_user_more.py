import pytest
import unittest
from src.routes.user import read_users_me
from src.models.userinfo import UserInfo


def test_read_users_me_returns_user():
    u = UserInfo(id=1, email="u@x.com", full_name="U")
    assert read_users_me(current_user=u) == u
