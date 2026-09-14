# tests/test_routes/test_user.py
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from sqlmodel import Session
from src.routes.user import create_user, login, get_current_user
from src.models.userinfo import UserCreate, UserInfo
from src.models.area import Area

# Фикстуры
@pytest.fixture
def mock_session():
    return Mock(spec=Session)

@pytest.fixture
def user_data():
    return UserCreate(email="test@example.com", full_name="Test User", password="Secret123!")

# 1. Успешная регистрация с созданием дефолтной области
@patch("src.routes.user.create_access_token")
def test_create_user_success_creates_default_area(mock_token, mock_session, user_data):
    mock_token.return_value = "token123"
    mock_session.exec.return_value.first.return_value = None  # email не занят

    # Simulate commit() and refresh() setting the user ID
    def set_user_id_on_refresh(user_obj):
        user_obj.id = 1  # Simulate DB-assigned ID
    
    mock_session.refresh.side_effect = set_user_id_on_refresh

    result = create_user(session=mock_session, user_in=user_data)

    assert result.access_token == "token123"
    # Проверяем, что добавили пользователя и область (2 add calls)
    assert mock_session.add.call_count == 2
    # Проверяем, что область называется "Work" и привязана к пользователю
    area_call = mock_session.add.call_args_list[1][0][0]
    assert isinstance(area_call, Area)
    assert area_call.name == "Work"
    assert area_call.user_id == 1 

# 2. Регистрация с уже существующим email → 400
def test_create_user_duplicate_email_raises_error(mock_session, user_data):
    # Симулируем, что пользователь уже есть
    existing_user = UserInfo(email=user_data.email)
    mock_session.exec.return_value.first.return_value = existing_user

    with pytest.raises(HTTPException) as exc:
        create_user(session=mock_session, user_in=user_data)
    assert exc.value.status_code == 400
    assert "already exists" in exc.value.detail.lower()

# 3. Хэширование пароля работает (интеграционный, но без БД)
def test_password_hashing_on_registration(mock_session, user_data):
    mock_session.exec.return_value.first.return_value = None

    with patch("src.routes.user.get_password_hash") as mock_hash:
        mock_hash.return_value = "hashed_123"
        create_user(session=mock_session, user_in=user_data)

    # Проверяем, что пароль был захэширован
    mock_hash.assert_called_once_with("Secret123!")
    # Проверяем, что пользователь сохраняется с хэшем
    user_added = mock_session.add.call_args_list[0][0][0]
    assert user_added.hashed_password == "hashed_123"
    assert not hasattr(user_added, "password")  # исходный пароль не хранится

# 4. Логин с правильным паролем возвращает токен
@patch("src.routes.user.create_access_token")
@patch("src.routes.user.verify_password")
def test_login_success(mock_verify, mock_token, mock_session):
    mock_verify.return_value = True
    mock_token.return_value = "token123"
    user = UserInfo(email="test@example.com", hashed_password="hash")
    mock_session.exec.return_value.first.return_value = user

    form_data = Mock(username="test@example.com", password="pass")
    result = login(session=mock_session, form_data=form_data)

    assert result.access_token == "token123"
    mock_session.add.assert_called_once_with(user)  # обновился last_login
    mock_session.commit.assert_called_once()

# 5. Получение текущего пользователя по токену (интеграция с security)
@patch("src.routes.user.decode_access_token")
def test_get_current_user_valid_token(mock_decode, mock_session):
    mock_decode.return_value = {"sub": "test@example.com"}
    user = UserInfo(email="test@example.com")
    mock_session.exec.return_value.first.return_value = user

    result = get_current_user(session=mock_session, token="valid_token")

    assert result == user
    mock_decode.assert_called_once_with("valid_token")
