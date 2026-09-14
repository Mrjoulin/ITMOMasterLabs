import os
import random
import string
from locust import HttpUser, task, between, events
import gevent


class FocusFlowUser(HttpUser):
    """
    Пользователь, который:
    1. При старте создаёт область (area) и сохраняет её ID.
    2. В основной задаче создаёт задачу (task) с случайным заголовком и сразу удаляет её.
    3. После завершения всех задач удаляет созданную область.
    """
    wait_time = between(1, 3)  # Пауза между выполнением задач
    host = "http://158.160.228.194/api/v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Флаг для отслеживания, нужно ли удалять area
        self.area_created = False
        self.area_id = None

    def on_start(self):
        """
        Инициализация: читаем токен, создаём область и сохраняем её ID.
        """
        # Получаем токен из переменной окружения
        self.token = os.getenv("FOCUSFLOW_TOKEN")
        if not self.token:
            raise Exception("❌ Токен не найден. Задайте переменную окружения FOCUSFLOW_TOKEN")

        # Устанавливаем заголовок авторизации для всех последующих запросов
        self.client.headers.update({
            "Authorization": f"Bearer {self.token}",
            'Content-Type': 'application/json'
        })

        # Создаём область (area)
        area_payload = {
            "name": self.random_title(10),
            "color": "blue"
        }

        with self.client.post("/areas/", json=area_payload, catch_response=True) as resp:
            if resp.status_code not in (200, 201):
                resp.failure(f"Failed to create area: {resp.status_code}")
                raise Exception("Не удалось создать область — прерываем инициализацию пользователя")
            try:
                data = resp.json()
                self.area_id = data.get("id")
                if not self.area_id:
                    raise ValueError("Ответ не содержит 'id'")
                self.area_created = True
                print(f"✅ Пользователь {id(self)} создал область с ID: {self.area_id}")
            except Exception as e:
                resp.failure(f"Ошибка парсинга ответа area: {e}")
                raise

    def on_stop(self):
        """
        Выполняется при остановке пользователя.
        Удаляем созданную область, если она была успешно создана.
        """
        if self.area_created and self.area_id:
            with self.client.delete(f"/areas/{self.area_id}", catch_response=True) as resp:
                if resp.status_code == 204:
                    print(f"✅ Пользователь {id(self)} удалил область {self.area_id}")
                    resp.success()
                else:
                    resp.failure(f"Failed to delete area {self.area_id}: {resp.status_code}")

    def random_title(self, length=20):
        """Генерирует случайную строку для заголовка задачи."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @task
    def create_and_delete_task(self):
        """
        Создаёт задачу с привязкой к сохранённой области,
        получает её ID и сразу удаляет.
        """
        if not self.area_created:
            return  # Если область не создана, не выполняем задачу

        # Генерируем случайный заголовок
        title = self.random_title()

        # Создаём задачу
        task_payload = {"title": title, "area_id": self.area_id}
        with self.client.post("/tasks/", json=task_payload, catch_response=True) as create_resp:
            if create_resp.status_code not in (200, 201):
                create_resp.failure(f"Create task failed: {create_resp.status_code}")
                return  # Не пытаемся удалять, если создание не удалось
            try:
                task_data = create_resp.json()
                task_id = task_data.get("id")
                if not task_id:
                    create_resp.failure("Ответ создания задачи не содержит 'id'")
                    return
            except Exception as e:
                create_resp.failure(f"Ошибка парсинга ответа задачи: {e}")
                return

        # Небольшая пауза перед удалением (опционально)
        gevent.sleep(0.5)  # имитация работы с задачей

        # Удаляем задачу
        with self.client.delete(f"/tasks/{task_id}", catch_response=True) as delete_resp:
            if delete_resp.status_code == 204:
                delete_resp.success()
            else:
                delete_resp.failure(f"Delete task failed: {delete_resp.status_code}")