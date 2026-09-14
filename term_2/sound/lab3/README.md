# audio-processing-labs-2026

## Лабораторная работа №3 (Neural TTS)

- **Ноутбук:** [src/lab3.ipynb](src/lab3.ipynb)
- **Данные (для дообучения):** `lab3_data/LJSpeech-1.1/` — нужны `metadata.csv` + папка `wavs/`
- **Текст для синтеза:** `lab3_text/lab3_test_sentences.txt` (создаётся автоматически)

### Модели

| Модель | Источник | Требования |
|--------|----------|------------|
| Tacotron2-DDC | `tts_models/en/ljspeech/tacotron2-DDC` (Coqui TTS) | — |
| VITS | `tts_models/en/ljspeech/vits` (Coqui TTS) | espeak-ng |

### Быстрый старт (Windows)

```powershell
# 1. Установить espeak-ng (нужен для VITS)
winget install eSpeak-NG.eSpeak-NG

# 2. Запустить ноутбук
.venv\Scripts\jupyter lab
# затем открыть src/lab3.ipynb
```

### Дообучение (опционально)

Скачать LJSpeech-1.1 (~2.6 ГБ) в нужный каталог:
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2 -C audio/lab3_data/
```
После этого ноутбук автоматически обнаружит датасет и установит `DO_TRAINING = True`.

### Содержание ноутбука

1. Корпус текстов (~200 слов, нейтральные / вопросы / восклицания / двоеточие / тире)
2. Синтез речи предобученными Tacotron2 и VITS
3. Прослушивание и сравнение аудио
4. Mel-кепстральные спектрограммы
5. Акустические признаки (MFCC, F0, спектральный центроид, RMS …)
6. Метрики: MCD, Log Spectral Distance
7. Дообучение Tacotron2 (2 конфигурации, гиперпараметры)
8. Логи обучения: loss, learning rate, MCD через шаги
9. Анимация эволюции mel-спектрограммы
10. Сравнение «до / после» дообучения
11. Сводная таблица и выводы