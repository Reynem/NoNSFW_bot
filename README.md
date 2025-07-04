# Телеграм-бот для модерации NSFW-контента

Это небольшой Телеграм-бот, который анализирует отправленные пользователями изображения и определяет, содержат ли они NSFW (Not Safe For Work) контент. Если изображение определяется как NSFW, бот автоматически удаляет сообщение.

## ✨ Возможности

  * **Анализ изображений**: Бот проверяет каждое отправленное фото на наличие NSFW-контента.
  * **Автоматическая модерация**: Сообщения с контентом, помеченным как NSFW, удаляются из чата.
  * **Высокая производительность**:
      * Использует асинхронную архитектуру (`asyncio`) для обработки запросов без блокировок.
      * Выполняет ресурсоемкую задачу анализа изображений в отдельном потоке (`ThreadPoolExecutor`).
      * Автоматически использует GPU (CUDA), если он доступен, для значительного ускорения обработки.

## ⚙️ Как это работает

Бот построен с использованием библиотеки `python-telegram-bot` и `transformers` от Hugging Face.

1.  **Получение изображения**: Бот принимает изображение от пользователя в чате Telegram.
2.  **Анализ**: Изображение передается в предобученную модель `Falconsai/nsfw_image_detection` для классификации. Эта модель была специально обучена для определения NSFW-контента.
3.  **Принятие решения**:
      * Если модель возвращает высокий показатель "nsfw" (в коде \> 0.8), бот удаляет исходное сообщение с изображением и сообщает об этом.
      * В противном случае, бот подтверждает, что контент безопасен

## 🚀 Установка и запуск

### 1\. Клонирование репозитория

```bash
git clone https://github.com/ваш-логин/ваш-репозиторий.git
cd ваш-репозиторий
```

### 2\. Создание виртуального окружения

Рекомендуется использовать виртуальное окружение для изоляции зависимостей проекта.

```bash
# Для Unix/macOS
python3 -m venv venv
source venv/bin/activate

# Для Windows
python -m venv venv
venv\Scripts\activate
```

### 3\. Установка зависимостей

Установите зависимости с помощью pip:

```bash
pip install -r requirements.txt
```

### 4\. Настройка

1.  Получите токен для вашего бота у [@BotFather](https://t.me/BotFather) в Telegram.

2.  Создайте файл `.env` в корневой папке проекта на примере example.env .

### 5\. Запуск бота

```bash
python main.py
```

После запуска вы увидите в консоли сообщения о загрузке модели и успешном запуске бота.

## Usage

1.  Найдите вашего бота в Telegram.
2.  Отправьте ему команду `/start`, чтобы получить приветственное сообщение.
3.  Отправьте любое изображение в чат. Бот проанализирует его и ответит результатом.

## 🛠️ Технологии

  * **Python 3**
  * **python-telegram-bot**: Библиотека для создания Телеграм-ботов.
  * **Hugging Face Transformers**: Для доступа к state-of-the-art моделям машинного обучения.
  * **Pillow (PIL)**: Библиотека для обработки изображений.
