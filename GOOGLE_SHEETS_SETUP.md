# 📊 Настройка Google Sheets для сохранения диалогов

Эта инструкция поможет вам настроить автоматическое сохранение диалогов с AI тьютором в Google Sheets.

---

## 🎯 Что это дает?

После настройки все диалоги будут автоматически сохраняться в Google таблицу:

| Дата | Время | Тема | Роль | Сообщение |
|------|-------|------|------|-----------|
| 21.10.2025 | 10:30:00 | Единицы измерения веса | Ученик | Привет! |
| 21.10.2025 | 10:30:05 | Единицы измерения веса | Тьютор | Привет! Давай разберем... |

Вы сможете:
- Отслеживать прогресс учеников
- Анализировать частые ошибки
- Фильтровать по темам и датам
- Делать статистику обучения

---

## 🚀 Быстрая настройка (5 минут)

### Шаг 1: Создайте Google Cloud проект

1. Перейдите на [Google Cloud Console](https://console.cloud.google.com/)
2. Нажмите **"Create Project"**
3. Введите название: `Math Tutor AI` (или любое другое)
4. Нажмите **"Create"**

### Шаг 2: Включите Google Sheets API

1. В боковом меню выберите **"APIs & Services"** → **"Library"**
2. Найдите **"Google Sheets API"**
3. Нажмите **"Enable"**
4. Аналогично включите **"Google Drive API"**

### Шаг 3: Создайте Service Account

1. Перейдите в **"APIs & Services"** → **"Credentials"**
2. Нажмите **"+ CREATE CREDENTIALS"** → **"Service account"**
3. Заполните:
   - **Service account name**: `math-tutor-bot`
   - **Service account ID**: (автоматически заполнится)
4. Нажмите **"CREATE AND CONTINUE"**
5. На шаге **"Grant this service account access to project"** выберите роль:
   - **Role**: `Editor` (или `Viewer` + permissions для Sheets)
6. Нажмите **"DONE"**

### Шаг 4: Скачайте JSON ключ

1. В списке **Service Accounts** найдите созданный аккаунт
2. Нажмите на три точки (⋮) справа → **"Manage keys"**
3. Нажмите **"ADD KEY"** → **"Create new key"**
4. Выберите тип **JSON**
5. Нажмите **"CREATE"** - файл автоматически скачается

**⚠️ ВАЖНО:** Этот файл содержит секретные данные! Не публикуйте его в открытом доступе.

### Шаг 5: Создайте Google таблицу

1. Откройте [Google Sheets](https://sheets.google.com/)
2. Создайте новую таблицу
3. Назовите её: **"Math Tutor Dialogs"** (или любое другое имя)
4. Скопируйте **URL** таблицы (из адресной строки)

### Шаг 6: Дайте доступ Service Account к таблице

1. Откройте скачанный JSON файл в текстовом редакторе
2. Найдите строку `"client_email"` - скопируйте email (выглядит как `math-tutor-bot@....iam.gserviceaccount.com`)
3. В вашей Google таблице нажмите **"Share"** (Поделиться)
4. Вставьте скопированный email
5. Выберите права: **"Editor"** (Редактор)
6. Снимите галочку **"Notify people"** (уведомление не нужно)
7. Нажмите **"Share"**

### Шаг 7: Настройте переменные окружения

#### Вариант A: Локальный запуск (.env файл)

1. Откройте ваш `.env` файл
2. Добавьте следующие переменные:

```bash
# Google Sheets настройки
GOOGLE_SHEETS_CREDENTIALS='{"type":"service_account","project_id":"...","private_key":"...","client_email":"..."}'
GOOGLE_SHEET_NAME="Math Tutor Dialogs"
```

**Как получить JSON строку:**
- Откройте скачанный JSON файл
- Скопируйте **весь** его содержимое (от `{` до `}`)
- Вставьте в одну строку после `GOOGLE_SHEETS_CREDENTIALS='`
- Закройте одинарной кавычкой `'`

**Пример:**
```bash
GOOGLE_SHEETS_CREDENTIALS='{"type":"service_account","project_id":"math-tutor-ai","private_key_id":"abc123...","private_key":"-----BEGIN PRIVATE KEY-----\nMIIE...","client_email":"math-tutor-bot@math-tutor-ai.iam.gserviceaccount.com","client_id":"1234567890"}'
GOOGLE_SHEET_NAME="Math Tutor Dialogs"
```

#### Вариант B: Streamlit Cloud

1. Перейдите в настройки вашего приложения на [Streamlit Cloud](https://share.streamlit.io/)
2. Нажмите **"⚙️ Settings"** → **"Secrets"**
3. Добавьте секрет в формате TOML:

```toml
# Google Sheets
GOOGLE_SHEETS_CREDENTIALS = '{"type":"service_account","project_id":"...","private_key":"...","client_email":"..."}'
GOOGLE_SHEET_NAME = "Math Tutor Dialogs"
```

4. Нажмите **"Save"**
5. Перезапустите приложение

---

## ✅ Проверка настройки

1. Запустите приложение
2. Проведите любой диалог
3. Нажмите кнопку **"📊 Сохранить в Google Sheets"**
4. Проверьте вашу Google таблицу - должны появиться новые строки

---

## 🔧 Альтернативные настройки

### Использовать другую таблицу для конкретного сохранения

Вы можете создать несколько таблиц (например, для разных учеников) и сохранять в них отдельно.

**В коде можно передать URL таблицы:**

```python
from utils import save_chat_to_sheets

save_chat_to_sheets(
    messages=messages,
    sheet_url="https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit",
    topic_title="Тема урока"
)
```

### Использовать файл вместо переменной окружения

Если не хотите хранить JSON в .env, сохраните его как файл:

1. Переименуйте скачанный файл в `google-credentials.json`
2. Положите в корень проекта (рядом с `streamlit_app.py`)
3. Добавьте в `.gitignore`:
   ```
   google-credentials.json
   ```
4. В `.env` укажите путь:
   ```bash
   GOOGLE_SHEETS_CREDENTIALS="/path/to/google-credentials.json"
   ```

---

## ❓ Частые проблемы

### ❌ "Таблица не найдена"
- Проверьте, что вы дали доступ Service Account email к таблице
- Проверьте название таблицы в `GOOGLE_SHEET_NAME`

### ❌ "Ошибка авторизации"
- Проверьте правильность JSON в `GOOGLE_SHEETS_CREDENTIALS`
- Убедитесь, что включены Google Sheets API и Google Drive API

### ❌ "Permission denied"
- Service Account должен иметь права **Editor** на таблицу
- Проверьте, что вы поделились таблицей с правильным email

### ❌ На Streamlit Cloud не работает
- Убедитесь, что добавили секреты в формате TOML (не .env)
- Перезапустите приложение после добавления секретов
- Проверьте логи приложения

---

## 🎓 Полезные ссылки

- [Google Cloud Console](https://console.cloud.google.com/)
- [Google Sheets API документация](https://developers.google.com/sheets/api)
- [gspread документация](https://docs.gspread.org/)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

---

## 💡 Советы

1. **Безопасность**: Никогда не коммитьте JSON credentials в Git
2. **Бэкап**: Периодически скачивайте таблицу как Excel для резервной копии
3. **Анализ**: Используйте Google Sheets формулы для анализа прогресса учеников
4. **Фильтры**: Создайте фильтры по темам и датам для удобного поиска

---

Если что-то не получается - проверьте логи приложения или откройте Issue на GitHub! 🚀
