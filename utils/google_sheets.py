"""
Интеграция с Google Sheets для сохранения диалогов
"""
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import os


def get_google_sheets_client(credentials_json: str = None):
    """
    Создает клиент для работы с Google Sheets

    Args:
        credentials_json: JSON строка с credentials или путь к файлу

    Returns:
        gspread.Client или None если не удалось авторизоваться
    """
    try:
        # Определяем scope для Google Sheets
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]

        # Если передан путь к файлу
        if credentials_json and os.path.isfile(credentials_json):
            creds = Credentials.from_service_account_file(credentials_json, scopes=scopes)
        # Если передана JSON строка
        elif credentials_json:
            creds_dict = json.loads(credentials_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        # Если ничего не передано - пытаемся из переменной окружения
        else:
            google_creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
            if not google_creds_json:
                return None

            creds_dict = json.loads(google_creds_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

        return gspread.authorize(creds)

    except Exception as e:
        print(f"Ошибка авторизации Google Sheets: {e}")
        return None


def save_chat_to_sheets(
    messages: list,
    sheet_url: str = None,
    sheet_name: str = None,
    topic_title: str = None,
    credentials_json: str = None
) -> bool:
    """
    Сохраняет диалог в Google Sheets

    Args:
        messages: Список сообщений [{"role": "user"/"assistant", "content": "..."}]
        sheet_url: URL Google Sheets (необязательно, можно использовать sheet_name)
        sheet_name: Название таблицы (если не указан sheet_url)
        topic_title: Название темы
        credentials_json: JSON с credentials

    Returns:
        True если успешно, False если ошибка

    Структура таблицы:
    | Дата | Время | Тема | Роль | Сообщение |
    """
    try:
        # Получаем клиент
        client = get_google_sheets_client(credentials_json)
        if not client:
            print("Не удалось авторизоваться в Google Sheets")
            return False

        # Открываем таблицу
        if sheet_url:
            sheet = client.open_by_url(sheet_url).sheet1
        elif sheet_name:
            sheet = client.open(sheet_name).sheet1
        else:
            # Используем название из переменной окружения
            default_sheet = os.getenv('GOOGLE_SHEET_NAME', 'Math Tutor Dialogs')
            sheet = client.open(default_sheet).sheet1

        # Проверяем, есть ли заголовки
        if sheet.row_count == 0 or not sheet.row_values(1):
            # Добавляем заголовки
            headers = ['Дата', 'Время', 'Тема', 'Роль', 'Сообщение']
            sheet.append_row(headers)

        # Подготавливаем данные для вставки
        now = datetime.now()
        date_str = now.strftime('%d.%m.%Y')
        time_str = now.strftime('%H:%M:%S')

        rows_to_add = []
        for msg in messages:
            role = "Ученик" if msg["role"] == "user" else "Тьютор"
            content = msg["content"]

            # Ограничиваем длину сообщения (Google Sheets имеет лимит 50000 символов на ячейку)
            if len(content) > 40000:
                content = content[:40000] + "... (обрезано)"

            row = [date_str, time_str, topic_title or "-", role, content]
            rows_to_add.append(row)

        # Добавляем все строки одним запросом (эффективнее)
        if rows_to_add:
            sheet.append_rows(rows_to_add)

        print(f"✅ Диалог сохранен в Google Sheets ({len(rows_to_add)} сообщений)")
        return True

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"❌ Таблица не найдена. Проверьте URL или название.")
        return False
    except gspread.exceptions.APIError as e:
        print(f"❌ Ошибка Google Sheets API: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка при сохранении в Google Sheets: {e}")
        return False


def create_new_sheet(sheet_name: str, credentials_json: str = None) -> str:
    """
    Создает новую Google таблицу для диалогов

    Args:
        sheet_name: Название таблицы
        credentials_json: JSON с credentials

    Returns:
        URL созданной таблицы или None
    """
    try:
        client = get_google_sheets_client(credentials_json)
        if not client:
            return None

        # Создаем новую таблицу
        spreadsheet = client.create(sheet_name)

        # Добавляем заголовки
        sheet = spreadsheet.sheet1
        headers = ['Дата', 'Время', 'Тема', 'Роль', 'Сообщение']
        sheet.append_row(headers)

        # Форматируем заголовки (жирный шрифт)
        sheet.format('A1:E1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
        })

        # Делаем таблицу доступной по ссылке (можно комментировать)
        spreadsheet.share('', perm_type='anyone', role='writer')

        print(f"✅ Таблица '{sheet_name}' создана")
        return spreadsheet.url

    except Exception as e:
        print(f"❌ Ошибка при создании таблицы: {e}")
        return None
