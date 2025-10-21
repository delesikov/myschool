"""
Экспорт диалогов в различных форматах
"""
from datetime import datetime


def format_chat_to_markdown(messages: list, topic_title: str = None) -> str:
    """
    Форматирует историю чата в красивый Markdown

    Args:
        messages: Список сообщений [{"role": "user"/"assistant", "content": "..."}]
        topic_title: Название темы (опционально)

    Returns:
        Отформатированная строка в Markdown
    """
    lines = []

    # Заголовок
    lines.append("# 📚 Диалог с AI Тьютором\n")

    # Дата и тема
    lines.append(f"**Дата:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    if topic_title:
        lines.append(f"**Тема:** {topic_title}\n")

    lines.append("\n---\n\n")

    # Диалог
    for i, msg in enumerate(messages, 1):
        role = msg["role"]
        content = msg["content"]

        # Эмодзи для роли
        if role == "user":
            icon = "👤"
            role_name = "Ученик"
        else:
            icon = "🤖"
            role_name = "Тьютор"

        # Форматирование сообщения
        lines.append(f"### {icon} {role_name} (сообщение {i})\n\n")
        lines.append(f"{content}\n\n")
        lines.append("---\n\n")

    # Футер
    lines.append("\n\n*Экспортировано из Математического помощника AI*\n")

    return "".join(lines)


def format_chat_to_text(messages: list, topic_title: str = None) -> str:
    """
    Форматирует историю чата в простой текстовый формат

    Args:
        messages: Список сообщений
        topic_title: Название темы (опционально)

    Returns:
        Простой текст
    """
    lines = []

    # Заголовок
    lines.append("=" * 60)
    lines.append("\n    ДИАЛОГ С AI ТЬЮТОРОМ")
    lines.append("\n" + "=" * 60 + "\n\n")

    # Дата и тема
    lines.append(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    if topic_title:
        lines.append(f"Тема: {topic_title}\n")

    lines.append("\n" + "-" * 60 + "\n\n")

    # Диалог
    for i, msg in enumerate(messages, 1):
        role = "УЧЕНИК" if msg["role"] == "user" else "ТЬЮТОР"
        content = msg["content"]

        lines.append(f"[{role}] (сообщение {i}):\n")
        lines.append(f"{content}\n\n")
        lines.append("-" * 60 + "\n\n")

    # Футер
    lines.append("\nЭкспортировано из Математического помощника AI\n")

    return "".join(lines)


def get_chat_filename(topic_title: str = None, format: str = "md") -> str:
    """
    Генерирует имя файла для экспорта

    Args:
        topic_title: Название темы
        format: Формат файла (md, txt)

    Returns:
        Имя файла
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if topic_title:
        # Очищаем название темы от спецсимволов
        clean_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_'
                              for c in topic_title)
        clean_title = clean_title.replace(' ', '_')
        return f"dialog_{clean_title}_{timestamp}.{format}"
    else:
        return f"dialog_{timestamp}.{format}"
