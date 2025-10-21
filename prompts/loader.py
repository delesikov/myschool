"""
Загрузчик промптов из Markdown файлов
"""
from pathlib import Path
from typing import Dict, Optional


def load_prompt(name: str, variables: Optional[Dict[str, str]] = None) -> str:
    """
    Загружает промпт из markdown файла

    Args:
        name: Имя файла промпта (без расширения .md)
        variables: Словарь переменных для замены в промпте (опционально)

    Returns:
        Текст промпта

    Example:
        >>> prompt = load_prompt('tutor_prompt')
        >>> formatted = prompt.format(chat_history="...", input="...")
    """
    prompt_path = Path(__file__).parent / f"{name}.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Промпт не найден: {prompt_path}")

    content = prompt_path.read_text(encoding='utf-8')

    # Если переданы переменные, сразу подставляем их
    if variables:
        content = content.format(**variables)

    return content


def load_all_prompts() -> Dict[str, str]:
    """
    Загружает все промпты из директории prompts/

    Returns:
        Словарь {имя_файла: содержимое}
    """
    prompts_dir = Path(__file__).parent
    prompts = {}

    for md_file in prompts_dir.glob("*.md"):
        prompt_name = md_file.stem
        prompts[prompt_name] = md_file.read_text(encoding='utf-8')

    return prompts
