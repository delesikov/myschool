"""
Модуль управления промптами
"""
from .loader import load_prompt, load_all_prompts

# Загружаем промпты из markdown файлов
TUTOR_PROMPT = load_prompt('tutor_prompt')              # Study Mode (свободный тьютор)
LEARN_MODE_PROMPT = load_prompt('learn_mode_prompt')   # Learn Mode (по схеме)
FEEDBACK_PROMPT = load_prompt('feedback_prompt')        # Финальный фидбек

__all__ = [
    'TUTOR_PROMPT',
    'LEARN_MODE_PROMPT',
    'FEEDBACK_PROMPT',
    'load_prompt',
    'load_all_prompts'
]
