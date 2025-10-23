"""
Данные тем для обучения
"""
from .new_topics import NEW_LEARNING_TOPICS
from .learning_topics import LEARNING_TOPICS  # Старый формат (legacy, используется для Study Mode)
from .grade_instructions import GRADE_INSTRUCTIONS, get_grade_instruction

# Основной источник тем - новый формат
TOPICS = NEW_LEARNING_TOPICS

__all__ = [
    'TOPICS',              # Новые темы (для Learn Mode)
    'LEARNING_TOPICS',     # Старые темы (legacy, только для Study Mode если нужно)
    'GRADE_INSTRUCTIONS',  # Инструкции для разных классов
    'get_grade_instruction' # Функция для получения инструкции по классу
]
