"""
Форматирование схем для промпта Learn Mode
"""


def format_schema(topic_data: dict) -> str:
    """
    Форматирует новую схему темы в текст для LEARN_MODE_PROMPT

    Структура схемы:
    {
        "title": "Название темы",
        "plan": "План урока...",
        "explanation": [
            {
                "info": "Теория",
                "action": "Вопрос",
                "solution": "Решение",
                "mistake_explanation": "Объяснение ошибок",
                "answer": "Ответ"
            }
        ],
        "boss": {
            "problem": "Условие задачи",
            "steps": [
                {
                    "step_num": 1,
                    "action": "...",
                    "solution": "...",
                    "answer": "..."
                }
            ],
            "final_answer": "Итоговый ответ"
        },
        "summary": "Конспект"
    }
    """
    parts = []

    # Заголовок
    title = topic_data.get('title', 'Тема')
    parts.append(f"# ТЕМА: {title}\n")
    parts.append("---\n\n")

    # 1. ПЛАН
    plan = topic_data.get('plan', '')
    if plan:
        parts.append("## 1. ПЛАН УРОКА\n\n")
        parts.append(f"{plan}\n\n")
        parts.append("---\n\n")

    # 2. ОСНОВНОЕ ОБЪЯСНЕНИЕ
    explanation = topic_data.get('explanation', [])
    if explanation:
        parts.append("## 2. ОСНОВНОЕ ОБЪЯСНЕНИЕ\n\n")
        parts.append("Веди ученика через эти блоки ПОСЛЕДОВАТЕЛЬНО:\n\n")

        for i, block in enumerate(explanation, 1):
            parts.append(f"### Блок {i}\n\n")

            # info
            if 'info' in block:
                parts.append(f"**[info] Объяснение:**\n{block['info']}\n\n")

            # action
            if 'action' in block:
                parts.append(f"**[action] Задай вопрос:**\n{block['action']}\n\n")

            # solution (для тьютора, не показывать ученику)
            if 'solution' in block:
                parts.append(f"**[solution] Правильное решение (используй для подсказок):**\n{block['solution']}\n\n")

            # mistake_explanation
            if 'mistake_explanation' in block:
                parts.append(f"**[mistake_explanation] Типичные ошибки:**\n")
                if isinstance(block['mistake_explanation'], dict):
                    for wrong_answer, explanation in block['mistake_explanation'].items():
                        parts.append(f"- Если ответ '{wrong_answer}': {explanation}\n")
                else:
                    parts.append(f"{block['mistake_explanation']}\n")
                parts.append("\n")

            # answer
            if 'answer' in block:
                parts.append(f"**[answer] Правильный ответ:** {block['answer']}\n\n")

            parts.append("---\n\n")

    # 3. ФИНАЛЬНЫЙ БОСС
    boss = topic_data.get('boss', {})
    if boss:
        parts.append("## 3. ФИНАЛЬНЫЙ БОСС 🎯\n\n")

        # Условие задачи
        if 'problem' in boss:
            parts.append(f"**[problem] Условие:**\n{boss['problem']}\n\n")

        # Шаги решения
        steps = boss.get('steps', [])
        if steps:
            parts.append("**Подзадачи (веди ученика пошагово):**\n\n")
            for step in steps:
                step_num = step.get('step_num', '?')
                parts.append(f"#### Шаг {step_num}\n\n")

                if 'action' in step:
                    parts.append(f"**[action]** {step['action']}\n\n")

                if 'solution' in step:
                    parts.append(f"**[solution]** {step['solution']}\n\n")

                if 'answer' in step:
                    parts.append(f"**[answer]** {step['answer']}\n\n")

        # Финальный ответ
        if 'final_answer' in boss:
            parts.append(f"**[final_answer] Итоговый правильный ответ:** {boss['final_answer']}\n\n")

        parts.append("---\n\n")

    # 4. КОНСПЕКТ
    summary = topic_data.get('summary', '')
    if summary:
        parts.append("## 4. КОНСПЕКТ (дай в конце урока)\n\n")
        parts.append(f"{summary}\n")

    return "".join(parts)


def format_feedback_context(topic_data: dict, chat_history: str) -> dict:
    """
    Подготавливает контекст для промпта фидбека

    Args:
        topic_data: Данные темы (новая структура)
        chat_history: История разговора

    Returns:
        Словарь с переменными для FEEDBACK_PROMPT
    """
    return {
        'topic_title': topic_data.get('title', ''),
        'topic_description': topic_data.get('description', ''),
        'chat_history': chat_history,
        'final_summary': topic_data.get('summary', '')
    }
