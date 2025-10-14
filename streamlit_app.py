import streamlit as st
import os
import sympy
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from data.learning_topics import LEARNING_TOPICS

load_dotenv()

st.set_page_config(page_title="Математический помощник AI", page_icon="🧮", layout="wide")

# ============= МАТЕМАТИЧЕСКИЕ ИНСТРУМЕНТЫ =============

def calculator(expression: str) -> str:
    try:
        import math
        result = eval(expression.strip(), {
            "__builtins__": {},
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs
        })
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

def symbolic_math(expression: str) -> str:
    try:
        x, y, z = sympy.symbols('x y z')
        local_dict = {
            'x': x, 'y': y, 'z': z, 'integrate': sympy.integrate, 'diff': sympy.diff,
            'solve': sympy.solve, 'limit': sympy.limit, 'sin': sympy.sin, 'cos': sympy.cos,
            'sqrt': sympy.sqrt, 'pi': sympy.pi, 'oo': sympy.oo
        }
        result = eval(expression.strip(), {"__builtins__": {}}, local_dict)
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

def equation_solver(equation: str) -> str:
    try:
        x = sympy.symbols('x')
        equations = [eq.strip() for eq in equation.split(',')]
        sympy_eqs = []
        for eq in equations:
            if '=' in eq:
                left, right = eq.split('=')
                sympy_eqs.append(sympy.sympify(left) - sympy.sympify(right))
            else:
                sympy_eqs.append(sympy.sympify(eq))
        solutions = sympy.solve(sympy_eqs[0], x)
        return f"Решение: {solutions}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# ============= ИНИЦИАЛИЗАЦИЯ АГЕНТА =============

@st.cache_resource
def init_bot(model_choice, yandex_key, gemini_key):
    tools = [
        Tool(name="Calculator", func=calculator, description="Числовые вычисления"),
        Tool(name="SymbolicMath", func=symbolic_math, description="Символьная математика"),
        Tool(name="EquationSolver", func=equation_solver, description="Решение уравнений")
    ]
    
    template = """Ты — дружелюбный математический помощник для детей.

ПРАВИЛА:
1. Если не математика - отвечай БЕЗ инструментов
2. Объясняй понятно для детей
3. Хвали за правильные ответы

Инструменты: {tool_names}
{tools}

Формат для математики:
Thought: нужен инструмент
Action: (инструмент)
Action Input: (данные)
Observation: (результат)
Final Answer: (объяснение)

Формат для общения:
Thought: обычный вопрос
Final Answer: (ответ)

История: {chat_history}
Вопрос: {input}
{agent_scratchpad}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "chat_history", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{t.name}: {t.description}" for t in tools]),
            "tool_names": ", ".join([t.name for t in tools])
        }
    )
    
    if model_choice == "YandexGPT 5.1 Pro":
        llm = ChatOpenAI(api_key=yandex_key, base_url="http://localhost:8520/v1",
                        model="yandexgpt/latest", temperature=0.3)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key,
                                    temperature=0.3, convert_system_message_to_human=True)
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, 
                        max_iterations=7, handle_parsing_errors=True)

# ============= ФУНКЦИИ =============

def check_answer(user_answer: str, correct_answer) -> bool:
    """Проверяет ответ ученика с помощью простого сравнения"""
    user_answer = user_answer.strip().lower().replace(" ", "")
    if isinstance(correct_answer, list):
        return any(ans.lower().replace(" ", "") in user_answer for ans in correct_answer)
    else:
        correct_answer = str(correct_answer).strip().lower().replace(" ", "")
        return correct_answer in user_answer or user_answer in correct_answer

def check_answer_with_llm(user_answer: str, correct_answer, model_choice, yandex_key, gemini_key, question_context: str = "") -> bool:
    """Проверяет ответ ученика с помощью LLM для семантического понимания"""
    
    # Сначала попробуем простую проверку
    if check_answer(user_answer, correct_answer):
        return True
    
    # Если простая проверка не сработала, используем LLM
    try:
        # Формируем правильные ответы
        if isinstance(correct_answer, list):
            correct_answers_str = " или ".join([f"'{ans}'" for ans in correct_answer])
        else:
            correct_answers_str = f"'{correct_answer}'"
        
        prompt = f"""Ты проверяешь ответ ученика на математический вопрос.

Контекст вопроса: {question_context}

Правильный ответ: {correct_answers_str}
Ответ ученика: "{user_answer}"

Задача: Определи, является ли ответ ученика правильным с учётом:
- Числа написанные словами (пятнадцать = 15, семь = 7)
- Синонимы (вычитание = минус = отнять)
- Математические выражения (2+3 = 5, 10-2 = 8)
- Разные формы записи дробей (1/2 = 0.5)

Ответь ТОЛЬКО одним словом: "ДА" если ответ правильный, "НЕТ" если неправильный."""

        if model_choice == "YandexGPT 5.1 Pro":
            llm = ChatOpenAI(api_key=yandex_key, base_url="http://localhost:8520/v1",
                            model="yandexgpt/latest", temperature=0)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key,
                                        temperature=0, convert_system_message_to_human=True)
        
        response = llm.invoke(prompt)
        result = response.content.strip().upper()
        
        return "ДА" in result or "YES" in result
    
    except Exception as e:
        print(f"LLM check failed: {e}")
        # Если LLM не сработал, возвращаем результат простой проверки
        return False

def show_learning_schema(topic_data):
    """Показывает схему темы для режима обучения"""
    with st.expander("📋 Посмотреть схему темы", expanded=False):
        st.markdown(f"**Тема:** {topic_data['title']}")
        st.markdown("---")
        st.markdown("### 1️⃣ Квиз-диагностика")
        for i, q in enumerate(topic_data['quiz'], 1):
            # Вопросы могут содержать LaTeX; рендерим их чуть крупнее для читаемости
            question_md = f"**Вопрос {i}:** {q['question']}"
            st.markdown(question_md)
            st.markdown(f"- Правильный ответ: {q['correct']}", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 2️⃣ Вводная теория")
        # Показываем полную вводную теорию (поддерживает LaTeX)
        # Показываем вводную теорию крупнее (поддерживает LaTeX)
        intro = topic_data.get('intro_theory', '')
        if intro:
            st.markdown(intro)

        st.markdown("---")
        st.markdown("### 3️⃣ Основная теория")
        # Основная теория — тоже крупнее
        main = topic_data.get('main_theory', '')
        if main:
            if isinstance(main, dict):
                st.markdown(f"**{main.get('title', '')}**")
                st.markdown(main.get('plan', ''))
                if 'examples' in main:
                    for i, ex in enumerate(main['examples'], 1):
                        st.markdown(f"\n**Пример {i}:**")
                        st.markdown(ex.get('explanation', ''))
                        st.markdown(f"*Вопрос: {ex.get('question', '')}*")
                        st.markdown(f"*Ответ: {ex.get('answer', '')}*")
            else:
                st.markdown(main)

        st.markdown("---")
        st.markdown("### 4️⃣ Босс (проверка)")
        st.markdown("2 варианта задач")
        
        st.markdown("---")
        st.markdown("### 5️⃣ Финиш")
        st.success("Короткий конспект")


# ============= ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ =============

if "mode" not in st.session_state:
    # Оставляем только режим изучения темы
    st.session_state.mode = "learn"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Для режима обучения
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "learning_stage" not in st.session_state:
    st.session_state.learning_stage = "quiz"
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []
if "boss_variant" not in st.session_state:
    st.session_state.boss_variant = None
if "boss_step" not in st.session_state:
    st.session_state.boss_step = 0
if "waiting_for_quiz_answer" not in st.session_state:
    st.session_state.waiting_for_quiz_answer = False
if "mistake_topics" not in st.session_state:
    st.session_state.mistake_topics = []
if "main_theory_step" not in st.session_state:
    st.session_state.main_theory_step = 0

# ============= UI =============

st.title("🧮 Математический помощник AI")
st.markdown("*Помогаю учить математику!*")
st.markdown("---")

# ============= SIDEBAR =============

with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Фиксированный режим: только изучение темы
    mode = st.radio(
        "🎯 Режим работы:",
        ["learn"],
        format_func=lambda x: {"learn": "📚 Изучить тему"}[x],
        key="mode_selector"
    )

    # Режим не меняется в рантайме (оставляем сообщения и состояние для режима learning)
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.messages = []
        st.session_state.current_topic = None
        st.session_state.learning_stage = "quiz"
        st.session_state.quiz_results = []
        st.session_state.boss_variant = None
        st.session_state.boss_step = 0
        st.session_state.waiting_for_quiz_answer = False
    
    st.markdown("---")
    
    # Выбор модели
    model_choice = st.selectbox("🤖 AI Модель", ["Google Gemini 2.5 Flash", "YandexGPT 5.1 Pro"])
    
    if model_choice == "YandexGPT 5.1 Pro":
        yandex_api_key = st.text_input("API ключ", value=os.getenv("YANDEX_API_KEY", ""), type="password")
        gemini_api_key = ""
    else:
        gemini_api_key = st.text_input("Google API ключ", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
        yandex_api_key = ""
        st.markdown("[Получить ключ →](https://aistudio.google.com/apikey)")
    
    st.markdown("---")
    
    # РЕЖИМ-ЗАВИСИМЫЙ КОНТЕНТ: только изучение темы
    st.header("📚 Выбери тему")
    for topic_id, topic_data in LEARNING_TOPICS.items():
        if st.button(topic_data['title'], key=f"topic_{topic_id}", use_container_width=True):
            st.session_state.current_topic = topic_id
            st.session_state.learning_stage = "quiz"
            st.session_state.quiz_results = []
            st.session_state.boss_variant = None
            st.session_state.boss_step = 0
            st.session_state.waiting_for_quiz_answer = True
            
            # Стартуем с квиза
            quiz = topic_data['quiz'][0]
            plan = """Вот наш план:
- Задам 3 вопроса, чтобы определить твой уровень;
- Объясню теорию (можешь задавать вопросы!);
- Дам решить задачу самому;
- Пришлю короткий конспект!

Готов? Поехали! 🚀
"""
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"**{topic_data['title']}**\n\n{topic_data['description']}\n\n{plan}\n\n" +
                          f"Давай сначала проверим, что ты уже знаешь! 🎯\n\n**Вопрос 1:** {quiz['question']}",
                "quiz_options": quiz['options']  # Сохраняем варианты ответов
            }]
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Начать заново", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_topic = None
        st.session_state.learning_stage = "quiz"
        st.session_state.quiz_results = []
        st.session_state.boss_variant = None
        st.session_state.boss_step = 0
        st.session_state.waiting_for_quiz_answer = False
        st.rerun()

# ============= ПРОВЕРКА API =============

current_key = yandex_api_key if model_choice == "YandexGPT 5.1 Pro" else gemini_api_key

if not current_key:
    st.warning(f"⚠️ Введите API ключ в настройках слева")
    st.info("💡 Для начала получите бесплатный ключ Gemini")
    st.stop()

# ============= ОСНОВНОЙ ИНТЕРФЕЙС =============

# Показываем режим
mode_badges = {
    "learn": "📚 Изучить тему"
}
st.info(f"**Текущий режим:** {mode_badges[st.session_state.mode]}")

# СХЕМЫ
if st.session_state.mode == "learn" and st.session_state.current_topic:
    topic_data = LEARNING_TOPICS[st.session_state.current_topic]
    show_learning_schema(topic_data)

# История сообщений
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Рендерим содержимое сообщения, поддерживая крупный LaTeX
        st.markdown(message["content"], unsafe_allow_html=True)

        # Если это сообщение с вариантами ответов квиза и оно последнее
        if (message["role"] == "assistant" and
            "quiz_options" in message and
            idx == len(st.session_state.messages) - 1 and
            st.session_state.waiting_for_quiz_answer):

            st.markdown("**Выбери ответ:**")
            cols = st.columns(2)
            for i, option in enumerate(message["quiz_options"]):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(f"{option}", key=f"quiz_opt_{idx}_{i}", use_container_width=True):
                        st.session_state.quiz_answer = option
                        st.session_state.waiting_for_quiz_answer = False
                        st.rerun()

        # Показываем кнопку перехода к теории после ответа на вопрос (если это последнее сообщение)
        if (message["role"] == "assistant" and
            "show_theory_button" in message and
            idx == len(st.session_state.messages) - 1):

            topic_data = LEARNING_TOPICS[st.session_state.current_topic]
            theory_button_text = f"🧠 {topic_data['title']}"
            if st.button(theory_button_text, key=f"theory_after_answer_{idx}", use_container_width=True):
                st.session_state.selected_question = "теория"
                st.rerun()

# Обработка ответа из кнопки квиза
if "quiz_answer" in st.session_state:
    question = st.session_state.quiz_answer
    del st.session_state.quiz_answer
    
    # Добавляем ответ пользователя
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Думаю..."):
            topic = LEARNING_TOPICS[st.session_state.current_topic]
            stage = st.session_state.learning_stage
            
            if stage == "quiz":
                quiz_index = len(st.session_state.quiz_results)
                quiz_q = topic['quiz'][quiz_index]

                # Проверяем, выбрал ли пользователь "Не знаю"
                is_dont_know = "не знаю" in question.lower()

                if is_dont_know:
                    # Отмечаем как неправильный ответ
                    st.session_state.quiz_results.append(False)
                    response = f"Ничего страшного! 😊 {quiz_q['explanation_template']}\n\n"
                else:
                    # Обычная проверка ответа
                    is_correct = check_answer_with_llm(
                        question,
                        quiz_q['correct'],
                        model_choice,
                        yandex_api_key,
                        gemini_api_key,
                        question_context=quiz_q['question']
                    )
                    st.session_state.quiz_results.append(is_correct)

                    if is_correct:
                        response = f"✅ Правильно!\n\n"
                    else:
                        response = f"❌ Не совсем. {quiz_q['explanation_template']}\n\n"
                
                # Следующий вопрос квиза или завершение
                next_quiz_index = quiz_index + 1
                if next_quiz_index < len(topic['quiz']):
                    next_q = topic['quiz'][next_quiz_index]
                    response += f"**Вопрос {next_quiz_index + 1}:** {next_q['question']}"
                    st.session_state.waiting_for_quiz_answer = True
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "quiz_options": next_q['options']
                    })
                else:
                    # Завершаем квиз
                    correct_count = sum(st.session_state.quiz_results)
                    if correct_count == len(topic['quiz']):
                        st.session_state.learning_stage = "main_theory"
                        st.session_state.main_theory_step = 0

                        # Формируем ответ с планом и первым примером
                        main_theory = topic['main_theory']
                        if isinstance(main_theory, dict):
                            response += f"\n\n🎉 Все правильно! Переходим к основной теории.\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                            if main_theory['examples']:
                                first_example = main_theory['examples'][0]
                                response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                        else:
                            response += f"\n\n🎉 Все правильно! Переходим к теории.\n\n---\n\n{main_theory}"

                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
                    else:
                        st.session_state.learning_stage = "choice"
                        response += "\n\nНе все ответы правильные. Что делаем дальше?"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
    
    st.rerun()

# ============= ОБРАБОТКА QUICK REPLY КНОПОК =============
# Сначала проверяем, была ли нажата кнопка в предыдущем цикле
if "selected_question" in st.session_state:
    question = st.session_state.selected_question
    del st.session_state.selected_question
    show_user_message = False

    # Обрабатываем вопрос сразу
    with st.chat_message("assistant"):
        with st.spinner("🤔 Думаю..."):
            if st.session_state.current_topic is None:
                response = "Пожалуйста, выбери тему из списка слева! 👈"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                topic = LEARNING_TOPICS[st.session_state.current_topic]
                stage = st.session_state.learning_stage

                if stage == "choice":
                    # Обработка выбора после квиза
                    if "разбор" in question.lower():
                        # Собираем темы, по которым были ошибки
                        incorrect_topics = []
                        for i, result in enumerate(st.session_state.quiz_results):
                            if not result:
                                topic_key = topic['quiz'][i].get('topic_key')
                                if topic_key and topic_key in topic['prerequisite_notes']:
                                    incorrect_topics.append(topic_key)

                        # Убираем дубликаты, сохраняя порядок
                        unique_topics = list(dict.fromkeys(incorrect_topics))

                        if unique_topics:
                            # Сохраняем темы для показа кнопок
                            st.session_state.mistake_topics = unique_topics
                            response = "📖 **Выбери тему для разбора:**\n\nНажми на кнопку ниже, чтобы изучить конспект по этой теме."
                            st.session_state.learning_stage = "waiting_for_topic_selection"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response, "show_theory_button": True})
                        else:
                            response = "Отлично! У тебя нет ошибок в темах. Переходим к основной теории!"
                            st.session_state.learning_stage = "main_theory"
                            st.session_state.main_theory_step = 0

                            main_theory = topic['main_theory']
                            if isinstance(main_theory, dict):
                                response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                                if main_theory['examples']:
                                    first_example = main_theory['examples'][0]
                                    response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                            else:
                                response += f"\n\n---\n\n{main_theory}"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                    elif "теория" in question.lower():
                        response = "Хорошо! Переходим к основной теории."
                        st.session_state.learning_stage = "main_theory"
                        st.session_state.main_theory_step = 0

                        main_theory = topic['main_theory']
                        if isinstance(main_theory, dict):
                            response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                            if main_theory['examples']:
                                first_example = main_theory['examples'][0]
                                response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                        else:
                            response += f"\n\n---\n\n{main_theory}"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "Используй кнопки 'Разобрать ошибки' или 'К теории'."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                elif stage == "waiting_for_topic_selection":
                    # Пользователь выбрал тему для разбора
                    # Проверяем, есть ли эта тема в prerequisite_notes
                    if question in topic['prerequisite_notes']:
                        note_data = topic['prerequisite_notes'][question]
                        response = f"{note_data['title']}\n\n{note_data['content']}\n\n---\n\n"

                        # Убираем просмотренную тему из списка
                        st.session_state.mistake_topics.remove(question)

                        # Если еще есть темы для разбора
                        if st.session_state.mistake_topics:
                            response += "Выбери следующую тему для продолжения."
                            # Остаемся на этапе waiting_for_topic_selection
                        else:
                            response += f"Если вопросов не осталось, то перейдем к разбору основной темы: **{topic['title']}**"
                            st.session_state.learning_stage = "waiting_after_notes"
                    else:
                        response = "Тема не найдена. Выбери тему из списка выше."

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "show_theory_button": True})

                elif stage == "waiting_after_notes":
                    if "теория" in question.lower():
                        response = f"Отлично! Переходим к вводной теории.\n\n---\n\n{topic['intro_theory']}"
                        st.session_state.learning_stage = "intro_theory"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()

# ============= QUICK REPLY КНОПКИ =============
# Показываем постоянные кнопки навигации сверху окна ввода
if st.session_state.current_topic:
    stage = st.session_state.learning_stage

    # Этап "choice" - выбор между разбором ошибок и теорией
    if stage == "choice":
        st.markdown("**Быстрые действия:**")
        topic_data = LEARNING_TOPICS[st.session_state.current_topic]
        cols = st.columns(2)
        with cols[0]:
            if st.button("📖 Разобрать ошибки", key="quick_mistakes", use_container_width=True):
                st.session_state.selected_question = "разбор"
                st.rerun()
        with cols[1]:
            # Используем название темы для кнопки
            theory_button_text = f"🧠 {topic_data['title']}"
            if st.button(theory_button_text, key="quick_theory", use_container_width=True):
                st.session_state.selected_question = "теория"
                st.rerun()

    # Этап "waiting_for_topic_selection" - показываем кнопки тем
    elif stage == "waiting_for_topic_selection":
        if "mistake_topics" in st.session_state and st.session_state.mistake_topics:
            st.markdown("**Темы, с которыми есть проблемы:**")
            topic_data = LEARNING_TOPICS[st.session_state.current_topic]

            # Определяем количество столбцов (1-2 в зависимости от количества тем)
            num_topics = len(st.session_state.mistake_topics)
            cols = st.columns(min(num_topics, 2))

            for idx, topic_key in enumerate(st.session_state.mistake_topics):
                note_data = topic_data['prerequisite_notes'].get(topic_key)
                if note_data:
                    col_idx = idx % 2
                    with cols[col_idx]:
                        if st.button(note_data['title'], key=f"topic_btn_{topic_key}", use_container_width=True):
                            st.session_state.selected_question = topic_key
                            st.rerun()

# Обработка обычного ввода с клавиатуры
question = st.chat_input("Напиши свой вопрос или ответ...")
show_user_message = True

if question:
    # Сохраняем сообщение пользователя только если это не саджест
    if show_user_message:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Думаю..."):
            # Обрабатываем только режим learning
            if st.session_state.current_topic is None:
                response = "Пожалуйста, выбери тему из списка слева! 👈"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                topic = LEARNING_TOPICS[st.session_state.current_topic]
                stage = st.session_state.learning_stage

                # Функция для обработки отвлечённого вопроса (без создания глобального агента)
                def handle_offtopic_question(_user_question, current_hint):
                    # Возвращаем короткий ответ без вызова внешнего агента
                    return f"Интересный вопрос! 😊 Но давай сначала закончим текущее задание.\n\n{current_hint}"

                if stage == "quiz":
                    # Обработка квиза (когда пользователь ввел текст вместо кнопки)
                    quiz_index = len(st.session_state.quiz_results)

                    if quiz_index >= len(topic['quiz']):
                        # Квиз уже завершен, переходим дальше
                        correct_count = sum(st.session_state.quiz_results)
                        if correct_count == len(topic['quiz']):
                            st.session_state.learning_stage = "main_theory"
                            st.session_state.main_theory_step = 0

                            # Формируем ответ с планом и первым примером
                            main_theory = topic['main_theory']
                            if isinstance(main_theory, dict):
                                response = f"🎉 Отлично! Переходим к основной теории.\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                                if main_theory['examples']:
                                    first_example = main_theory['examples'][0]
                                    response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                            else:
                                response = f"🎉 Отлично! Переходим к теории.\n\n---\n\n{main_theory}"
                        else:
                            st.session_state.learning_stage = "choice"
                            response = "\n\nНе все ответы правильные. Повторим?\n\n- Напиши **'Да'** - покажу конспект\n- Напиши **'Нет'** - сразу к теории"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        quiz_q = topic['quiz'][quiz_index]

                        # Сначала проверяем "Не знаю"
                        is_dont_know = "не знаю" in question.lower()

                        if is_dont_know:
                            # Отмечаем как неправильный ответ
                            st.session_state.quiz_results.append(False)
                            response = f"Ничего страшного! 😊 {quiz_q['explanation_template']}\n\n"
                        else:
                            # Проверяем с помощью LLM - может это правильный ответ
                            is_correct = check_answer_with_llm(
                                question,
                                quiz_q['correct'],
                                model_choice,
                                yandex_api_key,
                                gemini_api_key,
                                question_context=quiz_q['question']
                            )

                            # Также проверяем, похоже ли это на попытку ответить
                            is_quiz_answer = any(opt.lower().replace(" ", "") in question.lower().replace(" ", "")
                                               for opt in quiz_q['options'])

                            # Если ответ правильный ИЛИ это один из вариантов квиза, обрабатываем как ответ
                            if is_correct or is_quiz_answer:
                                st.session_state.quiz_results.append(is_correct)

                                if is_correct:
                                    response = f"✅ Правильно!\n\n"
                                else:
                                    response = f"❌ Не совсем. {quiz_q['explanation_template']}\n\n"
                            else:
                                # Это отвлеченный вопрос
                                current_hint = f"**Вопрос {quiz_index + 1}:** {quiz_q['question']}"
                                response = handle_offtopic_question(question, current_hint)
                                st.markdown(response)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response,
                                    "quiz_options": quiz_q['options']
                                })
                                st.session_state.waiting_for_quiz_answer = True
                                # Выходим из обработки, чтобы не продолжить с неправильной логикой
                                st.rerun()

                        # Если мы здесь, значит был ответ на квиз
                        # Следующий вопрос
                        next_quiz_index = quiz_index + 1
                        if next_quiz_index < len(topic['quiz']):
                            next_q = topic['quiz'][next_quiz_index]
                            response += f"**Вопрос {next_quiz_index + 1}:** {next_q['question']}"
                            st.session_state.waiting_for_quiz_answer = True
                            st.markdown(response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "quiz_options": next_q['options']
                            })
                            st.rerun()  # Перезагружаем чтобы показать кнопки
                        else:
                            # Квиз завершен
                            correct_count = sum(st.session_state.quiz_results)
                            if correct_count == len(topic['quiz']):
                                st.session_state.learning_stage = "main_theory"
                                st.session_state.main_theory_step = 0

                                # Формируем ответ с планом и первым примером
                                main_theory = topic['main_theory']
                                if isinstance(main_theory, dict):
                                    response += f"\n\n🎉 Все правильно! Переходим к основной теории.\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                                    if main_theory['examples']:
                                        first_example = main_theory['examples'][0]
                                        response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                                else:
                                    response += f"\n\n🎉 Все правильно! Переходим к теории.\n\n---\n\n{main_theory}"
                            else:
                                st.session_state.learning_stage = "choice"
                                response += "\n\nНе все ответы правильные. Повторим?\n\n- Напиши **'Да'** - покажу конспект\n- Напиши **'Нет'** - сразу к теории"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.rerun()  # Перезагружаем чтобы обновить интерфейс

                elif stage == "choice":
                    # Обработка выбора после квиза
                    if "разбор" in question.lower():
                        # Собираем темы, по которым были ошибки
                        incorrect_topics = []
                        for i, result in enumerate(st.session_state.quiz_results):
                            if not result:
                                topic_key = topic['quiz'][i].get('topic_key')
                                if topic_key:
                                    incorrect_topics.append(topic_key)

                        # Формируем конспект
                        notes_to_show = ""
                        for key in set(incorrect_topics):
                            note = topic['prerequisite_notes'].get(key)
                            if note:
                                notes_to_show += f"- {note}\n"

                        if notes_to_show:
                            response = f"📖 **Конспект для повторения:**\n\n{notes_to_show}\n\n---\n\nГотов? Напиши **'Понятно'** и переходим дальше!"
                            st.session_state.learning_stage = "waiting_after_notes"
                        else:
                            response = "Ошибок в темах с конспектами не найдено. Переходим к основной теории!"
                            st.session_state.learning_stage = "main_theory"
                            st.session_state.main_theory_step = 0

                            main_theory = topic['main_theory']
                            if isinstance(main_theory, dict):
                                response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                                if main_theory['examples']:
                                    first_example = main_theory['examples'][0]
                                    response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                            else:
                                response += f"\n\n---\n\n{main_theory}"

                    elif "теория" in question.lower():
                        response = "Хорошо! Переходим к основной теории."
                        st.session_state.learning_stage = "main_theory"
                        st.session_state.main_theory_step = 0

                        main_theory = topic['main_theory']
                        if isinstance(main_theory, dict):
                            response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                            if main_theory['examples']:
                                first_example = main_theory['examples'][0]
                                response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                        else:
                            response += f"\n\n---\n\n{main_theory}"
                    else:
                        # Отвлеченный вопрос
                        response = handle_offtopic_question(question, "Используй кнопки 'Разобрать ошибки' или 'К теории'.")

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                elif stage == "waiting_for_topic_selection":
                    # Пользователь может задавать вопросы по конспектам или переходить к теории
                    if "теория" in question.lower():
                        response = "Отлично! Переходим к основной теории."
                        st.session_state.learning_stage = "main_theory"
                        st.session_state.main_theory_step = 0

                        main_theory = topic['main_theory']
                        if isinstance(main_theory, dict):
                            response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                            if main_theory['examples']:
                                first_example = main_theory['examples'][0]
                                response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                        else:
                            response += f"\n\n---\n\n{main_theory}"
                    else:
                        # Используем агента для ответа на вопрос по изученным конспектам
                        agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)

                        # Формируем контекст из всех просмотренных конспектов
                        context_parts = []
                        # Получаем все темы кроме тех, что еще остались в mistake_topics
                        all_mistake_keys = ["дроби", "сложение_дробей", "НОК"]  # все возможные темы
                        viewed_topics = [key for key in all_mistake_keys
                                       if key not in st.session_state.get('mistake_topics', [])]

                        for topic_key in viewed_topics:
                            if topic_key in topic['prerequisite_notes']:
                                note_data = topic['prerequisite_notes'][topic_key]
                                context_parts.append(f"**{note_data['title']}**\n{note_data['content']}")

                        context = "\n\n".join(context_parts) if context_parts else "Ты изучаешь дроби."

                        # Создаем промпт для агента
                        agent_prompt = f"""Контекст изученного материала:
{context}

Вопрос ученика: {question}

Ответь на вопрос понятным языком для ребенка, используя изученный материал."""

                        try:
                            result = agent_executor.invoke({
                                "input": agent_prompt,
                                "chat_history": ""
                            })
                            response = result['output']
                        except Exception as e:
                            print(f"Agent error: {e}")
                            response = f"Отличный вопрос! 😊 {question}\n\nПопробую объяснить проще: если у тебя есть конкретный вопрос по конспекту, задай его, и я помогу разобраться!"

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "show_theory_button": True})

                elif stage == "waiting_after_notes":
                    # После просмотра всех конспектов
                    if "теория" in question.lower():
                        response = "Отлично! Переходим к основной теории."
                        st.session_state.learning_stage = "main_theory"
                        st.session_state.main_theory_step = 0

                        main_theory = topic['main_theory']
                        if isinstance(main_theory, dict):
                            response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                            if main_theory['examples']:
                                first_example = main_theory['examples'][0]
                                response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                        else:
                            response += f"\n\n---\n\n{main_theory}"
                    elif "понятно" in question.lower() or "готов" in question.lower() or "да" in question.lower():
                        response = "Отлично! Переходим к основной теории."
                        st.session_state.learning_stage = "main_theory"
                        st.session_state.main_theory_step = 0

                        main_theory = topic['main_theory']
                        if isinstance(main_theory, dict):
                            response += f"\n\n---\n\n**{main_theory['title']}**\n\n{main_theory['plan']}\n\n---\n\n"
                            if main_theory['examples']:
                                first_example = main_theory['examples'][0]
                                response += f"{first_example['explanation']}\n\n**Задание:** {first_example['question']}"
                        else:
                            response += f"\n\n---\n\n{main_theory}"
                    else:
                        # Используем агента для ответа на вопрос
                        agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)

                        # Формируем контекст из изученных конспектов
                        context_parts = []
                        for topic_key in ["дроби", "сложение_дробей", "НОК"]:
                            if topic_key in topic['prerequisite_notes']:
                                note_data = topic['prerequisite_notes'][topic_key]
                                context_parts.append(f"**{note_data['title']}**\n{note_data['content']}")

                        context = "\n\n".join(context_parts)

                        agent_prompt = f"""Контекст изученного материала:
{context}

Вопрос ученика: {question}

Ответь на вопрос понятным языком для ребенка, используя изученный материал."""

                        try:
                            result = agent_executor.invoke({
                                "input": agent_prompt,
                                "chat_history": ""
                            })
                            response = result['output']
                        except Exception as e:
                            print(f"Agent error: {e}")
                            response = f"Интересный вопрос! 😊 Давай продолжим обучение, и я отвечу на твои вопросы по ходу."

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "show_theory_button": True})

                elif stage == "main_theory":
                    # Обработка основной теории с примерами
                    main_theory = topic['main_theory']
                    if isinstance(main_theory, dict) and 'examples' in main_theory:
                        current_example = main_theory['examples'][st.session_state.main_theory_step]

                        # Проверяем ответ пользователя
                        is_correct = check_answer_with_llm(
                            question,
                            current_example['answer'],
                            model_choice,
                            yandex_api_key,
                            gemini_api_key,
                            question_context=current_example['question']
                        )

                        if is_correct:
                            # Правильно! Переходим к следующему примеру или к боссу
                            response = f"✅ Верно! "
                            st.session_state.main_theory_step += 1

                            if st.session_state.main_theory_step < len(main_theory['examples']):
                                # Есть еще примеры
                                next_example = main_theory['examples'][st.session_state.main_theory_step]
                                response += f"\n\n{next_example['explanation']}\n\n**Задание:** {next_example['question']}"
                            else:
                                # Все примеры пройдены, переходим к боссу
                                correct_quiz = sum(st.session_state.quiz_results)
                                variant_index = 0 if correct_quiz == len(topic['quiz']) else 1
                                st.session_state.boss_variant = variant_index
                                st.session_state.boss_step = 0
                                st.session_state.learning_stage = "boss"

                                variant = topic['boss']['variants'][variant_index]
                                response += f"\n\n{topic['boss']['intro']}\n\n**{variant['success_message']}**\n\n**Задача:** {variant['tasks'][0]['question']}"
                        else:
                            # Неправильно - объясняем ошибку и даем подсказку
                            response = f"🤔 Не совсем. Давай разберемся!\n\n**Подсказка:** {current_example['hint']}\n\nПопробуй еще раз решить: {current_example['question']}"
                    else:
                        # Старый формат - сразу к боссу
                        correct_quiz = sum(st.session_state.quiz_results)
                        variant_index = 0 if correct_quiz == len(topic['quiz']) else 1
                        st.session_state.boss_variant = variant_index
                        st.session_state.boss_step = 0
                        st.session_state.learning_stage = "boss"

                        variant = topic['boss']['variants'][variant_index]
                        response = f"{topic['boss']['intro']}\n\n**{variant['success_message']}**\n\n**Задача:** {variant['tasks'][0]['question']}"

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                elif stage == "boss":
                    # Обработка босса
                    variant = topic['boss']['variants'][st.session_state.boss_variant]
                    current_boss_step = st.session_state.boss_step

                    if current_boss_step >= len(variant['tasks']):
                        # Босс пройден
                        response = f"🎉 Отлично! Все задачи решены!\n\n---\n\n{topic['final_summary']}"
                        st.session_state.learning_stage = "finish"
                    else:
                        task = variant['tasks'][current_boss_step]
                        # Локальная проверка ответа
                        is_correct = check_answer(question, task['answer'])

                        if is_correct:
                            response = f"✅ Правильно! Ответ: {task['answer']}\n\n"
                            st.session_state.boss_step += 1

                            if st.session_state.boss_step < len(variant['tasks']):
                                next_task = variant['tasks'][st.session_state.boss_step]
                                response += f"**Следующая задача:** {next_task['question']}"
                            else:
                                response += f"🎉 Все задачи решены!\n\n---\n\n{topic['final_summary']}"
                                st.session_state.learning_stage = "finish"
                        else:
                            # Проверяем, это попытка ответа или вопрос
                            if any(char.isdigit() for char in question):
                                # Похоже на попытку ответить
                                response = f"🤔 Не совсем. **Подсказка:** {task['hint']}\n\nПопробуй еще раз!"
                            else:
                                # Отвлеченный вопрос
                                response = handle_offtopic_question(question, f"**Задача:** {task['question']}")

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                elif stage == "finish":
                    response = "Ты завершил тему! 🎉\n\nВыбери новую тему слева или задай уточняющий вопрос по ней!"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

# Вставляем CSS, чтобы увеличить размер формул
st.markdown(
    """
    <style>
    .stMarkdown .katex {
        font-size: 1.5em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)