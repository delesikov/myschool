import streamlit as st
import os
import sympy
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Математический помощник AI", page_icon="🧮", layout="wide")

# ============= БАЗА ДЛЯ РЕЖИМА "ИЗУЧИТЬ ТЕМУ" =============

LEARNING_TOPICS = {
    "topic1": {
        "title": "➕ Сложение обыкновенных дробей",
        "description": "Научимся складывать дроби с одинаковыми и разными знаменателями",
        
        # ЭТАП 1: Квиз-диагностика
        "quiz": [
            {
                "question": "Чему равен знаменатель у дроби 2/3?",
                "options": ["3", "2", "5", "Не знаю"],
                "correct": "3",
                "explanation_template": "Знаменатель дроби — это число под чертой. У дроби 2/3 знаменатель равен 3."
            },
            {
                "question": "Можешь сложить? 1/7 + 2/7 = ?",
                "options": ["3/14", "7/3", "3/7", "Не могу"],
                "correct": "3/7",
                "explanation_template": "Когда знаменатели одинаковые, складываем числители: 1/7 + 2/7 = (1+2)/7 = 3/7"
            },
            {
                "question": "Найди наименьшее общее кратное НОК(12, 15) = ?",
                "options": ["60", "27", "180", "3"],
                "correct": "60",
                "explanation_template": "НОК(12, 15) = 60. Это наименьшее число, которое делится и на 12, и на 15."
            }
        ],
        
        # Конспекты для повторения
        "prerequisite_notes": {
            "НОК": """**Наименьшее общее кратное (НОК)**

НОК двух чисел — это наименьшее число, которое делится на каждое из них.

**Как найти НОК(12, 15):**

1. Разложим числа на множители:
   - 12 = 2·2·3
   - 15 = 3·5

2. Множитель 3 есть и там, и там, возьмем его только один раз
3. Все множители перемножить: НОК(12,15) = 2·2·3·5 = 60

**Запомни:** Знаменатель дроби — это то, что написано под дробной чертой. А над ней — числитель.""",
            
            "дроби": """**Что такое дробь?**

На всякий случай напомню: знаменатель дроби — это то, что написано под дробной чертой. А над ней — числитель."""
        },
        
        # ЭТАП 2: Вводная теория
        "intro_theory": """**Давай я объясню тебе как складывать дроби.**

**Понятие дроби:**

Давай сложим 1/7 и 2/7. Сложи числители, а знаменатель оставь общим:

**1/7 + 2/7 = (1+2)/7 = 3/7**

Это работает, когда знаменатели одинаковые!

---

**А теперь попробуй сам:** Сможешь сложить так же 2/9 + 5/9 = ?""",
        
        # ЭТАП 3: Основная теория  
        "main_theory": """**Сложение с общим знаменателем**

Давай мы вместе сложим 1/7 и 2/7. Сложи числители, а знаменатель оставь общим.

**1/7 + 2/7 = (1+2)/7 = 3/7**

---

**Сложение с разными знаменателями**

Если знаменатели разные, нужно сначала привести дроби к общему знаменателю!

**Например: 1/7 + 2/5**

Действительно, знаменатель у этих дробей 9. 
Складываем числители и получаем 7.

**Ответ: 7/9** ✅

---

Теперь ты умеешь складывать дроби! Но чтобы тебе не пришлось перемножать большие числа, удобнее приводить дроби к **наименьшему общему знаменателю**.

Например, сложим 9/22 и 7/33. Очень не хочется умножать 9 на 33. Найдем наименьшее общее кратное знаменателей:

**НОК(22,33) = 66**

Общий знаменатель 66. Значит первую дробь надо умножить на 3, а вторую — на 2:

**9/22 + 7/33 = 27/66 + 14/66 = 41/66**""",
        
        # ЭТАП 4: Босс
        "boss": {
            "intro": "Мы почти у финиша! Реши самостоятельно:",
            "variants": [
                {
                    "variant": 1,
                    "tasks": [
                        {
                            "question": "5/9 + 1/6",
                            "answer": "13/18",
                            "hint": "НОК(9, 6) = 18. Приведи дроби к знаменателю 18"
                        }
                    ],
                    "success_message": "Если всё до этого получалось хорошо ✅"
                },
                {
                    "variant": 2,
                    "tasks": [
                        {
                            "question": "5/9 + 1/6",
                            "answer": "13/18",
                            "hint": "НОК(9, 6) = 18"
                        },
                        {
                            "question": "НОК(6, 9) = ?",
                            "answer": "18",
                            "hint": "Найди наименьшее общее кратное"
                        }
                    ],
                    "success_message": "Если до этого были ошибки ⚠️"
                }
            ]
        },
        
        # ЭТАП 5: Финиш
        "final_summary": """**🎉 Отлично! Теперь ты умеешь складывать дроби!**

**Держи короткий конспект:**

✅ **Найди НОК знаменателей**
✅ **Приведи дроби к общему знаменателю**  
✅ **Сложи числители, перепиши общий знаменатель**

---

У тебя остались ещё вопросы про сложение дробей? Или могу рассказать другую тему!

**Варианты:**
- Расскажи про умножение дробей
- Расскажи, как сравнивать дроби"""
    }
}

# ============= БАЗА ЗАДАЧ ДЛЯ ПОШАГОВОГО РЕШЕНИЯ =============

DEFAULT_TASKS = {
    "task1": {
        "title": "🍎 Задача про яблоки",
        "description": "У Маши было 15 яблок. Она отдала 7 яблок своему другу. Сколько яблок осталось у Маши?",
        "solution_steps": [
            {
                "step": 1,
                "hint": "Давай подумаем: сколько яблок было у Маши в начале?",
                "answer": "15",
                "explanation": "Правильно! У Маши было 15 яблок."
            },
            {
                "step": 2,
                "hint": "Хорошо! А сколько яблок она отдала другу?",
                "answer": "7",
                "explanation": "Верно! Она отдала 7 яблок."
            },
            {
                "step": 3,
                "hint": "Отлично! Теперь скажи: какое действие нужно сделать? Сложение или вычитание?",
                "answer": ["вычитание", "вычесть", "минус", "-"],
                "explanation": "Правильно! Нужно вычитание, потому что яблок стало меньше."
            },
            {
                "step": 4,
                "hint": "Супер! Теперь реши: 15 - 7 = ?",
                "answer": "8",
                "explanation": "🎉 Молодец! 15 - 7 = 8. У Маши осталось 8 яблок!"
            }
        ],
        "final_answer": "8 яблок"
    },
    "task2": {
        "title": "📐 Задача про периметр",
        "description": "Прямоугольник имеет длину 8 см и ширину 5 см. Найди периметр прямоугольника.",
        "solution_steps": [
            {
                "step": 1,
                "hint": "Сначала вспомним: какая формула периметра прямоугольника?",
                "answer": ["2*(a+b)", "2(a+b)", "(a+b)*2", "2a+2b"],
                "explanation": "Отлично! Периметр: P = 2×(длина + ширина)"
            },
            {
                "step": 2,
                "hint": "Какая длина прямоугольника?",
                "answer": "8",
                "explanation": "Верно! Длина = 8 см"
            },
            {
                "step": 3,
                "hint": "А какая ширина?",
                "answer": "5",
                "explanation": "Правильно! Ширина = 5 см"
            },
            {
                "step": 4,
                "hint": "Посчитай: длина + ширина = ?",
                "answer": "13",
                "explanation": "Хорошо! 8 + 5 = 13"
            },
            {
                "step": 5,
                "hint": "Теперь умножь на 2: 13 × 2 = ?",
                "answer": "26",
                "explanation": "🎉 Отлично! Периметр = 26 см!"
            }
        ],
        "final_answer": "26 см"
    }
}

if "tasks_database" not in st.session_state:
    st.session_state.tasks_database = DEFAULT_TASKS.copy()

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

def show_solution_schema(task_data):
    """Показывает схему решения для пошагового режима"""
    with st.expander("📋 Посмотреть схему решения", expanded=False):
        st.markdown(f"**Задача:** {task_data['description']}")
        st.markdown("---")
        for step in task_data['solution_steps']:
            st.markdown(f"**Шаг {step['step']}:**")
            st.markdown(f"- Вопрос: {step['hint']}")
            answer_text = step['answer'] if isinstance(step['answer'], str) else f"{step['answer'][0]} (или {', '.join(step['answer'][1:])})"
            st.markdown(f"- Ответ: `{answer_text}`")
            st.markdown(f"- Объяснение: {step['explanation']}")
            st.markdown("")
        st.success(f"**Итоговый ответ:** {task_data['final_answer']}")

def show_learning_schema(topic_data):
    """Показывает схему темы для режима обучения"""
    with st.expander("📋 Посмотреть схему темы", expanded=False):
        st.markdown(f"**Тема:** {topic_data['title']}")
        st.markdown("---")
        
        st.markdown("### 1️⃣ Квиз-диагностика")
        for i, q in enumerate(topic_data['quiz'], 1):
            st.markdown(f"**Вопрос {i}:** {q['question']}")
            st.markdown(f"- Правильный ответ: `{q['correct']}`")
        
        st.markdown("---")
        st.markdown("### 2️⃣ Вводная теория")
        st.info(topic_data['intro_theory'][:200] + "...")
        
        st.markdown("---")
        st.markdown("### 3️⃣ Основная теория")
        st.info(topic_data['main_theory'][:200] + "...")
        
        st.markdown("---")
        st.markdown("### 4️⃣ Босс (проверка)")
        st.markdown("2 варианта задач")
        
        st.markdown("---")
        st.markdown("### 5️⃣ Финиш")
        st.success("Короткий конспект")

# ============= ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ =============

if "mode" not in st.session_state:
    st.session_state.mode = "free"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Для пошагового режима
if "current_task" not in st.session_state:
    st.session_state.current_task = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

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

# ============= UI =============

st.title("🧮 Математический помощник AI")
st.markdown("*Помогаю учить математику!*")
st.markdown("---")

# ============= SIDEBAR =============

with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор режима
    mode = st.radio(
        "🎯 Режим работы:",
        ["free", "stepbystep", "learn"],
        format_func=lambda x: {
            "free": "📝 Свободный ввод",
            "stepbystep": "👣 Пошаговое решение",
            "learn": "📚 Изучить тему"
        }[x],
        key="mode_selector"
    )
    
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.messages = []
        st.session_state.current_task = None
        st.session_state.current_step = 0
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
    
    # РЕЖИМ-ЗАВИСИМЫЙ КОНТЕНТ
    if st.session_state.mode == "free":
        st.header("📚 Примеры задач")
        examples = [
            ("👋 Привет!", "Привет!"),
            ("🔢 Вычисли 25 × 4", "Вычисли 25 × 4"),
            ("📐 Реши x² - 9 = 0", "Реши уравнение x² - 9 = 0"),
        ]
        for label, question in examples:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state.selected_question = question
    
    elif st.session_state.mode == "stepbystep":
        st.header("📖 Выбери задачу")
        for task_id, task_data in st.session_state.tasks_database.items():
            if st.button(task_data["title"], key=f"task_{task_id}", use_container_width=True):
                st.session_state.current_task = task_id
                st.session_state.current_step = 0
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": f"**{task_data['title']}**\n\n{task_data['description']}\n\n" +
                              f"Давай решим вместе! 😊\n\n**Шаг 1:** {task_data['solution_steps'][0]['hint']}"
                }]
                st.rerun()
    
    else:  # learn режим
        st.header("📚 Выбери тему")
        for topic_id, topic_data in LEARNING_TOPICS.items():
            if st.button(topic_data["title"], key=f"topic_{topic_id}", use_container_width=True):
                st.session_state.current_topic = topic_id
                st.session_state.learning_stage = "quiz"
                st.session_state.quiz_results = []
                st.session_state.boss_variant = None
                st.session_state.boss_step = 0
                st.session_state.waiting_for_quiz_answer = True
                
                # Стартуем с квиза
                quiz = topic_data['quiz'][0]
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": f"**{topic_data['title']}**\n\n{topic_data['description']}\n\n" +
                              f"Давай сначала проверим, что ты уже знаешь! 🎯\n\n**Вопрос 1:** {quiz['question']}",
                    "quiz_options": quiz['options']  # Сохраняем варианты ответов
                }]
                st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Начать заново", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_task = None
        st.session_state.current_step = 0
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

# Инициализация агента (только для свободного режима)
if st.session_state.mode == "free":
    try:
        agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.stop()

# ============= ОСНОВНОЙ ИНТЕРФЕЙС =============

# Показываем режим
mode_badges = {
    "free": "📝 Свободный ввод",
    "stepbystep": "👣 Пошаговое решение",
    "learn": "📚 Изучить тему"
}
st.info(f"**Текущий режим:** {mode_badges[st.session_state.mode]}")

# СХЕМЫ
if st.session_state.mode == "stepbystep" and st.session_state.current_task:
    task_data = st.session_state.tasks_database[st.session_state.current_task]
    show_solution_schema(task_data)

if st.session_state.mode == "learn" and st.session_state.current_topic:
    topic_data = LEARNING_TOPICS[st.session_state.current_topic]
    show_learning_schema(topic_data)

# История сообщений
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
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
                        response += f"\n\n🎉 Все правильно! Переходим к теории.\n\n---\n\n{topic['main_theory']}"
                    else:
                        st.session_state.learning_stage = "choice"
                        response += "\n\nНе все ответы правильные. Повторим?\n\n- Напиши **'Да'** - покажу конспект\n- Напиши **'Нет'** - сразу к теории"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# Обработка обычного ввода
if "selected_question" in st.session_state:
    question = st.session_state.selected_question
    del st.session_state.selected_question
else:
    question = st.chat_input("Напиши свой вопрос или ответ...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Думаю..."):
            
            if st.session_state.mode == "free":
                # ===== РЕЖИМ СВОБОДНОГО ВВОДА =====
                try:
                    history = "\n".join([
                        f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                        for m in st.session_state.messages[-6:]
                    ])
                    
                    response = agent_executor.invoke({"input": question, "chat_history": history})
                    answer = response['output']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Ошибка: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            elif st.session_state.mode == "stepbystep":
                # ===== РЕЖИМ ПОШАГОВОГО РЕШЕНИЯ =====
                if st.session_state.current_task is None:
                    response = "Пожалуйста, выбери задачу из списка слева! 👈"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    task = st.session_state.tasks_database[st.session_state.current_task]
                    current_step_num = st.session_state.current_step
                    
                    if current_step_num >= len(task["solution_steps"]):
                        # Задача решена - свободный диалог
                        try:
                            history = "\n".join([
                                f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                                for m in st.session_state.messages[-6:]
                            ])
                            agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)
                            response_obj = agent_executor.invoke({"input": question, "chat_history": history})
                            response = response_obj['output']
                            response += "\n\n---\n\n💡 Хочешь решить ещё задачу? Выбери её слева!"
                        except:
                            response = f"🎉 Отлично! Мы решили задачу!\n\n**Ответ:** {task['final_answer']}\n\nХочешь решить еще одну? Выбери слева!"
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        step = task["solution_steps"][current_step_num]
                        
                        # Ключевые слова для запроса помощи
                        help_keywords = ["помощь", "помоги", "не понимаю", "не знаю", "объясни", 
                                       "подсказка", "что делать", "как решить", "не получается"]
                        asking_for_help = any(keyword in question.lower() for keyword in help_keywords)
                        
                        # Используем LLM для проверки ответа
                        is_correct = check_answer_with_llm(
                            question, 
                            step["answer"], 
                            model_choice, 
                            yandex_api_key, 
                            gemini_api_key,
                            question_context=step['hint']
                        )
                        
                        if is_correct:
                            # Правильный ответ
                            response = f"✅ {step['explanation']}\n\n"
                            st.session_state.current_step += 1
                            
                            if st.session_state.current_step < len(task["solution_steps"]):
                                next_step = task["solution_steps"][st.session_state.current_step]
                                response += f"**Шаг {next_step['step']}:** {next_step['hint']}"
                            else:
                                response += f"🎉 Молодец! Ты решил задачу!\n\n**Ответ:** {task['final_answer']}"
                            
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        elif asking_for_help:
                            # Ученик просит помощь
                            response = f"Конечно, помогу! 😊\n\n**Подсказка:** {step['hint']}\n\n"
                            
                            # Даём дополнительное объяснение
                            answer_str = step["answer"] if isinstance(step["answer"], str) else step["answer"][0]
                            
                            if "сколько" in step['hint'].lower() or "какая" in step['hint'].lower() or "какое" in step['hint'].lower():
                                response += f"Смотри внимательно на условие задачи! 👀"
                            else:
                                response += f"Попробуй подумать ещё раз, у тебя получится! 💪"
                            
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        else:
                            # Проверяем, это попытка ответа или вопрос для обсуждения
                            answer_str = step["answer"] if isinstance(step["answer"], str) else step["answer"][0]
                            
                            # Признаки попытки ответить (короткий ответ, есть цифры или ключевые слова из ожидаемого ответа)
                            is_short = len(question.split()) <= 5
                            has_numbers = any(char.isdigit() for char in question)
                            has_math_words = any(word in question.lower() for word in ["плюс", "минус", "умножить", "разделить", "равно", "вычитание", "сложение"])
                            
                            is_attempt = is_short and (has_numbers or has_math_words)
                            
                            if is_attempt:
                                # Это похоже на попытку ответить
                                response = f"🤔 Не совсем так. Давай подумаем вместе!\n\n**Подсказка:** {step['hint']}\n\nПопробуй ещё раз! Если нужна помощь, просто напиши 'помоги' 😊"
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                # Это вопрос для обсуждения - используем агента
                                try:
                                    history = "\n".join([
                                        f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                                        for m in st.session_state.messages[-6:]
                                    ])
                                    
                                    # Контекст задачи для агента
                                    context = f"Мы решаем задачу: {task['description']}. Сейчас на шаге {current_step_num + 1}: {step['hint']}"
                                    
                                    agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)
                                    response_obj = agent_executor.invoke({
                                        "input": f"Контекст: {context}\n\nВопрос ученика: {question}\n\nОтветь дружелюбно и понятно, но не давай прямой ответ на текущий шаг.", 
                                        "chat_history": history
                                    })
                                    response = response_obj['output']
                                    response += f"\n\n---\n\n💡 Теперь попробуй ответить на наш вопрос:\n\n**Шаг {step['step']}:** {step['hint']}"
                                except Exception as e:
                                    response = f"Интересный вопрос! 😊\n\nДавай я помогу: {step['explanation']}\n\n**Теперь попробуй:** {step['hint']}"
                                
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
            
            else:  # learn режим
                # ===== РЕЖИМ ИЗУЧЕНИЯ ТЕМЫ =====
                if st.session_state.current_topic is None:
                    response = "Пожалуйста, выбери тему из списка слева! 👈"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    topic = LEARNING_TOPICS[st.session_state.current_topic]
                    stage = st.session_state.learning_stage
                    
                    # Функция для обработки отвлечённого вопроса
                    def handle_offtopic_question(question, current_hint):
                        try:
                            # Используем агента для краткого ответа
                            history = "\n".join([
                                f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                                for m in st.session_state.messages[-4:]
                            ])
                            agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)
                            response_obj = agent_executor.invoke({
                                "input": f"Ответь КРАТКО (максимум 2 предложения): {question}", 
                                "chat_history": history
                            })
                            brief_answer = response_obj['output']
                            return f"{brief_answer}\n\n---\n\n💡 А теперь давай вернёмся к нашему заданию!\n\n{current_hint}"
                        except:
                            return f"Интересный вопрос! 😊 Но давай сначала закончим текущее задание, а потом я отвечу подробнее!\n\n{current_hint}"
                    
                    if stage == "quiz":
                        # Обработка квиза (когда пользователь ввел текст вместо кнопки)
                        quiz_index = len(st.session_state.quiz_results)
                        
                        if quiz_index >= len(topic['quiz']):
                            # Квиз уже завершен, переходим дальше
                            correct_count = sum(st.session_state.quiz_results)
                            if correct_count == len(topic['quiz']):
                                st.session_state.learning_stage = "main_theory"
                                response = f"🎉 Отлично! Переходим к теории.\n\n---\n\n{topic['main_theory']}"
                            else:
                                st.session_state.learning_stage = "choice"
                                response = "\n\nНе все ответы правильные. Повторим?\n\n- Напиши **'Да'** - покажу конспект\n- Напиши **'Нет'** - сразу к теории"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            quiz_q = topic['quiz'][quiz_index]
                            
                            # Проверяем, является ли это ответом на вопрос квиза
                            is_quiz_answer = any(opt.lower().replace(" ", "") in question.lower().replace(" ", "") 
                                               for opt in quiz_q['options'])
                            
                            if is_quiz_answer:
                                # Это ответ на квиз - используем LLM проверку
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
                                else:
                                    # Квиз завершен
                                    correct_count = sum(st.session_state.quiz_results)
                                    if correct_count == len(topic['quiz']):
                                        st.session_state.learning_stage = "main_theory"
                                        response += f"\n\n🎉 Все правильно! Переходим к теории.\n\n---\n\n{topic['main_theory']}"
                                    else:
                                        st.session_state.learning_stage = "choice"
                                        response += "\n\nНе все ответы правильные. Повторим?\n\n- Напиши **'Да'** - покажу конспект\n- Напиши **'Нет'** - сразу к теории"
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
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
                    
                    elif stage == "choice":
                        # Обработка выбора после квиза
                        if "да" in question.lower() or "вспомн" in question.lower():
                            response = f"📖 **Конспект для повторения:**\n\n{topic['prerequisite_notes']['НОК']}\n\n---\n\nГотов? Напиши **'Понятно'** и переходим дальше!"
                            st.session_state.learning_stage = "waiting_after_notes"
                        elif "нет" in question.lower() or "объясн" in question.lower():
                            response = f"Хорошо! Переходим к вводной теории.\n\n---\n\n{topic['intro_theory']}"
                            st.session_state.learning_stage = "intro_theory"
                        else:
                            # Отвлеченный вопрос
                            response = handle_offtopic_question(question, "Повторим темы?\n\n- Напиши **'Да'** - покажу конспект\n- Напиши **'Нет'** - сразу к теории")
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    elif stage == "waiting_after_notes":
                        # После просмотра конспекта
                        if "понятно" in question.lower() or "готов" in question.lower() or "да" in question.lower():
                            response = f"Отлично! Переходим к вводной теории.\n\n---\n\n{topic['intro_theory']}"
                            st.session_state.learning_stage = "intro_theory"
                        else:
                            response = handle_offtopic_question(question, "Готов продолжить? Напиши **'Понятно'** и переходим дальше!")
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    elif stage == "intro_theory":
                        # Проверяем практический вопрос из вводной теории с помощью LLM
                        is_correct = check_answer_with_llm(
                            question,
                            "7/9",
                            model_choice,
                            yandex_api_key,
                            gemini_api_key,
                            question_context="Сложи 2/9 + 5/9"
                        )
                        
                        if is_correct:
                            response = f"✅ Правильно! 2/9 + 5/9 = 7/9\n\nТеперь переходим к основной теории!\n\n---\n\n{topic['main_theory']}"
                            st.session_state.learning_stage = "main_theory"
                        else:
                            # Проверяем, это попытка ответа или отвлеченный вопрос
                            if any(char.isdigit() for char in question) and "/" in question:
                                # Похоже на попытку ответить
                                response = "🤔 Попробуй еще раз! Подсказка: складываем числители 2 + 5, знаменатель остается 9"
                            else:
                                # Отвлеченный вопрос
                                response = handle_offtopic_question(question, "**А теперь попробуй сам:** Сможешь сложить так же 2/9 + 5/9 = ?")
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    elif stage == "main_theory":
                        # После основной теории переходим к боссу
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
                            # Используем LLM для проверки ответа
                            is_correct = check_answer_with_llm(
                                question,
                                task['answer'],
                                model_choice,
                                yandex_api_key,
                                gemini_api_key,
                                question_context=task['question']
                            )
                            
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
                        # На финише отвечаем на любые вопросы более подробно
                        try:
                            history = "\n".join([
                                f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                                for m in st.session_state.messages[-6:]
                            ])
                            agent_executor = init_bot(model_choice, yandex_api_key, gemini_api_key)
                            response_obj = agent_executor.invoke({"input": question, "chat_history": history})
                            response = response_obj['output']
                            response += "\n\n---\n\n💡 Хочешь изучить новую тему? Выбери её слева!"
                        except:
                            response = "Ты завершил тему! 🎉\n\nВыбери новую тему слева или задай свободный вопрос!"
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
footer_text = {
    "free": "💡 Задавай любые математические вопросы!",
    "stepbystep": "💡 Выбери задачу и нажми '📋 Посмотреть схему'!",
    "learn": "💡 Изучай темы по шагам с квизом и практикой!"
}
st.markdown(footer_text[st.session_state.mode])