import streamlit as st
import os
import uuid
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from data import TOPICS  # Новый формат схем
from prompts import TUTOR_PROMPT, LEARN_MODE_PROMPT, FEEDBACK_PROMPT
from utils import format_schema, format_chat_to_markdown, get_chat_filename, save_chat_to_sheets

load_dotenv()

st.set_page_config(page_title="Математический помощник AI", page_icon="🧮", layout="wide")

# ============= УТИЛИТЫ =============

def parse_quick_replies(text):
    """
    Парсит маркер быстрых ответов из текста

    Формат: [QUICK_REPLIES: "Вариант 1" | "Вариант 2" | ...]

    Returns:
        tuple: (cleaned_text, list_of_replies)
    """
    pattern = r'\[QUICK_REPLIES:\s*(.+?)\]'
    match = re.search(pattern, text)

    if match:
        # Извлекаем варианты ответов
        replies_str = match.group(1)
        # Разбиваем по разделителю |
        replies = [r.strip().strip('"\'') for r in replies_str.split('|')]
        # Убираем маркер из текста
        cleaned_text = re.sub(pattern, '', text).strip()
        return cleaned_text, replies

    return text, []

# ============= ИНИЦИАЛИЗАЦИЯ АГЕНТА =============

@st.cache_resource
def init_bot(model_choice, yandex_key, gemini_key):
    """Инициализирует помощника - возвращает прямой LLM без инструментов"""
    if model_choice == "YandexGPT 5.1 Pro":
        llm = ChatOpenAI(api_key=yandex_key, base_url="http://localhost:8520/v1",
                        model="yandexgpt/latest", temperature=0.3)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key,
                                    temperature=0.3, convert_system_message_to_human=True)

    return llm

@st.cache_resource
def init_tutor(model_choice, yandex_key, gemini_key):
    """Инициализирует тьютора для Study Mode - возвращает прямой LLM без агента"""
    if model_choice == "YandexGPT 5.1 Pro":
        llm = ChatOpenAI(api_key=yandex_key, base_url="http://localhost:8520/v1",
                        model="yandexgpt/latest", temperature=0.6)
    else:
        # Используем Gemini без thinking mode для более быстрых ответов
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0.6,
            convert_system_message_to_human=True,
            model_kwargs={
                "thinking_config": {
                    "thinking_mode": "DISABLED"
                }
            }
        )

    return llm

# ============= ФУНКЦИИ =============
# (Старые функции удалены - используем промпт-подход)


# ============= ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ =============

if "mode" not in st.session_state:
    # По умолчанию режим изучения темы
    st.session_state.mode = "learn"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Для режима обучения
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "study_mode_initialized" not in st.session_state:
    st.session_state.study_mode_initialized = False
if "needs_feedback" not in st.session_state:
    st.session_state.needs_feedback = False

# ID сессии и метаданные для Google Sheets
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]  # Короткий уникальный ID
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now().strftime('%d.%m.%Y %H:%M:%S')

# Быстрые ответы (кнопки)
if "quick_replies" not in st.session_state:
    st.session_state.quick_replies = []
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

# ============= UI =============

# Заголовок в зависимости от режима
if st.session_state.mode == "learn":
    st.title("🎓 Study Mode")
    st.markdown("*Твой персональный тьютор*")
    st.markdown("---")
else:
    st.title("🧮 Математический помощник AI")
    st.markdown("*Помогаю учить математику!*")
    st.markdown("---")


# ============= SIDEBAR =============

with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор режима работы
    mode = st.radio(
        "🎯 Режим работы:",
        ["learn", "study"],
        format_func=lambda x: {"learn": "📚 Изучить тему", "study": "🎓 Study Mode (Тьютор)"}[x],
        key="mode_selector"
    )

    # Режим не меняется в рантайме (очищаем сообщения и состояние)
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.messages = []
        st.session_state.current_topic = None
        st.session_state.study_mode_initialized = False
        st.session_state.needs_feedback = False
        # Новая сессия
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_start = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    
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
    if mode == "learn":
        # Режим изучения темы
        st.header("📚 Выбери тему")
        for topic_id, topic_data in TOPICS.items():
            if st.button(topic_data['title'], key=f"topic_{topic_id}", use_container_width=True):
                st.session_state.current_topic = topic_id
                st.session_state.needs_feedback = False  # Сбрасываем при выборе новой темы
                # Новая сессия
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.session_start = datetime.now().strftime('%d.%m.%Y %H:%M:%S')

                # Приветственное сообщение с планом урока
                welcome_message = f"**{topic_data['title']}**\n\n{topic_data.get('description', '')}\n\n"

                # Добавляем план, если он есть
                if 'plan' in topic_data:
                    welcome_message += f"{topic_data['plan']}\n\n"

                welcome_message += "Готов? Поехали! 🚀"

                st.session_state.messages = [{
                    "role": "assistant",
                    "content": welcome_message
                }]
                st.rerun()
    else:
        # Study Mode - свободный тьютор
        st.header("🎓 Study Mode")
        st.markdown("*Задай любой вопрос по школьным предметам*")
        st.markdown("---")
        st.markdown("**Примеры вопросов:**")
        st.markdown("- Помоги разобраться с дробями")
        st.markdown("- Объясни квадратные уравнения")
        st.markdown("- Реши задачу по физике")

        if st.button("🆕 Начать новую тему", use_container_width=True):
            st.session_state.messages = []
            st.session_state.study_mode_initialized = False
            # Новая сессия
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.session_start = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Начать заново", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_topic = None
        st.session_state.study_mode_initialized = False
        st.session_state.needs_feedback = False
        # Новая сессия
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.session_start = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        st.rerun()

    # Кнопка экспорта диалога
    if len(st.session_state.messages) > 0:
        st.markdown("---")
        st.markdown("**💾 Экспорт диалога**")

        # Получаем название темы если есть
        topic_title = None
        if st.session_state.mode == "learn" and st.session_state.current_topic:
            topic_title = TOPICS[st.session_state.current_topic].get('title', 'Тема')

        # Форматируем диалог
        chat_markdown = format_chat_to_markdown(st.session_state.messages, topic_title)
        filename = get_chat_filename(topic_title, "md")

        # Кнопка скачивания
        st.download_button(
            label="📥 Скачать диалог (Markdown)",
            data=chat_markdown,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True
        )

        # Кнопка сохранения в Google Sheets
        if st.button("📊 Сохранить в Google Sheets", use_container_width=True):
            with st.spinner("Сохраняю в Google Sheets..."):
                success = save_chat_to_sheets(
                    messages=st.session_state.messages,
                    topic_title=topic_title,
                    session_id=st.session_state.session_id,
                    session_start=st.session_state.session_start
                )

                if success:
                    st.success(f"✅ Диалог сохранен в Google Sheets! (Session ID: {st.session_state.session_id})")
                else:
                    st.error("❌ Ошибка сохранения. Проверьте настройки Google Sheets в .env файле")
                    st.info("💡 Инструкция по настройке в файле GOOGLE_SHEETS_SETUP.md")

# ============= ПРОВЕРКА API =============

current_key = yandex_api_key if model_choice == "YandexGPT 5.1 Pro" else gemini_api_key

if not current_key:
    st.warning(f"⚠️ Введите API ключ в настройках слева")
    st.info("💡 Для начала получите бесплатный ключ Gemini")
    st.stop()

# ============= ОСНОВНОЙ ИНТЕРФЕЙС =============

# Показываем режим
mode_badges = {
    "learn": "📚 Изучить тему",
    "study": "🎓 Study Mode (Тьютор)"
}
st.info(f"**Текущий режим:** {mode_badges[st.session_state.mode]}")

# Инициализация Study Mode - модель сама генерирует первое приветствие
if st.session_state.mode == "study" and not st.session_state.study_mode_initialized and len(st.session_state.messages) == 0:
    # Вызываем модель с пустым input - она сама начнет диалог согласно TUTOR_PROMPT
    tutor_llm = init_tutor(model_choice, yandex_api_key, gemini_api_key)

    # Формируем промпт с пустой историей и пустым input
    full_prompt = TUTOR_PROMPT.replace("{chat_history}", "").replace("{input}", "")

    try:
        response_obj = tutor_llm.invoke(full_prompt)
        welcome_message = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

        # Парсим быстрые ответы
        cleaned_message, quick_replies = parse_quick_replies(welcome_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": cleaned_message
        })
        st.session_state.quick_replies = quick_replies
    except Exception as e:
        print(f"Study Mode init error: {e}")
        # Фоллбек на простое приветствие
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Привет! Я помогу тебе разобраться с любой темой 📚 Скажи, пожалуйста, в каком ты классе и что сегодня будем изучать?"
        })
        st.session_state.quick_replies = []

    st.session_state.study_mode_initialized = True
    st.rerun()

# История сообщений
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Рендерим содержимое сообщения, поддерживая крупный LaTeX
        st.markdown(message["content"], unsafe_allow_html=True)

# Быстрые ответы (кнопки)
if st.session_state.quick_replies:
    st.markdown("**💬 Быстрые ответы:**")
    cols = st.columns(len(st.session_state.quick_replies))

    for idx, reply in enumerate(st.session_state.quick_replies):
        with cols[idx]:
            if st.button(reply, key=f"quick_reply_{idx}", use_container_width=True):
                # Устанавливаем pending message для обработки
                st.session_state.pending_message = reply
                # Очищаем быстрые ответы
                st.session_state.quick_replies = []
                # Перезагружаем страницу для обработки ответа
                st.rerun()

# Обработка обычного ввода с клавиатуры или pending message
question = st.chat_input("Напиши свой вопрос или ответ...")

# Проверяем pending message (из кнопки быстрого ответа)
if st.session_state.pending_message:
    question = st.session_state.pending_message
    st.session_state.pending_message = None

if question:
    # Очищаем быстрые ответы (пользователь ввел текст вручную)
    st.session_state.quick_replies = []

    # Сохраняем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Думаю..."):
            # Обработка в зависимости от режима
            if st.session_state.mode == "study":
                # Study Mode - свободный тьютор (прямой вызов LLM без агента)
                tutor_llm = init_tutor(model_choice, yandex_api_key, gemini_api_key)

                # Формируем полный промпт с историей чата
                chat_history = "\n".join([
                    f"{'Ученик' if msg['role'] == 'user' else 'Тьютор'}: {msg['content']}"
                    for msg in st.session_state.messages[-5:]  # Последние 5 сообщений для контекста
                ])

                # Формируем полное сообщение для LLM
                full_prompt = TUTOR_PROMPT.replace("{chat_history}", chat_history).replace("{input}", question)

                try:
                    response_obj = tutor_llm.invoke(full_prompt)
                    response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

                    # Парсим быстрые ответы
                    response, quick_replies = parse_quick_replies(response)
                    st.session_state.quick_replies = quick_replies
                except Exception as e:
                    print(f"Tutor error: {e}")
                    response = "Извини, произошла ошибка. Попробуй переформулировать вопрос."
                    st.session_state.quick_replies = []

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Если есть быстрые ответы, делаем rerun чтобы кнопки появились
                if st.session_state.quick_replies:
                    st.rerun()

            elif st.session_state.current_topic is None:
                response = "Пожалуйста, выбери тему из списка слева! 👈"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Learn Mode - используем LEARN_MODE_PROMPT со схемой темы
                topic = TOPICS[st.session_state.current_topic]
                learn_llm = init_tutor(model_choice, yandex_api_key, gemini_api_key)

                # Форматируем схему темы
                schema = format_schema(topic)

                # Формируем историю чата
                chat_history = "\n".join([
                    f"{'Ученик' if msg['role'] == 'user' else 'Тьютор'}: {msg['content']}"
                    for msg in st.session_state.messages[-5:]  # Последние 5 сообщений
                ])

                # Формируем полный промпт
                full_prompt = LEARN_MODE_PROMPT.replace("{schema}", schema).replace("{chat_history}", chat_history).replace("{input}", question)

                try:
                    response_obj = learn_llm.invoke(full_prompt)
                    response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                except Exception as e:
                    print(f"Learn mode error: {e}")
                    response = "Извини, произошла ошибка. Попробуй переформулировать вопрос."

                # Проверяем маркер завершения урока
                # LLM добавляет [УРОК_ЗАВЕРШЕН] только после показа конспекта
                if "[УРОК_ЗАВЕРШЕН]" in response:
                    st.session_state.needs_feedback = True
                    # Убираем маркер из отображаемого текста (он служебный)
                    response = response.replace("[УРОК_ЗАВЕРШЕН]", "").strip()

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Если нужен фидбек, показываем его автоматически
                if st.session_state.needs_feedback:
                    # Формируем историю для фидбека (весь разговор)
                    full_chat_history = "\n".join([
                        f"{'Ученик' if msg['role'] == 'user' else 'Тьютор'}: {msg['content']}"
                        for msg in st.session_state.messages
                    ])

                    # Формируем промпт для фидбека
                    feedback_prompt = FEEDBACK_PROMPT.replace(
                        "{topic_title}", topic.get('title', '')
                    ).replace(
                        "{topic_description}", topic.get('description', '')
                    ).replace(
                        "{chat_history}", full_chat_history
                    ).replace(
                        "{final_summary}", topic.get('summary', '')
                    )

                    try:
                        feedback_obj = learn_llm.invoke(feedback_prompt)
                        feedback = feedback_obj.content if hasattr(feedback_obj, 'content') else str(feedback_obj)

                        # Показываем фидбек
                        st.markdown("\n\n---\n\n")
                        st.markdown(feedback)
                        st.session_state.messages.append({"role": "assistant", "content": f"\n\n---\n\n{feedback}"})

                        # Сбрасываем флаг
                        st.session_state.needs_feedback = False
                    except Exception as e:
                        print(f"Feedback error: {e}")

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