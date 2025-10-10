# Математический помощник с выбором модели
import os
from typing import Optional, Literal
import numexpr
import sympy
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def calculator(expression: str) -> str:
    try:
        result = numexpr.evaluate(expression.strip(), local_dict={'pi': 3.141592653589793, 'e': 2.718281828459045})
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

def symbolic_math(expression: str) -> str:
    try:
        x, y, z = sympy.symbols('x y z')
        local_dict = {
            'x': x, 'y': y, 'z': z, 'integrate': sympy.integrate, 'diff': sympy.diff,
            'solve': sympy.solve, 'limit': sympy.limit, 'expand': sympy.expand,
            'simplify': sympy.simplify, 'sin': sympy.sin, 'cos': sympy.cos,
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

tools = [
    Tool(name="Calculator", func=calculator, description="Числовые вычисления: '2+2', '45*89'"),
    Tool(name="SymbolicMath", func=symbolic_math, description="Символьная математика: 'integrate(x**2, x)'"),
    Tool(name="EquationSolver", func=equation_solver, description="Решение уравнений: 'x**2 - 5*x + 6'")
]

template = """Ты — математический помощник AI.

Инструменты: {tools}

Формат:
Thought: (рассуждения)
Action: (инструмент)
Action Input: (данные)
Observation: (результат)
Final Answer: (ответ)

История: {chat_history}
Вопрос: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "chat_history", "agent_scratchpad"],
    partial_variables={"tools": "\n".join([f"{t.name}: {t.description}" for t in tools])}
)

class MultiModelMathTutor:
    def __init__(self, model: Literal["yandex", "gemini"] = "gemini", 
                 yandex_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        self.model_name = model
        
        if model == "yandex":
            api_key = yandex_api_key or os.getenv("YANDEX_API_KEY")
            if not api_key:
                raise ValueError("Требуется YANDEX_API_KEY")
            self.llm = ChatOpenAI(api_key=api_key, base_url="http://localhost:8520/v1",
                                 model="yandexgpt/latest", temperature=0.1)
            print("✅ YandexGPT 5.1 Pro")
        else:
            api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Требуется GOOGLE_API_KEY")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key,
                                             temperature=0.1, convert_system_message_to_human=True)
            print("✅ Google Gemini 2.5 Flash")
        
        self.agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True,
                                          max_iterations=5, handle_parsing_errors=True)
        self.chat_history = []
    
    def ask(self, question: str) -> str:
        try:
            history_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in self.chat_history[-3:]])
            response = self.agent_executor.invoke({"input": question, "chat_history": history_str})
            answer = response['output']
            self.chat_history.append({"q": question, "a": answer})
            return answer
        except Exception as e:
            return f"Ошибка: {str(e)}"

if __name__ == "__main__":
    print("🤖 Математический помощник AI\n")
    print("1. YandexGPT 5.1 Pro")
    print("2. Google Gemini 2.5 Flash")
    choice = input("\nВыберите модель (1/2, по умолчанию 2): ").strip()
    
    model = "yandex" if choice == "1" else "gemini"
    
    try:
        bot = MultiModelMathTutor(model=model)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("Установите ключ в .env файл")
        exit(1)
    
    print("\n💡 Введите 'выход' для завершения\n")
    
    while True:
        user_input = input("\n🧮 Вопрос: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
        if user_input:
            print(f"\n{'='*60}")
            print(bot.ask(user_input))
            print(f"{'='*60}")