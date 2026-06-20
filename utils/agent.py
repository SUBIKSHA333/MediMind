from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_engine import load_vector_db, answer_question
from models.symptom_classifier import classify_symptoms
from nlp.nlp_pipeline import analyze_query

load_dotenv()

# LLM for the agent's reasoning
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Load vector DB once for the RAG tool
vector_db = load_vector_db()

# ---------- Define Tools ----------

def rag_tool_func(question):
    """Use this when the user asks a medical knowledge question that needs document context"""
    return answer_question(question, vector_db)

def symptom_tool_func(symptoms):
    """Use this when the user describes symptoms and wants a condition prediction"""
    label, score = classify_symptoms(symptoms)
    return f"Predicted condition: {label} (confidence: {score*100:.1f}%)"

def sentiment_tool_func(text):
    """Use this to analyze the emotional tone of a patient's message"""
    result = analyze_query(text)
    return f"Sentiment: {result['sentiment']}, Entities: {result['entities']}"

tools = [
    Tool(
        name="MedicalKnowledgeSearch",
        func=rag_tool_func,
        description="Useful for answering general medical questions using stored medical documents. Input should be a clear question."
    ),
    Tool(
        name="SymptomClassifier",
        func=symptom_tool_func,
        description="Useful when the user describes specific symptoms and wants to know the likely condition. Input should be the symptom description."
    ),
    Tool(
        name="SentimentAnalyzer",
        func=sentiment_tool_func,
        description="Useful for understanding how worried or calm the patient sounds. Input should be the patient's message."
    ),
]

# ---------- Agent Prompt ----------
prompt = PromptTemplate.from_template("""
You are MediMind, a helpful medical AI assistant. Answer the user's question as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
# ---------- Build the Agent ----------
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=4
)

def ask_agent(question):
    """Main entry point - the agent decides which tool(s) to use"""
    result = agent_executor.invoke({"input": question})
    return result["output"]

# Test it
if __name__ == "__main__":
    print("\n" + "="*50)
    answer = ask_agent("I have severe headache and blurred vision, what could it be?")
    print(f"\n🤖 Final Answer: {answer}")