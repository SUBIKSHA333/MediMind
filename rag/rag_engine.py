from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.llm import ask_llm
import os

# Step 1 — Load and split documents
def load_documents(file_path):
    """Load a PDF or text file"""
    print(f"📄 Loading document: {file_path}")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# Step 2 — Create vector database
def create_vector_db(chunks):
    """Convert chunks to vectors and store in FAISS"""
    print("🔄 Creating embeddings...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("rag/vector_store")
    print("✅ Vector database created and saved!")
    return vector_db

# Step 3 — Load existing vector database
def load_vector_db():
    """Load saved vector database"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.load_local(
        "rag/vector_store", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_db

# Step 4 — Answer questions using RAG
def answer_question(question, vector_db):
    """Find relevant chunks and answer question using LLM"""
    print(f"\n❓ Question: {question}")
    
    # Find relevant document chunks
    relevant_chunks = vector_db.similarity_search(question, k=3)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Build prompt with context
    prompt = f"""Based on the following medical information, answer the question.

Context:
{context}

Question: {question}

Give a clear, helpful answer based on the context above."""
    
    answer = ask_llm(prompt)
    print(f"🤖 Answer: {answer}")
    return answer

# Test it
if __name__ == "__main__":
    # Create a sample medical text file for testing
    sample_text = """
    Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin.
    Symptoms include increased thirst, frequent urination, fatigue, and blurred vision.
    Type 1 diabetes requires insulin therapy. Type 2 diabetes can be managed with diet and exercise.
    
    Hypertension or high blood pressure is when the force of blood against artery walls is too high.
    Symptoms include headaches, shortness of breath, and nosebleeds.
    Treatment includes lifestyle changes and medications like ACE inhibitors.
    
    Asthma is a condition where airways narrow and swell producing extra mucus.
    Symptoms include shortness of breath, chest tightness, wheezing and coughing.
    Treatment includes inhalers and avoiding triggers.
    """
    
    # Save sample text
    with open("rag/sample_medical.txt", "w") as f:
        f.write(sample_text)
    
    # Run the RAG pipeline
    chunks = load_documents("rag/sample_medical.txt")
    vector_db = create_vector_db(chunks)
    answer_question("What are the symptoms of diabetes?", vector_db)
    answer_question("How is asthma treated?", vector_db)