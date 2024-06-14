
# from fastapi import FastAPI, UploadFile, File
# from typing import List
# import uvicorn
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from pydantic import BaseModel
# from dotenv import load_dotenv

# load_dotenv()
# genai_api_key = os.getenv("GOOGLE_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# app = FastAPI()

# class QuestionRequest(BaseModel):
#     question: str

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf.file)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(api_key=genai_api_key, model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def load_vector_store():
#     embeddings = GoogleGenerativeAIEmbeddings(api_key=genai_api_key, model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     return vector_store

# def get_retrieval_chain():
#     llm = ChatGoogleGenerativeAI(api_key=genai_api_key, model="gemini-pro", temperature=1)
#     prompt = ChatPromptTemplate.from_template("""
#     Answer the following question based only on the provided context. 
#     Think step by step before providing a detailed answer. 
#     I will tip you $1000 if the user finds the answer helpful. 
#     <context>
#     {context}
#     </context>
#     Question: {input}""")
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     vector_store = load_vector_store()
#     retriever = vector_store.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     return retrieval_chain

# @app.post("/process-pdf/")
# async def process_pdf(files: List[UploadFile] = File(...)):
#     pdf_docs = files
#     raw_text = get_pdf_text(pdf_docs)
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)
#     return {"message": "PDF processed and vector store created"}

# @app.post("/ask-question/")
# async def ask_question(payload: QuestionRequest):
#     question = payload.question
#     retrieval_chain = get_retrieval_chain()
#     response = retrieval_chain.invoke({"input": question})
#     return {"answer": response['answer']}

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)

#second app
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uvicorn
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")
            raise HTTPException(status_code=400, detail="Failed to read one or more PDF files.")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(api_key=genai_api_key, model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logging.info("Vector store created and saved locally.")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vector store.")

def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(api_key=genai_api_key, model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        raise HTTPException(status_code=500, detail="Failed to load vector store.")

def get_retrieval_chain():
    try:
        llm = ChatGoogleGenerativeAI(api_key=genai_api_key, model="gemini-pro", temperature=1)
        prompt = ChatPromptTemplate.from_template("""                                        
        Answer the following question based only on the provided context friendly and conversational tone. 
        Think step by step before providing a detailed answer. 
        I will tip you $1000 if the user finds the answer helpful. 
        please do not print the steps that you are taking just give me the answer .
        <context>
        {context}
        </context>
        Question: {input}
                                                  """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain
    except Exception as e:
        logging.error(f"Error creating retrieval chain: {e}")
        raise HTTPException(status_code=500, detail="Failed to create retrieval chain.")

@app.post("/process-pdf/")
async def process_pdf(files: List[UploadFile] = File(...)):
    try:
        pdf_docs = files
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return {"message": "PDF processed and vector store created"}
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF files.")

@app.post("/ask-question/")
async def ask_question(payload: QuestionRequest):
    try:
        question = payload.question
        retrieval_chain = get_retrieval_chain()
        response = retrieval_chain.invoke({"input": question})
        return {"answer": response['answer']}
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail="Failed to get an answer to the question.")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


#next  code down

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from typing import List
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import uvicorn

# load_dotenv()
# app = FastAPI()

# class QuestionRequest(BaseModel):
#     question: str

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf.file)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def create_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# @app.post("/process-pdf/")
# async def process_pdf(files: List[UploadFile] = File(...)):
#     pdf_docs = files
#     raw_text = get_pdf_text(pdf_docs)
#     text_chunks = get_text_chunks(raw_text)
#     create_vector_store(text_chunks)
#     return {"message": "PDF processed and vector store created"}

# def load_vector_store():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     return vector_store

# def retrieve_relevant_docs(question):
#     vector_store = load_vector_store()
#     docs = vector_store.similarity_search(question)
#     return docs

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     If the answer is not in the provided context, say "The answer is not available in the context".
#     Context:
#     {context}
#     Question: {question}
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# @app.post("/ask-question/")
# async def ask_question(payload: QuestionRequest):
#     question = payload.question
#     docs = retrieve_relevant_docs(question)
#     if not docs:
#         raise HTTPException(status_code=404, detail="No relevant documents found")

#     context = "\n\n".join([doc.page_content for doc in docs])
#     chain = get_conversational_chain()
#     response = chain({"context": context, "question": question, "input_documents": docs}, return_only_outputs=True)
#     return {"answer": response["output_text"]}

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)
