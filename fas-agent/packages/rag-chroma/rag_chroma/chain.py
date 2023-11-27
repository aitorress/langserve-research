from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader

# Load
from langchain.document_loaders import WebBaseLoader
import os

print("Current Working Directory:", os.getcwd())

loader = UnstructuredMarkdownLoader("fas_info.md")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(documents=all_splits, 
                                    collection_name="rag-chroma-fas",
                                    embedding=OpenAIEmbeddings(),
                                    )
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)