from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader
import os

# Get the directory of the current script (__file__ is the path to the current script)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the fas_info.md file
# Assuming fas_info.md is in the root directory of the project
fas_info_path = os.path.join(script_dir, 'fas_info.md')

loader = UnstructuredMarkdownLoader(fas_info_path)
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(documents=all_splits, 
                                    collection_name="rag-chroma",
                                    embedding=OpenAIEmbeddings(),
                                    )
retriever = vectorstore.as_retriever()

# RAG prompt
template = """You are a friendly concierge at Norwegian Cruise Line that is incredibly knowlegeable about Free At Sea (FAS).

You always try to give guests as much information as possible related to their question. 

Answer the question based only on the following context, don't make anything up:
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