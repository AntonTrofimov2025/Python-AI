import os  
from dotenv import load_dotenv 

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
# from langchain_core.embeddings import Embeddings

# class EmbeddingModel(Embeddings):

#     def __init__(
#         self,
#         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     ):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)

#         self.model.eval()


#     def _embed(self, text):

#         tokens = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         )

#         with torch.no_grad():
#             outputs = self.model(**tokens)

#         token_embeddings = outputs.last_hidden_state
#         attention_mask = tokens["attention_mask"]

#         # [batch, seq_len] -> [batch, seq_len, 1]
#         mask = attention_mask.unsqueeze(-1)

#         # Mean pooling
#         embeddings = (
#             token_embeddings * mask
#         ).sum(dim=1) / mask.sum(dim=1)

#         # [1, hidden_size] -> [hidden_size]
#         embedding = embeddings.squeeze(0)

#         return embedding.numpy().tolist()


#     def embed_documents(self, texts):
#         return [
#             self._embed(text)
#             for text in texts
#         ]


#     def embed_query(self, text):
#         return self._embed(text)


from pathlib import Path
file = Path().home().joinpath('Downloads', 'PD_Tech_JobSearch_17.pdf')

def read_pdf():
    loader = PyPDFLoader(file)
    pages = loader.load()
    return pages  




load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

result = read_pdf()
# print(*(line for line in result), sep='\n')
# embeddings = EmbeddingModel()

vector_store = InMemoryVectorStore.from_documents(result, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key))
# vector_store = InMemoryVectorStore.from_documents(result, embeddings)

docs = vector_store.similarity_search("langchain", k=2)


for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content}\n')