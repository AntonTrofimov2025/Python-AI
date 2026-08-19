from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv

import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)

def get_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs
    # return WebBaseLoader(url).load()



prompt = ChatPromptTemplate.from_template("Напишите краткое изложение следующего текста: {context}")


chain = create_stuff_documents_chain(llm, prompt)


user_url = input('Введите адрес веб-страницы для парсинга: ')
result = chain.invoke({"context": get_docs(user_url)})


print(result)