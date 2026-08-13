from dotenv import load_dotenv
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

load_from_url = WebBaseLoader('https://ru.wikipedia.org/wiki/%D0%92%D0%BE%D0%B9%D0%BD%D0%B0_%D0%B8_%D0%BC%D0%B8%D1%80')

docs = load_from_url.load()

prompt = ChatPromptTemplate.from_template('Сделай-ка краткое summery указанной веб-страницы из Википедии :).'
                                          ' Веб-страница: {context}')

# prompt = ChatPromptTemplate.from_template(
#     'Изучи текст веб-страницы Википедии: {context}\n\n'
#     'Сделай краткое, но емкое summery по следующим пунктам:\n'
#     '1. 📜 **Основная суть**: О чем статья в 2-3 предложениях.\n'
#     '2. 🔑 **Ключевые факты**: 3-5 самых важных тезисов или дат.\n'
#     '3. 📊 **Интересная деталь**: Один необычный или примечательный факт из текста.\n\n'
#     'Отвечай на русском языке, используй форматирование Markdown и эмодзи для наглядности.'
# )

chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# print(chain.input_schema.schema())
result = chain.invoke({'context': docs})

print(result)