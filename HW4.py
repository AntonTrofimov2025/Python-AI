import os
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, \
    stop_after_delay
import logging
import requests

logging.basicConfig(level=logging.INFO)

# For future. Is not used, for now.
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)

# Инициализация локальной модели эмбеддингов
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Памятка:
#
# multiplier=1 — базовый коэффициент умножения.
# min=2 — первая пауза после сбоя будет 2 секунды.
# max=10 — максимальное время ожидания между попытками не превысит 10 секунд
# stop_after_attempt(3) - заставит функцию перезапуститься максимум 3 раза, если внутри произойдет ЛЮБАЯ ошибка
# (stop_after_attempt(3) | stop_after_delay(30)) - Пробуем до 3 раз ИЛИ пока суммарно не пройдет 30 секунд

# С retry_if_exception_type мы перезапускаем функцию ТОЛЬКО если упала сетевая ошибка
# (например, ConnectionError или Timeout) от библиотеки requests.

@retry(stop=(stop_after_attempt(3) | stop_after_delay(30)), wait=wait_exponential(multiplier=1, min=2, max=10),
       before_sleep=before_sleep_log(logging.getLogger(), logging.INFO),
       retry=retry_if_exception_type((requests.exceptions.ConnectTimeout,
       requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError)))
def get_embedding(text):
    tokens = tokenizer(
        text,
        return_tensors="pt"
    )
    attention_mask = tokens["attention_mask"]
    with torch.no_grad():
        outputs = model(**tokens)

    token_embeddings = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1)
    sentence_embedding = (
                                 token_embeddings * mask
                         ).sum(dim=1) / mask.sum(dim=1)
    return sentence_embedding


# Получение embedding для заданных текстовых значений
vector_1 = get_embedding("Я люблю программирование.")
vector_2 = get_embedding("Кодинг – это моё хобби.")
print(vector_1.shape)
print(vector_2.shape)

# Вывод полученных векторов с поясняющими сообщениями
# print("Я люблю программирование:", vector_1)
# print("Кодинг – это моё хобби:", vector_2)


# ---  Секция семантического поиска  ---

# 1. Создание набора текстов для поиска
texts_to_index = [
    "Кошка сидит на окне",
    "Собака играет в парке",
    "Ананас растет в тропиках",
    "Кот спит на диване",
    "Пес лает на почтальона",
    "Фрукт ананас очень вкусный",
    "Домашняя кошка любит ласку",
    "Верный пес охраняет дом",
    "Спелый ананас полон витаминов",
    "Кошки любят рыбу и мясо",
    "Основной рацион кошек - это белок",
    "Чем кормить котенка?",
    "Лучший корм для кошек - сбалансированный",
    "Коты едят сухой и влажный корм",
    "Нельзя кормить кошку шоколадом",
    "Молоко не всегда полезно для кошек",
    "Я обожаю программировать.",
    "Программирование – это то, что мне очень нравится.",
    "Меня увлекает разработка программного обеспечения.",
    "Я испытываю страсть к написанию кода.",
    "Программирование приносит мне огромное удовольствие.",
    "Мне интересно заниматься программированием.",
    "Моё хобби – это кодинг.",
    "В свободное время я занимаюсь программированием.",
    "Кодинг – это моё любимое увлечение.",
    "Я увлекаюсь кодингом на досуге.",
    "Программирование – это моё хобби и страсть.",
    "Когда есть свободное время, я кодирую.",
    "Кодинг - это моё хобби, которым я наслаждаюсь."
]

# 2. Получение embedding для каждого текста и сохранение их в списке
embeddings_list = [get_embedding(text).detach().cpu().numpy() for text in texts_to_index]
embeddings_array = np.vstack(embeddings_list).astype('float32')    # Преобразуем в numpy array для FAISS

# 3. Создание FAISS индекса
dimension = embeddings_array.shape[1]   # Размерность embedding
index = faiss.IndexFlatL2(dimension)    # Используем IndexFlatL2 для L2 дистанции (евклидова)
# noinspection PyArgumentList
index.add(embeddings_array)             # Добавляем embeddings в индекс


# 4. Функция для выполнения семантического поиска
def semantic_search(query, index, texts, k=2):
    """
    Выполняет семантический поиск по индексу FAISS.

    :param query: Поисковый запрос (строка).
    :param index: FAISS индекс.
    :param texts: Список текстов, которые были проиндексированы.
    :param k: Количество ближайших соседей для поиска (по умолчанию 2).
    :return: Список из k наиболее релевантных текстов.
    """
    query_embedding = get_embedding(query).reshape(1, -1)   # Получаем embedding для запроса и меняем размерность
    D, I = index.search(query_embedding, k)                 # Ищем k ближайших соседа
    results = [texts[i] for i in I[0]]                      # Получаем тексты по индексам
    return results


# 5. Пример использования семантического поиска
search_query = "Питание кошек"
search_results = semantic_search(search_query, index, texts_to_index, k=4)

print("\n--- Результаты семантического поиска ---")
print(f"Запрос: '{search_query}'")
print("Найденные соответствия:")
for result in search_results:
    print(f"- {result}")
