import os
import faiss
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


model_name = "distilbert-base-multilingual-cased"


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModel.from_pretrained(model_name)


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
    return sentence_embedding.squeeze(0).numpy()



vector_1 = get_embedding("Я люблю программирование.")
print(vector_1.shape)
# vector_2 = get_embedding("Кодинг – это моё хобби.")

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


embeddings_list = [get_embedding(text) for text in texts_to_index]
embeddings_array = np.array(embeddings_list)    


dimension = embeddings_array.shape[1]   
index = faiss.IndexFlatL2(dimension)    
index.add(embeddings_array)             
# print('Index shape:\n')
# print(embeddings_array.shape)


def semantic_search(query, index, texts, k=2):
    """
    Выполняет семантический поиск по индексу FAISS.

    :param query: Поисковый запрос (строка).
    :param index: FAISS индекс.
    :param texts: Список текстов, которые были проиндексированы.
    :param k: Количество ближайших соседей для поиска (по умолчанию 2).
    :return: Список из k наиболее релевантных текстов.
    """
    query_embedding = get_embedding(query).reshape(1, -1)   
    D, I = index.search(query_embedding, k)                 
    results = [texts[i] for i in I[0]]                      
    return results



search_query = "Питание кошек"
search_results = semantic_search(search_query, index, texts_to_index, k=4)

print("\n--- Результаты семантического поиска ---")
print(f"Запрос: '{search_query}'")
print("Найденные соответствия:")
for result in search_results:
    print(f"- {result}")