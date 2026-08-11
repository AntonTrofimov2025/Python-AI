import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


model_name = "distilbert-base-multilingual-cased"


tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModel.from_pretrained(model_name)


model.eval()


text1 = 'Я изучаю искусственный интеллект.'
text2 = 'cat'
text3 = 'coffee'

def get_tokens(string):
    tokens = tokenizer(
        string,
        return_tensors="pt"
    )
    return tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]), tokens

output, tokens = get_tokens(text1)
print(output)
print(tokens)
# print("Tokens:")
# print(get_tokens(text1))

# tokens1 = get_tokens(text1)[1]
# tokens2 = get_tokens(text2)[1]
# tokens3 = get_tokens(text3)[1]


# def get_embedding(tokens):
#     with torch.no_grad():
#         outputs = model(**tokens)

#     embeddings = outputs.last_hidden_state
#     return embeddings

# def get_sentence_embedding(tokens):
#     attention_mask = tokens["attention_mask"]

#     with torch.no_grad():
#         outputs = model(**tokens)

#     token_embeddings = outputs.last_hidden_state

#     mask = attention_mask.unsqueeze(-1)

#     sentence_embedding = (
#         token_embeddings * mask
#     ).sum(dim=1) / mask.sum(dim=1)
#     return sentence_embedding

# # print("\nEmbeddings shape:")
# emb1 = get_embedding(tokens1)[0, 1]
# emb2 = get_embedding(tokens2)[0, 1]
# emb3 = get_embedding(tokens3)[0, 1]

# def get_similarity(vector1, vector2):
#     return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

# cos12 = get_similarity(emb1, emb2)
# cos13 = get_similarity(emb1, emb3)
# print(cos12)
# print(cos13)

# sentence_emb1 = get_sentence_embedding(tokens1)
# print(sentence_emb1)