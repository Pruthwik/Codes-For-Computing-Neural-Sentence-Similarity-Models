import torch
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F


tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()


def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


with torch.no_grad():
    english_sentences = [
    "dog",
    "Puppies are nice.",
    "I enjoy taking long walks along the beach with my dog.",
    ]
    english_inputs = tokenizer(english_sentences, return_tensors="pt", padding=True)
    english_outputs = model(**english_inputs)
    english_embeddings = english_outputs.pooler_output
    italian_sentences = [
    "cane",
    "I cuccioli sono carini.",
    "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.",
    ]
    japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]
    italian_inputs = tokenizer(italian_sentences, return_tensors="pt", padding=True)
    japanese_inputs = tokenizer(japanese_sentences, return_tensors="pt", padding=True)
    italian_outputs = model(**italian_inputs)
    japanese_outputs = model(**japanese_inputs)
    italian_embeddings = italian_outputs.pooler_output
    japanese_embeddings = japanese_outputs.pooler_output
    print(torch.diagonal(similarity(english_embeddings, italian_embeddings)))
    print(torch.diagonal(similarity(english_embeddings, japanese_embeddings)))
    print(torch.diagonal(similarity(italian_embeddings, japanese_embeddings)))
