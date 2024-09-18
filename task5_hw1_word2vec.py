# 计算出词向量
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

seed = 42
random.seed(seed)
np.random.seed(seed)

train_df = pd.read_csv("./data/train_set.csv", sep='\t')

sentences = []
for sentence in train_df['text']:
    sentences.append(sentence.split())

num_features = 200     # Word vector dimensionality
num_workers = 8       # Number of threads to run in parallel

model = Word2Vec(sentences, workers=num_workers, vector_size=num_features)

model.save("./tmp/word2vec.bin")

# load model
model = Word2Vec.load("./tmp/word2vec.bin")

# convert format
model.wv.save_word2vec_format('./tmp/word2vec.txt', binary=False)