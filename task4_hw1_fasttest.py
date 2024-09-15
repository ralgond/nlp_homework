import pandas as pd
from sklearn.metrics import f1_score
import fasttext

# 转换为FastText需要的格式
train_df = pd.read_csv("./data/train_set.csv", sep="\t", nrows=50000)
train_df['label_ft'] = "__label__" + train_df['label'].astype(str)
train_df[['text', 'label_ft']].iloc[:-5000].to_csv("train.csv", index=None, header=None, sep="\t")

# lr=0.1: 0.7518090546084483
# lr=1.0: 0.8427563275710632
# lr=1.0-dim=80: 0.844064150530288
# lr=1.0-dim=80-loss=softmax: 0.876463710502484
# lr=1.0-dim=80-loss=softmax-minCount=2: 0.8784109019957421
# nrows=50000-lr=1.0-dim=80-loss=softmax-minCount=2: 0.9175976346881222
model = fasttext.train_supervised("train.csv", lr=1.0, wordNgrams=2,
                                  verbose=2, minCount=2, epoch=40, dim=80, loss="softmax")
val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))