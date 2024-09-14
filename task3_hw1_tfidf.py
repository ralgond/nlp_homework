import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv("./data/train_set.csv", sep="\t", nrows=20000)

tfdif = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
# ngram_range=(1,3)-origin: 0.882072720898739
# ngram_range=(1,3)-max_features=5000: 0.8946843376608505
# ngram_range=(1,4)-max_features=5000: 0.8929799496893134
# ngram_range=(1,2)-max_features=5000: 0.8925488624508778
train_cnt = tfdif.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_cnt[:15000], train_df['label'].values[:15000])

pred = clf.predict(train_cnt[15000:])
print(f1_score(train_df['label'].values[15000:], pred, average='macro'))
