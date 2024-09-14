import pandas as pd
from lightgbm import LGBMClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv("./data/train_set.csv", sep="\t", nrows=20000)

tfdif = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
train_cnt = tfdif.fit_transform(train_df['text'])

# clf = RandomForestClassifier() # 0.8098234790068418
# clf = RandomForestClassifier() # 0.8297925143002833
clf = LGBMClassifier(learning_rate=0.01, max_depth=4, verbosity=-1)
clf.fit(train_cnt[:15000], train_df['label'].values[:15000])

pred = clf.predict(train_cnt[15000:])
print(f1_score(train_df['label'].values[15000:], pred, average='macro'))
