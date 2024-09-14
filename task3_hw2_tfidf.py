import pandas as pd
from lightgbm import LGBMClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score

train_df = pd.read_csv("./data/train_set.csv", sep="\t", nrows=20000)

tfdif = TfidfVectorizer(ngram_range=(1,3), max_features=5000, sublinear_tf=True)
train_cnt = tfdif.fit_transform(train_df['text'])


# clf = RandomForestClassifier() # 0.8297925143002833
# clf = LGBMClassifier(n_estimators=200, num_leaves=64, learning_rate=0.01, max_depth=4, verbosity=-1)
# clf = BernoulliNB() # 0.6660894255683578
# clf = GaussianNB() # error
# clf = KNeighborsClassifier() # timeout
clf = LinearSVC(multi_class="ovr") # 0.9172201740621652
clf.fit(train_cnt[:15000], train_df['label'].values[:15000])

pred = clf.predict(train_cnt[15000:])
print(f1_score(train_df['label'].values[15000:], pred, average='macro'))
