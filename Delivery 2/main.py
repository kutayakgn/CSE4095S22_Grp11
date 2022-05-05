import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2

stop_words = set(stopwords.words('turkish'))
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
df = pd.read_csv("modified.csv", encoding = "latin-1")
df.columns = ['v1', 'v2']


category_id_df = df[['v2', 'v1']]
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['v1', 'v2']].values)

fig = plt.figure(figsize=(8,6))
df.groupby('v1').v2.count().plot.bar(ylim=0)
plt.show()
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stop_words)
features = tfidf.fit_transform(df.v2).toarray()
labels = df.v1
print(features.shape)

#i used the chi-squared test to find the terms that are the most correlated with each of the categories:
# N = 2
# for v1, v1 in sorted(category_to_id.items()):
#   features_chi2 = chi2(features, labels == v1)
#   indices = np.argsort(features_chi2[0])
#   feature_names = np.array(tfidf.get_feature_names_out())[indices]
#   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#   print("# '{}':".format(v1))
#   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.20, random_state=20)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    #LogisticRegression(random_state=0,solver='lbfgs', max_iter=100),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
print(cv_df.groupby('model_name').accuracy.mean())

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['v1'].unique()))