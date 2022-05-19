import numpy as n
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras
import keras.backend as K
from keras.layers import *
from keras.losses import *
from keras.models import *
from keras.callbacks import *
from keras.activations import *
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train.csv", encoding="latin-1")
test = pd.read_csv("test.csv", encoding="latin-1")

print(train.head())


def get_preprocessing_func():
    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    def preprocessing_func(sent):
        return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(sent)]

    return preprocessing_func


X = train['v2'].apply(get_preprocessing_func()).values
y = train['v1'].values
X_test = test['v2'].apply(get_preprocessing_func()).values


def prepare_tokenizer_and_weights(X):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(X)

    weights = np.zeros((len(tokenizer.word_index) + 1, 300))
    with open("cc.tr.300.vec", encoding="utf-8") as f:
        next(f)
        for l in f:
            w = l.split(' ')
            if w[0] in tokenizer.word_index:
                weights[tokenizer.word_index[w[0]]] = np.array([float(x) for x in w[1:301]])
    return tokenizer, weights


tokenizer, weights = prepare_tokenizer_and_weights(np.append(X, X_test))
X_seq = tokenizer.texts_to_sequences(X)
MAX_LEN = max(map(lambda x: len(x), X_seq))
X_seq = pad_sequences(X_seq, MAX_LEN)
MAX_ID = len(tokenizer.word_index)
print('MAX_LEN=', MAX_LEN)
print('MAX_ID=', MAX_ID)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)


def make_fast_text():
    fast_text = Sequential()
    fast_text.add(InputLayer((MAX_LEN,)))
    fast_text.add(Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True))
    fast_text.add(SpatialDropout1D(0.5))
    fast_text.add(GlobalMaxPooling1D())
    fast_text.add(Dropout(0.5))
    fast_text.add(Dense(32,activation='softmax'))
    return fast_text

fast_texts = [make_fast_text() for i in range(3)]
fast_texts[0].summary()

for fast_text in fast_texts:
    X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.2)
    fast_text.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    fast_text.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
                 epochs=10,
                 verbose=2)

# def make_model_lstm():
#     model_lstm = Sequential()
#     model_lstm.add(InputLayer((MAX_LEN,)))
#     model_lstm.add(Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True))
#     model_lstm.add(SpatialDropout1D(0.5))
#     model_lstm.add(Bidirectional(LSTM(300, return_sequences=True)))
#     model_lstm.add(BatchNormalization())
#     model_lstm.add(SpatialDropout1D(0.5))
#     model_lstm.add(Bidirectional(LSTM(300)))
#     model_lstm.add(BatchNormalization())
#     model_lstm.add(Dropout(0.5))
#     model_lstm.add(Dense(32,activation='softmax'))
#     return model_lstm
#
# model_lstms = [make_model_lstm() for i in range(1)]
# model_lstms[0].summary()
#
# for model_lstm in model_lstms:
#     X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.2)
#     model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     model_lstm.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
#                  callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=0)],
#                  epochs=1,
#                  verbose=2)

# def make_model_cnn():
#     inputs = Input((MAX_LEN,))
#     x = Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[weights], trainable=True)(inputs)
#     x = SpatialDropout1D(0.5)(x)
#     x = Conv1D(300, kernel_size=5,activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = SpatialDropout1D(0.5)(x)
#     x = MaxPooling1D(pool_size=2, strides=2)(x)
#     x = Conv1D(300, kernel_size=5,activation='relu')(x)
#     x = GlobalMaxPooling1D()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     outputs = Dense(32,activation='softmax')(x)
#     model_cnn = Model(inputs, outputs)
#     return model_cnn
#
# model_cnns = [make_model_cnn() for i in range(3)]
# model_cnns[0].summary()
#
# for model_cnn in model_cnns:
#     X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(X_seq, y, test_size=0.1)
#     model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     model_cnn.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
#                  callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
#                  epochs=5,
#                  verbose=2)

def make_model_bagged(models):
    inputs = Input((MAX_LEN,))
    outputs = average([model(inputs) for model in models])
    return Model(inputs, outputs)
model_bagged = make_model_bagged(fast_texts)
model_bagged.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
y_prob = model_bagged.predict(X_seq)
y_predict = np.argmax(y_prob, axis=1)
print(classification_report(y, y_predict))
sns.heatmap(confusion_matrix(y, y_predict));
plt.show()