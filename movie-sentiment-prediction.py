# 11/12/2020
# source: https://machinelearningmastery.com/
# predict-sentiment-movie-reviews-using-deep-learning/

# imports
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence

TOP_WORDS = 5000
MAX_WORDS = 500
#load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=TOP_WORDS)

X_train = sequence.pad_sequences(X_train, maxlen = MAX_WORDS)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_WORDS)

# create model
model = Sequential()
model.add(Embedding(TOP_WORDS, 32, input_length=MAX_WORDS))
mode.add(Flatten())
model.add(Dense(250, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.Summary()

# Fit
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch=128, verbose=2)

# eval
scores = model.evaluate(X_test, y_test, verbose=0)
print("accuracy: %.2f%%"%(scores[1]*100))
