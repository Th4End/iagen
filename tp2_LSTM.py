import pandas as pd
import numpy as np
import string
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

df = pd.read_excel("grimms_tale.xlsx")

def nettoyer_texte(texte):
    if isinstance(texte, str):
        texte = texte.lower().translate(str.maketrans("", "", string.punctuation))
        texte = " ".join(texte.split())
    return texte

df["Story"] = df["Story"].apply(nettoyer_texte)

VOCAB_SIZE = 10000
MAX_LENGTH = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df["Story"].dropna())

sequences = tokenizer.texts_to_sequences(df["Story"].dropna())

input_sequences = []
for seq in sequences:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i+1])

input_sequences = pad_sequences(input_sequences, maxlen=MAX_LENGTH, padding='pre', dtype=np.uint16)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=VOCAB_SIZE)

lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9)

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LENGTH-1),
    Bidirectional(LSTM(128, return_sequences=True)),  
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(VOCAB_SIZE, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

model.fit(X, y, epochs=15, batch_size=64, verbose=1, callbacks=[early_stopping])

model.save("lstm_text_gen_optim_v2.h5")

def generer_phrase_lstm(longueur_phrase=20, temperature=0.8):
    start_text = random.choice(df["Story"].dropna().tolist())
    words = start_text.split()[:5]

    for _ in range(longueur_phrase):
        sequence = tokenizer.texts_to_sequences([" ".join(words)])
        sequence = pad_sequences(sequence, maxlen=MAX_LENGTH-1, padding='pre', dtype=np.uint16)

        predictions = model.predict(sequence)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        next_word_index = np.random.choice(range(VOCAB_SIZE), p=predictions)
        next_word = tokenizer.index_word.get(next_word_index, "")

        if not next_word:
            break

        words.append(next_word)

    return " ".join(words)

phrase_generee = generer_phrase_lstm(20, temperature=0.8)
print(phrase_generee)
