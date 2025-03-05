import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def run_nlp():
    max_features = 10000  
    maxlen = 200          

    
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test  = pad_sequences(x_test, maxlen=maxlen)


    model = tf.keras.Sequential([
        layers.Embedding(max_features, 32, input_length=maxlen),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\nTraining NLP model (IMDB)...")
    model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.2, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("\nNLP Test accuracy: {:.2f}%".format(test_acc * 100))

    y_pred = model.predict(x_test)
    y_pred_labels = (y_pred > 0.5).astype("int32").flatten()

    print("\nClassification Report (NLP):")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix (NLP):")
    print(confusion_matrix(y_test, y_pred_labels))

if __name__ == '__main__':
    run_nlp()
