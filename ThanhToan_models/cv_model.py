import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def run_cv():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

 
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training CV model (CIFAR-10)...")
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("\nCV Test accuracy: {:.2f}%".format(test_acc * 100))


    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\nClassification Report (CV):")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix (CV):")
    print(confusion_matrix(y_test, y_pred_labels))

if __name__ == '__main__':
    run_cv()
