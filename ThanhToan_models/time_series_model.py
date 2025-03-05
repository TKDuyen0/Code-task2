import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

def generate_time_series_data(n_samples=2000, timesteps=100, noise_amp=1.5):
    X = np.zeros((n_samples, timesteps, 1))
    y = np.zeros((n_samples,))
    for i in range(n_samples):
        freq = np.random.choice([1, 2])
        y[i] = 0 if freq == 1 else 1
        x = np.linspace(0, 2 * np.pi, timesteps)

        X[i, :, 0] = np.sin(freq * x) + noise_amp * np.random.randn(timesteps)
    return X, y

def run_time_series():
    X, y = generate_time_series_data(n_samples=2000, timesteps=100, noise_amp=1.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(100, 1)),
        layers.Dense(2, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nTraining Modified Time Series model (Target ~80% Accuracy)...")
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=2)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\nModified Time Series Test accuracy: {:.2f}%".format(test_acc * 100))
    
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype("int32").flatten()
    
    print("\nClassification Report (Modified Time Series):")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix (Modified Time Series):")
    print(confusion_matrix(y_test, y_pred_labels))
    
if __name__ == '__main__':
    run_time_series()
