import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def run_nlp():
    # Đường dẫn đến file CSV đã tải về trên máy tính của bạn.
    dataset_path = r"C:\Download\imdb-dataset-of-50k-movie-reviews\IMDB Dataset.csv"
    
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(dataset_path)
    
    # Chuyển đổi nhãn: 'positive' thành 1 và 'negative' thành 0
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Lấy dữ liệu đánh giá và nhãn
    reviews = df['review'].values
    labels = df['sentiment'].values
    
    # Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
    x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
    
    # Thiết lập tokenizer với số từ tối đa là 10.000
    max_features = 10000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    
    # Chuyển đổi văn bản thành chuỗi số (sequences)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    
    # Padding các sequences về độ dài cố định (200 từ)
    maxlen = 200
    x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen)
    x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen)
    
    # Xây dựng mô hình Neural Network
    model = tf.keras.Sequential([
        layers.Embedding(max_features, 32, input_length=maxlen),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nTraining NLP model (Local IMDB Dataset)...")
    # Huấn luyện mô hình với validation split 20%
    history = model.fit(x_train_pad, y_train, epochs=5, batch_size=512, validation_split=0.2, verbose=2)
    
    # Vẽ đồ thị hiển thị độ chính xác
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], marker='o', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Đánh giá mô hình trên tập test
    test_loss, test_acc = model.evaluate(x_test_pad, y_test, verbose=0)
    print("\nNLP Test accuracy: {:.2f}%".format(test_acc * 100))
    
    # Dự đoán và in báo cáo phân loại
    y_pred = model.predict(x_test_pad)
    y_pred_labels = (y_pred > 0.5).astype("int32").flatten()
    
    print("\nClassification Report (NLP):")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix (NLP):")
    print(confusion_matrix(y_test, y_pred_labels))

if __name__ == '__main__':
    run_nlp()
