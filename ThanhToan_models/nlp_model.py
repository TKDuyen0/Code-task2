import kagglehub  # Chỉ cần dùng nếu cần tải dataset trong môi trường chưa có sẵn
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def run_nlp():
    # Nếu bạn chạy trên Kaggle Kernel, dataset đã được mount sẵn nên có thể dùng đường dẫn trực tiếp.
    # Nếu không, uncomment dòng dưới để tải dataset qua kagglehub.
    # path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    dataset_path = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(dataset_path)
    
    # Chuyển đổi nhãn: 'positive' thành 1 và 'negative' thành 0
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Lấy cột review và sentiment
    reviews = df['review'].values
    labels = df['sentiment'].values
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
    
    # Thiết lập tokenizer với số từ tối đa là 10.000
    max_features = 10000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    
    # Chuyển đổi các review thành sequences (chuỗi số)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    
    # Padding để các sequences có độ dài cố định (ở đây là 200 từ)
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
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nTraining NLP model (Kaggle IMDB Dataset)...")
    model.fit(x_train_pad, y_train, epochs=5, batch_size=512, validation_split=0.2, verbose=2)
    
    test_loss, test_acc = model.evaluate(x_test_pad, y_test, verbose=0)
    print("\nNLP Test accuracy: {:.2f}%".format(test_acc * 100))
    
    # Dự đoán trên tập test và in báo cáo phân loại
    y_pred = model.predict(x_test_pad)
    y_pred_labels = (y_pred > 0.5).astype("int32").flatten()
    
    print("\nClassification Report (NLP):")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix (NLP):")
    print(confusion_matrix(y_test, y_pred_labels))

if __name__ == '__main__':
    run_nlp()
