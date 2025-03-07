import tensorflow as tf
from tensorflow.keras import layers, models  # Import các lớp cần thiết từ Keras của TensorFlow
from sklearn.metrics import classification_report, confusion_matrix  # Dùng để đánh giá mô hình
from sklearn.model_selection import train_test_split  # Dùng để chia tách dữ liệu thành tập huấn luyện và kiểm tra
import numpy as np  # Thư viện hỗ trợ tính toán với mảng số
import matplotlib.pyplot as plt  # Dùng để vẽ biểu đồ
import seaborn as sns  # Thư viện hiển thị biểu đồ heatmap

def generate_time_series_data(n_samples=2000, timesteps=100, noise_amp=1.5):
    """
    Tạo dữ liệu chuỗi thời gian với nhiễu ngẫu nhiên.
    
    Args:
        n_samples (int): Số lượng mẫu dữ liệu cần tạo.
        timesteps (int): Số bước thời gian cho mỗi mẫu.
        noise_amp (float): Hệ số khuếch đại của nhiễu Gauss.
        
    Returns:
        X (np.ndarray): Dữ liệu chuỗi thời gian có kích thước (n_samples, timesteps, 1).
        y (np.ndarray): Nhãn của dữ liệu, 0 hoặc 1, tùy thuộc vào tần số của hàm sin.
    """
    # Khởi tạo mảng dữ liệu và nhãn với các giá trị ban đầu là 0
    X = np.zeros((n_samples, timesteps, 1))
    y = np.zeros((n_samples,))
    
    # Lặp qua từng mẫu để sinh dữ liệu
    for i in range(n_samples):
        # Chọn ngẫu nhiên tần số 1 hoặc 2
        freq = np.random.choice([1, 2])
        # Gán nhãn: nếu tần số là 1 thì nhãn 0, nếu là 2 thì nhãn 1
        y[i] = 0 if freq == 1 else 1
        
        # Tạo vector x với các giá trị phân bố đều từ 0 đến 2π
        x = np.linspace(0, 2 * np.pi, timesteps)
        
        # Tính giá trị hàm sin nhân với tần số và cộng thêm nhiễu Gauss
        X[i, :, 0] = np.sin(freq * x) + noise_amp * np.random.randn(timesteps)
    
    return X, y

def run_time_series():
    """
    Sinh dữ liệu, xây dựng, huấn luyện, đánh giá mô hình và hiển thị ma trận nhầm lẫn.
    """
    # Sinh dữ liệu chuỗi thời gian
    X, y = generate_time_series_data(n_samples=2000, timesteps=100, noise_amp=1.5)
    
    # Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Xây dựng mô hình mạng nơ-ron đơn giản
    model = tf.keras.Sequential([
        # Lớp Flatten chuyển đổi dữ liệu đầu vào (100, 1) thành vector 1 chiều (100,)
        layers.Flatten(input_shape=(100, 1)),
        # Lớp Dense với 2 neuron và hàm kích hoạt ReLU để học các đặc trưng phi tuyến
        layers.Dense(2, activation='relu'),
        # Lớp Dropout với tỷ lệ 0.5 để giảm hiện tượng overfitting
        layers.Dropout(0.5),
        # Lớp Dense đầu ra với 1 neuron và hàm kích hoạt sigmoid để đưa ra xác suất của lớp nhị phân
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Biên dịch mô hình với bộ tối ưu Adam, hàm mất mát binary_crossentropy và metric đánh giá là accuracy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # In thông báo bắt đầu huấn luyện mô hình
    print("\nTraining Modified Time Series model (Target ~80% Accuracy)...")
    
    # Huấn luyện mô hình với tập huấn luyện, sử dụng 10% dữ liệu làm validation, 3 epoch và batch size là 32
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=2)
    
    # Đánh giá mô hình trên tập kiểm tra
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\nModified Time Series Test accuracy: {:.2f}%".format(test_acc * 100))
    
    # Dự đoán nhãn cho tập kiểm tra
    y_pred = model.predict(X_test)
    # Chuyển xác suất dự đoán thành nhãn 0 hoặc 1 với ngưỡng 0.5
    y_pred_labels = (y_pred > 0.5).astype("int32").flatten()
    
    # In báo cáo phân loại chi tiết
    print("\nClassification Report (Modified Time Series):")
    print(classification_report(y_test, y_pred_labels))
    
    # Tính toán và in ma trận nhầm lẫn
    print("Confusion Matrix (Modified Time Series):")
    cm = confusion_matrix(y_test, y_pred_labels)
    print(cm)
    
    # Vẽ biểu đồ ma trận nhầm lẫn sử dụng seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Ma trận nhầm lẫn (Modified Time Series)")
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.show()

# Nếu file được chạy trực tiếp thì thực hiện hàm run_time_series
if __name__ == '__main__':
    run_time_series()
