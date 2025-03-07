import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# Đường dẫn đến thư mục dữ liệu train và test
train_dir = r"C:\Download\orange-classifier\old_oranges_data_1\old_oranges_data\train_set"
test_dir = r"C:\Download\orange-classifier\old_oranges_data_1\old_oranges_data\test_set"

# Kiểm tra xem thư mục có tồn tại hay không
assert os.path.exists(train_dir), "Thư mục train không tồn tại."
assert os.path.exists(test_dir), "Thư mục test không tồn tại."

# Tạo bộ sinh dữ liệu cho tập train với data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Tạo bộ sinh dữ liệu cho tập test (chỉ rescale)
test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho tập train
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Tạo generator cho tập test, đặt shuffle=False để thứ tự file không bị xáo trộn (quan trọng cho việc dự đoán)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Xây dựng kiến trúc của model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Biên dịch model với optimizer Adam và loss binary_crossentropy
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Sử dụng EarlyStopping để tránh overfitting, theo dõi val_loss với patience = 5
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Huấn luyện model trên tập train và kiểm tra trên tập test (validation)
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stopping]
)

# Đánh giá model trên tập test
test_loss, test_acc = model.evaluate(test_generator)
print(f"Độ chính xác trên tập test: {test_acc * 100:.2f}%")

# Sinh dự đoán cho tập test
predictions = (model.predict(test_generator) > 0.5).astype("int32").flatten()
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# In báo cáo phân loại
print(classification_report(true_labels, predictions, target_names=class_labels))

# Vẽ đồ thị Accuracy của huấn luyện và xác thực
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Accuracy huấn luyện')
plt.plot(history.history['val_accuracy'], label='Accuracy xác thực')
plt.title("Đồ thị Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Vẽ đồ thị Loss của huấn luyện và xác thực
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Loss huấn luyện')
plt.plot(history.history['val_loss'], label='Loss xác thực')
plt.title("Đồ thị Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Reset lại test_generator (nếu cần) và dự đoán lại cho file submission
test_generator.reset()
predictions = (model.predict(test_generator) > 0.5).astype("int32").flatten()

# Lấy tên file hình ảnh (loại bỏ phần mở rộng)
image_names = [os.path.splitext(os.path.basename(path))[0] for path in test_generator.filenames]

# Tạo DataFrame cho file submission với tên ảnh và nhãn dự đoán
submission_df = pd.DataFrame({
    "image_name": image_names, 
    "label": predictions         
})

# Lưu file submission vào đường dẫn "C:\Download\orange-classifier"
submission_file = r"C:\Download\orange-classifier\submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"File submission đã được tạo: {submission_file}")

# In ra cấu trúc của model (summary)
model.summary()

# Lấy và in tên các lớp (model layers) của model
layer_names = [layer.name for layer in model.layers]
print("Các tên của các lớp trong model:")
for name in layer_names:
    print(name)
