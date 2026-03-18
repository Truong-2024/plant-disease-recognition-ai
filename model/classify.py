# model/classify.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array # Giữ lại img_to_array
import numpy as np
import os
from PIL import Image
import io # Thêm import này để làm việc với dữ liệu bytes trong bộ nhớ

# --- Kiểm tra và báo cáo việc sử dụng GPU ---
print("--- Kiểm tra GPU cho TensorFlow ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e) # Lỗi trong quá trình cấu hình GPU
else:
    print("Không tìm thấy GPU vật lý. TensorFlow sẽ chạy trên CPU.")
print("---------------------------------")

# Load mô hình phân loại cây cảnh của bạn
# Đảm bảo đường dẫn tới file mô hình là chính xác
plant_classifier_model = load_model('model/mobilenetv2_plant_classifier.h5')

# Định nghĩa tên các lớp (loại cây) theo thứ tự mà mô hình đã được huấn luyện
plant_class_names = [
    "Lưỡi hổ", "Lan ý", "Môn trường sinh", "Lan tổ điểu", "Thường xuân", "Đuôi công tím",
    "Trầu bà", "Phát tài", "Phú quý", "Đuôi công sọc xanh dài", "Ngọc ngân", "Kim tiền",
    "Oai hùng", "Vạn niên thanh", "Đuôi công nữ thần xanh", "Ổ rồng", "Trầu bà cung đàn"
]

def classify_plant(image_input):
    """
    Phân loại loại cây từ hình ảnh đầu vào bằng mô hình đã load.
    image_input có thể là:
    - Đường dẫn file (string)
    - Dữ liệu ảnh dưới dạng bytes (nhận từ request trong Flask)
    Trả về tên cây, độ tin cậy và chỉ số của lớp dự đoán.
    """
    if isinstance(image_input, str): # Nếu đầu vào là đường dẫn file
        # Sử dụng PIL để mở ảnh từ đường dẫn
        img = Image.open(image_input)
    elif isinstance(image_input, (bytes, io.BytesIO)): # Nếu đầu vào là dữ liệu bytes
        # Sử dụng BytesIO để tạo một "file" trong bộ nhớ từ bytes, sau đó mở bằng PIL
        img = Image.open(io.BytesIO(image_input))
    else:
        raise ValueError("Đầu vào ảnh không hợp lệ. Phải là đường dẫn file hoặc dữ liệu bytes.")

    # Resize ảnh về kích thước mong muốn sau khi đã mở bằng PIL
    img = img.resize((224, 224))
    
    # Chuyển đổi ảnh PIL sang numpy array và chuẩn hóa
    img_array = img_to_array(img) / 255.0  # Chuẩn hóa về [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

    predictions = plant_classifier_model.predict(img_array)[0]
    top_index = np.argmax(predictions)
    confidence = predictions[top_index] * 100

    # Giới hạn độ tin cậy nếu quá cao (tùy chọn, có thể bỏ nếu không cần)
    if confidence > 99.99:
        confidence = 99
    
    return {
        "plant_name": plant_class_names[top_index],
        "confidence": f"{confidence:.2f}%",
        "predicted_class_index": int(top_index)
    }

# --- Ví dụ sử dụng (có thể bỏ khi tích hợp với Flask) ---
if __name__ == "__main__":
    # Đặt đường dẫn đến một ảnh thử nghiệm
    test_image_path = 'test_plant_image.jpg' 
    
    if not os.path.exists(test_image_path):
        print(f"Lỗi: Không tìm thấy ảnh thử nghiệm tại {test_image_path}")
        print("Vui lòng cập nhật 'test_image_path' đến một ảnh cây có sẵn.")
    else:
        print(f"\nĐang xử lý ảnh: {test_image_path}")
        try:
            # Ví dụ sử dụng với đường dẫn file
            classification_result_path = classify_plant(test_image_path)
            print("Kết quả phân loại cây (từ đường dẫn):", classification_result_path)

            # Ví dụ sử dụng với dữ liệu bytes (nếu bạn muốn thử nghiệm độc lập)
            with open(test_image_path, 'rb') as f:
                image_bytes_for_test = f.read()
            classification_result_bytes = classify_plant(image_bytes_for_test)
            print("Kết quả phân loại cây (từ bytes):", classification_result_bytes)

        except Exception as e:
            print(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")