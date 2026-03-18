from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid # Để tạo tên file duy nhất
import io # Để làm việc với dữ liệu bytes

# Import các hàm từ các module model của bạn
# Đảm bảo đường dẫn import là chính xác
from model.classify import classify_plant 
from model.detect import detect_diseases_with_gradcam 

app = Flask(__name__)
CORS(app) # Cho phép CORS

# Cấu hình thư mục lưu trữ ảnh tải lên tạm thời (có thể giữ hoặc bỏ tùy nhu cầu)
UPLOAD_FOLDER = 'temp_uploads' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """
    Route mặc định để hiển thị trang HTML chính của ứng dụng.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint để xử lý yêu cầu dự đoán.
    Nhận MỘT ảnh từ form, chạy qua các mô hình AI và trả về kết quả.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh trong yêu cầu."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Không có file ảnh được chọn."}), 400

    # Đọc dữ liệu ảnh dưới dạng bytes TRỰC TIẾP từ request
    image_bytes = file.read()
    
    # Ghi chú: Có thể bỏ qua việc lưu file tạm thời nếu cả hai model đều xử lý bytes trực tiếp.
    # unique_filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    # filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    # with open(filepath, 'wb') as f:
    #     f.write(image_bytes)

    try:
        # --- Chạy mô hình Phân loại cây cảnh (MobileNetV2) ---
        # Hàm classify_plant đã được chỉnh sửa để nhận image_bytes
        classification_result = classify_plant(image_bytes)

        # --- Chạy mô hình Phát hiện bệnh (YOLO) và tạo Grad-CAM heatmap ---
        # NHẬN BỐN GIÁ TRỊ TỪ HÀM detect_diseases_with_gradcam
        detections, original_image_b64, detected_image_b64, heatmap_image_b64 = \
            detect_diseases_with_gradcam(image_bytes)
        
        # Tổng hợp kết quả để trả về frontend
        response_data = {
            "success": True,
            # "filename": unique_filename, # Bỏ nếu không lưu file tạm
            
            # Kết quả từ mô hình Phân loại cây cảnh
            "plant_classification": classification_result, 

            # Kết quả từ mô hình Phát hiện (YOLO bounding box)
            "detection_results": detections,
            "detected_image": detected_image_b64, # Ảnh YOLO với bounding box

            # Kết quả Grad-CAM
            "original_image_for_gradcam": original_image_b64, # Ảnh gốc cho Grad-CAM
            "heatmap_image": heatmap_image_b64 # Ảnh Grad-CAM heatmap
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi khi xử lý ảnh: {str(e)}"}), 500
    # Ghi chú: Nếu bạn có code lưu file tạm thời, hãy bỏ phần xóa file vào đây
    # finally:
    #     if 'filepath' in locals() and os.path.exists(filepath):
    #         os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)