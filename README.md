# 🪴 AI-Powered Plant Care: Classification & Disease Detection

Dự án Nghiên cứu Khoa học ứng dụng các thuật toán Học sâu (Deep Learning) để tự động hóa việc phân loại các loài cây cảnh và nhận diện sớm các loại bệnh trên lá. Hệ thống hỗ trợ người dùng không chuyên chăm sóc cây cảnh hiệu quả thông qua hình ảnh.

## 🚀 Điểm nổi bật kỹ thuật (Technical Highlights)
Phân loại loài cây: Nhận diện 16 loại cây cảnh phổ biến trong nhà (Trầu bà, Kim tiền, Lưỡi hổ, Lan ý, Phú quý...).

Phát hiện bệnh lý: Nhận diện và khoanh vùng 11 loại trạng thái bệnh như đốm lá, cháy mép lá, thán thư, rệp sáp, vàng lá....

Xử lý đa mô hình: Kết hợp các kiến trúc CNN tiên tiến để tối ưu hóa giữa độ chính xác và tốc độ xử lý trên thiết bị di động.

Giải thích mô hình: Tích hợp GradCAM++ để trực quan hóa vùng bệnh, giúp người dùng hiểu rõ tại sao AI đưa ra kết quả đó.

## 🚀 Kiến trúc hệ thống
Dự án triển khai và so sánh hai nhóm mô hình chính:

Phân loại cây (Classification): EfficientNet-B0, ResNet-50, VGG-16, MobileNetV2, CNN và SVM.

Nhận diện bệnh (Object Detection): YOLOv8, Fast R-CNN và RetinaNet.

## 🖼️ Demo & Kết quả thực nghiệm
### Giao diện hệ thống
Khi người dùng chưa tải ảnh lên, giao diện hiển thị khung tương tác chính với nút “Phân loại và nhận diện bệnh”, cho phép người dùng kéo/thả hoặc nhấn để chọn ảnh từ thiết bị.

