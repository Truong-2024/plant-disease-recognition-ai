from ultralytics import YOLO
import cv2
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

# Tải mô hình YOLO để phát hiện bệnh. Đảm bảo đường dẫn chính xác.
# Thay đổi đường dẫn đến file mô hình .pt của bạn
model_yolo_detection = YOLO('model/best_yolov8.pt') 

# Định nghĩa tên các lớp bệnh theo thứ tự mà mô hình đã được huấn luyện
disease_classes = [
    "Cháy lá", "Đốm lá", "Khỏe mạnh", "Côn trùng/ sâu bọ",
    "Vàng lá", "Bệnh thán thư", "Cháy mép lá", "Đốm vàng"
]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Thay đổi kích thước và đệm ảnh trong khi vẫn đáp ứng các ràng buộc bội số stride
    shape = im.shape[:2]  # kích thước hiện tại [chiều cao, chiều rộng]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Tỷ lệ scale (mới / cũ)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # chỉ scale nhỏ lại, không scale lớn lên (để có mAP tốt hơn)
        r = min(r, 1.0)

    # Tính toán phần đệm
    ratio = r, r  # tỷ lệ chiều rộng, chiều cao
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # đệm chiều rộng, chiều cao
    if auto:  # hình chữ nhật tối thiểu
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # đệm chiều rộng, chiều cao
    elif scaleFill:  # kéo giãn
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # tỷ lệ chiều rộng, chiều cao

    dw /= 2  # chia đệm thành 2 phía
    dh /= 2

    if shape[::-1] != new_unpad:  # thay đổi kích thước
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # thêm đường viền
    return im, ratio, (top, bottom, left, right)

class ActivationsAndGradients:
    """ Lớp để trích xuất activations và
    đăng ký gradients từ các lớp trung gian được nhắm mục tiêu """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Vì https://github.com/pytorch/pytorch/issues/61519,
            # chúng ta không sử dụng backward hook để ghi lại gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # Bạn chỉ có thể đăng ký hook trên tensor yêu cầu grad.
            return

        # Gradients được tính theo thứ tự ngược lại
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result)

class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result)

class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result)

class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        return data.max()

class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'Thông tin lớp mô hình: {model_names}')
        model = model_yolo.model.to(device) # Sử dụng trực tiếp .model để lấy Backbone và Head
        # model.info() # Dòng này có thể gây nhiều thông báo, bỏ ghi chú nếu cần gỡ lỗi kiến trúc mô hình
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False

        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"Không hỗ trợ tác vụ ({task}).")

        target_layers = [model.model[l] for l in layer] # Lấy các lớp mục tiêu từ backbone của mô hình
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Chuẩn hóa CAM nằm trong khoảng [0, 1]
        bên trong mỗi hộp giới hạn, và bằng 0 bên ngoài các hộp giới hạn."""
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_bytes): 
        # Đọc ảnh từ bytes
        try:
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_original_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # Ảnh gốc OpenCV (BGR)
            if img_original_cv2 is None:
                raise ValueError("Không thể giải mã ảnh từ bytes.")
        except Exception as e:
            print(f"Lỗi: Không thể đọc ảnh từ bytes. {e}")
            # Trả về các giá trị None để frontend không bị lỗi tham chiếu
            return [], None, None, None 

        original_img_shape = img_original_cv2.shape[:2] # Lưu kích thước ảnh gốc

        # Tạo bản sao của ảnh gốc để vẽ bounding box lên đó
        img_with_boxes_cv2 = img_original_cv2.copy() 

        # Chuẩn bị ảnh cho Grad-CAM và YOLO
        img_padded, _, (top, bottom, left, right) = letterbox(img_original_cv2, new_shape=(self.img_size, self.img_size), auto=True)
        img_for_cam = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB) # Giữ cho việc hiển thị CAM
        img_for_cam_float = np.float32(img_for_cam) / 255.0 # Chuẩn hóa về [0, 1] cho Grad-CAM
        tensor = torch.from_numpy(np.transpose(img_for_cam_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        detections = []
        
        # --- Thực hiện YOLO detection để lấy bounding box ---
        # Chạy detection trên ảnh đã được letterbox
        results_yolo = model_yolo_detection(img_padded)[0] 
        
        boxes_for_gradcam_renormalization = []

        if results_yolo.boxes:
            for box in results_yolo.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = disease_classes[class_id]
                
                detections.append({
                    "disease": class_name,
                    "confidence": f"{score*100:.2f}%"
                })
                boxes_for_gradcam_renormalization.append([x1, y1, x2, y2]) # Thu thập các box để dùng cho renormalize CAM

                # Vẽ hộp giới hạn lên ảnh dành cho bounding box (img_with_boxes_cv2)
                cv2.rectangle(img_with_boxes_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2) # Xanh lá
                cv2.putText(img_with_boxes_cv2, f"{class_name} {score*100:.1f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Nếu không tìm thấy hộp nào, thêm kết quả "Khỏe mạnh"
            detections.append({"disease": "Khỏe mạnh", "confidence": "100.00%"})
            # Có thể vẽ chữ "Khỏe mạnh" lên ảnh gốc hoặc ảnh bounding box nếu muốn
            # cv2.putText(img_with_boxes_cv2, "Khỏe mạnh", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # --- Tạo Grad-CAM heatmap ---
        try:
            grayscale_cam = self.method(tensor, [self.target]) # Tạo CAM
            grayscale_cam = grayscale_cam[0, :]
            
            threshold = 0.4 # Ngưỡng để lọc nhiễu trong heatmap
            grayscale_cam[grayscale_cam < threshold] = 0

            # Áp dụng renormalize CAM nếu được bật và có hộp giới hạn
            if self.renormalize and self.task in ['detect', 'segment', 'pose'] and len(boxes_for_gradcam_renormalization) > 0:
                # `renormalize_cam_in_bounding_boxes` mong đợi boxes là list of lists hoặc numpy array
                cam_image_overlayed_rgb = self.renormalize_cam_in_bounding_boxes(
                    np.array(boxes_for_gradcam_renormalization), img_for_cam_float, grayscale_cam
                )
            else:
                # Nếu không renormalize hoặc không có box, chỉ overlay bình thường
                cam_image_overlayed_rgb = show_cam_on_image(img_for_cam_float, grayscale_cam, use_rgb=True)

            # Chuyển đổi ảnh overlayed sang BGR (OpenCV) và thang màu 0-255
            cam_image_overlayed_cv2 = cv2.cvtColor((cam_image_overlayed_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Xóa phần đệm (letterbox) và resize về kích thước ảnh gốc cho ảnh Grad-CAM
            cam_image_final = cam_image_overlayed_cv2[top:cam_image_overlayed_cv2.shape[0] - bottom, left:cam_image_overlayed_cv2.shape[1] - right]
            cam_image_final = cv2.resize(cam_image_final, (original_img_shape[1], original_img_shape[0]))

        except Exception as e:
            print(f"Lỗi khi tạo Grad-CAM: {e}. Trả về ảnh gốc cho heatmap.")
            # Nếu Grad-CAM thất bại, trả về ảnh gốc cho phần heatmap
            cam_image_final = img_original_cv2.copy() 


        # --- Xử lý ảnh gốc và ảnh có bounding box để trả về ---
        # Xóa phần đệm và resize ảnh YOLO với bounding box về kích thước gốc
        img_with_boxes_final = img_with_boxes_cv2[top:img_with_boxes_cv2.shape[0] - bottom, left:img_with_boxes_cv2.shape[1] - right]
        img_with_boxes_final = cv2.resize(img_with_boxes_final, (original_img_shape[1], original_img_shape[0]))

        # --- Mã hóa các ảnh sang Base64 ---
        # Ảnh gốc (dùng cho Grad-CAM trên frontend)
        _, original_encoded = cv2.imencode('.png', img_original_cv2)
        original_image_b64 = base64.b64encode(original_encoded).decode('utf-8')

        # Ảnh có bounding box (từ YOLO)
        _, detected_encoded = cv2.imencode('.png', img_with_boxes_final)
        detected_image_b64 = base64.b64encode(detected_encoded).decode('utf-8')

        # Ảnh heatmap (đã overlay lên ảnh gốc)
        _, heatmap_encoded = cv2.imencode('.png', cam_image_final)
        heatmap_image_b64 = base64.b64encode(heatmap_encoded).decode('utf-8')

        return detections, original_image_b64, detected_image_b64, heatmap_image_b64

    def __call__(self, img_bytes):
        return self.process(img_bytes)

def get_gradcam_params():
    params = {
        'weight': 'model/best_yolov8.pt', # Đường dẫn đến file trọng số mô hình YOLOv8 của bạn
        'device': 'cpu', # 'cuda' nếu bạn có GPU, 'cpu' nếu không
        'method': 'GradCAMPlusPlus', # Các tùy chọn: GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [3, 4, 7], # Các lớp ví dụ, bạn có thể cần thử nghiệm để có kết quả tốt nhất
        'backward_type': 'all', # 'class', 'box', 'all'
        'conf_threshold': 0.3, # Ngưỡng tin cậy cho việc phát hiện
        'ratio': 0.03, # Tỷ lệ để chọn mục tiêu trong Grad-CAM, thường là một giá trị nhỏ
        'show_result': True, # Có hiển thị các phát hiện trên ảnh CAM không
        'renormalize': True, # Chuẩn hóa bản đồ nhiệt trong các hộp giới hạn
        'task':'detect', # Tác vụ của mô hình của bạn
        'img_size':640, # Kích thước ảnh đầu vào cho mô hình
    }
    return params

# Khởi tạo bộ xử lý Grad-CAM một lần
gradcam_processor = yolo_heatmap(**get_gradcam_params())

def detect_diseases_with_gradcam(image_bytes):
    """
    Phát hiện các bệnh trong ảnh bằng mô hình YOLO và tạo bản đồ nhiệt Grad-CAM.
    Vẽ các hộp giới hạn và phủ bản đồ nhiệt lên ảnh.
    Trả về danh sách các phát hiện và các ảnh đã xử lý dưới dạng chuỗi Base64.
    """
    return gradcam_processor(image_bytes)