document.addEventListener("DOMContentLoaded", () => {
    // --- Lấy các phần tử DOM ---
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const predictBtn = document.getElementById("predict-btn");
    const loadingSpinner = document.getElementById("loading");
    const errorMessage = document.getElementById("error-message"); // Phần tử hiển thị thông báo lỗi

    const imagePreview = document.getElementById("image-preview"); // Ảnh xem trước được chọn
    const imagePlaceholder = document.getElementById("image-placeholder"); // Placeholder cho ảnh xem trước

    // Các phần tử hiển thị kết quả từ server (đã khớp với index.html)
    const detectionResultImg = document.getElementById("detection-result-image"); // Ảnh phát hiện bệnh (YOLO)
    const detectionResultPlaceholder = document.getElementById(
        "detection-result-placeholder"
    ); // Placeholder cho ảnh phát hiện
    const detectionResultsList = document.getElementById(
        "detection-results-list"
    ); // Danh sách phát hiện (ví dụ: Cháy lá 95%)

    // Kết quả Phân loại Cây (Mô hình MobileNetV2)
    const classificationResultText = document.getElementById(
        "classification-result-text"
    ); // Text hiển thị tên cây và độ tin cậy
    const classificationCardBody = document.getElementById(
        "classification-card-body"
    ); // Phần thân card chứa kết quả phân loại
    const classificationPlaceholder = document.getElementById(
        "classification-placeholder"
    ); // Placeholder cho phân loại

    // Các phần tử hiển thị ảnh gốc và heatmap của Grad-CAM
    const gradcamOriginalImage = document.getElementById(
        "gradcam-original-image"
    );
    const gradcamOriginalPlaceholder = document.getElementById(
        "gradcam-original-placeholder"
    );
    const gradcamHeatmapImage = document.getElementById("gradcam-heatmap-image");
    const gradcamHeatmapPlaceholder = document.getElementById(
        "gradcam-heatmap-placeholder"
    );

    // Các phần tử liên quan đến camera
    const captureBtn = document.getElementById("capture-btn");
    const video = document.getElementById("camera-stream");
    const canvas = document.getElementById("snapshot-canvas");

    let stream = null; // Để lưu trữ luồng camera
    let selectedFile = null; // Để lưu trữ file ảnh được chọn hoặc chụp

    // --- Xử lý sự kiện kéo/thả và chọn file ---

    // Reset giá trị input để có thể chọn lại cùng file nếu muốn
    fileInput.addEventListener("click", () => {
        fileInput.value = null;
    });

    // Highlight vùng tải lên khi kéo file qua
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault(); // Ngăn chặn hành vi mặc định (mở file)
        uploadArea.classList.add("dragover");
    });

    // Bỏ highlight khi rời file khỏi vùng tải lên
    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    // Xử lý khi thả file vào vùng tải lên
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault(); // Ngăn chặn hành vi mặc định
        uploadArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFiles(files); // Sử dụng hàm handleFiles để xử lý file
        }
    });

    // Xử lý khi người dùng chọn file bằng input
    fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

    // --- Hàm xử lý các file được chọn hoặc kéo thả ---
    function handleFiles(files) {
        selectedFile = files[0]; // Chỉ lấy file đầu tiên

        // Reset UI khi chọn ảnh mới
        hideMessage(errorMessage);
        hideAllResults(); // Ẩn tất cả các kết quả trước đó

        // Ẩn camera stream nếu đang hiển thị
        if (stream) {
            video.srcObject.getTracks().forEach(track => track.stop());
            stream = null;
            video.style.display = "none";
        }

        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result; // Hiển thị ảnh xem trước
                imagePreview.style.display = "block";
                imagePlaceholder.style.display = "none";
            };
            reader.readAsDataURL(selectedFile);
        } else {
            imagePreview.src = "#"; // Xóa ảnh xem trước
            imagePreview.style.display = "none";
            imagePlaceholder.style.display = "block";
        }
    }

    // --- Xử lý sự kiện chụp ảnh từ camera ---
    captureBtn.addEventListener("click", async () => {
        // Nếu camera chưa bật, bật camera
        if (!stream) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.style.display = "block";
                imagePreview.style.display = "none"; // Ẩn ảnh xem trước nếu đang bật camera
                imagePlaceholder.style.display = "none"; // Ẩn placeholder khi bật camera
            } catch (err) {
                alert("Không thể truy cập camera: " + err.message);
                console.error("Lỗi truy cập camera:", err);
            }
            return; // Dừng lại sau khi bật camera, đợi click tiếp theo để chụp
        }

        // Nếu camera đã bật, chụp ảnh
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Hiển thị ảnh chụp
        const imageDataURL = canvas.toDataURL("image/png");
        imagePreview.src = imageDataURL;
        imagePreview.style.display = "block";
        imagePlaceholder.style.display = "none";

        // Tắt camera sau khi chụp
        video.srcObject.getTracks().forEach(track => track.stop());
        stream = null;
        video.style.display = "none";

        // Chuyển ảnh đã chụp thành đối tượng File và gán vào selectedFile
        const blob = await (await fetch(imageDataURL)).blob();
        selectedFile = new File([blob], "captured_image.png", { type: "image/png" });

        // Reset UI và ẩn kết quả cũ khi chụp ảnh mới
        hideMessage(errorMessage);
        hideAllResults();
    });

    // --- Xử lý sự kiện nhấn nút "Phân loại và nhận diện bệnh" ---
    predictBtn.addEventListener("click", async () => {
        if (!selectedFile) {
            showMessage(errorMessage, "Vui lòng chọn một ảnh hoặc chụp ảnh để dự đoán.", "warning");
            return;
        }

        // Hiển thị loading spinner và xóa kết quả cũ
        hideMessage(errorMessage);
        showSpinner(loadingSpinner);
        hideAllResults(); // Ẩn tất cả kết quả trước đó

        // Tạo đối tượng FormData để gửi file lên server
        const formData = new FormData();
        formData.append("file", selectedFile); // Backend mong đợi key là 'file'

        try {
            // Gửi yêu cầu POST tới API /predict
            const res = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            // Kiểm tra nếu request không thành công
            if (!res.ok) {
                // Thử đọc lỗi từ JSON nếu server trả về
                const errorData = await res.json().catch(() => ({})); // Bắt lỗi nếu không phải JSON
                const errorMsg =
                    errorData.error || `Lỗi HTTP! Trạng thái: ${res.status}`;
                throw new Error(errorMsg);
            }

            // Nhận kết quả JSON từ server
            const data = await res.json();

            if (data.success) {
                // --- Hiển thị kết quả Phát hiện Bệnh (YOLO) ---
                if (data.detected_image) {
                    detectionResultImg.src = `data:image/png;base64,${data.detected_image}`;
                    showImage(detectionResultImg, detectionResultPlaceholder);
                } else {
                    hideImage(detectionResultImg, detectionResultPlaceholder);
                }

                // Xóa các kết quả phát hiện cũ
                detectionResultsList.innerHTML = "";
                if (data.detection_results && data.detection_results.length > 0) {
                    data.detection_results.forEach((detection) => {
                        const li = document.createElement("li");
                        li.textContent = `${detection.disease}: ${detection.confidence}`;
                        detectionResultsList.appendChild(li);
                    });
                } else {
                    const li = document.createElement("li");
                    li.textContent =
                        "Không phát hiện bệnh (có thể khỏe mạnh hoặc không có đối tượng).";
                    li.classList.add("text-muted");
                    detectionResultsList.appendChild(li);
                }

                // --- Hiển thị kết quả Phân loại Cây (MobileNetV2) ---
                if (data.plant_classification) {
                    classificationResultText.innerHTML = `
                                <h6>Loài cây dự đoán: <span class="text-primary">${data.plant_classification.plant_name}</span></h6>
                                <p>Độ tin cậy: <span class="text-success">${data.plant_classification.confidence}</span></p>
                            `;
                    showClassificationResult(); // Hiển thị phần thân card
                } else {
                    hideClassificationResult();
                }

                // --- Hiển thị ảnh Gốc (cho Grad-CAM) ---
                if (data.original_image_for_gradcam) {
                    gradcamOriginalImage.src = `data:image/png;base64,${data.original_image_for_gradcam}`;
                    showImage(gradcamOriginalImage, gradcamOriginalPlaceholder);
                } else {
                    hideImage(gradcamOriginalImage, gradcamOriginalPlaceholder);
                }

                // --- Hiển thị Heatmap Grad-CAM ---
                if (data.heatmap_image) {
                    gradcamHeatmapImage.src = `data:image/png;base64,${data.heatmap_image}`;
                    showImage(gradcamHeatmapImage, gradcamHeatmapPlaceholder);
                } else {
                    hideImage(gradcamHeatmapImage, gradcamHeatmapPlaceholder);
                }
            } else {
                // Xử lý lỗi từ server
                showMessage(
                    errorMessage,
                    data.error || "Có lỗi không xác định xảy ra từ server.",
                    "danger"
                );
            }
        } catch (error) {
            console.error("Lỗi trong quá trình dự đoán:", error);
            showMessage(
                errorMessage,
                `Đã xảy ra lỗi: ${error.message}. Vui lòng thử lại.`,
                "danger"
            );
        } finally {
            // Luôn ẩn loading spinner và reset trạng thái sau khi dự đoán xong
            hideSpinner(loadingSpinner);
        }
    });

    // --- Các hàm hỗ trợ ---

    function showSpinner(spinnerElement) {
        spinnerElement.style.display = "inline-block"; // hoặc 'block' tùy thuộc vào CSS
    }

    function hideSpinner(spinnerElement) {
        spinnerElement.style.display = "none";
    }

    function showMessage(element, message, type) {
        element.textContent = message;
        element.className = `alert alert-${type} mt-3`; // Ví dụ: 'alert-danger', 'alert-warning'
        element.style.display = "block";
    }

    function hideMessage(element) {
        element.style.display = "none";
        element.textContent = "";
    }

    // Hàm hiển thị ảnh và ẩn placeholder
    function showImage(imageElement, placeholderElement) {
        imageElement.style.display = "block";
        placeholderElement.style.display = "none";
    }

    // Hàm ẩn ảnh và hiển thị placeholder
    function hideImage(imageElement, placeholderElement) {
        imageElement.style.display = "none";
        placeholderElement.style.display = "block";
        imageElement.src = ""; // Xóa ảnh cũ
    }

    // Hàm hiển thị kết quả phân loại
    function showClassificationResult() {
        classificationCardBody.style.display = "block";
        classificationPlaceholder.style.display = "none";
    }

    // Hàm ẩn kết quả phân loại
    function hideClassificationResult() {
        classificationCardBody.style.display = "none";
        classificationPlaceholder.style.display = "block";
        classificationResultText.innerHTML = ""; // Xóa nội dung
    }

    // Hàm ẩn tất cả các kết quả (dùng khi reset UI)
    function hideAllResults() {
        // Ẩn ảnh phát hiện
        hideImage(detectionResultImg, detectionResultPlaceholder);
        detectionResultsList.innerHTML = ""; // Xóa nội dung danh sách

        // Ẩn kết quả phân loại
        hideClassificationResult();

        // Ẩn ảnh Grad-CAM gốc và heatmap
        hideImage(gradcamOriginalImage, gradcamOriginalPlaceholder);
        hideImage(gradcamHeatmapImage, gradcamHeatmapPlaceholder);
    }

    // Khởi tạo UI khi tải trang
    hideSpinner(loadingSpinner);
    hideMessage(errorMessage);
    hideAllResults(); // Đảm bảo tất cả kết quả bị ẩn lúc đầu
});