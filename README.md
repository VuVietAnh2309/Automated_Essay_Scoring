# Automated Essay Scoring (Prep Test)

## 📌 Overview
Đây là hướng tiếp cận của tôi cho bài **Prep Test** trong vòng 4 ngày.  
Mục tiêu: Xây dựng hệ thống **chấm điểm bài luận tự động (AES)** với đầu ra là điểm số dựa trên nội dung bài viết.

---

## 🏗️ Approach

### 1. Research & Problem
- Tham khảo các hướng đi từ mô hình ngôn ngữ (BERT, DeBERTa) và mô hình truyền thống (ML).
- Bài toán AES: dữ liệu không đầy đủ cho 4 nhãn chi tiết (TR, CC, LR, GR) → khó làm multi-head regression như paper gốc.
- Giải pháp: **kết hợp mô hình ngôn ngữ (trích đặc trưng)** và **mô hình học máy (LGBM)** để dự đoán điểm số.

---

### 2. Feature Engineering
Các đặc trưng được chọn:
- **Embedding features**: trích từ DeBERTa/transformer encoder.
- **Grammar & Range (GR)**:
  - Tỉ lệ lỗi ngữ pháp / chính tả (LanguageTool).
  - Tỉ lệ câu dài (>= threshold từ).
  - Tỉ lệ câu phức.
  - Tỉ lệ câu bắt đầu bằng liên từ phụ thuộc (Although, Because, While,...).
- **Coherence & Cohesion (CC)**:
  - Liên kết giữa các câu, độ mượt văn bản.
- **Lexical Resource (LR)**:
  - Độ đa dạng từ vựng, tỉ lệ từ vựng hiếm.
- **Task Response (TR)**:
  - Độ dài bài viết, số lượng câu/đoạn.

👉 Cuối cùng, sau khi thử nghiệm nhiều mô hình ML cơ bản, tôi chọn **LGBM** vì:
- Phù hợp dữ liệu tabular + embedding gộp.
- Ít nhạy với chuẩn hóa.
- Tốc độ huấn luyện nhanh.
- Kiểm soát overfitting tốt, dễ tối ưu bằng CV.
- Hữu ích khi dữ liệu hạn chế.

---

### 3. Model Training
- **Encoder**: DeBERTa để lấy sentence/document embeddings.
- **Regressor**: LightGBM để dự đoán điểm số.
- **Evaluation metrics**:
  - Accuracy
  - F1-score
  - QWK (Quadratic Weighted Kappa)

---

### 4. Deployment & Integration
- Chọn **BentoML** để đóng gói mô hình → tạo thành service API.
- Các bước:
  1. **Đóng gói**: gom mô hình + pre/post-processing thành service.
  2. **API**: định nghĩa endpoint `POST /score` → nhận bài viết, trả về điểm.
  3. **Triển khai**: BentoML build Docker image → chạy trên server hoặc deploy Kubernetes/Cloud.
- Vấn đề thực tế: **xử lý nhiều request đồng thời** → BentoML đã hỗ trợ serving models.

---

## 📊 Results
- Hệ thống chạy ổn định trong test cơ bản.
- Các đặc trưng kết hợp ML + embedding cho kết quả khả quan.
- Còn hạn chế: dữ liệu ít, chưa chuẩn bị đủ cho multi-head scoring.

---

## 🚀 Next Steps
- Thử nghiệm thêm với multi-task learning (predict TR/CC/LR/GR riêng).
- Mở rộng dataset để cải thiện độ tin cậy.
- Tối ưu serving để đáp ứng concurrency cao hơn.

---

## 🙌 Acknowledgement
- Bài thi hoàn thành trong 4 ngày, lần đầu tiếp cận AES nên còn nhiều thiếu sót.  
- Xin cảm ơn anh Tiến đã hướng dẫn và hỗ trợ trong quá trình làm bài.

