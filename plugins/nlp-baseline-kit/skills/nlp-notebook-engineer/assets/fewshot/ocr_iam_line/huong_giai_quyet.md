# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

Đầu vào là ảnh **một dòng chữ viết tay tiếng Anh**, đầu ra là chuỗi chữ trong dòng đó. Ví dụ minh họa định dạng: ảnh ghi “hello world” → text “hello world”. Đây không phải OCR tiếng Việt hoặc đọc cả trang với phân tích bố cục.

Metric chính **CER** là tổng số thao tác thêm/xóa/thay ký tự cần để biến dự đoán thành gold, chia tổng ký tự gold; WER tính tương tự theo từ. Cả hai càng thấp càng tốt và có thể lớn hơn 1 nếu nhiều lỗi. Tính trên toàn split, không trung bình đơn giản CER từng batch.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) có image và text, số dòng theo card đã ghi nhận là 6.482/976/2.915 train/validation/test. Mẫu nguồn có chiều cao 128; đã khảo sát mẫu ảnh/nhãn, chưa quét toàn bộ phân bố tỷ lệ ảnh/độ dài hoặc xác minh trùng người viết. HF schema không có writer_id nên giữ split nguồn và không khẳng định writer-disjoint.

Ảnh có độ rộng thay đổi, chữ viết nối nét, nền và phong cách người viết khác nhau. Resize mọi dòng về một hình vuông dễ làm méo hoặc mất ký tự; việc biến ảnh thành một chuỗi đủ dài quan trọng hơn chỉ tăng số lớp CNN. Alphabet phải fit train để không dùng transcript test bổ sung nhãn; ký tự chưa gặp vẫn phải được tính lỗi khi đánh giá.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| CNN scratch + RNN + CTC | Không cần annotation vị trí từng ký tự | Encoder yếu hơn có thể khó thích ứng biến thiên chữ viết. |
| Encoder ảnh pretrained + BiLSTM + CTC | Tận dụng feature thị giác, ngữ cảnh hai phía và loss không cần alignment ký tự | Phải sửa stride/độ dài CTC; ImageNet khác miền chữ viết. |
| Attention encoder–decoder scratch | Có thể học giải mã chuỗi linh hoạt | Đắt hơn và cần kiểm alignment/decoder; không được dùng OCR pretrained hoàn chỉnh cho nhiệm vụ này. |

## 3. Tiền xử lý dữ liệu tối ưu

1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn.

2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học.

3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference.

4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC**. Encoder torchvision/resnet18 dùng `ResNet18_Weights.IMAGENET1K_V1`, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. CTC cộng xác suất nhiều alignment hợp lệ, không cần biết ký tự nằm ở pixel nào; greedy decoder gộp lặp và bỏ blank theo độ dài ảnh thật.

Giữ tỷ lệ và stride nhỏ theo chiều ngang xử lý trực tiếp yêu cầu độ dài CTC. BiLSTM có thể hỗ trợ nét chữ mơ hồ dựa trên hai phía, encoder pretrained chỉ cung cấp feature ảnh; alphabet và recognizer học mới. Không dùng full pretrained OCR model hoặc language decoder. Viết tay khác miền ImageNet và không có language model nên vẫn có thể sai từ hiếm.

[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 8 | Số mẫu mỗi bước | 8 hạn chế chi phí dòng biến thiên độ rộng, encoder và BiLSTM. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.001 | Độ lớn cập nhật | 0.001 cho BiLSTM/CTC mới, tách LR1e-4 của encoder ảnh để recognizer học nhanh hơn mức đổi visual features. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `BACKBONE_LEARNING_RATE` | 0.0001 | Fine-tune encoder ảnh | Nhỏ hơn LR phần mới để giữ feature pretrained trong khi thích nghi ảnh đúng miền. | Tăng dễ phá feature cũ; giảm làm encoder thích nghi chậm hoặc gần frozen. |
| `DROPOUT` | 0.2 | Điều chuẩn activation | 0.2 thêm điều chuẩn vừa phải cho học chuỗi mà không bỏ quá nhiều tín hiệu mỗi lượt. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 5 | Giới hạn norm gradient | Ngưỡng 5 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `IMAGE_HEIGHT` | 64 | Chiều cao resize giữ tỷ lệ | 64 tiết kiệm so với ảnh nguồn cao 128 nhưng phải giữ đủ bước CTC bằng stride ngang /4. | Tăng giữ nét và tăng số bước/nhớ; giảm có thể làm mất nét hoặc target bất khả thi. |
| `HIDDEN_DIM` | 256 | Hidden BiLSTM mỗi chiều | 256 đề xuất để phân giải ngữ cảnh nét chữ từ feature 512 mà không cần decoder ngôn ngữ lớn. | Tăng tốn nhớ/overfit; giảm hạn chế ngữ cảnh của chữ mơ hồ. |
| `EPOCHS` | 50 | Trần ngân sách huấn luyện | 50 cho recognizer CTC học mới trên khoảng6.5 nghìn dòng; encoder pretrained không có sẵn ánh xạ ký tự nên vẫn cần học chuỗi. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 8 | Dừng sớm theo validation | 8 epoch không cải thiện là dung sai đề xuất cho metric cer; chưa được sweep để chứng minh giá trị này tốt nhất. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |

Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
