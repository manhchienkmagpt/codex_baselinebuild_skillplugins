# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

Bài toán đang triển khai là **nhận dạng thực thể trên tài liệu đã có OCR words và tọa độ**, không phải tự đọc chữ từ ảnh. Model nhận ảnh trang, chuỗi từ và các hộp bao; trả các cụm HEADER, QUESTION hoặc ANSWER. Ví dụ minh họa: trường “Name: Alice” có thể được gán “Name” là QUESTION và “Alice” là ANSWER. Việc nối hai cụm thành cặp key–value là nhiệm vụ khác.

Metric chính **entity span micro-F1** đòi cả ranh giới cụm từ và loại thực thể đúng, càng cao càng tốt. Nguồn HF này không có gold links, nên K–V linking F1 không xác định; không được coi khoảng cách hình học là nhãn liên kết thật.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) có 149 tài liệu train và 50 test với image, words, bboxes, ner_tags. Box dùng hệ tọa độ 0..1000; có bảy BIO tags: O và B/I cho HEADER, QUESTION, ANSWER. Baseline chia 149 tài liệu thành 119 train/30 validation theo seed cố định, giữ 50 test.

Dữ liệu rất ít ở cấp trang, bố cục và hình thức chữ có thể mang tín hiệu quan trọng ngoài nội dung từ. Giữ mọi token của một trang cùng split; so trùng bằng nội dung words+boxes thay vì chỉ dùng ID vì ID có thể được đánh lại theo split. Đã khảo sát schema và mẫu ảnh/nhãn, chưa quét toàn bộ trang hoặc biết phân bố đủ mọi kiểu form.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| Word + layout + RNN/softmax | Nhẹ và tận dụng vị trí trường | Khó xử lý OOV, kiểu chữ và quan hệ nhãn liên tiếp. |
| Word/char/layout/ảnh + BiLSTM-CRF | Kết hợp nội dung, hình học, hình thức và cấu trúc chuỗi | Nhiều thành phần, dễ quá khớp với 149 tài liệu nguồn. |
| Kiến trúc Transformer tài liệu scratch | Có thể mô hình hóa quan hệ xa giữa vùng | Ít trang để học từ đầu; không được dùng LayoutLM pretrained theo cấu hình hiện tại. |

## 3. Tiền xử lý dữ liệu tối ưu

1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn.

2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu.

3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác.

4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF**. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans.

Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; CRF giảm chuỗi nhãn sai cấu trúc. Tất cả khởi tạo mới, không pretrained. Mô hình không giải quyết OCR đầu vào hoặc linking; độ phức tạp là một rủi ro trên tập nhỏ nên validation theo trang rất quan trọng.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 4 | Số mẫu mỗi bước | 4 hạn chế tensor toàn trang384 và chuỗi từ không bị cắt. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.001 | Độ lớn cập nhật | 0.001 cho các nhánh word/char/ảnh/CRF đều scratch; với ít trang, dropout và early stopping cần ngăn cập nhật nhanh biến thành học thuộc. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `DROPOUT` | 0.3 | Điều chuẩn activation | 0.3 là điều chuẩn đề xuất vì mô hình đa nguồn/decoder có capacity lớn hoặc tập source hạn chế. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 1 | Giới hạn norm gradient | Ngưỡng 1 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `IMAGE_SIZE` | 384 | Chi tiết trang cho CNN/RoI | 384 giữ nhiều hình thức chữ hơn ảnh nhỏ mà còn hạn chế chi phí toàn trang. | Tăng rõ chi tiết nhưng tốn nhớ; giảm khiến vùng từ sau RoI kém phân biệt. |
| `MAX_WORD_CHARS` | 32 | Trần ký tự của từ | 32 giới hạn tensor char-CNN cho mỗi token; không cắt chuỗi từ/tài liệu. | Tăng giữ hậu tố từ dài nhưng tốn nhớ; giảm có thể mất dấu hiệu hình thái. |
| `HIDDEN_DIM` | 128 | Ngữ cảnh BiLSTM mỗi chiều | 128 giới hạn capacity của mô hình đa nguồn trên rất ít trang. | Tăng dễ quá khớp bố cục; giảm có thể thiếu ngữ cảnh thực thể dài. |
| `EPOCHS` | 80 | Trần ngân sách huấn luyện | 80 cho phép nhiều lượt khi một epoch chỉ có119 trang train; trần cao đi cùng dừng sớm, không phải bằng chứng cần train80. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 12 | Dừng sớm theo validation | 12 cho validation chỉ30 trang, tránh phản ứng quá nhanh với dao động entity F1, nhưng vẫn có nguy cơ quá khớp. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |

Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
