# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

Một câu đánh giá có thể khen món ăn nhưng chê phục vụ. Bài toán cần tìm đúng cụm từ chỉ từng khía cạnh, gọi là **aspect**, rồi xác định cảm xúc dành cho cụm đó. Đầu vào là câu gốc; đầu ra là các khoảng ký tự và nhãn negative, neutral, positive hoặc conflict. Ví dụ minh họa: “Great food, slow service.” → “food”: positive; “service”: negative.

Metric chính là **aspect span micro-F1**: cân bằng giữa việc tìm đủ aspect và tránh tìm nhầm, chỉ tính đúng khi cả đầu và cuối cụm khớp chính xác. F1 càng cao càng tốt. Polarity macro-F1 đánh giá cảm xúc trên các span có nhãn, coi bốn lớp ngang nhau; test của nguồn này không có polarity nên không được công bố polarity F1 trên test. Các F1 trong baseline dùng thang 0–1.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[SetFit SemEval Restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) có các trường text, span, label, ordinal. Khảo sát đã ghi nhận 3.693 hàng aspect train và 1.134 hàng test, tương ứng 2.019 và 606 câu nguyên văn. Một câu có thể có nhiều hàng annotation; ordinal là lần xuất hiện của đúng chuỗi span, không phải thứ tự mọi aspect trong câu. Nhãn train gồm 2.164 positive, 805 negative, 633 neutral và 91 conflict trước gộp trùng; lớp conflict rất ít. Toàn bộ label test là chuỗi rỗng, nhưng còn span/ordinal để chấm extraction.

Có ba annotation train kết thúc giữa một token regex. Vì vậy biểu diễn chỉ theo từ không thể khớp mọi ranh giới; dùng ký tự làm đơn vị dự đoán có căn cứ trực tiếp từ dữ liệu. Gộp annotation trùng của cùng câu, giữ nguyên nhãn nguồn; nguồn chỉ chứa câu có aspect nên khả năng nhận biết câu hoàn toàn không có aspect chưa được đại diện đầy đủ. Loại nhóm train trùng câu test theo giao thức đang dùng, rồi tách 20% train theo câu để làm validation; không chia các aspect của một câu sang hai tập.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| Đặc trưng từ + CRF | Nhẹ và dễ kiểm soát cấu trúc nhãn | Khó biểu diễn chính xác ranh giới giữa token nếu dự đoán theo từ. |
| Ký tự + RNN/softmax | Giữ được offset gốc và đơn giản | Ngữ cảnh một chiều và giải mã từng vị trí dễ tạo chuỗi nhãn thiếu nhất quán. |
| Ký tự/từ + BiLSTM-CRF | Giữ offset ký tự, bổ sung ngữ nghĩa từ và ràng buộc chuỗi | Nhiều tham số hơn, cần điều chuẩn vì chỉ có khoảng hai nghìn câu train nguồn. |

## 3. Tiền xử lý dữ liệu tối ưu

1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc.

2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold.

3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó.

4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **embedding ký tự + embedding từ → BiLSTM hai lớp → CRF**, kèm head polarity cho mỗi span. BiLSTM đọc cả trái và phải để hiểu “slow” đang nói về “service”; CRF là bộ giải mã chọn cả chuỗi BIO hợp lệ thay vì chọn từng ký tự riêng. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán.

Lựa chọn giải quyết đồng thời hai khó khăn đã thấy: offset không trùng biên từ và nhiều aspect/cảm xúc trong cùng câu. Trọng số loss BIO từ tần suất train giúp giảm áp đảo của O, nhưng lớp conflict vẫn thiếu dữ liệu. CRF không biểu diễn aspect lồng nhau; khảo sát nguồn chưa thấy span chồng lấn. Toàn bộ mô hình và vocabulary học từ đầu.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 16 | Số mẫu mỗi bước | 16 giữ budget cho chuỗi ký tự dài và loss span/polarity, thay vì chọn theo số từ ngắn hơn. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.002 | Độ lớn cập nhật | 0.002 là khởi đầu tương đối chủ động cho embedding/CRF học mới trên corpus nhỏ; clipping1 và dropout0.3 hỗ trợ ổn định, chưa phải kết quả sweep. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `DROPOUT` | 0.3 | Điều chuẩn activation | 0.3 là điều chuẩn đề xuất vì mô hình đa nguồn/decoder có capacity lớn hoặc tập source hạn chế. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 1 | Giới hạn norm gradient | Ngưỡng 1 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `CHAR_EMBED_DIM` | 48 | Biểu diễn ký tự | 48 chiều dành dung lượng cho chữ/dấu và kiểu viết trong corpus nhỏ; chưa có ablation chứng minh 48 tốt nhất. | Tăng có thể học hình thái tốt hơn nhưng dễ quá khớp; giảm mất chi tiết ký tự. |
| `WORD_EMBED_DIM` | 128 | Ngữ nghĩa từ | 128 chiều bổ sung nghĩa từ cho nhãn theo ký tự mà không làm embedding quá lớn với số câu hiện có. | Tăng tốn tham số cho từ hiếm; giảm có thể làm polarity khó phân biệt. |
| `HIDDEN_DIM` | 128 | Ngữ cảnh BiLSTM mỗi chiều | 128 giữ thông tin ngữ cảnh hai phía với capacity vừa phải so với khoảng hai nghìn câu nguồn. | Tăng làm head/CRF phụ thuộc biểu diễn lớn hơn và dễ quá khớp; giảm có thể bỏ ngữ cảnh xa. |
| `NUM_LAYERS` | 2 | Độ sâu chuỗi | Hai lớp cho phép kết hợp dấu hiệu ký tự/từ và ngữ cảnh; sâu hơn chưa có căn cứ dữ liệu. | Tăng làm khó tối ưu trên tập nhỏ; giảm bớt capacity nhưng tiết kiệm thời gian. |
| `POLARITY_LOSS_WEIGHT` | 0.25 | Cân bằng hai nhiệm vụ | 0.25 đặt extraction, metric chính, ở vai trò ưu tiên và vẫn truyền tín hiệu polarity. | Tăng có thể hy sinh extraction để học polarity; giảm khiến head cảm xúc học yếu. |
| `EPOCHS` | 40 | Trần ngân sách huấn luyện | 40 là trần cho embeddings/BiLSTM/CRF học mới với ít câu; không khẳng định cần đủ40, dùng aspect F1 validation để chọn. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 8 | Dừng sớm theo validation | 8 epoch không cải thiện là dung sai đề xuất cho metric aspect F1; chưa được sweep để chứng minh giá trị này tốt nhất. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |

Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
