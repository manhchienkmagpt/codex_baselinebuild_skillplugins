# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

**Extractive question answering** tìm câu trả lời bằng cách chọn một đoạn liên tiếp trong context đã cho. Ví dụ minh họa: context “Paris is the capital of France.”, question “What is the capital of France?” → “Paris”. Model dự đoán vị trí bắt đầu/kết thúc, không tự tạo kiến thức ngoài context.

Metric chính **Exact Match (EM)** yêu cầu text trả lời trùng gold sau chuẩn hóa chính thức; token F1 bổ sung mức khớp một phần. Cả hai càng cao càng tốt. Context đưa vào model giữ nguyên offset; chuẩn hóa đánh giá chỉ áp dụng khi so text, không sửa vị trí annotation. EM và token F1 được báo theo thang 0–100 trong baseline.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[SQuAD v1.1 qua TFDS](https://www.tensorflow.org/datasets/catalog/squad) có id/title/context/question/answers, mỗi answer gồm text và answer_start. Khảo sát JSON gốc trước đây ghi nhận 87.599 train questions trên 442 titles và 10.570 dev questions trên 48 titles; kiểm offset gold không thấy mismatch. Có 262 annotation train và 76 dev có biên nằm giữa token regex.

Nhiều câu hỏi chia sẻ cùng đoạn/bài Wikipedia, nên chia ngẫu nhiên câu hỏi dễ rò rỉ context. Baseline lấy 10% train titles làm validation nội bộ và giữ official dev làm held-out vì test không có nhãn công khai. Các answer có biên giữa token là giới hạn biểu diễn thật của head token-span; không tự sửa gold hoặc bỏ chúng để nâng EM.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| Word RNN + vector câu hỏi | Chi phí thấp, có điều kiện theo question | Một vector hỏi cố định khó khớp từng từ của context. |
| Word/char + tương tác context–question kiểu BiDAF | So khớp chi tiết hai chuỗi và thêm dấu hiệu từ lạ | Tăng tính toán; dự đoán biên token vẫn không bao phủ mọi gold substring. |
| Transformer scratch hoặc head span ký tự | Có thể cải thiện tương tác hoặc độ mịn ranh giới | Cần thí nghiệm/chi phí khác; chưa có bằng chứng để thay implementation hiện tại. |

## 3. Tiền xử lý dữ liệu tối ưu

1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.

2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.

3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.

4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads**. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; attention giúp từng token context đọc phần question liên quan. Toàn bộ mô hình học từ đầu, không dùng BERT/tokenizer pretrained.

Câu hỏi và đoạn văn cần so khớp ở mức token nên tương tác hai chiều có căn cứ hơn chỉ một vector question. Char-CNN giảm phụ thuộc vào word vocabulary giới hạn. Bản hiện tại vẫn không thể chọn chính xác biên giữa một token; đây là trần biểu diễn trên một số gold EM. Lựa chọn là thiết kế scratch có cơ sở, chưa chứng minh tối ưu toàn bộ SQuAD.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 24 | Số mẫu mỗi bước | 24 hạn chế tensor context384 và context–question attention/char features. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.001 | Độ lớn cập nhật | 0.001 cho embeddings, attention và span heads đều scratch; clipping1 đi kèm để hạn chế bước lớn trên nhiều windows. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `DROPOUT` | 0.2 | Điều chuẩn activation | 0.2 thêm điều chuẩn vừa phải cho học chuỗi mà không bỏ quá nhiều tín hiệu mỗi lượt. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 1 | Giới hạn norm gradient | Ngưỡng 1 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `VOCAB_SIZE` | 40000 | Trần word vocabulary | 40.000 cân bằng độ phủ nội dung Wikipedia với embedding trainable; char-CNN bổ sung phần từ hiếm. | Tăng giảm UNK nhưng thêm nhiều embedding ít dữ liệu; giảm tăng phụ thuộc char features. |
| `MAX_CONTEXT_TOKENS` | 384 | Độ dài window | 384 giữ một vùng context đủ rộng trong chi phí attention context–question cố định. | Tăng tốn nhớ và tính toán; giảm tạo nhiều windows hoặc mất liên hệ xa. |
| `WINDOW_STEP` | 256 | Bước trượt window | 256 tạo overlap 128 với window384, lớn hơn answer cap64, giảm nguy cơ cắt ngang một gold span vừa cap. | Tăng giảm overlap/nhanh hơn nhưng dễ mất span; giảm tăng số windows và chi phí. |
| `MAX_QUESTION_TOKENS` | 64 | Trần question | 64 kiểm soát chiều tương tác attention; đây là cap công khai, không sửa câu hỏi gold. | Tăng giữ câu hỏi dài hơn nhưng đắt; giảm có thể mất điều kiện quan trọng. |
| `MAX_ANSWER_TOKENS` | 64 | Trần joint span | 64 bao phủ độ dài gold tối đa đã đo theo regex ở khảo sát trước và tránh span quá dài. | Tăng mở nhiều span sai ứng viên; giảm có thể loại câu trả lời hợp lệ. |
| `MAX_WORD_CHARS` | 32 | Trần char feature | 32 giới hạn chi phí char-CNN cho nhiều token/window; không bỏ token context. | Tăng giữ thêm dạng từ dài; giảm có thể mất hậu tố/biến thể. |
| `HIDDEN_DIM` | 128 | Hidden mỗi chiều | 128 kết hợp word/char và attention nhưng giữ dung lượng vừa cho học từ đầu. | Tăng đắt/quá khớp; giảm có thể giảm khả năng khớp context–question. |
| `EPOCHS` | 20 | Trần ngân sách huấn luyện | 20 là trần khi mỗi câu hỏi có thể tạo nhiều windows; đánh giá EM validation thay vì coi số lượt lớn tự động tốt hơn. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 5 | Dừng sớm theo validation | 5 epoch không cải thiện là dung sai đề xuất cho metric exact_match; chưa được sweep để chứng minh giá trị này tốt nhất. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |

Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
