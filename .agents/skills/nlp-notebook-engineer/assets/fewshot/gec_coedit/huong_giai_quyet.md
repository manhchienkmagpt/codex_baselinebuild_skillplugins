# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

**Grammatical error correction** là sửa lỗi ngữ pháp trong câu mà giữ ý gốc. Đầu vào là câu tiếng Anh có thể sai; đầu ra là câu đã sửa. Ví dụ minh họa: “He go to school.” → “He goes to school.” Mô hình cần biết cả khi nào nên sửa và khi nào nên giữ nguyên.

Metric chính **ERRANT F0.5** so các phép sửa với tham chiếu và ưu tiên precision hơn recall: sửa đúng quan trọng hơn sửa thật nhiều. Điểm càng cao càng tốt. GLEU bổ sung đối chiếu câu nguồn/đích. Một câu có nhiều cách sửa đúng, nên phải giữ mọi correction tham chiếu đã có. ERRANT F0.5 và GLEU được báo theo thang 0–100 trong baseline.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[grammarly/coedit](https://huggingface.co/datasets/grammarly/coedit) chứa nhiều nhiệm vụ chỉnh sửa; chỉ chọn task=gec. Khảo sát đã ghi nhận 19.823 hàng GEC train trong 69.071 hàng train và 485 hàng GEC validation. Sau nhóm câu nguồn có 19.696 train sources, trong đó 60 nhóm có hai corrections khác nhau; official validation có 485 nguồn.

Các src chứa instruction ở đầu câu. Nếu không tách đúng, model có thể học cách lặp prompt thay vì sửa ngữ pháp. Baseline nhận diện 27 prefix đã kiểm tra, nhóm theo câu nguồn sau bỏ prefix và giữ các target khác nhau. Tách 10% nhóm train thành 1.970 validation nội bộ, học từ 17.726 nhóm; 485 official validation làm held-out. Mọi correction của cùng nguồn ở cùng tập; không trộn thêm BEA/CoNLL để làm giàu train.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| Seq2seq RNN context cố định | Học trực tiếp câu sai → câu đúng | Dễ quên tên/từ nguồn và sửa quá mức khi dữ liệu hạn chế. |
| BiGRU + attention + pointer-copy | Có đường giữ lại token nguồn và tập trung vào vị trí sửa | Copy không tự quyết định sửa đúng; vẫn có thể overcorrect. |
| Transformer scratch | Có thể học phụ thuộc toàn câu | Cần dữ liệu/điều chuẩn tốt; không dùng pretrained grammar model theo chính sách hiện tại. |

## 3. Tiền xử lý dữ liệu tối ưu

Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.

1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu.

2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer.

3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm.

4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **BiGRU encoder + attention cộng + GRU decoder với pointer-copy**, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold.

Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Attention giảm nút thắt hidden cuối. Copy vẫn có thể giữ nguyên lỗi, còn decoder có thể sửa quá mức; ERRANT F0.5 validation là tín hiệu lựa chọn. spaCy en_core_web_sm 3.8.0 chỉ là evaluator ERRANT đã được cho phép riêng, không cung cấp feature hay sinh output cho task model.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 32 | Số mẫu mỗi bước | 32 cân bằng số câu GEC mỗi bước với encoder/decoder và pointer attention. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.001 | Độ lớn cập nhật | 0.001 cho GRU và cổng copy mới trên tập GEC vừa; clipping1 và validation F0.5 kiểm việc học quá mạnh/sửa quá mức. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `DROPOUT` | 0.3 | Điều chuẩn activation | 0.3 là điều chuẩn đề xuất vì mô hình đa nguồn/decoder có capacity lớn hoặc tập source hạn chế. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 1 | Giới hạn norm gradient | Ngưỡng 1 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `VOCAB_SIZE` | 8000 | Trần BPE | 8.000 cùng byte fallback hạn chế head và hỗ trợ từ nguồn lỗi/hiếm trên khoảng 18 nghìn source học. | Tăng làm chuỗi ngắn hơn nhưng thêm token hiếm khó học; giảm làm chuỗi dài hơn. |
| `MAX_SOURCE_TOKENS` | 160 | Trần câu nguồn | 160 kiểm soát chi phí theo token BPE; đây là giới hạn model chứ không phải bảo đảm mọi câu đều vừa. | Tăng giữ câu dài nhưng tốn nhớ; giảm làm mất phần cần sửa. |
| `MAX_TARGET_TOKENS` | 160 | Trần target train | 160 giữ budget tương ứng source cho sửa câu; gold metric vẫn nguyên vẹn. | Tăng tốn bước decoder; giảm có thể cắt correction hợp lệ. |
| `HIDDEN_DIM` | 192 | Hidden mỗi chiều encoder | 192 tạo memory/decoder 384, cân bằng context với tập GEC không lớn. | Tăng dễ quá khớp và chậm; giảm có thể quên quan hệ ngữ pháp. |
| `BEAM_SIZE` | 3 | Số giả thuyết giải mã | 3 là đề xuất cân bằng tìm chuỗi tốt hơn greedy với chi phí nhỏ hơn beam rộng. | Tăng chậm, chưa chắc tăng F0.5; giảm nhanh hơn nhưng bỏ bớt phương án. |
| `LENGTH_PENALTY` | 0.6 | Điều chỉnh xu hướng chuỗi ngắn | 0.6 điều chỉnh điểm beam theo độ dài để tránh ưu tiên quá mức câu ngắn. | Tăng có thể thiên câu dài hơn; giảm dễ bỏ bớt nội dung, cần validation. |
| `EPOCHS` | 40 | Trần ngân sách huấn luyện | 40 là trần học từ đầu trên khoảng17.7 nghìn source train; checkpoint theo F0.5 để không kéo dài việc overcorrect. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 8 | Dừng sớm theo validation | 8 epoch không cải thiện là dung sai đề xuất cho metric errant_f05; chưa được sweep để chứng minh giá trị này tốt nhất. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |

Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
