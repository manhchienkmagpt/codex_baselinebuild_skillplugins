# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

Bài toán dịch máy nhận một câu **tiếng Bồ Đào Nha** và tạo câu **tiếng Anh** giữ nghĩa. Ví dụ minh họa: “Olá, mundo.” → “Hello, world.” Đầu ra phải tự sinh từ source, không được nhìn câu dịch tham chiếu lúc inference.

Metric chính **SacreBLEU** đo khớp cụm từ với reference theo tokenizer 13a, điểm 0–100 càng cao càng tốt; chrF++ bổ sung đánh giá theo ký tự/từ. COMET nếu bật chỉ chấm sau sinh và có pretrained evaluator được cho phép riêng, không tham gia model dịch hoặc chọn candidates trên test.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[TED HRLR Translate pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate) có 51.785/1.193/1.803 cặp train/validation/test theo builder nguồn đã ghi nhận. Đường phục hồi hiện có dùng [neulab/ted_hrlr](https://huggingface.co/datasets/neulab/ted_hrlr), cùng corpus PT→EN; mỗi Parquet có thêm một hàng rỗng cần lọc như builder. Không coi đây là đổi sang một bộ TED/ngôn ngữ khác.

Khoảng 52 nghìn cặp là quy mô hạn chế cho Transformer học hoàn toàn từ đầu; tên riêng, từ hiếm và khác trật tự ngôn ngữ cần được giữ. Sau NFC/gộp whitespace, khảo sát đã phát hiện source trùng giữa các split. Baseline giữ toàn test, loại source validation trùng test rồi loại train trùng val/test; protocol này khác benchmark native chưa loại trùng. Không có talk/speaker ID đủ để chứng minh disjoint theo người nói.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| RNN encoder–decoder context cố định | Ít tham số và trực quan | Khó giữ nghĩa/trật tự ở câu dài. |
| GRU có attention | Inductive bias chuỗi có thể hợp dữ liệu nhỏ | Khả năng song song hóa và truy cập ngữ cảnh vẫn hạn chế hơn attention toàn chuỗi. |
| Transformer encoder–decoder scratch | Self/cross-attention truy cập quan hệ xa và học trật tự dịch | Dễ quá khớp nếu cấu hình quá lớn; cần warmup/mask/điều chuẩn đúng. |

## 3. Tiền xử lý dữ liệu tối ưu

Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.

1. Giữ đúng cặp pt/en và split corpus, lọc cặp rỗng; NFC và gộp khoảng trắng nhưng giữ case/dấu câu. Loại trùng source chéo split theo giao thức đã nêu trước khi fit.

2. Fit chung BPE SentencePiece 8.000 cho source/target chỉ bằng train, có byte fallback. Chia đơn vị dưới từ giúp biểu diễn biến thể hình thái và từ hiếm mà không dùng tokenizer pretrained.

3. Cap source/target/decoding ở 128 token, thêm BOS/EOS/PAD đúng vị trí. Gold dùng chấm BLEU/chrF++ vẫn đầy đủ; theo dõi phần bị cap để hiểu giới hạn model.

4. Mask PAD ở source/target và dùng causal mask cho decoder để không nhìn token tương lai. Loss bỏ PAD; inference beam chỉ dựa source và xác suất model.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **Transformer encoder–decoder scratch**: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Chỉ dùng kiến trúc PyTorch khởi tạo mới; không text weights/embedding/tokenizer pretrained.

Attention toàn chuỗi phù hợp việc đổi trật tự và nối các phần xa nhau; cấu hình 256/bốn lớp giữ quy mô thấp hơn Transformer lớn trước dữ liệu hạn chế. Byte fallback và label smoothing hỗ trợ từ hiếm/điều chuẩn, nhưng không tạo kiến thức dịch sẵn. GRU attention có thể cạnh tranh hơn trên dữ liệu nhỏ; lựa chọn hiện tại chưa được chứng minh thắng các ứng viên trong một so sánh đầy đủ.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 64 | Số mẫu mỗi bước | 64 là đề xuất cho Transformer256 và cap128; không coi đây là phép đo vừa GPU32GB. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.0005 | Độ lớn cập nhật | 5e-4 kết hợp warmup1000/inverse-sqrt cho Transformer256 học từ đầu; chọn cả lịch LR thay vì coi riêng một con số bảo đảm ổn định. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `DROPOUT` | 0.2 | Điều chuẩn activation | 0.2 thêm điều chuẩn vừa phải cho học chuỗi mà không bỏ quá nhiều tín hiệu mỗi lượt. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 1 | Giới hạn norm gradient | Ngưỡng 1 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `VOCAB_SIZE` | 8000 | Trần BPE chung | 8.000 giữ head/embedding có kích thước vừa và chia được từ Bồ Đào Nha/Anh hiếm. | Tăng giảm độ dài chuỗi nhưng thêm token ít dữ liệu; giảm kéo dài chuỗi và chi phí attention. |
| `EMBED_DIM` | 256 | Chiều biểu diễn Transformer | 256 chia hết cho 8 heads và hạn chế capacity so với tập khoảng 52 nghìn cặp. | Tăng tốn tham số/nhớ và có thể quá khớp; giảm hạn chế biểu diễn dịch. |
| `MAX_SOURCE_TOKENS` | 128 | Trần source | 128 giới hạn attention bậc hai trong ngân sách cố định; không khẳng định bao phủ mọi câu. | Tăng giữ nhiều ngữ cảnh với chi phí attention tăng; giảm dễ mất nghĩa cuối câu. |
| `MAX_TARGET_TOKENS` | 128 | Trần target train | 128 cân bằng budget với source, giữ BOS/EOS và mask đúng. | Tăng thêm bước dự đoán; giảm có thể cắt câu dịch đích. |
| `WARMUP_STEPS` | 1000 | Lịch tăng LR ban đầu | 1.000 bước tăng dần trước lịch inverse-sqrt nhằm giảm cập nhật quá mạnh khi attention/embedding mới khởi tạo. | Tăng kéo dài giai đoạn học chậm; giảm có thể làm giai đoạn đầu kém ổn định. |
| `LABEL_SMOOTHING` | 0.1 | Điều chuẩn phân phối đích | 0.1 giảm ép toàn bộ xác suất vào đúng một token khi chỉ có một reference. | Tăng quá mức làm mềm tín hiệu dịch; giảm tăng tự tin/quá khớp. |
| `BEAM_SIZE` | 4 | Độ rộng beam | 4 khảo sát thêm phương án dịch so với greedy trong chi phí inference vừa phải. | Tăng không bảo đảm BLEU tăng và làm chậm; giảm dễ bỏ phương án tốt. |
| `LENGTH_PENALTY` | 0.6 | Điều chỉnh độ dài beam | 0.6 giảm thiên vị câu ngắn trong tổng log-probability. | Tăng có thể sinh dài hơn; giảm có thể bỏ ý, cần validation. |
| `EPOCHS` | 45 | Trần ngân sách huấn luyện | 45 cấp ngân sách cho Transformer học mới từ khoảng52 nghìn cặp; epoch tốt thực tế phải do BLEU validation quyết định. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 8 | Dừng sớm theo validation | 8 epoch không cải thiện là dung sai đề xuất cho metric bleu; chưa được sweep để chứng minh giá trị này tốt nhất. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |
| Lớp/heads/feed-forward | 4 encoder + 4 decoder; 8 heads; FF=1024 | Capacity attention | Ghép với embedding256 để có nhiều góc nhìn nhưng hạn chế quy mô trên corpus nhỏ; đúng cấu hình MODEL_CONFIG của code. | Tăng độ sâu/FF làm chậm và dễ quá khớp; giảm có thể mất khả năng biểu diễn. |

Đánh giá độ nhạy nên giữ protocol cố định và thay một yếu tố như warmup, vocab hoặc beam; chỉ chốt bằng BLEU validation. Không mở rộng tokenizer/text pretraining hoặc dùng COMET/test để lựa chọn trái giao thức.
