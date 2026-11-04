# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

Bài toán **image captioning** yêu cầu nhìn một ảnh rồi viết một câu mô tả nội dung chính. Đầu vào chỉ có ảnh; đầu ra là một caption tiếng Anh, không phải danh sách nhãn đối tượng. Ví dụ minh họa định dạng: ảnh một người đạp xe → “A person is riding a bicycle.”

Metric chính **CIDEr** thưởng các cụm từ phù hợp với nhiều mô tả của con người và giảm ảnh hưởng của cụm quá phổ biến. Điểm càng cao càng tốt; baseline dùng điểm raw của công cụ COCO, không tự nhân 100. Một ảnh có nhiều cách mô tả đúng, vì vậy đánh giá phải giữ đủ references của ảnh.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[COCO Captions 2014 qua CocoCaptions](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html) có 82.783 ảnh train2014 và 40.504 ảnh val2014 theo thông tin nguồn đã ghi nhận. Mỗi ảnh có nhiều caption. Baseline học từ train2014; chia val2014 theo image ID và seed cố định thành 5.000 ảnh validation, phần còn lại 35.504 ảnh held-out có nhãn. Đây không phải COCO hidden test hoặc Karpathy split.

Cùng một ảnh có thể tạo nhiều cặp train, nhưng mọi caption của ảnh phải cùng split. Khi đánh giá, mỗi ảnh chỉ sinh một dự đoán và dùng toàn bộ captions gốc làm references. Khó khăn là vật thể nhỏ, quan hệ giữa các vật và tên hiếm; một câu trôi chảy vẫn có thể mô tả chi tiết không tồn tại. Dự án đã khảo sát annotation và mẫu ảnh, chưa quét toàn bộ ảnh hay chứng minh không có ảnh gần trùng giữa các ID.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| CNN + RNN, vector ảnh cố định | Nhẹ và dễ huấn luyện | Một vector khó giữ đủ chi tiết cho từng từ. |
| Encoder ảnh + attention + LSTM | Mỗi bước sinh từ có thể nhìn vùng ảnh phù hợp | Tốn thêm tính toán; attention không tự bảo đảm mô tả đúng thực tế. |
| Encoder ảnh + Transformer decoder scratch | Có thể học quan hệ chuỗi mạnh hơn | Nhiều tham số và chi phí; không có bằng chứng ở đây rằng vượt LSTM trong cùng ngân sách. |

## 3. Tiền xử lý dữ liệu tối ưu

1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh.

2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng.

3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp.

4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch**. Encoder dùng torchvision/resnet50 với `ResNet50_Weights.IMAGENET1K_V2`; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản.

Ảnh có cấu trúc thị giác phong phú nên encoder đã học đặc trưng ảnh là điểm khởi đầu hợp lý, còn attention giúp tránh ép mọi vật vào một vector. Decoder và tokenizer đều học mới: ImageNet classifier không tự hoàn thành nhiệm vụ sinh caption, phù hợp chính sách pretrained theo thành phần. Giới hạn còn lại là caption chung chung, OOV và hallucination; CIDEr cao cũng không tự chứng minh mọi chi tiết đều đúng.

[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.

Đây là lựa chọn thiết kế hiện có trong `baseline.ipynb`, được đề xuất có cơ sở theo dữ liệu và chính sách mô hình. Chưa có so sánh huấn luyện đầy đủ giữa các ứng viên để gọi là mô hình tốt nhất đã đo. Mục tiêu thiết kế là một GPU 32 GB với cấu hình cố định, chưa phải chứng nhận bộ nhớ hoặc thời gian chạy thực tế.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `BATCH_SIZE` | 32 | Số mẫu mỗi bước | 32 cân bằng số cặp caption mỗi bước với feature ảnh256 và decoder512. | Tăng giảm nhiễu gradient nhưng tốn nhớ; giảm tiết kiệm nhớ và có thể làm cập nhật nhiễu hơn. |
| `LEARNING_RATE` | 0.0003 | Độ lớn cập nhật | 3e-4 dành cho attention/LSTM/head học mới, cao hơn LR1e-5 của layer4 để không ép encoder pretrained thay đổi cùng mức. | Tăng có thể học nhanh nhưng dao động/phân kỳ; giảm ổn định hơn nhưng hội tụ chậm. |
| `BACKBONE_LEARNING_RATE` | 1e-05 | Fine-tune encoder ảnh | Nhỏ hơn LR phần mới để giữ feature pretrained trong khi thích nghi ảnh đúng miền. | Tăng dễ phá feature cũ; giảm làm encoder thích nghi chậm hoặc gần frozen. |
| `DROPOUT` | 0.3 | Điều chuẩn activation | 0.3 là điều chuẩn đề xuất vì mô hình đa nguồn/decoder có capacity lớn hoặc tập source hạn chế. | Tăng giảm đồng thích nghi nhưng có thể underfit; giảm giữ tín hiệu nhưng dễ quá khớp. |
| `WEIGHT_DECAY` | 0.0001 | Điều chuẩn trọng số | 1e-4 phạt nhẹ độ lớn trọng số; đây là mức khởi đầu, chưa có sweep xác nhận tốt nhất. | Tăng hạn chế trọng số mạnh hơn nhưng có thể underfit; giảm có thể tăng quá khớp. |
| `GRAD_CLIP` | 5 | Giới hạn norm gradient | Ngưỡng 5 hạn chế gradient lớn bất thường trong học chuỗi/decoder; không thay cho chọn LR đúng. | Tăng ít chặn hơn; giảm chặn nhiều và có thể làm học quá chậm. |
| `IMAGE_SIZE` | 256 | Chi tiết thị giác | 256 là điểm cân bằng đề xuất giữa giữ đối tượng sau letterbox và chi phí feature map ResNet50. | Tăng giữ chi tiết nhưng tăng bộ nhớ/tính toán; giảm dễ mất vật thể nhỏ. |
| `VOCAB_SIZE` | 10000 | Trần vocabulary caption | 10.000 giới hạn head sinh từ trong corpus caption có nhiều từ lặp; vocabulary vẫn chỉ lấy từ train. | Tăng giảm UNK nhưng tăng tham số và từ hiếm ít được học; giảm mất mô tả chi tiết. |
| `MIN_WORD_FREQUENCY` | 3 | Lọc từ hiếm khi fit | Ngưỡng 3 giảm head cho các dạng từ rất ít xuất hiện; đây là giả thuyết điều chuẩn cần validation. | Tăng làm nhiều từ thành UNK; giảm tăng độ phủ nhưng thêm nhiễu/tham số. |
| `MAX_TOKENS` | 40 | Trần caption train và decoding | 40 token gồm giới hạn EOS phù hợp mục tiêu mô tả một câu ngắn; không cắt gold khi chấm. | Tăng có thể bao phủ caption dài nhưng chậm hơn; giảm có thể bỏ thông tin đích. |
| `HIDDEN_DIM` | 512 | Dung lượng decoder | 512 lớn hơn embedding/memory 256 để kết hợp ngữ cảnh từ với vùng ảnh. | Tăng có thể quá khớp và tốn thời gian; giảm dễ sinh câu chung chung. |
| `EPOCHS` | 20 | Trần ngân sách huấn luyện | 20 giới hạn ngân sách khi mỗi ảnh train có nhiều caption và backbone chỉ fine-tune một phần; chưa đo số epoch hội tụ. | Tăng chỉ hữu ích nếu validation còn cải thiện; giảm có thể dừng trước khi học đủ. |
| `PATIENCE` | 5 | Dừng sớm theo validation | 5 epoch không cải thiện là dung sai đề xuất cho metric cider; chưa được sweep để chứng minh giá trị này tốt nhất. | Tăng tốn thời gian khi đã quá khớp; giảm dễ dừng do dao động nhất thời. |

Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
