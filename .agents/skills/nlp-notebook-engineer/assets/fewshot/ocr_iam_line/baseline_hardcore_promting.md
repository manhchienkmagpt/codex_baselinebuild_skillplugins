# Prompt hardcore: OCR IAM-line — bản tối ưu

<!-- prompt-guide
Format: 4
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
Constraint workflow: extract-before-edit
Reference workflow: report-first
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Lượt phân tích dùng Code cần sửa, Code liên quan, Nguồn tham khảo và context. Lượt sửa chỉ cần Code cần sửa, kết quả phân tích và context; gửi thêm nguồn khi còn điều chưa rõ. Module mới dùng ý tưởng, nguồn tham khảo và context trực tiếp. Tham số trong nguồn là gợi ý để chọn phù hợp. Chỉ tóm tắt ở bước có liên kết dùng tiếp; không giới hạn cứng số dòng/token của bản tóm tắt. Nếu cả phiên không đủ chỗ, chia theo thành phần hoàn chỉnh.

## Bước 1: Load dữ liệu và chia train/validation/test

<!-- module
ID: s01
Group: data
Kind: code
Depends: none
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Reference report: huong_giai_quyet.md#lines=5-5
Reference report: huong_giai_quyet.md#lines=7-7
Reference report: huong_giai_quyet.md#lines=11-11
Reference report: huong_giai_quyet.md#lines=13-13
Reference code: baseline.ipynb#cell=5; lines=1-1
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=22-24
Reference code: baseline.ipynb#cell=5; lines=38-39
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Đầu vào là ảnh **một dòng chữ viết tay tiếng Anh**, đầu ra là chuỗi chữ trong dòng đó. Ví dụ minh họa định dạng: ảnh ghi “hello world” → text “hello world”. Đây không phải OCR tiếng Việt hoặc đọc cả trang với phân tích bố cục.
```

Trích báo cáo:

```markdown
Metric chính **CER** là tổng số thao tác thêm/xóa/thay ký tự cần để biến dự đoán thành gold, chia tổng ký tự gold; WER tính tương tự theo từ. Cả hai càng thấp càng tốt và có thể lớn hơn 1 nếu nhiều lỗi. Tính trên toàn split, không trung bình đơn giản CER từng batch.
```

Trích báo cáo:

```markdown
[Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) có image và text, số dòng theo card đã ghi nhận là 6.482/976/2.915 train/validation/test. Mẫu nguồn có chiều cao 128; đã khảo sát mẫu ảnh/nhãn, chưa quét toàn bộ phân bố tỷ lệ ảnh/độ dài hoặc xác minh trùng người viết. HF schema không có writer_id nên giữ split nguồn và không khẳng định writer-disjoint.
```

Trích báo cáo:

```markdown
Ảnh có độ rộng thay đổi, chữ viết nối nét, nền và phong cách người viết khác nhau. Resize mọi dòng về một hình vuông dễ làm méo hoặc mất ký tự; việc biến ảnh thành một chuỗi đủ dài quan trọng hơn chỉ tăng số lớp CNN. Alphabet phải fit train để không dùng transcript test bổ sung nhãn; ký tự chưa gặp vẫn phải được tính lỗi khi đánh giá.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
```

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
SOURCE = {'row': 1, 'task': 'OCR dòng chữ viết tay tiếng Anh', 'dataset': 'Teklia/IAM-line', 'revision': 'fbdad97500ce54635c0d1ba306bf535cb40656cf', 'url': 'https://huggingface.co/datasets/Teklia/IAM-line'}
DATASET_ID = SOURCE["dataset"]
DATASET_REVISION = SOURCE["revision"]
```

```python
PRIMARY_METRIC = "cer"
DIRECTION = "min"
```

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s01) và context ở trên.

```text
Tạo module theo ý tưởng:
Load dữ liệu và chia tập theo mô tả bài toán.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.
```

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 2: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh

<!-- module
ID: s02
Group: preprocessing
Kind: code
Depends: s01
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=8; lines=12-13
Target: baseline.ipynb#cell=10; lines=2-5
Target: baseline.ipynb#cell=10; lines=11-23
Target: baseline.ipynb#cell=10; lines=53-56
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Nguồn tham khảo

<a id="references-s02"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn.
```

Trích báo cáo:

```markdown
2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học.
```

Trích báo cáo:

```markdown
3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference.
```

Trích báo cáo:

```markdown
4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s02) và context ở trên.

```text
Tạo module theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 3: Tiền xử lí dữ liệu — Dataset và batch

<!-- module
ID: s03
Group: preprocessing
Kind: code
Depends: s01, s02
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=25-52
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn.
```

Trích báo cáo:

```markdown
2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học.
```

Trích báo cáo:

```markdown
3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference.
```

Trích báo cáo:

```markdown
4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s03) và context ở trên.

```text
Tạo module theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]
```

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 4: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=12; lines=3-33
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: architecture
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Code liên quan

Giao diện hiện có để nối với module mới: RawImageTextModel: __init__(); forward(rows).

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC**. Encoder torchvision/resnet18 dùng `ResNet18_Weights.IMAGENET1K_V1`, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. CTC cộng xác suất nhiều alignment hợp lệ, không cần biết ký tự nằm ở pixel nào; greedy decoder gộp lặp và bỏ blank theo độ dài ảnh thật.
```

Trích báo cáo:

```markdown
Giữ tỷ lệ và stride nhỏ theo chiều ngang xử lý trực tiếp yêu cầu độ dài CTC. BiLSTM có thể hỗ trợ nét chữ mơ hồ dựa trên hai phía, encoder pretrained chỉ cung cấp feature ảnh; alphabet và recognizer học mới. Không dùng full pretrained OCR model hoặc language decoder. Viết tay khác miền ImageNet và không có language model nên vẫn có thể sai từ hiếm.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s04) và context ở trên.

```text
Tạo module theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]
```

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 5: Siêu tham số

<!-- module
ID: s05
Group: hyperparameters
Kind: code
Depends: s03, s04
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=64-64
Reference code: baseline.ipynb#cell=5; lines=28-29
Reference code: baseline.ipynb#cell=5; lines=32-39
Reference missing: no
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s04)

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.
```

Trích báo cáo:

```markdown
Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BATCH_SIZE = 8
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 5.0
PRIMARY_METRIC = "cer"
DIRECTION = "min"
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s05) và context ở trên.

```text
Tạo module theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Tóm tắt context để dùng tiếp

<a id="context-s05"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 6: Loss, optimizer và training loop

<!-- module
ID: s06
Group: training
Kind: code
Depends: s01, s03, s04, s05
Budget: input=650; output=950; reserve=200
Summary: no
Constraints: no
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=64-64
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=38-39
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config uses: splits, preprocess_2, architecture, training_config
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s04)
- [Siêu tham số](#context-s05)

### Nguồn tham khảo

<a id="references-s06"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC**. Encoder torchvision/resnet18 dùng `ResNet18_Weights.IMAGENET1K_V1`, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. CTC cộng xác suất nhiều alignment hợp lệ, không cần biết ký tự nằm ở pixel nào; greedy decoder gộp lặp và bỏ blank theo độ dài ảnh thật.
```

Trích báo cáo:

```markdown
Giữ tỷ lệ và stride nhỏ theo chiều ngang xử lý trực tiếp yêu cầu độ dài CTC. BiLSTM có thể hỗ trợ nét chữ mơ hồ dựa trên hai phía, encoder pretrained chỉ cung cấp feature ảnh; alphabet và recognizer học mới. Không dùng full pretrained OCR model hoặc language decoder. Viết tay khác miền ImageNet và không có language model nên vẫn có thể sai từ hiếm.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Trích báo cáo:

```markdown
Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.
```

Trích báo cáo:

```markdown
Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
PRIMARY_METRIC = "cer"
DIRECTION = "min"
```

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s06) và context ở trên.

```text
Tạo module theo ý tưởng:
Bổ sung huấn luyện cho mô hình và dữ liệu đã có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]

Context s05 — Siêu tham số:
[Dán tóm tắt s05]
```
