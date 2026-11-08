# Prompt hardcore: COCO Captions 2014 — bản tối ưu

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
Reference code: baseline.ipynb#cell=5; lines=22-22
Reference code: baseline.ipynb#cell=5; lines=40-41
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Bài toán **image captioning** yêu cầu nhìn một ảnh rồi viết một câu mô tả nội dung chính. Đầu vào chỉ có ảnh; đầu ra là một caption tiếng Anh, không phải danh sách nhãn đối tượng. Ví dụ minh họa định dạng: ảnh một người đạp xe → “A person is riding a bicycle.”
```

Trích báo cáo:

```markdown
Metric chính **CIDEr** thưởng các cụm từ phù hợp với nhiều mô tả của con người và giảm ảnh hưởng của cụm quá phổ biến. Điểm càng cao càng tốt; baseline dùng điểm raw của công cụ COCO, không tự nhân 100. Một ảnh có nhiều cách mô tả đúng, vì vậy đánh giá phải giữ đủ references của ảnh.
```

Trích báo cáo:

```markdown
[COCO Captions 2014 qua CocoCaptions](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html) có 82.783 ảnh train2014 và 40.504 ảnh val2014 theo thông tin nguồn đã ghi nhận. Mỗi ảnh có nhiều caption. Baseline học từ train2014; chia val2014 theo image ID và seed cố định thành 5.000 ảnh validation, phần còn lại 35.504 ảnh held-out có nhãn. Đây không phải COCO hidden test hoặc Karpathy split.
```

Trích báo cáo:

```markdown
Cùng một ảnh có thể tạo nhiều cặp train, nhưng mọi caption của ảnh phải cùng split. Khi đánh giá, mỗi ảnh chỉ sinh một dự đoán và dùng toàn bộ captions gốc làm references. Khó khăn là vật thể nhỏ, quan hệ giữa các vật và tên hiếm; một câu trôi chảy vẫn có thể mô tả chi tiết không tồn tại. Dự án đã khảo sát annotation và mẫu ảnh, chưa quét toàn bộ ảnh hay chứng minh không có ảnh gần trùng giữa các ID.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
```

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
SOURCE = {'row': 2, 'task': 'Image Captioning', 'dataset': 'MS COCO Captions 2014', 'url': 'https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html', 'reader': 'torchvision.datasets.CocoCaptions', 'split_protocol': 'train2014; val2014 chia image-disjoint validation5000/held-out còn lại'}
```

```python
PRIMARY_METRIC = "cider"
DIRECTION = "max"
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
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
Target: baseline.ipynb#cell=10; lines=2-8
Target: baseline.ipynb#cell=10; lines=14-27
Target: baseline.ipynb#cell=10; lines=54-55
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
1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh.
```

Trích báo cáo:

```markdown
2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng.
```

Trích báo cáo:

```markdown
3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp.
```

Trích báo cáo:

```markdown
4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
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
Target: baseline.ipynb#cell=10; lines=29-53
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
1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh.
```

Trích báo cáo:

```markdown
2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng.
```

Trích báo cáo:

```markdown
3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp.
```

Trích báo cáo:

```markdown
4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
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

## Bước 4: Xử lí mô hình — Encoder độc lập

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=12; lines=2-57
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: encoder
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
Chọn **ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch**. Encoder dùng torchvision/resnet50 với `ResNet50_Weights.IMAGENET1K_V2`; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản.
```

Trích báo cáo:

```markdown
Ảnh có cấu trúc thị giác phong phú nên encoder đã học đặc trưng ảnh là điểm khởi đầu hợp lý, còn attention giúp tránh ép mọi vật vào một vector. Decoder và tokenizer đều học mới: ImageNet classifier không tự hoàn thành nhiệm vụ sinh caption, phù hợp chính sách pretrained theo thành phần. Giới hạn còn lại là caption chung chung, OOV và hallucination; CIDEr cao cũng không tự chứng minh mọi chi tiết đều đúng.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s04) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Encoder độc lập hoàn chỉnh để mô hình chính sử dụng.

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

## Bước 5: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s05
Group: model
Kind: code
Depends: s03, s04
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=12; lines=2-57
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Encoder độc lập](#context-s04)

### Code liên quan

Giao diện hiện có để nối với module mới: RawImageTextModel: __init__(); forward(rows).

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch**. Encoder dùng torchvision/resnet50 với `ResNet50_Weights.IMAGENET1K_V2`; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản.
```

Trích báo cáo:

```markdown
Ảnh có cấu trúc thị giác phong phú nên encoder đã học đặc trưng ảnh là điểm khởi đầu hợp lý, còn attention giúp tránh ép mọi vật vào một vector. Decoder và tokenizer đều học mới: ImageNet classifier không tự hoàn thành nhiệm vụ sinh caption, phù hợp chính sách pretrained theo thành phần. Giới hạn còn lại là caption chung chung, OOV và hallucination; CIDEr cao cũng không tự chứng minh mọi chi tiết đều đúng.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s05) và context ở trên.

```text
Tạo module theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Encoder độc lập:
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

## Bước 6: Siêu tham số

<!-- module
ID: s06
Group: hyperparameters
Kind: code
Depends: s03, s05
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=67-67
Reference code: baseline.ipynb#cell=5; lines=24-25
Reference code: baseline.ipynb#cell=5; lines=34-41
Reference missing: no
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s05)

### Nguồn tham khảo

<a id="references-s06"></a>

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
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.0003
BACKBONE_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 5.0
PRIMARY_METRIC = "cider"
DIRECTION = "max"
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s06) và context ở trên.

```text
Tạo module theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s05 — Model và giao diện sử dụng:
[Dán tóm tắt s05]
```

### Tóm tắt context để dùng tiếp

<a id="context-s06"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 7: Loss, optimizer và training loop

<!-- module
ID: s07
Group: training
Kind: code
Depends: s01, s03, s05, s06
Budget: input=650; output=950; reserve=200
Summary: no
Constraints: no
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=67-67
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=40-41
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference code: baseline.ipynb#cell=15; lines=7-10
Reference missing: no
Config uses: splits, preprocess_2, architecture, training_config
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s05)
- [Siêu tham số](#context-s06)

### Nguồn tham khảo

<a id="references-s07"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch**. Encoder dùng torchvision/resnet50 với `ResNet50_Weights.IMAGENET1K_V2`; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản.
```

Trích báo cáo:

```markdown
Ảnh có cấu trúc thị giác phong phú nên encoder đã học đặc trưng ảnh là điểm khởi đầu hợp lý, còn attention giúp tránh ép mọi vật vào một vector. Decoder và tokenizer đều học mới: ImageNet classifier không tự hoàn thành nhiệm vụ sinh caption, phù hợp chính sách pretrained theo thành phần. Giới hạn còn lại là caption chung chung, OOV và hallucination; CIDEr cao cũng không tự chứng minh mọi chi tiết đều đúng.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
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
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
PRIMARY_METRIC = "cider"
DIRECTION = "max"
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

```python
def compute_loss(model,batch):
    tokens=batch["tokens"]
    logits=model(batch["images"],tokens[:,:-1])
    return nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),tokens[:,1:].reshape(-1),ignore_index=PAD_ID)
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s07) và context ở trên.

```text
Tạo module theo ý tưởng:
Bổ sung huấn luyện cho mô hình và dữ liệu đã có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s05 — Model và giao diện sử dụng:
[Dán tóm tắt s05]

Context s06 — Siêu tham số:
[Dán tóm tắt s06]
```
