# Prompt hardcore: FUNSD — nhận dạng thực thể tài liệu — bản tối ưu

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
Reference code: baseline.ipynb#cell=5; lines=27-27
Reference code: baseline.ipynb#cell=5; lines=40-41
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Bài toán đang triển khai là **nhận dạng thực thể trên tài liệu đã có OCR words và tọa độ**, không phải tự đọc chữ từ ảnh. Model nhận ảnh trang, chuỗi từ và các hộp bao; trả các cụm HEADER, QUESTION hoặc ANSWER. Ví dụ minh họa: trường “Name: Alice” có thể được gán “Name” là QUESTION và “Alice” là ANSWER. Việc nối hai cụm thành cặp key–value là nhiệm vụ khác.
```

Trích báo cáo:

```markdown
Metric chính **entity span micro-F1** đòi cả ranh giới cụm từ và loại thực thể đúng, càng cao càng tốt. Nguồn HF này không có gold links, nên K–V linking F1 không xác định; không được coi khoảng cách hình học là nhãn liên kết thật.
```

Trích báo cáo:

```markdown
[nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) có 149 tài liệu train và 50 test với image, words, bboxes, ner_tags. Box dùng hệ tọa độ 0..1000; có bảy BIO tags: O và B/I cho HEADER, QUESTION, ANSWER. Baseline chia 149 tài liệu thành 119 train/30 validation theo seed cố định, giữ 50 test.
```

Trích báo cáo:

```markdown
Dữ liệu rất ít ở cấp trang, bố cục và hình thức chữ có thể mang tín hiệu quan trọng ngoài nội dung từ. Giữ mọi token của một trang cùng split; so trùng bằng nội dung words+boxes thay vì chỉ dùng ID vì ID có thể được đánh lại theo split. Đã khảo sát schema và mẫu ảnh/nhãn, chưa quét toàn bộ trang hoặc biết phân bố đủ mọi kiểu form.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
```

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
SOURCE = {'row': 4, 'task': 'FUNSD conditional entity recognition, không có gold K-V links', 'dataset': 'nielsr/funsd', 'revision': '7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5', 'url': 'https://huggingface.co/datasets/nielsr/funsd'}
DATASET_ID = SOURCE["dataset"]
DATASET_REVISION = SOURCE["revision"]
```

```python
VAL_FRACTION = 0.20
```

```python
PRIMARY_METRIC = "entity_f1"
DIRECTION = "max"
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
Target: baseline.ipynb#cell=10; lines=1-4
Target: baseline.ipynb#cell=10; lines=9-11
Target: baseline.ipynb#cell=10; lines=47-47
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
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
1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn.
```

Trích báo cáo:

```markdown
2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu.
```

Trích báo cáo:

```markdown
3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác.
```

Trích báo cáo:

```markdown
4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
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
Target: baseline.ipynb#cell=10; lines=13-46
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
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
1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn.
```

Trích báo cáo:

```markdown
2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu.
```

Trích báo cáo:

```markdown
3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác.
```

Trích báo cáo:

```markdown
4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
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

## Bước 4: Xử lí mô hình — Thành phần EntityCRF

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=12; lines=1-45
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference missing: no
Config owns: component_EntityCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Code liên quan

Giao diện hiện có để nối với module mới: RawDocumentModel: __init__(); forward(rows).

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF**. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans.
```

Trích báo cáo:

```markdown
Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; CRF giảm chuỗi nhãn sai cấu trúc. Tất cả khởi tạo mới, không pretrained. Mô hình không giải quyết OCR đầu vào hoặc linking; độ phức tạp là một rủi ro trên tập nhỏ nên validation theo trang rất quan trọng.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s04) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Thành phần EntityCRF hoàn chỉnh để mô hình chính sử dụng.

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
Target: baseline.ipynb#cell=13; lines=2-42
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, component_EntityCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Thành phần EntityCRF](#context-s04)

### Code liên quan

Giao diện hiện có để nối với module mới: RawDocumentModel: __init__(); forward(rows).

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF**. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans.
```

Trích báo cáo:

```markdown
Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; CRF giảm chuỗi nhãn sai cấu trúc. Tất cả khởi tạo mới, không pretrained. Mô hình không giải quyết OCR đầu vào hoặc linking; độ phức tạp là một rủi ro trên tập nhỏ nên validation theo trang rất quan trọng.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s05) và context ở trên.

```text
Tạo module theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Thành phần EntityCRF:
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
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=62-62
Reference code: baseline.ipynb#cell=5; lines=30-31
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
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
```

```python
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "entity_f1"
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
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=62-62
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=40-41
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
Chọn **word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF**. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans.
```

Trích báo cáo:

```markdown
Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; CRF giảm chuỗi nhãn sai cấu trúc. Tất cả khởi tạo mới, không pretrained. Mô hình không giải quyết OCR đầu vào hoặc linking; độ phức tạp là một rủi ro trên tập nhỏ nên validation theo trang rất quan trọng.
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
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
PRIMARY_METRIC = "entity_f1"
DIRECTION = "max"
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
