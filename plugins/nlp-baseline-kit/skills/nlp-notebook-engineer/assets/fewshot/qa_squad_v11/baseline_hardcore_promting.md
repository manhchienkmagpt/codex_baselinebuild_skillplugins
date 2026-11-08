# Prompt hardcore: SQuAD v1.1 — QA trích xuất span — bản tối ưu

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
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=5; lines=21-23
Reference code: baseline.ipynb#cell=5; lines=26-26
Reference code: baseline.ipynb#cell=5; lines=44-44
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
**Extractive question answering** tìm câu trả lời bằng cách chọn một đoạn liên tiếp trong context đã cho. Ví dụ minh họa: context “Paris is the capital of France.”, question “What is the capital of France?” → “Paris”. Model dự đoán vị trí bắt đầu/kết thúc, không tự tạo kiến thức ngoài context.
```

Trích báo cáo:

```markdown
Metric chính **Exact Match (EM)** yêu cầu text trả lời trùng gold sau chuẩn hóa chính thức; token F1 bổ sung mức khớp một phần. Cả hai càng cao càng tốt. Context đưa vào model giữ nguyên offset; chuẩn hóa đánh giá chỉ áp dụng khi so text, không sửa vị trí annotation. EM và token F1 được báo theo thang 0–100 trong baseline.
```

Trích báo cáo:

```markdown
[SQuAD v1.1 qua TFDS](https://www.tensorflow.org/datasets/catalog/squad) có id/title/context/question/answers, mỗi answer gồm text và answer_start. Khảo sát JSON gốc trước đây ghi nhận 87.599 train questions trên 442 titles và 10.570 dev questions trên 48 titles; kiểm offset gold không thấy mismatch. Có 262 annotation train và 76 dev có biên nằm giữa token regex.
```

Trích báo cáo:

```markdown
Nhiều câu hỏi chia sẻ cùng đoạn/bài Wikipedia, nên chia ngẫu nhiên câu hỏi dễ rò rỉ context. Baseline lấy 10% train titles làm validation nội bộ và giữ official dev làm held-out vì test không có nhãn công khai. Các answer có biên giữa token là giới hạn biểu diễn thật của head token-span; không tự sửa gold hoặc bỏ chúng để nâng EM.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
```

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
SOURCE = {'row': 3, 'task': 'Extractive QA', 'dataset': 'SQuAD v1.1', 'tfds_builder': 'squad/v1.1:3.0.0', 'original_hint': 'squad/v1.1.0 (đã sửa config thành v1.1, builder version3.0.0)', 'url': 'https://www.tensorflow.org/datasets/catalog/squad'}
DATA_MODE = "tfds"
SOURCE["data_mode"] = DATA_MODE
```

```python
VAL_FRACTION = 0.10
```

```python
PRIMARY_METRIC = "exact_match"
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
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
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=10; lines=5-25
Target: baseline.ipynb#cell=10; lines=32-43
Target: baseline.ipynb#cell=10; lines=69-77
Target: baseline.ipynb#cell=10; lines=94-100
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=2-2
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
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
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
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=2-2
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
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
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
Target: baseline.ipynb#cell=12; lines=1-41
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: architecture
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Code liên quan

Giao diện hiện có để nối với module mới: RawSpanModel: __init__(); forward(rows).

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads**. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; attention giúp từng token context đọc phần question liên quan. Toàn bộ mô hình học từ đầu, không dùng BERT/tokenizer pretrained.
```

Trích báo cáo:

```markdown
Câu hỏi và đoạn văn cần so khớp ở mức token nên tương tác hai chiều có căn cứ hơn chỉ một vector question. Char-CNN giảm phụ thuộc vào word vocabulary giới hạn. Bản hiện tại vẫn không thể chọn chính xác biên giữa một token; đây là trần biểu diễn trên một số gold EM. Lựa chọn là thiết kế scratch có cơ sở, chưa chứng minh tối ưu toàn bộ SQuAD.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
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
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=66-66
Reference code: baseline.ipynb#cell=5; lines=36-44
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
BATCH_SIZE = 24
EVAL_BATCH_SIZE = 24
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "exact_match"
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
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=66-66
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=5; lines=44-44
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference code: baseline.ipynb#cell=15; lines=17-21
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
Chọn **word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads**. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; attention giúp từng token context đọc phần question liên quan. Toàn bộ mô hình học từ đầu, không dùng BERT/tokenizer pretrained.
```

Trích báo cáo:

```markdown
Câu hỏi và đoạn văn cần so khớp ở mức token nên tương tác hai chiều có căn cứ hơn chỉ một vector question. Char-CNN giảm phụ thuộc vào word vocabulary giới hạn. Bản hiện tại vẫn không thể chọn chính xác biên giữa một token; đây là trần biểu diễn trên một số gold EM. Lựa chọn là thiết kế scratch có cơ sở, chưa chứng minh tối ưu toàn bộ SQuAD.
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
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
PRIMARY_METRIC = "exact_match"
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
```

```python
def compute_loss(model,batch):
    start,end=model(batch);targets=batch["targets"]
    loss=nn.functional.cross_entropy(start,targets[:,0],reduction="none")+nn.functional.cross_entropy(end,targets[:,1],reduction="none")
    weights=torch.where(targets[:,0]==0,torch.full_like(loss,NEGATIVE_WINDOW_WEIGHT),torch.ones_like(loss))
    return (loss*weights).sum()/weights.sum()
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
