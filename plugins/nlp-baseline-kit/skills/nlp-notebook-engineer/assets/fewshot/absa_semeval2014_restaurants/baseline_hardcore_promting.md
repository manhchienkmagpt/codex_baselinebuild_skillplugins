# Prompt hardcore: Trích xuất aspect và phân loại polarity — bản tối ưu

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
Reference code: baseline.ipynb#cell=5; lines=1-5
Reference code: baseline.ipynb#cell=5; lines=12-12
Reference code: baseline.ipynb#cell=5; lines=27-30
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Một câu đánh giá có thể khen món ăn nhưng chê phục vụ. Bài toán cần tìm đúng cụm từ chỉ từng khía cạnh, gọi là **aspect**, rồi xác định cảm xúc dành cho cụm đó. Đầu vào là câu gốc; đầu ra là các khoảng ký tự và nhãn negative, neutral, positive hoặc conflict. Ví dụ minh họa: “Great food, slow service.” → “food”: positive; “service”: negative.
```

Trích báo cáo:

```markdown
Metric chính là **aspect span micro-F1**: cân bằng giữa việc tìm đủ aspect và tránh tìm nhầm, chỉ tính đúng khi cả đầu và cuối cụm khớp chính xác. F1 càng cao càng tốt. Polarity macro-F1 đánh giá cảm xúc trên các span có nhãn, coi bốn lớp ngang nhau; test của nguồn này không có polarity nên không được công bố polarity F1 trên test. Các F1 trong baseline dùng thang 0–1.
```

Trích báo cáo:

```markdown
[SetFit SemEval Restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) có các trường text, span, label, ordinal. Khảo sát đã ghi nhận 3.693 hàng aspect train và 1.134 hàng test, tương ứng 2.019 và 606 câu nguyên văn. Một câu có thể có nhiều hàng annotation; ordinal là lần xuất hiện của đúng chuỗi span, không phải thứ tự mọi aspect trong câu. Nhãn train gồm 2.164 positive, 805 negative, 633 neutral và 91 conflict trước gộp trùng; lớp conflict rất ít. Toàn bộ label test là chuỗi rỗng, nhưng còn span/ordinal để chấm extraction.
```

Trích báo cáo:

```markdown
Có ba annotation train kết thúc giữa một token regex. Vì vậy biểu diễn chỉ theo từ không thể khớp mọi ranh giới; dùng ký tự làm đơn vị dự đoán có căn cứ trực tiếp từ dữ liệu. Gộp annotation trùng của cùng câu, giữ nguyên nhãn nguồn; nguồn chỉ chứa câu có aspect nên khả năng nhận biết câu hoàn toàn không có aspect chưa được đại diện đầy đủ. Loại nhóm train trùng câu test theo giao thức đang dùng, rồi tách 20% train theo câu để làm validation; không chia các aspect của một câu sang hai tập.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
DATASET_ID = "tomaarsen/setfit-absa-semeval-restaurants"
DATASET_REVISION = "8885372fe73256f96bb60f65b550538ac5c26047"
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
VAL_FRACTION = 0.20
```

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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
Target: baseline.ipynb#cell=9; lines=5-31
Target: baseline.ipynb#cell=9; lines=73-74
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=27-30
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
1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc.
```

Trích báo cáo:

```markdown
2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold.
```

Trích báo cáo:

```markdown
3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó.
```

Trích báo cáo:

```markdown
4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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
Target: baseline.ipynb#cell=9; lines=34-42
Target: baseline.ipynb#cell=9; lines=45-62
Target: baseline.ipynb#cell=9; lines=65-72
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=27-30
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
1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc.
```

Trích báo cáo:

```markdown
2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold.
```

Trích báo cáo:

```markdown
3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó.
```

Trích báo cáo:

```markdown
4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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

## Bước 4: Xử lí mô hình — Thành phần LinearChainCRF

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=11; lines=1-53
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=4-5
Reference code: baseline.ipynb#cell=5; lines=27-30
Reference missing: no
Config owns: component_LinearChainCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Code liên quan

Giao diện hiện có để nối với module mới: RawAspectModel: __init__(); forward(texts).

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **embedding ký tự + embedding từ → BiLSTM hai lớp → CRF**, kèm head polarity cho mỗi span. BiLSTM đọc cả trái và phải để hiểu “slow” đang nói về “service”; CRF là bộ giải mã chọn cả chuỗi BIO hợp lệ thay vì chọn từng ký tự riêng. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán.
```

Trích báo cáo:

```markdown
Lựa chọn giải quyết đồng thời hai khó khăn đã thấy: offset không trùng biên từ và nhiều aspect/cảm xúc trong cùng câu. Trọng số loss BIO từ tần suất train giúp giảm áp đảo của O, nhưng lớp conflict vẫn thiếu dữ liệu. CRF không biểu diễn aspect lồng nhau; khảo sát nguồn chưa thấy span chồng lấn. Toàn bộ mô hình và vocabulary học từ đầu.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s04) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Thành phần LinearChainCRF hoàn chỉnh để mô hình chính sử dụng.

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
Target: baseline.ipynb#cell=12; lines=1-43
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=4-5
Reference code: baseline.ipynb#cell=5; lines=27-30
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, component_LinearChainCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Thành phần LinearChainCRF](#context-s04)

### Code liên quan

Giao diện hiện có để nối với module mới: RawAspectModel: __init__(); forward(texts).

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **embedding ký tự + embedding từ → BiLSTM hai lớp → CRF**, kèm head polarity cho mỗi span. BiLSTM đọc cả trái và phải để hiểu “slow” đang nói về “service”; CRF là bộ giải mã chọn cả chuỗi BIO hợp lệ thay vì chọn từng ký tự riêng. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán.
```

Trích báo cáo:

```markdown
Lựa chọn giải quyết đồng thời hai khó khăn đã thấy: offset không trùng biên từ và nhiều aspect/cảm xúc trong cùng câu. Trọng số loss BIO từ tần suất train giúp giảm áp đảo của O, nhưng lớp conflict vẫn thiếu dữ liệu. CRF không biểu diễn aspect lồng nhau; khảo sát nguồn chưa thấy span chồng lấn. Toàn bộ mô hình và vocabulary học từ đầu.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s05) và context ở trên.

```text
Tạo module theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Thành phần LinearChainCRF:
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
Reference report: huong_giai_quyet.md#lines=64-64
Reference code: baseline.ipynb#cell=5; lines=13-19
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
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
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
Reference report: huong_giai_quyet.md#lines=64-64
Reference code: baseline.ipynb#cell=5; lines=4-5
Reference code: baseline.ipynb#cell=5; lines=27-30
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
Chọn **embedding ký tự + embedding từ → BiLSTM hai lớp → CRF**, kèm head polarity cho mỗi span. BiLSTM đọc cả trái và phải để hiểu “slow” đang nói về “service”; CRF là bộ giải mã chọn cả chuỗi BIO hợp lệ thay vì chọn từng ký tự riêng. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán.
```

Trích báo cáo:

```markdown
Lựa chọn giải quyết đồng thời hai khó khăn đã thấy: offset không trùng biên từ và nhiều aspect/cảm xúc trong cùng câu. Trọng số loss BIO từ tần suất train giúp giảm áp đảo của O, nhưng lớp conflict vẫn thiếu dữ liệu. CRF không biểu diễn aspect lồng nhau; khảo sát nguồn chưa thấy span chồng lấn. Toàn bộ mô hình và vocabulary học từ đầu.
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
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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
