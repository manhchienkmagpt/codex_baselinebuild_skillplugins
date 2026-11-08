# Prompt hardcore: Sửa ngữ pháp CoEdIT GEC — bản tối ưu

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
Reference code: baseline.ipynb#cell=5; lines=21-21
Reference code: baseline.ipynb#cell=5; lines=24-26
Reference code: baseline.ipynb#cell=5; lines=45-45
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
**Grammatical error correction** là sửa lỗi ngữ pháp trong câu mà giữ ý gốc. Đầu vào là câu tiếng Anh có thể sai; đầu ra là câu đã sửa. Ví dụ minh họa: “He go to school.” → “He goes to school.” Mô hình cần biết cả khi nào nên sửa và khi nào nên giữ nguyên.
```

Trích báo cáo:

```markdown
Metric chính **ERRANT F0.5** so các phép sửa với tham chiếu và ưu tiên precision hơn recall: sửa đúng quan trọng hơn sửa thật nhiều. Điểm càng cao càng tốt. GLEU bổ sung đối chiếu câu nguồn/đích. Một câu có nhiều cách sửa đúng, nên phải giữ mọi correction tham chiếu đã có. ERRANT F0.5 và GLEU được báo theo thang 0–100 trong baseline.
```

Trích báo cáo:

```markdown
[grammarly/coedit](https://huggingface.co/datasets/grammarly/coedit) chứa nhiều nhiệm vụ chỉnh sửa; chỉ chọn task=gec. Khảo sát đã ghi nhận 19.823 hàng GEC train trong 69.071 hàng train và 485 hàng GEC validation. Sau nhóm câu nguồn có 19.696 train sources, trong đó 60 nhóm có hai corrections khác nhau; official validation có 485 nguồn.
```

Trích báo cáo:

```markdown
Các src chứa instruction ở đầu câu. Nếu không tách đúng, model có thể học cách lặp prompt thay vì sửa ngữ pháp. Baseline nhận diện 27 prefix đã kiểm tra, nhóm theo câu nguồn sau bỏ prefix và giữ các target khác nhau. Tách 10% nhóm train thành 1.970 validation nội bộ, học từ 17.726 nhóm; 485 official validation làm held-out. Mọi correction của cùng nguồn ở cùng tập; không trộn thêm BEA/CoNLL để làm giàu train.
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
SOURCE = {'row': 4, 'task': 'Grammar Error Correction', 'dataset': 'grammarly/coedit', 'revision': 'e9a255c33ef910bc33a9d2b522653fa87521583e', 'url': 'https://huggingface.co/datasets/grammarly/coedit'}
```

```python
SOURCE["data_mode"] = DATA_MODE
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'spaCy', 'model_id': 'en_core_web_sm', 'revision': '3.8.0', 'training': 'frozen', 'sha256': '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85', 'purpose': 'ERRANT annotation, không tạo task output'}]
VAL_FRACTION = 0.1
```

```python
PRIMARY_METRIC = "errant_f05"
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
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=11; lines=3-13
Target: baseline.ipynb#cell=11; lines=19-21
Target: baseline.ipynb#cell=11; lines=46-50
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=11; lines=2-2
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
Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.
```

Trích báo cáo:

```markdown
1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu.
```

Trích báo cáo:

```markdown
2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer.
```

Trích báo cáo:

```markdown
3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm.
```

Trích báo cáo:

```markdown
4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu.
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
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=11; lines=23-45
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=11; lines=2-2
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
Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.
```

Trích báo cáo:

```markdown
1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu.
```

Trích báo cáo:

```markdown
2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer.
```

Trích báo cáo:

```markdown
3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm.
```

Trích báo cáo:

```markdown
4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu.
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
Target: baseline.ipynb#cell=13; lines=1-63
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference missing: no
Config owns: encoder
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Code liên quan

Giao diện hiện có để nối với module mới: RawTextModel: __init__(); forward(texts).

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **BiGRU encoder + attention cộng + GRU decoder với pointer-copy**, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold.
```

Trích báo cáo:

```markdown
Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Attention giảm nút thắt hidden cuối. Copy vẫn có thể giữ nguyên lỗi, còn decoder có thể sửa quá mức; ERRANT F0.5 validation là tín hiệu lựa chọn. spaCy en_core_web_sm 3.8.0 chỉ là evaluator ERRANT đã được cho phép riêng, không cung cấp feature hay sinh output cho task model.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
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
Target: baseline.ipynb#cell=13; lines=1-63
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Encoder độc lập](#context-s04)

### Code liên quan

Giao diện hiện có để nối với module mới: RawTextModel: __init__(); forward(texts).

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **BiGRU encoder + attention cộng + GRU decoder với pointer-copy**, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold.
```

Trích báo cáo:

```markdown
Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Attention giảm nút thắt hidden cuối. Copy vẫn có thể giữ nguyên lỗi, còn decoder có thể sửa quá mức; ERRANT F0.5 validation là tín hiệu lựa chọn. spaCy en_core_web_sm 3.8.0 chỉ là evaluator ERRANT đã được cho phép riêng, không cung cấp feature hay sinh output cho task model.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
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
Reference code: baseline.ipynb#cell=5; lines=27-28
Reference code: baseline.ipynb#cell=5; lines=38-45
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
EVAL_BATCH_SIZE = 16
```

```python
LABEL_SMOOTHING = 0.0
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "errant_f05"
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
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=67-67
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=5; lines=45-45
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference code: baseline.ipynb#cell=17; lines=8-14
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
Chọn **BiGRU encoder + attention cộng + GRU decoder với pointer-copy**, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold.
```

Trích báo cáo:

```markdown
Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Attention giảm nút thắt hidden cuối. Copy vẫn có thể giữ nguyên lỗi, còn decoder có thể sửa quá mức; ERRANT F0.5 validation là tín hiệu lựa chọn. spaCy en_core_web_sm 3.8.0 chỉ là evaluator ERRANT đã được cho phép riêng, không cung cấp feature hay sinh output cho task model.
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
PRIMARY_METRIC = "errant_f05"
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

```python
def compute_loss(model,batch):
    logp=model(batch["source"],batch["lengths"],batch["target"][:,:-1])
    target=batch["target"][:,1:]
    losses=-logp.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    if LABEL_SMOOTHING:losses=(1-LABEL_SMOOTHING)*losses+LABEL_SMOOTHING*(-logp.mean(-1))
    mask=target!=PAD_ID
    return losses[mask].mean()
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
