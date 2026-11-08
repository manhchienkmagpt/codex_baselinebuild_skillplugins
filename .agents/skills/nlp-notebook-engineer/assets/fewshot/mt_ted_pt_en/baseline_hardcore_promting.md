# Prompt hardcore: Dịch máy TED PT→EN — bản tối ưu

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
Bài toán dịch máy nhận một câu **tiếng Bồ Đào Nha** và tạo câu **tiếng Anh** giữ nghĩa. Ví dụ minh họa: “Olá, mundo.” → “Hello, world.” Đầu ra phải tự sinh từ source, không được nhìn câu dịch tham chiếu lúc inference.
```

Trích báo cáo:

```markdown
Metric chính **SacreBLEU** đo khớp cụm từ với reference theo tokenizer 13a, điểm 0–100 càng cao càng tốt; chrF++ bổ sung đánh giá theo ký tự/từ. COMET nếu bật chỉ chấm sau sinh và có pretrained evaluator được cho phép riêng, không tham gia model dịch hoặc chọn candidates trên test.
```

Trích báo cáo:

```markdown
[TED HRLR Translate pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate) có 51.785/1.193/1.803 cặp train/validation/test theo builder nguồn đã ghi nhận. Đường phục hồi hiện có dùng [neulab/ted_hrlr](https://huggingface.co/datasets/neulab/ted_hrlr), cùng corpus PT→EN; mỗi Parquet có thêm một hàng rỗng cần lọc như builder. Không coi đây là đổi sang một bộ TED/ngôn ngữ khác.
```

Trích báo cáo:

```markdown
Khoảng 52 nghìn cặp là quy mô hạn chế cho Transformer học hoàn toàn từ đầu; tên riêng, từ hiếm và khác trật tự ngôn ngữ cần được giữ. Sau NFC/gộp whitespace, khảo sát đã phát hiện source trùng giữa các split. Baseline giữ toàn test, loại source validation trùng test rồi loại train trùng val/test; protocol này khác benchmark native chưa loại trùng. Không có talk/speaker ID đủ để chứng minh disjoint theo người nói.
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
SOURCE = {'row': 1, 'task': 'Machine Translation PT→EN', 'dataset': 'TED HRLR Translate pt_to_en', 'tfds_builder': 'ted_hrlr_translate/pt_to_en:1.0.0', 'url': 'https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate', 'recovery_dataset': 'neulab/ted_hrlr', 'recovery_revision': 'cf21722111c95aa73c30e13df430032db463736d'}
```

```python
SOURCE["data_mode"] = DATA_MODE
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'Unbabel/wmt22-comet-da', 'revision': '2760a223ac957f30acfb18c8aa649b01cf1d75f2', 'training': 'frozen', 'purpose': 'COMET; chỉ final scoring'}, {'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'FacebookAI/xlm-roberta-large', 'revision': 'c23d21b0620b635a76227c604d44e43a9f0ee389', 'training': 'frozen', 'purpose': 'COMET encoder config/tokenizer; weights trong metric checkpoint'}]
VAL_FRACTION = 0.1
```

```python
PRIMARY_METRIC = "bleu"
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
1. Giữ đúng cặp pt/en và split corpus, lọc cặp rỗng; NFC và gộp khoảng trắng nhưng giữ case/dấu câu. Loại trùng source chéo split theo giao thức đã nêu trước khi fit.
```

Trích báo cáo:

```markdown
2. Fit chung BPE SentencePiece 8.000 cho source/target chỉ bằng train, có byte fallback. Chia đơn vị dưới từ giúp biểu diễn biến thể hình thái và từ hiếm mà không dùng tokenizer pretrained.
```

Trích báo cáo:

```markdown
3. Cap source/target/decoding ở 128 token, thêm BOS/EOS/PAD đúng vị trí. Gold dùng chấm BLEU/chrF++ vẫn đầy đủ; theo dõi phần bị cap để hiểu giới hạn model.
```

Trích báo cáo:

```markdown
4. Mask PAD ở source/target và dùng causal mask cho decoder để không nhìn token tương lai. Loss bỏ PAD; inference beam chỉ dựa source và xác suất model.
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
1. Giữ đúng cặp pt/en và split corpus, lọc cặp rỗng; NFC và gộp khoảng trắng nhưng giữ case/dấu câu. Loại trùng source chéo split theo giao thức đã nêu trước khi fit.
```

Trích báo cáo:

```markdown
2. Fit chung BPE SentencePiece 8.000 cho source/target chỉ bằng train, có byte fallback. Chia đơn vị dưới từ giúp biểu diễn biến thể hình thái và từ hiếm mà không dùng tokenizer pretrained.
```

Trích báo cáo:

```markdown
3. Cap source/target/decoding ở 128 token, thêm BOS/EOS/PAD đúng vị trí. Gold dùng chấm BLEU/chrF++ vẫn đầy đủ; theo dõi phần bị cap để hiểu giới hạn model.
```

Trích báo cáo:

```markdown
4. Mask PAD ở source/target và dùng causal mask cho decoder để không nhìn token tương lai. Loss bỏ PAD; inference beam chỉ dựa source và xác suất model.
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
Target: baseline.ipynb#cell=13; lines=1-52
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
Chọn **Transformer encoder–decoder scratch**: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Chỉ dùng kiến trúc PyTorch khởi tạo mới; không text weights/embedding/tokenizer pretrained.
```

Trích báo cáo:

```markdown
Attention toàn chuỗi phù hợp việc đổi trật tự và nối các phần xa nhau; cấu hình 256/bốn lớp giữ quy mô thấp hơn Transformer lớn trước dữ liệu hạn chế. Byte fallback và label smoothing hỗ trợ từ hiếm/điều chuẩn, nhưng không tạo kiến thức dịch sẵn. GRU attention có thể cạnh tranh hơn trên dữ liệu nhỏ; lựa chọn hiện tại chưa được chứng minh thắng các ứng viên trong một so sánh đầy đủ.
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
Target: baseline.ipynb#cell=13; lines=1-52
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
Chọn **Transformer encoder–decoder scratch**: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Chỉ dùng kiến trúc PyTorch khởi tạo mới; không text weights/embedding/tokenizer pretrained.
```

Trích báo cáo:

```markdown
Attention toàn chuỗi phù hợp việc đổi trật tự và nối các phần xa nhau; cấu hình 256/bốn lớp giữ quy mô thấp hơn Transformer lớn trước dữ liệu hạn chế. Byte fallback và label smoothing hỗ trợ từ hiếm/điều chuẩn, nhưng không tạo kiến thức dịch sẵn. GRU attention có thể cạnh tranh hơn trên dữ liệu nhỏ; lựa chọn hiện tại chưa được chứng minh thắng các ứng viên trong một so sánh đầy đủ.
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
Reference report: huong_giai_quyet.md#lines=70-70
Reference code: baseline.ipynb#cell=5; lines=27-28
Reference code: baseline.ipynb#cell=5; lines=38-47
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
Đánh giá độ nhạy nên giữ protocol cố định và thay một yếu tố như warmup, vocab hoặc beam; chỉ chốt bằng BLEU validation. Không mở rộng tokenizer/text pretraining hoặc dùng COMET/test để lựa chọn trái giao thức.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 16
```

```python
LABEL_SMOOTHING = 0.1
EPOCHS = 45
PATIENCE = 8
LEARNING_RATE = 0.0005
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "bleu"
RUN_COMET = True
COMET_BATCH_SIZE = 8
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
Reference report: huong_giai_quyet.md#lines=70-70
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
Chọn **Transformer encoder–decoder scratch**: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Chỉ dùng kiến trúc PyTorch khởi tạo mới; không text weights/embedding/tokenizer pretrained.
```

Trích báo cáo:

```markdown
Attention toàn chuỗi phù hợp việc đổi trật tự và nối các phần xa nhau; cấu hình 256/bốn lớp giữ quy mô thấp hơn Transformer lớn trước dữ liệu hạn chế. Byte fallback và label smoothing hỗ trợ từ hiếm/điều chuẩn, nhưng không tạo kiến thức dịch sẵn. GRU attention có thể cạnh tranh hơn trên dữ liệu nhỏ; lựa chọn hiện tại chưa được chứng minh thắng các ứng viên trong một so sánh đầy đủ.
```

Trích báo cáo:

```markdown
Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.
```

Trích báo cáo:

```markdown
Đánh giá độ nhạy nên giữ protocol cố định và thay một yếu tố như warmup, vocab hoặc beam; chỉ chốt bằng BLEU validation. Không mở rộng tokenizer/text pretraining hoặc dùng COMET/test để lựa chọn trái giao thức.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
PRIMARY_METRIC = "bleu"
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
