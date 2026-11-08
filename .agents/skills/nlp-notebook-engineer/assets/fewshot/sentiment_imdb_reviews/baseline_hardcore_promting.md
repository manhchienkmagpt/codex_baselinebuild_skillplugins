# Prompt hardcore: Phân loại cảm xúc IMDb — optimized

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
Reference code: baseline.ipynb#cell=3; lines=13-15
Reference code: baseline.ipynb#cell=3; lines=59-60
Reference code: baseline.ipynb#cell=3; lines=74-74
Reference missing: no
Config owns: splits
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Bài toán phân loại cảm xúc nhận một review phim tiếng Anh và trả nhãn **negative** hoặc **positive**. Ví dụ minh họa: “The acting is excellent.” → positive. Không có lớp neutral; nhận xét trung tính hoặc review tiếng Việt nằm ngoài phạm vi dữ liệu đang dùng.
```

Trích báo cáo:

```markdown
Metric chính **Macro F1** lấy trung bình F1 của hai lớp, nên cần nhận đúng cả khen lẫn chê; điểm 0–1 càng cao càng tốt. Accuracy được báo bổ sung. Nhãn 0 là neg, 1 là pos; không nhầm Macro F1 với F1 chỉ của positive.
```

Trích báo cáo:

```markdown
[IMDb Reviews trong TFDS](https://www.tensorflow.org/datasets/catalog/imdb_reviews) có text:str và label:int. Theo nguồn/khảo sát trước, có 25.000 train và 25.000 test, mỗi split cân bằng hai lớp; tập unsupervised 50.000 không được dùng trong phương pháp này. Dữ liệu là review phân cực mạnh nên không đại diện mọi sắc thái cảm xúc.
```

Trích báo cáo:

```markdown
Nội dung review có HTML, từ phủ định, cách viết đa dạng và có thể dài. Baseline tách validation khoảng 20% bằng StratifiedGroupKFold theo nhóm văn bản chuẩn hóa để bản sao không vượt biên train/validation. Lần khảo sát trước ghi nhận 123 nhóm nội dung không rỗng giao nhau giữa official train/test; giữ test chuẩn nhưng không coi đây là đánh giá hoàn toàn sạch leakage. Không có mã phim ổn định để chia theo phim.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
SOURCE = 'https://www.tensorflow.org/datasets/catalog/imdb_reviews'
LABEL_NAMES = ['neg', 'pos']
```

```python
PRIMARY_METRIC = "macro_f1"
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
Target: baseline.ipynb#cell=5; lines=1-4
Target: baseline.ipynb#cell=7; lines=1-23
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=3; lines=60-60
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
TF-IDF biến văn bản thành vector thưa: mỗi chiều là một từ/cụm, trọng số giảm cho từ xuất hiện quá phổ biến trong nhiều văn bản. N-gram là nhóm các đơn vị liền nhau, chẳng hạn bigram là hai từ liên tiếp.
```

Trích báo cáo:

```markdown
1. NFKC, HTML unescape, đổi thẻ br thành khoảng trắng rồi gộp whitespace. Không xóa stopword hoặc từ phủ định như “not”, vì chúng có thể đảo sentiment.
```

Trích báo cáo:

```markdown
2. Chia nhóm train/validation trước khi fit; lowercase do vectorizer thực hiện. Không cắt review theo độ dài nên những từ sentiment ở cuối vẫn có thể đóng góp.
```

Trích báo cáo:

```markdown
3. Fit TF-IDF word unigram/bigram và char_wb n-gram 3–5 trên train. Word bigram giữ cụm như “not good”; character features giúp biến thể từ/cách viết. Giới hạn vocabulary và min_df để kiểm soát nhiễu/chi phí.
```

Trích báo cáo:

```markdown
4. Tính binary NB log-count ratio từ document presence của train với smoothing, rồi nhân đặc trưng TF-IDF trước LinearSVC. Ratio có thể âm nên không đưa ma trận này vào MultinomialNB. Chỉ transform val/test; mẫu rỗng vẫn được chấm.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
LABEL_NAMES = ['neg', 'pos']
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

## Bước 3: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=9; lines=2-4
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference report: huong_giai_quyet.md#lines=43-43
Reference report: huong_giai_quyet.md#lines=50-50
Reference code: baseline.ipynb#cell=3; lines=14-15
Reference code: baseline.ipynb#cell=3; lines=60-60
Reference missing: no
Config owns: architecture
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s02)

### Code liên quan

Giao diện hiện có để nối với module mới: RawClassifier: __init__(num_classes); forward(texts).

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **NB-SVM sparse**: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features.
```

Trích báo cáo:

```markdown
Review sentiment thường có tín hiệu từ/cụm rõ, nên một bộ phân loại tuyến tính với biểu diễn tốt là lựa chọn có căn cứ cho khoảng 20 nghìn mẫu train học và chính sách scratch. Không cần ép dùng neural network vì có GPU; phương pháp này chạy CPU. Giới hạn là bag-of-features không biểu diễn đầy đủ mỉa mai và cấu trúc dài.
```

Trích báo cáo:

```markdown
**Bằng chứng đã được ghi nhận:** lần chạy đầy đủ trước đây của optimized trên cùng phương pháp/giao thức (20.000 train học / 5.000 validation / 25.000 test) chọn C=4 trong ba ứng viên theo Macro F1 validation. Kết quả này được giữ từ báo cáo trước, không phải lần huấn luyện mới trong lần viết này.
```

Trích báo cáo:

```markdown
Đây là kết quả một seed/giao thức đã ghi nhận, không phải chứng minh tốt nhất trên mọi mô hình, mọi C hoặc nguồn dữ liệu khác. Không dùng test để chọn lại C.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
LABEL_NAMES = ['neg', 'pos']
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s03) và context ở trên.

```text
Tạo module theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

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

## Bước 4: Siêu tham số

<!-- module
ID: s04
Group: hyperparameters
Kind: code
Depends: s02, s03
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Reference report: huong_giai_quyet.md#lines=56-56
Reference report: huong_giai_quyet.md#lines=70-70
Reference code: baseline.ipynb#cell=3; lines=68-68
Reference code: baseline.ipynb#cell=3; lines=71-74
Reference missing: no
Config owns: training_config
Config uses: preprocess_1, architecture
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s02)
- [Model và giao diện sử dụng](#context-s03)

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Các giá trị dưới đây khớp `baseline.ipynb`. C=4 là tốt nhất đã đo **trong lưới ba C ở lần chạy đã ghi nhận**; các trần features và tham số solver còn lại là cấu hình có cơ sở, chưa được chứng minh tối ưu riêng bằng ablation.
```

Trích báo cáo:

```markdown
Cần giữ nguyên split/metric khi đánh giá thêm một thay đổi như bỏ char hoặc thay min_df, để biết lợi ích đến từ đâu. Không gọi MAX_ITER hoặc việc chạy CPU là nguyên nhân bảo đảm F1 tối ưu.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
C_VALUES = (0.25, 1.0, 4.0)
```

```python
MAX_ITER = 10000
TOLERANCE = 1e-4
USE_NB_RATIO = True
PRIMARY_METRIC = "macro_f1"
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s04) và context ở trên.

```text
Tạo module theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]

Context s03 — Model và giao diện sử dụng:
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

## Bước 5: Loss, optimizer và training loop

<!-- module
ID: s05
Group: training
Kind: code
Depends: s01, s02, s03, s04
Budget: input=650; output=950; reserve=200
Summary: no
Constraints: no
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference report: huong_giai_quyet.md#lines=43-43
Reference report: huong_giai_quyet.md#lines=50-50
Reference report: huong_giai_quyet.md#lines=56-56
Reference report: huong_giai_quyet.md#lines=70-70
Reference code: baseline.ipynb#cell=3; lines=14-15
Reference code: baseline.ipynb#cell=3; lines=60-60
Reference code: baseline.ipynb#cell=3; lines=74-74
Reference missing: no
Config uses: splits, preprocess_1, architecture, training_config
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)
- [Model và giao diện sử dụng](#context-s03)
- [Siêu tham số](#context-s04)

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **NB-SVM sparse**: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features.
```

Trích báo cáo:

```markdown
Review sentiment thường có tín hiệu từ/cụm rõ, nên một bộ phân loại tuyến tính với biểu diễn tốt là lựa chọn có căn cứ cho khoảng 20 nghìn mẫu train học và chính sách scratch. Không cần ép dùng neural network vì có GPU; phương pháp này chạy CPU. Giới hạn là bag-of-features không biểu diễn đầy đủ mỉa mai và cấu trúc dài.
```

Trích báo cáo:

```markdown
**Bằng chứng đã được ghi nhận:** lần chạy đầy đủ trước đây của optimized trên cùng phương pháp/giao thức (20.000 train học / 5.000 validation / 25.000 test) chọn C=4 trong ba ứng viên theo Macro F1 validation. Kết quả này được giữ từ báo cáo trước, không phải lần huấn luyện mới trong lần viết này.
```

Trích báo cáo:

```markdown
Đây là kết quả một seed/giao thức đã ghi nhận, không phải chứng minh tốt nhất trên mọi mô hình, mọi C hoặc nguồn dữ liệu khác. Không dùng test để chọn lại C.
```

Trích báo cáo:

```markdown
Các giá trị dưới đây khớp `baseline.ipynb`. C=4 là tốt nhất đã đo **trong lưới ba C ở lần chạy đã ghi nhận**; các trần features và tham số solver còn lại là cấu hình có cơ sở, chưa được chứng minh tối ưu riêng bằng ablation.
```

Trích báo cáo:

```markdown
Cần giữ nguyên split/metric khi đánh giá thêm một thay đổi như bỏ char hoặc thay min_df, để biết lợi ích đến từ đâu. Không gọi MAX_ITER hoặc việc chạy CPU là nguyên nhân bảo đảm F1 tối ưu.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
LABEL_NAMES = ['neg', 'pos']
```

```python
PRIMARY_METRIC = "macro_f1"
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s05) và context ở trên.

```text
Tạo module theo ý tưởng:
Bổ sung huấn luyện cho mô hình và dữ liệu đã có.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]

Context s03 — Model và giao diện sử dụng:
[Dán tóm tắt s03]

Context s04 — Siêu tham số:
[Dán tóm tắt s04]
```
