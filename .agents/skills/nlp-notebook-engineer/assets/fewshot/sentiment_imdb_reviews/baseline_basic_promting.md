# Prompt basic: Phân loại cảm xúc IMDb — optimized

<!-- prompt-guide
Format: 4
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
Constraint workflow: extract-before-edit
Reference workflow: report-first
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Lượt phân tích dùng Code cần sửa, Code liên quan, Nguồn tham khảo và context. Lượt sửa chỉ cần Code cần sửa, kết quả phân tích và context; gửi thêm nguồn khi còn điều chưa rõ. Module mới dùng ý tưởng, nguồn tham khảo và context trực tiếp. Tham số trong nguồn là gợi ý để chọn phù hợp. Chỉ tóm tắt ở bước có liên kết dùng tiếp; không giới hạn cứng số dòng/token của bản tóm tắt. Nếu cả phiên không đủ chỗ, chia theo thành phần hoàn chỉnh.

## Bước 1: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh

<!-- module
ID: s01
Group: preprocessing
Kind: code
Depends: none
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=5; lines=1-4
Source: baseline_basic.ipynb#cell=7; lines=1-9
Related source: baseline_basic.ipynb#cell=3; lines=64-64
Related source: baseline_basic.ipynb#cell=5; lines=75-75
Related source: baseline_basic.ipynb#cell=5; lines=85-86
Related source: baseline_basic.ipynb#cell=9; lines=12-31
Related source: baseline_basic.ipynb#cell=11; lines=17-17
Target: baseline.ipynb#cell=5; lines=1-4
Target: baseline.ipynb#cell=7; lines=1-23
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=3; lines=13-15
Reference code: baseline.ipynb#cell=3; lines=59-60
Reference code: baseline.ipynb#cell=3; lines=74-74
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

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

### Prompt trích ràng buộc

<a id="constraints-s01"></a>

Gửi cùng [nguồn tham khảo](#references-s01), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s01) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def normalize_text(text):
    text = html.unescape(unicodedata.normalize("NFKC", text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    return " ".join(text.split())
```

```python
def build_features():
    return CountVectorizer(tokenizer=str.split, token_pattern=None,
        lowercase=False, min_df=1, max_features=WORD_MAX_FEATURES, dtype=np.int64)

features = build_features()
X_train = features.fit_transform(train_text)
X_val = features.transform(val_text)
ratio = np.ones(X_train.shape[1], dtype=np.float32)
print("Kích thước đặc trưng train/validation:", X_train.shape, X_val.shape)
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
WORD_MAX_FEATURES = 40000
```

```python
texts_all = [normalize_text(r["text"]) for r in raw_train]
```

```python
train_text = [raw_train[i]["text"] for i in train_idx]
val_text = [raw_train[i]["text"] for i in val_idx]
```

```python
for parameter in PARAMETERS:
    classifier = make_classifier(parameter)
    # Không chấp nhận nghiệm SVM chưa hội tụ rồi âm thầm báo kết quả.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(X_train, y_train)
    val_pred = classifier.predict(X_val)
    scores = classification_metrics(y_val, val_pred)
    history.append({"parameter": float(parameter), **scores})
    print("Tham số:", parameter, "| validation:", scores)
    if scores[PRIMARY_METRIC] > best_score:
        best_score = scores[PRIMARY_METRIC]
        # Chỉ lưu các thành phần đã fit cần để inference; cấu hình/provenance ở notebook.
        joblib.dump({
            "features": features,
            "ratio": ratio,
            "classifier": classifier,
            "label_names": LABEL_NAMES,
            "validation": scores,
        }, CHECKPOINT_PATH)
```

```python
test_groups = {sha(normalize_text(t).casefold()) for t in test_text if normalize_text(t)}
```

Giao diện dùng trong notebook: make_classifier(parameter).

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

## Bước 2: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s02
Group: model
Kind: code
Depends: s01
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=9; lines=2-3
Related source: baseline_basic.ipynb#cell=9; lines=12-31
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

- [Biểu diễn và căn chỉnh](#context-s01)

### Nguồn tham khảo

<a id="references-s02"></a>

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

### Prompt trích ràng buộc

<a id="constraints-s02"></a>

Gửi cùng [nguồn tham khảo](#references-s02), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s02) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def make_classifier(parameter):
    return MultinomialNB(alpha=parameter)
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
for parameter in PARAMETERS:
    classifier = make_classifier(parameter)
    # Không chấp nhận nghiệm SVM chưa hội tụ rồi âm thầm báo kết quả.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(X_train, y_train)
    val_pred = classifier.predict(X_val)
    scores = classification_metrics(y_val, val_pred)
    history.append({"parameter": float(parameter), **scores})
    print("Tham số:", parameter, "| validation:", scores)
    if scores[PRIMARY_METRIC] > best_score:
        best_score = scores[PRIMARY_METRIC]
        # Chỉ lưu các thành phần đã fit cần để inference; cấu hình/provenance ở notebook.
        joblib.dump({
            "features": features,
            "ratio": ratio,
            "classifier": classifier,
            "label_names": LABEL_NAMES,
            "validation": scores,
        }, CHECKPOINT_PATH)
```

Giao diện dùng trong notebook: make_classifier(parameter).

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

## Bước 3: Siêu tham số

<!-- module
ID: s03
Group: hyperparameters
Kind: code
Depends: s01, s02
Budget: input=1000; output=600; reserve=200
Summary: no
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=3; lines=67-67
Source: baseline_basic.ipynb#cell=3; lines=70-71
Reference report: huong_giai_quyet.md#lines=56-56
Reference report: huong_giai_quyet.md#lines=70-70
Reference code: baseline.ipynb#cell=3; lines=68-68
Reference code: baseline.ipynb#cell=3; lines=71-74
Reference missing: no
Config owns: training_config
Config uses: preprocess_1, architecture
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)
- [Model và giao diện sử dụng](#context-s02)

### Nguồn tham khảo

<a id="references-s03"></a>

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

### Prompt trích ràng buộc

<a id="constraints-s03"></a>

Gửi cùng [nguồn tham khảo](#references-s03), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s03) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]

Context s02 — Model và giao diện sử dụng:
[Dán tóm tắt s02]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
C_VALUES = (0.25, 1.0, 4.0)
```

```python
MAX_ITER = 10000
TOLERANCE = 1e-4
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

Giao diện dùng trong notebook: make_classifier(parameter).
