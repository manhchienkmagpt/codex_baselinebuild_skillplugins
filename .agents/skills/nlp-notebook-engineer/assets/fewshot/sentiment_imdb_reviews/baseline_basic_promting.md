# Prompt basic: Phân loại cảm xúc IMDb — optimized

<!-- prompt-guide
Format: 4
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Chỉ tóm tắt ở bước có liên kết dùng tiếp; web không đặt trần output riêng nên giới hạn độ dài tóm tắt là yêu cầu mềm.

## Bước 1: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh

<!-- module
ID: s01
Group: preprocessing
Kind: code
Depends: none
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=5; lines=1-4
Source: baseline_basic.ipynb#cell=7; lines=1-9
Target: baseline.ipynb#cell=5; lines=1-4
Target: baseline.ipynb#cell=7; lines=1-23
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: normalize_text, build_features.
Ý tưởng tiền xử lí: 1. NFKC, HTML unescape, đổi thẻ br thành khoảng trắng rồi gộp whitespace. Không xóa stopword hoặc từ phủ định như “not”, vì chúng có thể đảo sentiment. 2. Chia nhóm train/validation trước khi fit; lowercase do vectorizer thực hiện. Không cắt review theo độ dài nên những từ sentiment ở cuối vẫn có thể đóng góp. 3. Fit TF-IDF word unigram/bigram và char_wb n-gram 3–5 trên train. Word bigram giữ cụm như “not good”; character features giúp biến thể từ/cách viết. Giới hạn vocabulary và min_df để kiểm soát nhiễu/chi phí. 4. Tính binary NB log-count ratio từ document presence của train với smoothing, rồi nhân đặc trưng TF-IDF trước LinearSVC. Ratio có thể âm nên không đưa ma trận này vào MultinomialNB. Chỉ transform val/test; mẫu rỗng vẫn được chấm.
Nguồn: "https://www.tensorflow.org/datasets/catalog/imdb_reviews".
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=9; lines=2-3
Target: baseline.ipynb#cell=9; lines=2-4
Config owns: architecture
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn NB-SVM sparse: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def make_classifier(parameter):
    return MultinomialNB(alpha=parameter)
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Source: baseline_basic.ipynb#cell=3; lines=67-67
Source: baseline_basic.ipynb#cell=3; lines=70-71
Config owns: training_config
Config uses: preprocess_1, architecture
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)
- [Model và giao diện sử dụng](#context-s02)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng.
Phương pháp: Chọn NB-SVM sparse: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features. Metric: macro_f1. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]

Context s02 — Model và giao diện sử dụng:
[Dán tóm tắt s02]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
C_VALUES = (0.25, 1.0, 4.0)
```

```python
MAX_ITER = 10000
TOLERANCE = 1e-4
```
