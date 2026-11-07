# Prompt hardcore: Phân loại cảm xúc IMDb — optimized

<!-- prompt-guide
Format: 4
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Chỉ tóm tắt ở bước có liên kết dùng tiếp; web không đặt trần output riêng nên giới hạn độ dài tóm tắt là yêu cầu mềm.

## Bước 1: Load dữ liệu và chia train/validation/test

<!-- module
ID: s01
Group: data
Kind: code
Depends: none
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Config owns: splits
-->

### Prompt tạo hoặc sửa module

```text
Bổ sung loader train/validation/test, giữ nguồn/schema và split chính thức hoặc quy tắc nhóm/seed của bài. Giữ ID, nhãn/references và held-out gốc; kiểm nhóm trùng, chưa fit biểu diễn.
Nguồn/schema/split: "https://www.tensorflow.org/datasets/catalog/imdb_reviews". IMDb Reviews trong TFDS có text:str và label:int. Theo nguồn/khảo sát trước, có 25.000 train và 25.000 test, mỗi split cân bằng hai lớp; tập unsupervised 50.000 không được dùng trong phương pháp này. Dữ liệu là review phân cực mạnh nên không đại diện mọi sắc thái cảm xúc. Nội dung review có HTML, từ phủ định, cách viết đa dạng và có thể dài. Baseline tách validation khoảng 20% bằng StratifiedGroupKFold theo nhóm văn bản chuẩn hóa để bản sao không vượt biên train/validation. Lần khảo sát trước ghi nhận 123 nhóm nội dung không rỗng giao nhau giữa official train/test; giữ test chuẩn nhưng không coi đây là đánh giá hoàn toàn sạch leakage. Không có mã phim ổn định để chia theo phim.
Thiết lập split cần giữ: {"SEED":42}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=5; lines=1-4
Target: baseline.ipynb#cell=7; lines=1-23
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. NFKC, HTML unescape, đổi thẻ br thành khoảng trắng rồi gộp whitespace. Không xóa stopword hoặc từ phủ định như “not”, vì chúng có thể đảo sentiment. 2. Chia nhóm train/validation trước khi fit; lowercase do vectorizer thực hiện. Không cắt review theo độ dài nên những từ sentiment ở cuối vẫn có thể đóng góp. 3. Fit TF-IDF word unigram/bigram và char_wb n-gram 3–5 trên train. Word bigram giữ cụm như “not good”; character features giúp biến thể từ/cách viết. Giới hạn vocabulary và min_df để kiểm soát nhiễu/chi phí. 4. Tính binary NB log-count ratio từ document presence của train với smoothing, rồi nhân đặc trưng TF-IDF trước LinearSVC. Ratio có thể âm nên không đưa ma trận này vào MultinomialNB. Chỉ transform val/test; mẫu rỗng vẫn được chấm.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=9; lines=2-4
Config owns: architecture
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn NB-SVM sparse: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawClassifier: __init__(num_classes); forward(texts).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]
```

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Config owns: training_config
Config uses: preprocess_1, architecture
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s02)
- [Model và giao diện sử dụng](#context-s03)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Trả cấu hình cố định để bước training dùng; chưa viết vòng train.
Phương pháp: Chọn NB-SVM sparse: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features. Metric: macro_f1. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]

Context s03 — Model và giao diện sử dụng:
[Dán tóm tắt s03]
```

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Config uses: splits, preprocess_1, architecture, training_config
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)
- [Model và giao diện sử dụng](#context-s03)
- [Siêu tham số](#context-s04)

### Prompt tạo hoặc sửa module

```text
Bổ sung huấn luyện thật vào notebook hardcore: loss đúng output/mask/target, optimizer chỉ cho tham số được học, train/eval và chọn checkpoint bằng metric validation. Tận dụng loader/model/config trong context; không tạo lại chúng. Với sparse dùng fit/predict và chọn tham số validation, không bịa optimizer/backward. Với neural kiểm forward/loss hữu hạn, backward và một bước update; reload config/weights và trạng thái biểu diễn rồi dùng lại inference hiện có. Giữ ID, metric và chiều tốt hơn của metric; test chỉ chấm sau khi chọn xong, không chọn tham số bằng test.
Phương pháp: Chọn NB-SVM sparse: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features. Metric: macro_f1. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]

Context s03 — Model và giao diện sử dụng:
[Dán tóm tắt s03]

Context s04 — Siêu tham số:
[Dán tóm tắt s04]
```
