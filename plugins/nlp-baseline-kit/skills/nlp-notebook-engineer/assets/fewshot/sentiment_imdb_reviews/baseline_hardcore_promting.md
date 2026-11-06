# Prompt hardcore: IMDb: TF-IDF từ/ký tự và NB-SVM

<!-- prompt-guide
Format: 3
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: sha(text)

<!-- module
ID: m003
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=3; lines=25-26
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết sha(text). Tạo hash ổn định của văn bản để nhận diện nhóm trùng lặp. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m003"></a>

```text
Tóm tắt giao diện sha(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của sha(text) tại đây.
```

## Bước 2: read_jsonl(path)

<!-- module
ID: m005
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=3; lines=31-36
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết read_jsonl(path). Đọc UTF-8 JSONL, giữ thứ tự bản ghi và báo rõ lỗi thiếu file hoặc dữ liệu không hợp lệ. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m005"></a>

```text
Tóm tắt giao diện read_jsonl(path) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của read_jsonl(path) tại đây.
```

## Bước 3: normalize_text(text)

<!-- module
ID: m009
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=1-4
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết normalize_text(text). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. TFDS imdb_reviews; local JSONL train/test có id, text, label với neg=0, pos=1. Chuẩn hóa HTML/NFC, gộp khoảng trắng; giữ test chuẩn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện normalize_text(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của normalize_text(text) tại đây.
```

## Bước 4: canonical_rows(rows, split)

<!-- module
ID: m010
Depends: m003
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=6-18
Summary budget: input=1200; output=500; reserve=300
Config owns: LABEL_NAMES
Config uses: LABEL_NAMES
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [sha(text)](#context-m003)

### Prompt tạo hoặc sửa module

```text
Viết canonical_rows(rows, split). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. TFDS imdb_reviews; local JSONL train/test có id, text, label với neg=0, pos=1. Chuẩn hóa HTML/NFC, gộp khoảng trắng; giữ test chuẩn. Giữ ràng buộc: LABEL_NAMES=['neg', 'pos']. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m003 — sha(text):
[Dán tóm tắt m003]
```

### Tóm tắt context để dùng tiếp

<a id="context-m010"></a>

```text
Tóm tắt giao diện canonical_rows(rows, split) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m003 — sha(text):
[Dán tóm tắt m003]
```

```python
# Dán code hiện tại của canonical_rows(rows, split) tại đây.
```

## Bước 5: load_raw(split) — phần 1/3

<!-- module
ID: m011
Depends: m005, m010
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=20-24
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_DIR, TASK
Config uses: DATA_DIR, LABEL_NAMES, TASK
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m005)
2. [canonical_rows(rows, split)](#context-m010)

### Prompt tạo hoặc sửa module

```text
Viết load_raw(split) — phần 1/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ ràng buộc: DATA_DIR=Path('data/sentiment_imdb_reviews'). Tự chọn TASK phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 1/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

### Tóm tắt context để dùng tiếp

<a id="context-m011"></a>

```text
Tóm tắt giao diện load_raw(split) — phần 1/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

```python
# Dán code hiện tại của load_raw(split) — phần 1/3 tại đây.
```

## Bước 6: load_raw(split) — phần 2/3

<!-- module
ID: m012
Depends: m005, m010, m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=25-71
Summary budget: input=1200; output=500; reserve=300
Config uses: DATA_DIR, LABEL_NAMES, TASK
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m005)
2. [canonical_rows(rows, split)](#context-m010)
3. [load_raw(split) — phần 1/3](#context-m011)

### Prompt tạo hoặc sửa module

```text
Viết load_raw(split) — phần 2/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Dùng lại DATA_DIR, LABEL_NAMES, TASK từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 2/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m011 — load_raw(split) — phần 1/3:
[Dán tóm tắt m011]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện load_raw(split) — phần 2/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m011 — load_raw(split) — phần 1/3:
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của load_raw(split) — phần 2/3 tại đây.
```

## Bước 7: load_raw(split) — phần 3/3

<!-- module
ID: m013
Depends: m005, m010, m011, m012
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=72-72
Summary budget: input=1200; output=500; reserve=300
Config uses: DATA_DIR, LABEL_NAMES, TASK
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m005)
2. [canonical_rows(rows, split)](#context-m010)
3. [load_raw(split) — phần 1/3](#context-m011)
4. [load_raw(split) — phần 2/3](#context-m012)

### Prompt tạo hoặc sửa module

```text
Viết load_raw(split) — phần 3/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Dùng lại DATA_DIR, LABEL_NAMES, TASK từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 3/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m011 — load_raw(split) — phần 1/3:
[Dán tóm tắt m011]

Context m012 — load_raw(split) — phần 2/3:
[Dán tóm tắt m012]
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện load_raw(split) — phần 3/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m011 — load_raw(split) — phần 1/3:
[Dán tóm tắt m011]

Context m012 — load_raw(split) — phần 2/3:
[Dán tóm tắt m012]
```

```python
# Dán code hiện tại của load_raw(split) — phần 3/3 tại đây.
```

## Bước 8: Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…

<!-- module
ID: m014
Depends: m003, m009, m010, m013
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=74-94
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, SOURCE
Config uses: LABEL_NAMES, SEED, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [sha(text)](#context-m003)
2. [normalize_text(text)](#context-m009)
3. [canonical_rows(rows, split)](#context-m010)
4. [load_raw(split) — phần 3/3](#context-m013)

### Prompt tạo hoặc sửa module

```text
Viết Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…. Giữ ràng buộc: SEED=42. Khai báo cùng phần này: SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]
```

```python
# Dán code hiện tại của Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train… tại đây.
```

## Bước 9: Đọc dữ liệu, chuẩn hóa và chia validation

<!-- module
ID: m015
Depends: m010, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=95-99
Summary budget: input=1200; output=500; reserve=300
Config uses: LABEL_NAMES
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)
2. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Đọc dữ liệu, chuẩn hóa và chia validation. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Đọc dữ liệu, chuẩn hóa và chia validation từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Đọc dữ liệu, chuẩn hóa và chia validation tại đây.
```

## Bước 10: build_features()

<!-- module
ID: m016
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=1-8
Summary budget: input=1200; output=500; reserve=300
Config owns: CHAR_MAX_FEATURES, MIN_DF, WORD_MAX_FEATURES
Config uses: CHAR_MAX_FEATURES, MIN_DF, WORD_MAX_FEATURES
-->

### Prompt tạo hoặc sửa module

```text
Viết build_features(). Kết hợp TF-IDF word và char_wb bằng FeatureUnion, chỉ fit train; tự chọn ngram_range và trọng số hai nhánh phù hợp. Tự chọn CHAR_MAX_FEATURES, MIN_DF, WORD_MAX_FEATURES phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện build_features() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của build_features() tại đây.
```

## Bước 11: Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…

<!-- module
ID: m017
Depends: m014, m016
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=10-23
Summary budget: input=1200; output=500; reserve=300
Config owns: NB_RATIO_ALPHA, USE_NB_RATIO
Config uses: NB_RATIO_ALPHA, USE_NB_RATIO
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
2. [build_features()](#context-m016)

### Prompt tạo hoặc sửa module

```text
Viết Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…. Tạo NB log-ratio nhị phân từ presence train với smoothing dương được chọn phù hợp; nhân cùng ratio vào CSR train và validation. Giữ ràng buộc: USE_NB_RATIO=True. Tự chọn NB_RATIO_ALPHA phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m016 — build_features():
[Dán tóm tắt m016]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Biểu diễn văn bản chỉ học trên train — X_train, X_val, features… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m016 — build_features():
[Dán tóm tắt m016]
```

```python
# Dán code hiện tại của Biểu diễn văn bản chỉ học trên train — X_train, X_val, features… tại đây.
```

## Bước 12: make_classifier(parameter)

<!-- module
ID: m019
Depends: m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=2-4
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_ITER, TOLERANCE
Config uses: MAX_ITER, SEED, TOLERANCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết make_classifier(parameter). Tạo classifier LinearSVC theo API đã chốt; kiểm hội tụ và dùng cấu hình lựa chọn trên validation. Tự chọn MAX_ITER, TOLERANCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện make_classifier(parameter) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của make_classifier(parameter) tại đây.
```

## Bước 13: classification_metrics(y, pred)

<!-- module
ID: m020
Depends: m010
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=6-8
Summary budget: input=1200; output=500; reserve=300
Config uses: LABEL_NAMES
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)

### Prompt tạo hoặc sửa module

```text
Viết classification_metrics(y, pred). Viết classification_metrics(y,pred) trả macro_f1 và accuracy, giữ đủ nhãn và zero_division bằng0. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. classification_metrics(y,pred) giữ interface hiện tại. Kết quả giữ các trường accuracy, macro_f1. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện classification_metrics(y, pred) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

```python
# Dán code hiện tại của classification_metrics(y, pred) tại đây.
```

## Bước 14: Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…

<!-- module
ID: m021
Depends: m010, m014, m017, m019, m020
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=10-35
Summary budget: input=1200; output=500; reserve=300
Config owns: C_VALUES, OUTPUT_DIR, PARAMETERS, PRIMARY_METRIC
Config uses: C_VALUES, LABEL_NAMES, OUTPUT_DIR, PARAMETERS, PRIMARY_METRIC
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)
2. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
3. [Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…](#context-m017)
4. [make_classifier(parameter)](#context-m019)
5. [classification_metrics(y, pred)](#context-m020)

### Prompt tạo hoặc sửa module

```text
Viết Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…. Fit pipeline trên train theo các ứng viên siêu tham số được chọn; chọn bằng metric validation và lưu đủ biểu diễn, classifier và nhãn để reload. Không fit lại trên test hoặc train+validation. Giữ ràng buộc: PRIMARY_METRIC='macro_f1'. Tự chọn C_VALUES phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: OUTPUT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; PARAMETERS được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m017 — Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…:
[Dán tóm tắt m017]

Context m019 — make_classifier(parameter):
[Dán tóm tắt m019]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m017 — Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…:
[Dán tóm tắt m017]

Context m019 — make_classifier(parameter):
[Dán tóm tắt m019]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]
```

```python
# Dán code hiện tại của Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier… tại đây.
```

## Bước 15: Nạp lại, đánh giá test một lần và dự đoán — best

<!-- module
ID: m022
Depends: m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=2-2
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết Nạp lại, đánh giá test một lần và dự đoán — best. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện Nạp lại, đánh giá test một lần và dự đoán — best từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của Nạp lại, đánh giá test một lần và dự đoán — best tại đây.
```

## Bước 16: predict_texts(texts)

<!-- module
ID: m023
Depends: m009, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=4-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_text(text)](#context-m009)
2. [Nạp lại, đánh giá test một lần và dự đoán — best](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết predict_texts(texts). Giữ predict_texts(texts), reload features, ratio và classifier; đối chiếu validation trước khi chấm test. Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện predict_texts(texts) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của predict_texts(texts) tại đây.
```

## Bước 17: Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…

<!-- module
ID: m024
Depends: m003, m009, m010, m013, m014, m020, m021, m022, m023
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=10-29
Summary budget: input=1200; output=500; reserve=300
Config uses: LABEL_NAMES, PRIMARY_METRIC
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [sha(text)](#context-m003)
2. [normalize_text(text)](#context-m009)
3. [canonical_rows(rows, split)](#context-m010)
4. [load_raw(split) — phần 3/3](#context-m013)
5. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
6. [classification_metrics(y, pred)](#context-m020)
7. [Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…](#context-m021)
8. [Nạp lại, đánh giá test một lần và dự đoán — best](#context-m022)
9. [predict_texts(texts)](#context-m023)

### Prompt tạo hoặc sửa module

```text
Viết Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Dùng lại LABEL_NAMES, PRIMARY_METRIC từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]
```

```python
# Dán code hiện tại của Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val… tại đây.
```

## Bước 18: Nạp lại, đánh giá test một lần và dự đoán — f, pred, row

<!-- module
ID: m025
Depends: m010, m014, m021, m022, m023, m024
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=30-38
Summary budget: input=1200; output=500; reserve=300
Config owns: VERSIONS
Config uses: LABEL_NAMES, OUTPUT_DIR, VERSIONS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)
2. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
3. [Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…](#context-m021)
4. [Nạp lại, đánh giá test một lần và dự đoán — best](#context-m022)
5. [predict_texts(texts)](#context-m023)
6. [Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…](#context-m024)

### Prompt tạo hoặc sửa module

```text
Viết Nạp lại, đánh giá test một lần và dự đoán — f, pred, row. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Khai báo cùng phần này: VERSIONS được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại LABEL_NAMES, OUTPUT_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]

Context m024 — Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…:
[Dán tóm tắt m024]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện Nạp lại, đánh giá test một lần và dự đoán — f, pred, row từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]

Context m024 — Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…:
[Dán tóm tắt m024]
```

```python
# Dán code hiện tại của Nạp lại, đánh giá test một lần và dự đoán — f, pred, row tại đây.
```

## Bước 19: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m015, m025
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc dữ liệu, chuẩn hóa và chia validation](#context-m015)
2. [Nạp lại, đánh giá test một lần và dự đoán — f, pred, row](#context-m025)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m015 — Đọc dữ liệu, chuẩn hóa và chia validation:
[Dán tóm tắt m015]

Context m025 — Nạp lại, đánh giá test một lần và dự đoán — f, pred, row:
[Dán tóm tắt m025]
```
