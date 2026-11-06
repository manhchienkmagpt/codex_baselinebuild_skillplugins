# Prompt hardcore: SemEval restaurants: character BIO, word features, BiLSTM-CRF và polarity

<!-- prompt-guide
Format: 3
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…

<!-- module
ID: m004
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=1-41
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m004"></a>

```text
Tóm tắt giao diện Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR… tại đây.
```

## Bước 2: load_raw_data()

<!-- module
ID: m006
Depends: m004
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=1-22
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

### Prompt tạo hoặc sửa module

```text
Viết load_raw_data(). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Tóm tắt context để dùng tiếp

<a id="context-m006"></a>

```text
Tóm tắt giao diện load_raw_data() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

```python
# Dán code hiện tại của load_raw_data() tại đây.
```

## Bước 3: group_rows(rows, split) — phần 1/3

<!-- module
ID: m007
Depends: m004
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=25-27
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

### Prompt tạo hoặc sửa module

```text
Viết group_rows(rows, split) — phần 1/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Đây là phần 1/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Tóm tắt context để dùng tiếp

<a id="context-m007"></a>

```text
Tóm tắt giao diện group_rows(rows, split) — phần 1/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

```python
# Dán code hiện tại của group_rows(rows, split) — phần 1/3 tại đây.
```

## Bước 4: group_rows(rows, split) — phần 2/3

<!-- module
ID: m008
Depends: m004, m007
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=28-52
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [group_rows(rows, split) — phần 1/3](#context-m007)

### Prompt tạo hoặc sửa module

```text
Viết group_rows(rows, split) — phần 2/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Đây là phần 2/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m007 — group_rows(rows, split) — phần 1/3:
[Dán tóm tắt m007]
```

### Tóm tắt context để dùng tiếp

<a id="context-m008"></a>

```text
Tóm tắt giao diện group_rows(rows, split) — phần 2/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m007 — group_rows(rows, split) — phần 1/3:
[Dán tóm tắt m007]
```

```python
# Dán code hiện tại của group_rows(rows, split) — phần 2/3 tại đây.
```

## Bước 5: group_rows(rows, split) — phần 3/3

<!-- module
ID: m009
Depends: m004, m008
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=53-60
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [group_rows(rows, split) — phần 2/3](#context-m008)

### Prompt tạo hoặc sửa module

```text
Viết group_rows(rows, split) — phần 3/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Đây là phần 3/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m008 — group_rows(rows, split) — phần 2/3:
[Dán tóm tắt m008]
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện group_rows(rows, split) — phần 3/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m008 — group_rows(rows, split) — phần 2/3:
[Dán tóm tắt m008]
```

```python
# Dán code hiện tại của group_rows(rows, split) — phần 3/3 tại đây.
```

## Bước 6: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m010
Depends: m004, m006, m009
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=63-81
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [load_raw_data()](#context-m006)
3. [group_rows(rows, split) — phần 3/3](#context-m009)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m006 — load_raw_data():
[Dán tóm tắt m006]

Context m009 — group_rows(rows, split) — phần 3/3:
[Dán tóm tắt m009]
```

### Tóm tắt context để dùng tiếp

<a id="context-m010"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m006 — load_raw_data():
[Dán tóm tắt m006]

Context m009 — group_rows(rows, split) — phần 3/3:
[Dán tóm tắt m009]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 7: Tải dữ liệu, nhóm câu và chia validation

<!-- module
ID: m012
Depends: m010
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=87-94
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m010)

### Prompt tạo hoặc sửa module

```text
Viết Tải dữ liệu, nhóm câu và chia validation. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện Tải dữ liệu, nhóm câu và chia validation từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]
```

```python
# Dán code hiện tại của Tải dữ liệu, nhóm câu và chia validation tại đây.
```

## Bước 8: words_with_offsets(text)

<!-- module
ID: m014
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=5-6
Summary budget: input=1200; output=500; reserve=300
Config owns: WORD_PATTERN
Config uses: WORD_PATTERN
-->

### Prompt tạo hoặc sửa module

```text
Viết words_with_offsets(text). Bổ sung words_with_offsets và word_vocab train-only, giữ trục thời gian ký tự nguyên văn. PAD0 UNK1; word regex words+punct; WORD_MIN_FREQ1; char_vocab train-only; không normalize/cắt câu làm lệch span. Tự chọn WORD_PATTERN phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện words_with_offsets(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của words_with_offsets(text) tại đây.
```

## Bước 9: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m015
Depends: m004, m010, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=8-15
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Chuẩn hóa và chia dữ liệu](#context-m010)
3. [words_with_offsets(text)](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m014 — words_with_offsets(text):
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m014 — words_with_offsets(text):
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Bảng ký tự và trạng thái tokenizer tại đây.
```

## Bước 10: encode_record(record)

<!-- module
ID: m016
Depends: m004, m014, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=17-31
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [words_with_offsets(text)](#context-m014)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết encode_record(record). Nâng encode_record để word ID phủ đúng khoảng ký tự, giữ BIO và spans gốc. Kết quả giữ các trường char_ids, record, tags, word_ids. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m014 — words_with_offsets(text):
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện encode_record(record) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m014 — words_with_offsets(text):
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của encode_record(record) tại đây.
```

## Bước 11: AspectDataset.__init__(self, records)

<!-- module
ID: m017
Depends: m016
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=35-36
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [encode_record(record)](#context-m016)

### Prompt tạo hoặc sửa module

```text
Viết AspectDataset.__init__(self, records). Khởi tạo các thành phần và trạng thái cần cho class. Viết method trong class AspectDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — encode_record(record):
[Dán tóm tắt m016]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện AspectDataset.__init__(self, records) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — encode_record(record):
[Dán tóm tắt m016]
```

```python
# Dán code hiện tại của AspectDataset.__init__(self, records) tại đây.
```

## Bước 12: AspectDataset.__len__(self)

<!-- module
ID: m018
Depends: m017
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=38-39
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectDataset.__init__(self, records)](#context-m017)

### Prompt tạo hoặc sửa module

```text
Viết AspectDataset.__len__(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class AspectDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]
```

### Tóm tắt context để dùng tiếp

<a id="context-m018"></a>

```text
Tóm tắt giao diện AspectDataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]
```

```python
# Dán code hiện tại của AspectDataset.__len__(self) tại đây.
```

## Bước 13: AspectDataset.__getitem__(self, index)

<!-- module
ID: m019
Depends: m017
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=41-42
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectDataset.__init__(self, records)](#context-m017)

### Prompt tạo hoặc sửa module

```text
Viết AspectDataset.__getitem__(self, index). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class AspectDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện AspectDataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]
```

```python
# Dán code hiện tại của AspectDataset.__getitem__(self, index) tại đây.
```

## Bước 14: Ghép giao diện class AspectDataset

<!-- module
ID: m020
Depends: m017, m018, m019
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=9; lines=34-42
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectDataset.__init__(self, records)](#context-m017)
2. [AspectDataset.__len__(self)](#context-m018)
3. [AspectDataset.__getitem__(self, index)](#context-m019)

<a id="context-m020"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class AspectDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]

Context m018 — AspectDataset.__len__(self):
[Dán tóm tắt m018]

Context m019 — AspectDataset.__getitem__(self, index):
[Dán tóm tắt m019]
```

## Bước 15: collate_batch(items)

<!-- module
ID: m021
Depends: m004
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=45-62
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Cập nhật collate_batch nhận word IDs, masks và gold polarity spans hợp lệ. lengths CPU, mask bool; gold_spans[M,4] chỉ polarity có nhãn; records giữ sid,offsets; AspectDataset/make_loader dùng encode_record mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 16: make_loader(records, shuffle=False, batch_size)

<!-- module
ID: m022
Depends: m004, m020, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=65-68
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class AspectDataset](#context-m020)
3. [collate_batch(items)](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(records, shuffle=False, batch_size). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m020 — Ghép giao diện class AspectDataset:
[Dán tóm tắt m020]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện make_loader(records, shuffle=False, batch_size) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m020 — Ghép giao diện class AspectDataset:
[Dán tóm tắt m020]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của make_loader(records, shuffle=False, batch_size) tại đây.
```

## Bước 17: Tạo DataLoader cho các split

<!-- module
ID: m023
Depends: m004, m010, m015, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=70-74
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Chuẩn hóa và chia dữ liệu](#context-m010)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)
4. [make_loader(records, shuffle=False, batch_size)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m022 — make_loader(records, shuffle=False, batch_size):
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m022 — make_loader(records, shuffle=False, batch_size):
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 18: LinearChainCRF.__init__(self)

<!-- module
ID: m024
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=3-11
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết LinearChainCRF.__init__(self). Dựng LinearChainCRF O/B/I với start/end và cấm I đầu câu, cấm O sang I. Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class LinearChainCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện LinearChainCRF.__init__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của LinearChainCRF.__init__(self) tại đây.
```

## Bước 19: LinearChainCRF.constrained(self)

<!-- module
ID: m025
Depends: m024
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=13-15
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [LinearChainCRF.__init__(self)](#context-m024)

### Prompt tạo hoặc sửa module

```text
Viết LinearChainCRF.constrained(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Viết method trong class LinearChainCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện LinearChainCRF.constrained(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]
```

```python
# Dán code hiện tại của LinearChainCRF.constrained(self) tại đây.
```

## Bước 20: LinearChainCRF.neg_log_likelihood(self, emissions, tags, mask)

<!-- module
ID: m026
Depends: m024, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=17-32
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [LinearChainCRF.__init__(self)](#context-m024)
2. [LinearChainCRF.constrained(self)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết LinearChainCRF.neg_log_likelihood(self, emissions, tags, mask). Viết CRF neg_log_likelihood bằng gold path và logsumexp partition, chỉ tính ký tự thật. Viết method trong class LinearChainCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]

Context m025 — LinearChainCRF.constrained(self):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện LinearChainCRF.neg_log_likelihood(self, emissions, tags, mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]

Context m025 — LinearChainCRF.constrained(self):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của LinearChainCRF.neg_log_likelihood(self, emissions, tags, mask) tại đây.
```

## Bước 21: LinearChainCRF.decode(self, emissions, mask)

<!-- module
ID: m027
Depends: m024, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=34-53
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [LinearChainCRF.__init__(self)](#context-m024)
2. [LinearChainCRF.constrained(self)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết LinearChainCRF.decode(self, emissions, mask). Viết decode Viterbi có constraints và mask, backtrack theo chiều dài thật. Viết method trong class LinearChainCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]

Context m025 — LinearChainCRF.constrained(self):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện LinearChainCRF.decode(self, emissions, mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]

Context m025 — LinearChainCRF.constrained(self):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của LinearChainCRF.decode(self, emissions, mask) tại đây.
```

## Bước 22: Ghép giao diện class LinearChainCRF

<!-- module
ID: m028
Depends: m024, m025, m026, m027
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=11; lines=1-53
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [LinearChainCRF.__init__(self)](#context-m024)
2. [LinearChainCRF.constrained(self)](#context-m025)
3. [LinearChainCRF.neg_log_likelihood(self, emissions, tags, mask)](#context-m026)
4. [LinearChainCRF.decode(self, emissions, mask)](#context-m027)

<a id="context-m028"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class LinearChainCRF kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m024 — LinearChainCRF.__init__(self):
[Dán tóm tắt m024]

Context m025 — LinearChainCRF.constrained(self):
[Dán tóm tắt m025]

Context m026 — LinearChainCRF.neg_log_likelihood(self, emissions, tags, mask):
[Dán tóm tắt m026]

Context m027 — LinearChainCRF.decode(self, emissions, mask):
[Dán tóm tắt m027]
```

## Bước 23: AspectModel.__init__(self, cfg, n_chars, n_words)

<!-- module
ID: m029
Depends: m004, m028
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=2-15
Summary budget: input=1200; output=500; reserve=300
Config owns: MODEL_CONFIG
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class LinearChainCRF](#context-m028)

### Prompt tạo hoặc sửa module

```text
Viết AspectModel.__init__(self, cfg, n_chars, n_words). Thay constructor AspectModel bằng char/word embedding, BiLSTM số lớp được chọn, tag head, CRF và polarity head. Khai báo cùng phần này: MODEL_CONFIG có các khóa char_dim, word_dim, hidden_dim, num_layers, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m028 — Ghép giao diện class LinearChainCRF:
[Dán tóm tắt m028]
```

### Tóm tắt context để dùng tiếp

<a id="context-m029"></a>

```text
Tóm tắt giao diện AspectModel.__init__(self, cfg, n_chars, n_words) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m028 — Ghép giao diện class LinearChainCRF:
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của AspectModel.__init__(self, cfg, n_chars, n_words) tại đây.
```

## Bước 24: AspectModel.forward(self, batch)

<!-- module
ID: m030
Depends: m021, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=17-26
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m021)
2. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết AspectModel.forward(self, batch). Viết forward ghép char/word embedding theo ký tự, packed BiLSTM và BIO emissions. Giữ thứ tự đầu ra: hidden, kết quả 2. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện AspectModel.forward(self, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của AspectModel.forward(self, batch) tại đây.
```

## Bước 25: AspectModel.extraction_loss(self, emissions, batch)

<!-- module
ID: m031
Depends: m021, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=28-29
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m021)
2. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết AspectModel.extraction_loss(self, emissions, batch). Viết extraction_loss và decode của AspectModel để gọi CRF đã dựng, giữ chữ ký caller hiện tại. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện AspectModel.extraction_loss(self, emissions, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của AspectModel.extraction_loss(self, emissions, batch) tại đây.
```

## Bước 26: AspectModel.decode(self, emissions, mask)

<!-- module
ID: m032
Depends: m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=31-32
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết AspectModel.decode(self, emissions, mask). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m032"></a>

```text
Tóm tắt giao diện AspectModel.decode(self, emissions, mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của AspectModel.decode(self, emissions, mask) tại đây.
```

## Bước 27: AspectModel.classify_spans(self, hidden, spans, mask)

<!-- module
ID: m033
Depends: m004, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=34-43
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết AspectModel.classify_spans(self, hidden, spans, mask). Nâng classify_spans dùng hai biên, mean span và mean câu có mask cho polarity. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện AspectModel.classify_spans(self, hidden, spans, mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của AspectModel.classify_spans(self, hidden, spans, mask) tại đây.
```

## Bước 28: Ghép giao diện class AspectModel

<!-- module
ID: m034
Depends: m029, m030, m031, m032, m033
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=1-43
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)
2. [AspectModel.forward(self, batch)](#context-m030)
3. [AspectModel.extraction_loss(self, emissions, batch)](#context-m031)
4. [AspectModel.decode(self, emissions, mask)](#context-m032)
5. [AspectModel.classify_spans(self, hidden, spans, mask)](#context-m033)

<a id="context-m034"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class AspectModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]

Context m030 — AspectModel.forward(self, batch):
[Dán tóm tắt m030]

Context m031 — AspectModel.extraction_loss(self, emissions, batch):
[Dán tóm tắt m031]

Context m032 — AspectModel.decode(self, emissions, mask):
[Dán tóm tắt m032]

Context m033 — AspectModel.classify_spans(self, hidden, spans, mask):
[Dán tóm tắt m033]
```

## Bước 29: Cấu hình kiến trúc mô hình

<!-- module
ID: m035
Depends: m004, m015, m029, m034
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=1-5
Summary budget: input=1200; output=500; reserve=300
Config uses: MODEL_CONFIG
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)
3. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)
4. [Ghép giao diện class AspectModel](#context-m034)

### Prompt tạo hoặc sửa module

```text
Viết Cấu hình kiến trúc mô hình. Dùng lại MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện Cấu hình kiến trúc mô hình từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]
```

```python
# Dán code hiện tại của Cấu hình kiến trúc mô hình tại đây.
```

## Bước 30: move_batch(batch)

<!-- module
ID: m036
Depends: m004, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=1-3
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [collate_batch(items)](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết move_batch(batch). Viết riêng move_batch theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. move_batch nhận (batch); Chuyển tensor sang device trừ lengths giữ CPU, giữ metadata records nguyên dạng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện move_batch(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của move_batch(batch) tại đây.
```

## Bước 31: bio_to_spans(tags)

<!-- module
ID: m037
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=6-15
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết bio_to_spans(tags). Viết riêng bio_to_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. bio_to_spans nhận (tags); Đọc O/B/I thành character span end exclusive; I sau O xem B, không trim khoảng trắng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m037"></a>

```text
Tóm tắt giao diện bio_to_spans(tags) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của bio_to_spans(tags) tại đây.
```

## Bước 32: set_prf(gold, predicted)

<!-- module
ID: m038
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=18-24
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết set_prf(gold, predicted). Viết riêng set_prf theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. set_prf nhận (gold, predicted); TP từ giao tập gold/predicted, precision/recall/F1 từ tổng set sizes, empty chia0 trả0. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Kết quả giữ các trường f1, gold, precision, predicted, recall, tp. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện set_prf(gold, predicted) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của set_prf(gold, predicted) tại đây.
```

## Bước 33: classification_metrics(gold, predicted)

<!-- module
ID: m039
Depends: m004
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=27-45
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

### Prompt tạo hoặc sửa module

```text
Viết classification_metrics(gold, predicted). Viết riêng classification_metrics theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. classification_metrics nhận (gold, predicted); Tính confusion matrix bốn polarity, macro-F1 gồm mọi lớp; thiếu gold trả None và count0, không bịa nhãn. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Kết quả giữ các trường accuracy, confusion_matrix, count, macro_f1, micro_f1, per_class. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Tóm tắt context để dùng tiếp

<a id="context-m039"></a>

```text
Tóm tắt giao diện classification_metrics(gold, predicted) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

```python
# Dán code hiện tại của classification_metrics(gold, predicted) tại đây.
```

## Bước 34: evaluate(model, loader, include_predictions=False) — phần 1/3

<!-- module
ID: m040
Depends: m004, m034, m036, m037, m038, m039
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=48-53
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class AspectModel](#context-m034)
3. [move_batch(batch)](#context-m036)
4. [bio_to_spans(tags)](#context-m037)
5. [set_prf(gold, predicted)](#context-m038)
6. [classification_metrics(gold, predicted)](#context-m039)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, include_predictions=False) — phần 1/3. Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad. Đây là phần 1/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m038 — set_prf(gold, predicted):
[Dán tóm tắt m038]

Context m039 — classification_metrics(gold, predicted):
[Dán tóm tắt m039]
```

### Tóm tắt context để dùng tiếp

<a id="context-m040"></a>

```text
Tóm tắt giao diện evaluate(model, loader, include_predictions=False) — phần 1/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m038 — set_prf(gold, predicted):
[Dán tóm tắt m038]

Context m039 — classification_metrics(gold, predicted):
[Dán tóm tắt m039]
```

```python
# Dán code hiện tại của evaluate(model, loader, include_predictions=False) — phần 1/3 tại đây.
```

## Bước 35: evaluate(model, loader, include_predictions=False) — phần 2/3

<!-- module
ID: m041
Depends: m004, m034, m036, m037, m038, m039, m040
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=54-82
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class AspectModel](#context-m034)
3. [move_batch(batch)](#context-m036)
4. [bio_to_spans(tags)](#context-m037)
5. [set_prf(gold, predicted)](#context-m038)
6. [classification_metrics(gold, predicted)](#context-m039)
7. [evaluate(model, loader, include_predictions=False) — phần 1/3](#context-m040)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, include_predictions=False) — phần 2/3. Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad. Đây là phần 2/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m038 — set_prf(gold, predicted):
[Dán tóm tắt m038]

Context m039 — classification_metrics(gold, predicted):
[Dán tóm tắt m039]

Context m040 — evaluate(model, loader, include_predictions=False) — phần 1/3:
[Dán tóm tắt m040]
```

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện evaluate(model, loader, include_predictions=False) — phần 2/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m038 — set_prf(gold, predicted):
[Dán tóm tắt m038]

Context m039 — classification_metrics(gold, predicted):
[Dán tóm tắt m039]

Context m040 — evaluate(model, loader, include_predictions=False) — phần 1/3:
[Dán tóm tắt m040]
```

```python
# Dán code hiện tại của evaluate(model, loader, include_predictions=False) — phần 2/3 tại đây.
```

## Bước 36: evaluate(model, loader, include_predictions=False) — phần 3/3

<!-- module
ID: m042
Depends: m004, m034, m036, m037, m038, m039, m041
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=83-87
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class AspectModel](#context-m034)
3. [move_batch(batch)](#context-m036)
4. [bio_to_spans(tags)](#context-m037)
5. [set_prf(gold, predicted)](#context-m038)
6. [classification_metrics(gold, predicted)](#context-m039)
7. [evaluate(model, loader, include_predictions=False) — phần 2/3](#context-m041)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, include_predictions=False) — phần 3/3. Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad. Đây là phần 3/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m038 — set_prf(gold, predicted):
[Dán tóm tắt m038]

Context m039 — classification_metrics(gold, predicted):
[Dán tóm tắt m039]

Context m041 — evaluate(model, loader, include_predictions=False) — phần 2/3:
[Dán tóm tắt m041]
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện evaluate(model, loader, include_predictions=False) — phần 3/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m038 — set_prf(gold, predicted):
[Dán tóm tắt m038]

Context m039 — classification_metrics(gold, predicted):
[Dán tóm tắt m039]

Context m041 — evaluate(model, loader, include_predictions=False) — phần 2/3:
[Dán tóm tắt m041]
```

```python
# Dán code hiện tại của evaluate(model, loader, include_predictions=False) — phần 3/3 tại đây.
```

## Bước 37: Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs

<!-- module
ID: m043
Depends: m004, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=1-2
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Cấu hình kiến trúc mô hình](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs. Khởi tạo optimizer AdamW cho tham số có thể học. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m035 — Cấu hình kiến trúc mô hình:
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m043"></a>

```text
Tóm tắt giao diện Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m035 — Cấu hình kiến trúc mô hình:
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs tại đây.
```

## Bước 38: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m044
Depends: m004, m015, m023, m035, m036, m042, m043
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=3-47
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)
3. [Tạo DataLoader cho các split](#context-m023)
4. [Cấu hình kiến trúc mô hình](#context-m035)
5. [move_batch(batch)](#context-m036)
6. [evaluate(model, loader, include_predictions=False) — phần 3/3](#context-m042)
7. [Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs](#context-m043)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m035 — Cấu hình kiến trúc mô hình:
[Dán tóm tắt m035]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m042 — evaluate(model, loader, include_predictions=False) — phần 3/3:
[Dán tóm tắt m042]

Context m043 — Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs:
[Dán tóm tắt m043]
```

### Tóm tắt context để dùng tiếp

<a id="context-m044"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m035 — Cấu hình kiến trúc mô hình:
[Dán tóm tắt m035]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m042 — evaluate(model, loader, include_predictions=False) — phần 3/3:
[Dán tóm tắt m042]

Context m043 — Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs:
[Dán tóm tắt m043]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 39: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m045
Depends: m004, m010, m022, m034, m042
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=2-18
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Chuẩn hóa và chia dữ liệu](#context-m010)
3. [make_loader(records, shuffle=False, batch_size)](#context-m022)
4. [Ghép giao diện class AspectModel](#context-m034)
5. [evaluate(model, loader, include_predictions=False) — phần 3/3](#context-m042)

### Prompt tạo hoặc sửa module

```text
Viết Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m022 — make_loader(records, shuffle=False, batch_size):
[Dán tóm tắt m022]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m042 — evaluate(model, loader, include_predictions=False) — phần 3/3:
[Dán tóm tắt m042]
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m022 — make_loader(records, shuffle=False, batch_size):
[Dán tóm tắt m022]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]

Context m042 — evaluate(model, loader, include_predictions=False) — phần 3/3:
[Dán tóm tắt m042]
```

```python
# Dán code hiện tại của Bảng ký tự và trạng thái tokenizer tại đây.
```

## Bước 40: Tải lại và đánh giá cuối — results

<!-- module
ID: m046
Depends: m004, m012, m045
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=19-25
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Tải dữ liệu, nhóm câu và chia validation](#context-m012)
3. [Bảng ký tự và trạng thái tokenizer](#context-m045)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá cuối — results. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m012 — Tải dữ liệu, nhóm câu và chia validation:
[Dán tóm tắt m012]

Context m045 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m045]
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá cuối — results từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m012 — Tải dữ liệu, nhóm câu và chia validation:
[Dán tóm tắt m012]

Context m045 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m045]
```

```python
# Dán code hiện tại của Tải lại và đánh giá cuối — results tại đây.
```

## Bước 41: predict_text(text)

<!-- module
ID: m047
Depends: m004, m016, m021, m036, m037, m045
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=21; lines=1-14
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [encode_record(record)](#context-m016)
3. [collate_batch(items)](#context-m021)
4. [move_batch(batch)](#context-m036)
5. [bio_to_spans(tags)](#context-m037)
6. [Bảng ký tự và trạng thái tokenizer](#context-m045)

### Prompt tạo hoặc sửa module

```text
Viết predict_text(text). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Hàm predict_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m016 — encode_record(record):
[Dán tóm tắt m016]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m045 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m045]
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện predict_text(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m016 — encode_record(record):
[Dán tóm tắt m016]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m036 — move_batch(batch):
[Dán tóm tắt m036]

Context m037 — bio_to_spans(tags):
[Dán tóm tắt m037]

Context m045 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m045]
```

```python
# Dán code hiện tại của predict_text(text) tại đây.
```

## Bước 42: Dự đoán câu mới

<!-- module
ID: m048
Depends: m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=21; lines=16-16
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [predict_text(text)](#context-m047)

### Prompt tạo hoặc sửa module

```text
Viết Dự đoán câu mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m047 — predict_text(text):
[Dán tóm tắt m047]
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện Dự đoán câu mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m047 — predict_text(text):
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của Dự đoán câu mới tại đây.
```

## Bước 43: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m044, m046, m048
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vòng huấn luyện, validation và lưu checkpoint](#context-m044)
2. [Tải lại và đánh giá cuối — results](#context-m046)
3. [Dự đoán câu mới](#context-m048)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m044 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m044]

Context m046 — Tải lại và đánh giá cuối — results:
[Dán tóm tắt m046]

Context m048 — Dự đoán câu mới:
[Dán tóm tắt m048]
```
