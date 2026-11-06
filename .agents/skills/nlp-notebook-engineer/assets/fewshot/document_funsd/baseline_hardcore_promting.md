# Prompt hardcore: FUNSD: word/char/layout/ROI, BiLSTM và CRF

<!-- prompt-guide
Format: 3
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: open_rgb(value)

<!-- module
ID: m007
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=5-15
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết open_rgb(value). Đọc ảnh từ path, bytes hoặc PIL và chuyển RGB; báo rõ ảnh hỏng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m007"></a>

```text
Tóm tắt giao diện open_rgb(value) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của open_rgb(value) tại đây.
```

## Bước 2: Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…

<!-- module
ID: m011
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=2-14
Summary budget: input=1200; output=500; reserve=300
Config owns: ARTIFACT_DIR, CACHE_DIR, DATASET_ID, DATASET_REVISION, LOCAL_DATA_DIR, SOURCE, USE_LOCAL_DATA, VARIANT
Config uses: ARTIFACT_DIR, CACHE_DIR, DATASET_ID, DATASET_REVISION, LOCAL_DATA_DIR, SOURCE, USE_LOCAL_DATA, VARIANT
-->

### Prompt tạo hoặc sửa module

```text
Viết Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…. Giữ ràng buộc: DATASET_ID=SOURCE['dataset']; DATASET_REVISION=SOURCE['revision']. Tự chọn USE_LOCAL_DATA, VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m011"></a>

```text
Tóm tắt giao diện Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags… tại đây.
```

## Bước 3: canonical_bio(tags)

<!-- module
ID: m012
Depends: m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=16-24
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)

### Prompt tạo hoặc sửa module

```text
Viết canonical_bio(tags). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện canonical_bio(tags) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của canonical_bio(tags) tại đây.
```

## Bước 4: adapt(dataset, split)

<!-- module
ID: m013
Depends: m012
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=26-44
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_bio(tags)](#context-m012)

### Prompt tạo hoặc sửa module

```text
Viết adapt(dataset, split). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Nguồn dữ liệu dataset: nielsr/funsd; revision: 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — canonical_bio(tags):
[Dán tóm tắt m012]
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện adapt(dataset, split) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — canonical_bio(tags):
[Dán tóm tắt m012]
```

```python
# Dán code hiện tại của adapt(dataset, split) tại đây.
```

## Bước 5: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m014
Depends: m011, m013
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=46-54
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, VAL_FRACTION
Config uses: SEED, VAL_FRACTION
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [adapt(dataset, split)](#context-m013)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: nielsr/funsd; revision: 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: SEED=42; VAL_FRACTION=0.2. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m013 — adapt(dataset, split):
[Dán tóm tắt m013]
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m013 — adapt(dataset, split):
[Dán tóm tắt m013]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 6: document_key(row)

<!-- module
ID: m015
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=56-57
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết document_key(row). Tạo khóa tài liệu từ nội dung words và boxes để phát hiện tài liệu trùng giữa các split, thay vì chỉ dựa ID. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện document_key(row) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của document_key(row) tại đây.
```

## Bước 7: Đọc FUNSD và chia theo tài liệu

<!-- module
ID: m016
Depends: m014, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=58-65
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [document_key(row)](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết Đọc FUNSD và chia theo tài liệu. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — document_key(row):
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện Đọc FUNSD và chia theo tài liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — document_key(row):
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của Đọc FUNSD và chia theo tài liệu tại đây.
```

## Bước 8: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m017
Depends: m011, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=1-8
Summary budget: input=1200; output=500; reserve=300
Config owns: IMAGE_SIZE, MAX_WORD_CHARS
Config uses: IMAGE_SIZE, MAX_WORD_CHARS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [Chuẩn hóa và chia dữ liệu](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Tự chọn IMAGE_SIZE, MAX_WORD_CHARS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Bảng ký tự và trạng thái tokenizer tại đây.
```

## Bước 9: restore_tokenizer(state)

<!-- module
ID: m018
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=9-11
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết restore_tokenizer(state). Bổ sung char vocabulary train-only và lưu cùng word_vocab, TAG_NAMES để reload. PAD0 UNK1; TOKENIZER_STATE word_vocab,char_vocab,tag_names; MAX_WORD_CHARS32,IMAGE_SIZE384; không học vocabulary từ validation/test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m018"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 10: FormDataset.__init__(self, records)

<!-- module
ID: m019
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=14-15
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết FormDataset.__init__(self, records). Bổ sung RGB image tensor và chars vào FormDataset/collate_batch, giữ box/tag/word alignment. Viết method trong class FormDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện FormDataset.__init__(self, records) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của FormDataset.__init__(self, records) tại đây.
```

## Bước 11: FormDataset.__len__(self)

<!-- module
ID: m020
Depends: m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=16-17
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [FormDataset.__init__(self, records)](#context-m019)

### Prompt tạo hoặc sửa module

```text
Viết FormDataset.__len__(self). Bổ sung RGB image tensor và chars vào FormDataset/collate_batch, giữ box/tag/word alignment. Viết method trong class FormDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện FormDataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của FormDataset.__len__(self) tại đây.
```

## Bước 12: FormDataset.__getitem__(self, index)

<!-- module
ID: m021
Depends: m007, m017, m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=18-21
Summary budget: input=1200; output=500; reserve=300
Config uses: IMAGE_SIZE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [open_rgb(value)](#context-m007)
2. [Bảng ký tự và trạng thái tokenizer](#context-m017)
3. [FormDataset.__init__(self, records)](#context-m019)

### Prompt tạo hoặc sửa module

```text
Viết FormDataset.__getitem__(self, index). Bổ sung RGB image tensor và chars vào FormDataset/collate_batch, giữ box/tag/word alignment. Giữ thứ tự đầu ra: image, record. Dùng lại IMAGE_SIZE từ context, không chọn hoặc khai báo lại giá trị. Viết method trong class FormDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện FormDataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của FormDataset.__getitem__(self, index) tại đây.
```

## Bước 13: Ghép giao diện class FormDataset

<!-- module
ID: m022
Depends: m019, m020, m021
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=10; lines=13-21
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [FormDataset.__init__(self, records)](#context-m019)
2. [FormDataset.__len__(self)](#context-m020)
3. [FormDataset.__getitem__(self, index)](#context-m021)

<a id="context-m022"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class FormDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]

Context m020 — FormDataset.__len__(self):
[Dán tóm tắt m020]

Context m021 — FormDataset.__getitem__(self, index):
[Dán tóm tắt m021]
```

## Bước 14: collate_batch(items)

<!-- module
ID: m023
Depends: m017
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=23-42
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_WORD_CHARS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m017)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường boxes, chars, ids, images, lengths, mask, records, tags, words. Dùng lại MAX_WORD_CHARS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 15: to_device(batch)

<!-- module
ID: m008
Depends: m023
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=17-19
Summary budget: input=1200; output=500; reserve=300
Config owns: DEVICE
Config uses: DEVICE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)

### Prompt tạo hoặc sửa module

```text
Viết to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]
```

### Tóm tắt context để dùng tiếp

<a id="context-m008"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 16: make_loader(dataset, train=False)

<!-- module
ID: m009
Depends: m014, m023
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=21-24
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [collate_batch(items)](#context-m023)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m023 — collate_batch(items):
[Dán tóm tắt m023]
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m023 — collate_batch(items):
[Dán tóm tắt m023]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 17: Tạo DataLoader cho các split

<!-- module
ID: m024
Depends: m009, m014, m017, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=44-47
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m009)
2. [Chuẩn hóa và chia dữ liệu](#context-m014)
3. [Bảng ký tự và trạng thái tokenizer](#context-m017)
4. [Ghép giao diện class FormDataset](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m022 — Ghép giao diện class FormDataset:
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m022 — Ghép giao diện class FormDataset:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 18: EntityCRF.__init__(self, names)

<!-- module
ID: m025
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=2-15
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết EntityCRF.__init__(self, names). Dựng EntityCRF với start/end/transitions và ràng buộc BIO theo TAG_NAMES. Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class EntityCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện EntityCRF.__init__(self, names) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của EntityCRF.__init__(self, names) tại đây.
```

## Bước 19: EntityCRF.potentials(self)

<!-- module
ID: m026
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=16-17
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [EntityCRF.__init__(self, names)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết EntityCRF.potentials(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Viết method trong class EntityCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện EntityCRF.potentials(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của EntityCRF.potentials(self) tại đây.
```

## Bước 20: EntityCRF.loss(self, emissions, tags, mask)

<!-- module
ID: m027
Depends: m025, m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=18-29
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [EntityCRF.__init__(self, names)](#context-m025)
2. [EntityCRF.potentials(self)](#context-m026)

### Prompt tạo hoặc sửa module

```text
Viết EntityCRF.loss(self, emissions, tags, mask). Viết loss CRF bằng logsumexp partition và gold path có mask, chuẩn hóa theo lengths. Viết method trong class EntityCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]

Context m026 — EntityCRF.potentials(self):
[Dán tóm tắt m026]
```

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện EntityCRF.loss(self, emissions, tags, mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]

Context m026 — EntityCRF.potentials(self):
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của EntityCRF.loss(self, emissions, tags, mask) tại đây.
```

## Bước 21: EntityCRF.decode(self, emissions, mask)

<!-- module
ID: m028
Depends: m025, m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=30-45
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [EntityCRF.__init__(self, names)](#context-m025)
2. [EntityCRF.potentials(self)](#context-m026)

### Prompt tạo hoặc sửa module

```text
Viết EntityCRF.decode(self, emissions, mask). Viết Viterbi có mask và backtracking, trả chuỗi nhãn đúng độ dài mỗi tài liệu. Viết method trong class EntityCRF kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]

Context m026 — EntityCRF.potentials(self):
[Dán tóm tắt m026]
```

### Tóm tắt context để dùng tiếp

<a id="context-m028"></a>

```text
Tóm tắt giao diện EntityCRF.decode(self, emissions, mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]

Context m026 — EntityCRF.potentials(self):
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của EntityCRF.decode(self, emissions, mask) tại đây.
```

## Bước 22: Ghép giao diện class EntityCRF

<!-- module
ID: m029
Depends: m025, m026, m027, m028
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=1-45
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [EntityCRF.__init__(self, names)](#context-m025)
2. [EntityCRF.potentials(self)](#context-m026)
3. [EntityCRF.loss(self, emissions, tags, mask)](#context-m027)
4. [EntityCRF.decode(self, emissions, mask)](#context-m028)

<a id="context-m029"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class EntityCRF kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m025 — EntityCRF.__init__(self, names):
[Dán tóm tắt m025]

Context m026 — EntityCRF.potentials(self):
[Dán tóm tắt m026]

Context m027 — EntityCRF.loss(self, emissions, tags, mask):
[Dán tóm tắt m027]

Context m028 — EntityCRF.decode(self, emissions, mask):
[Dán tóm tắt m028]
```

## Bước 23: FormModel.__init__(self, cfg)

<!-- module
ID: m031
Depends: m011, m017, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=3-16
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, HIDDEN_DIM
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [Bảng ký tự và trạng thái tokenizer](#context-m017)
3. [Ghép giao diện class EntityCRF](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết FormModel.__init__(self, cfg). Thay constructor FormModel bằng word/char/layout/visual CNN, BiLSTM và EntityCRF. Tự chọn DROPOUT, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa n_words, n_chars, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m029 — Ghép giao diện class EntityCRF:
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện FormModel.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m029 — Ghép giao diện class EntityCRF:
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của FormModel.__init__(self, cfg) tại đây.
```

## Bước 24: FormModel.forward(self, batch)

<!-- module
ID: m032
Depends: m023, m031
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=17-38
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [FormModel.__init__(self, cfg)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết FormModel.forward(self, batch). Viết phần đầu forward trích char-CNN masked pooling và layout từ tọa độ cùng kích thước box. Viết phần ROI Align theo boxes và ghép visual features về đúng word của từng tài liệu. Hoàn tất forward: ghép word,char,layout,ROI, chạy packed BiLSTM và tag head, nối loss/decode CRF. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m032"></a>

```text
Tóm tắt giao diện FormModel.forward(self, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của FormModel.forward(self, batch) tại đây.
```

## Bước 25: FormModel.loss(self, emissions, batch)

<!-- module
ID: m033
Depends: m023, m031
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=39-40
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [FormModel.__init__(self, cfg)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết FormModel.loss(self, emissions, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện FormModel.loss(self, emissions, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của FormModel.loss(self, emissions, batch) tại đây.
```

## Bước 26: FormModel.decode(self, emissions, batch)

<!-- module
ID: m034
Depends: m023, m031
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=41-42
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [FormModel.__init__(self, cfg)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết FormModel.decode(self, emissions, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện FormModel.decode(self, emissions, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của FormModel.decode(self, emissions, batch) tại đây.
```

## Bước 27: Ghép giao diện class FormModel

<!-- module
ID: m035
Depends: m031, m032, m033, m034
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=2-42
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [FormModel.__init__(self, cfg)](#context-m031)
2. [FormModel.forward(self, batch)](#context-m032)
3. [FormModel.loss(self, emissions, batch)](#context-m033)
4. [FormModel.decode(self, emissions, batch)](#context-m034)

<a id="context-m035"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class FormModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]

Context m032 — FormModel.forward(self, batch):
[Dán tóm tắt m032]

Context m033 — FormModel.loss(self, emissions, batch):
[Dán tóm tắt m033]

Context m034 — FormModel.decode(self, emissions, batch):
[Dán tóm tắt m034]
```

## Bước 28: build_model(cfg, initialize_pretrained=True)

<!-- module
ID: m036
Depends: m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=43-44
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class FormModel](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết build_model(cfg, initialize_pretrained=True). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện build_model(cfg, initialize_pretrained=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của build_model(cfg, initialize_pretrained=True) tại đây.
```

## Bước 29: entity_spans(tags)

<!-- module
ID: m038
Depends: m011, m012
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=1-10
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [canonical_bio(tags)](#context-m012)

### Prompt tạo hoặc sửa module

```text
Viết entity_spans(tags). Viết riêng entity_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. entity_spans nhận (tags); Sửa BIO bằng canonical_bio, đóng entity khi O/B/khác type; trả start_word,end_word exclusive,type, giữ thứ tự words. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m012 — canonical_bio(tags):
[Dán tóm tắt m012]
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện entity_spans(tags) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m012 — canonical_bio(tags):
[Dán tóm tắt m012]
```

```python
# Dán code hiện tại của entity_spans(tags) tại đây.
```

## Bước 30: compute_loss(model, batch)

<!-- module
ID: m039
Depends: m023, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=12-13
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [Ghép giao diện class FormModel](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết compute_loss(model, batch). Viết loss scalar nối model và batch theo hợp đồng, bỏ padding và giữ gradient; kiểm loss hữu hạn. AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. Dùng các API forward/loss đã dựng, không dùng gold trong đường suy luận. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m039"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 31: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m040
Depends: m008, m035, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=15-37
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Ghép giao diện class FormModel](#context-m035)
3. [entity_spans(tags)](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, return_predictions=False). Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]

Context m038 — entity_spans(tags):
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m040"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]

Context m038 — entity_spans(tags):
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 32: make_optimizer(model)

<!-- module
ID: m041
Depends: m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=3-11
Summary budget: input=1200; output=500; reserve=300
Config owns: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
Config uses: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class FormModel](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết make_optimizer(model). Khởi tạo optimizer AdamW cho tham số có thể học. Tách nhóm backbone và task head theo cấu trúc model. Tự chọn BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện make_optimizer(model) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của make_optimizer(model) tại đây.
```

## Bước 33: Huấn luyện và checkpoint — best_score, model, optimizer…

<!-- module
ID: m042
Depends: m008, m031, m036, m041
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=1-6
Summary budget: input=1200; output=500; reserve=300
Config owns: DIRECTION, EPOCHS
Config uses: DEVICE, DIRECTION, EPOCHS, MODEL_CONFIG
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [FormModel.__init__(self, cfg)](#context-m031)
3. [build_model(cfg, initialize_pretrained=True)](#context-m036)
4. [make_optimizer(model)](#context-m041)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — best_score, model, optimizer…. Giữ ràng buộc: DIRECTION='max'. Tự chọn EPOCHS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]

Context m036 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m036]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, model, optimizer… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]

Context m036 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m036]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — best_score, model, optimizer… tại đây.
```

## Bước 34: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m043
Depends: m008, m011, m017, m024, m039, m040, m042
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=7-44
Summary budget: input=1200; output=500; reserve=300
Config owns: CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DIRECTION, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
3. [Bảng ký tự và trạng thái tokenizer](#context-m017)
4. [Tạo DataLoader cho các split](#context-m024)
5. [compute_loss(model, batch)](#context-m039)
6. [evaluate(model, loader, return_predictions=False)](#context-m040)
7. [Huấn luyện và checkpoint — best_score, model, optimizer…](#context-m042)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='entity_f1'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, DIRECTION, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m024 — Tạo DataLoader cho các split:
[Dán tóm tắt m024]

Context m039 — compute_loss(model, batch):
[Dán tóm tắt m039]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m042 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m042]
```

### Tóm tắt context để dùng tiếp

<a id="context-m043"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m024 — Tạo DataLoader cho các split:
[Dán tóm tắt m024]

Context m039 — compute_loss(model, batch):
[Dán tóm tắt m039]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m042 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m042]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 35: Reload và đánh giá test — best_model, checkpoint, inference_config…

<!-- module
ID: m044
Depends: m008, m011, m016, m017, m018, m024, m031, m036, m040, m043
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=21; lines=1-18
Summary budget: input=1200; output=500; reserve=300
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
3. [Đọc FUNSD và chia theo tài liệu](#context-m016)
4. [Bảng ký tự và trạng thái tokenizer](#context-m017)
5. [restore_tokenizer(state)](#context-m018)
6. [Tạo DataLoader cho các split](#context-m024)
7. [FormModel.__init__(self, cfg)](#context-m031)
8. [build_model(cfg, initialize_pretrained=True)](#context-m036)
9. [evaluate(model, loader, return_predictions=False)](#context-m040)
10. [Vòng huấn luyện, validation và lưu checkpoint](#context-m043)

### Prompt tạo hoặc sửa module

```text
Viết Reload và đánh giá test — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m016 — Đọc FUNSD và chia theo tài liệu:
[Dán tóm tắt m016]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m018 — restore_tokenizer(state):
[Dán tóm tắt m018]

Context m024 — Tạo DataLoader cho các split:
[Dán tóm tắt m024]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]

Context m036 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m036]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m043 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m043]
```

### Tóm tắt context để dùng tiếp

<a id="context-m044"></a>

```text
Tóm tắt giao diện Reload và đánh giá test — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m016 — Đọc FUNSD và chia theo tài liệu:
[Dán tóm tắt m016]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m018 — restore_tokenizer(state):
[Dán tóm tắt m018]

Context m024 — Tạo DataLoader cho các split:
[Dán tóm tắt m024]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]

Context m036 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m036]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m043 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m043]
```

```python
# Dán code hiện tại của Reload và đánh giá test — best_model, checkpoint, inference_config… tại đây.
```

## Bước 36: Reload và đánh giá test

<!-- module
ID: m045
Depends: m044
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=21; lines=19-19
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Reload và đánh giá test — best_model, checkpoint, inference_config…](#context-m044)

### Prompt tạo hoặc sửa module

```text
Viết Reload và đánh giá test. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m044 — Reload và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện Reload và đánh giá test từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m044 — Reload và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của Reload và đánh giá test tại đây.
```

## Bước 37: parse_form(image, words, bboxes)

<!-- module
ID: m046
Depends: m008, m022, m023, m038, m044
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=23; lines=1-14
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Ghép giao diện class FormDataset](#context-m022)
3. [collate_batch(items)](#context-m023)
4. [entity_spans(tags)](#context-m038)
5. [Reload và đánh giá test — best_model, checkpoint, inference_config…](#context-m044)

### Prompt tạo hoặc sửa module

```text
Viết parse_form(image, words, bboxes). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Hàm parse_form nhận (image, words, bboxes), giữ chữ ký và output schema đã chốt, không cần gold. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m022 — Ghép giao diện class FormDataset:
[Dán tóm tắt m022]

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m038 — entity_spans(tags):
[Dán tóm tắt m038]

Context m044 — Reload và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện parse_form(image, words, bboxes) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m022 — Ghép giao diện class FormDataset:
[Dán tóm tắt m022]

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m038 — entity_spans(tags):
[Dán tóm tắt m038]

Context m044 — Reload và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của parse_form(image, words, bboxes) tại đây.
```

## Bước 38: Phân tích tài liệu mới — sample

<!-- module
ID: m047
Depends: m014, m046
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=23; lines=16-17
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [parse_form(image, words, bboxes)](#context-m046)

### Prompt tạo hoặc sửa module

```text
Viết Phân tích tài liệu mới — sample. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m046 — parse_form(image, words, bboxes):
[Dán tóm tắt m046]
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện Phân tích tài liệu mới — sample từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m046 — parse_form(image, words, bboxes):
[Dán tóm tắt m046]
```

```python
# Dán code hiện tại của Phân tích tài liệu mới — sample tại đây.
```

## Bước 39: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m045, m047
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Reload và đánh giá test](#context-m045)
2. [Phân tích tài liệu mới — sample](#context-m047)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m045 — Reload và đánh giá test:
[Dán tóm tắt m045]

Context m047 — Phân tích tài liệu mới — sample:
[Dán tóm tắt m047]
```
