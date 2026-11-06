# Prompt hardcore: IAM-line: ResNet18, BiLSTM và CTC

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

## Bước 2: Dữ liệu và split — raw

<!-- module
ID: m011
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=2-10
Summary budget: input=1200; output=500; reserve=300
Config owns: ARTIFACT_DIR, CACHE_DIR, DATASET_ID, DATASET_REVISION, LOCAL_DATA_DIR, SOURCE, USE_LOCAL_DATA, VARIANT
Config uses: ARTIFACT_DIR, CACHE_DIR, DATASET_ID, DATASET_REVISION, LOCAL_DATA_DIR, SOURCE, USE_LOCAL_DATA, VARIANT
-->

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu và split — raw. Giữ ràng buộc: DATASET_ID=SOURCE['dataset']; DATASET_REVISION=SOURCE['revision']. Tự chọn USE_LOCAL_DATA, VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m011"></a>

```text
Tóm tắt giao diện Dữ liệu và split — raw từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Dữ liệu và split — raw tại đây.
```

## Bước 3: normalize_text(text)

<!-- module
ID: m012
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=12-13
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết normalize_text(text). Chuẩn hóa văn bản theo quy ước nhãn của bài toán; dùng nhất quán khi xử lý dữ liệu và đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện normalize_text(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của normalize_text(text) tại đây.
```

## Bước 4: adapt_split(dataset, split)

<!-- module
ID: m013
Depends: m012
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=15-25
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_text(text)](#context-m012)

### Prompt tạo hoặc sửa module

```text
Viết adapt_split(dataset, split). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Nguồn dữ liệu dataset: Teklia/IAM-line; revision: fbdad97500ce54635c0d1ba306bf535cb40656cf. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — normalize_text(text):
[Dán tóm tắt m012]
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện adapt_split(dataset, split) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — normalize_text(text):
[Dán tóm tắt m012]
```

```python
# Dán code hiện tại của adapt_split(dataset, split) tại đây.
```

## Bước 5: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m014
Depends: m011, m013
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=27-35
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và split — raw](#context-m011)
2. [adapt_split(dataset, split)](#context-m013)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: Teklia/IAM-line; revision: fbdad97500ce54635c0d1ba306bf535cb40656cf. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m013 — adapt_split(dataset, split):
[Dán tóm tắt m013]
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m013 — adapt_split(dataset, split):
[Dán tóm tắt m013]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 6: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m015
Depends: m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=1-9
Summary budget: input=1200; output=500; reserve=300
Config owns: IMAGE_HEIGHT
Config uses: IMAGE_HEIGHT
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Tự chọn IMAGE_HEIGHT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Bảng ký tự và trạng thái tokenizer tại đây.
```

## Bước 7: restore_tokenizer(state)

<!-- module
ID: m016
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=11-14
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết restore_tokenizer(state). Giữ alphabet train-only, blank/PAD và UNK; lưu mapping để reload CTC nhất quán. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 8: image_tensor(value)

<!-- module
ID: m017
Depends: m007, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=16-23
Summary budget: input=1200; output=500; reserve=300
Config uses: IMAGE_HEIGHT
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [open_rgb(value)](#context-m007)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết image_tensor(value). Nâng image_tensor sang RGB chiều cao theo cấu hình giữ tỷ lệ, chuẩn ImageNet; không cắt chiều rộng làm mất chữ. Dùng lại IMAGE_HEIGHT từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện image_tensor(value) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của image_tensor(value) tại đây.
```

## Bước 9: OCRDataset.__init__(self, records)

<!-- module
ID: m018
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=26-27
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết OCRDataset.__init__(self, records). Khởi tạo các thành phần và trạng thái cần cho class. Viết method trong class OCRDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m018"></a>

```text
Tóm tắt giao diện OCRDataset.__init__(self, records) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của OCRDataset.__init__(self, records) tại đây.
```

## Bước 10: OCRDataset.__len__(self)

<!-- module
ID: m019
Depends: m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=28-29
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [OCRDataset.__init__(self, records)](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết OCRDataset.__len__(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class OCRDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện OCRDataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của OCRDataset.__len__(self) tại đây.
```

## Bước 11: OCRDataset.__getitem__(self, index)

<!-- module
ID: m020
Depends: m015, m017, m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=30-33
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)
2. [image_tensor(value)](#context-m017)
3. [OCRDataset.__init__(self, records)](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết OCRDataset.__getitem__(self, index). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, target, record. Viết method trong class OCRDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện OCRDataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của OCRDataset.__getitem__(self, index) tại đây.
```

## Bước 12: Ghép giao diện class OCRDataset

<!-- module
ID: m021
Depends: m018, m019, m020
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=10; lines=25-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [OCRDataset.__init__(self, records)](#context-m018)
2. [OCRDataset.__len__(self)](#context-m019)
3. [OCRDataset.__getitem__(self, index)](#context-m020)

<a id="context-m021"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class OCRDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]

Context m019 — OCRDataset.__len__(self):
[Dán tóm tắt m019]

Context m020 — OCRDataset.__getitem__(self, index):
[Dán tóm tắt m020]
```

## Bước 13: collate_batch(items)

<!-- module
ID: m022
Depends: m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=35-48
Summary budget: input=1200; output=500; reserve=300
Config uses: IMAGE_HEIGHT
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Cập nhật padding nền trắng đã chuẩn hóa và kiểm tra số bước CTC kể cả ký tự lặp. Kết quả giữ các trường ids, images, target_lengths, targets, texts, widths. Dùng lại IMAGE_HEIGHT từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 14: to_device(batch)

<!-- module
ID: m008
Depends: m022
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

1. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m008"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 15: make_loader(dataset, train=False)

<!-- module
ID: m009
Depends: m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=21-24
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Giữ ràng buộc: SEED=42. Tự chọn BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: EVAL_BATCH_SIZE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 16: Tạo DataLoader cho các split

<!-- module
ID: m023
Depends: m009, m014, m015, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=50-56
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m009)
2. [Chuẩn hóa và chia dữ liệu](#context-m014)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)
4. [Ghép giao diện class OCRDataset](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m021 — Ghép giao diện class OCRDataset:
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m021 — Ghép giao diện class OCRDataset:
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 17: OCRModel.__init__(self, cfg, initialize_pretrained=True)

<!-- module
ID: m025
Depends: m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=4-22
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, HIDDEN_DIM
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết OCRModel.__init__(self, cfg, initialize_pretrained=True). Thay CNN scratch bằng ResNet18 ImageNet V1, chỉnh stride để chiều ngang giảm4 và chiều cao giảm32. OCRModel(cfg,initialize_pretrained=True); conv1(2,1), maxpool(2,2), layer2(2,1),layer3(2,2),layer4(2,1), cả conv1/downsample; tải sai weights phải lỗi, fc=Identity. Bổ sung BiLSTM số lớp được chọn và CTC head vào constructor OCRModel, giữ cfg và build_model. Tự chọn DROPOUT, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa hidden_dim, dropout, num_classes; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class OCRModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện OCRModel.__init__(self, cfg, initialize_pretrained=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của OCRModel.__init__(self, cfg, initialize_pretrained=True) tại đây.
```

## Bước 18: OCRModel.forward(self, images, widths)

<!-- module
ID: m026
Depends: m022, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=24-33
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [OCRModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết OCRModel.forward(self, images, widths). Thay forward để lấy feature map ResNet, trung bình chiều cao và chạy packed BiLSTM theo widths//4. Giữ thứ tự đầu ra: kết quả 1, lengths. Viết method trong class OCRModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện OCRModel.forward(self, images, widths) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của OCRModel.forward(self, images, widths) tại đây.
```

## Bước 19: Ghép giao diện class OCRModel

<!-- module
ID: m027
Depends: m025, m026
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=3-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [OCRModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
2. [OCRModel.forward(self, images, widths)](#context-m026)

<a id="context-m027"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class OCRModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m026 — OCRModel.forward(self, images, widths):
[Dán tóm tắt m026]
```

## Bước 20: build_model(cfg, initialize_pretrained=True)

<!-- module
ID: m028
Depends: m027
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=35-36
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class OCRModel](#context-m027)

### Prompt tạo hoặc sửa module

```text
Viết build_model(cfg, initialize_pretrained=True). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]
```

### Tóm tắt context để dùng tiếp

<a id="context-m028"></a>

```text
Tóm tắt giao diện build_model(cfg, initialize_pretrained=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]
```

```python
# Dán code hiện tại của build_model(cfg, initialize_pretrained=True) tại đây.
```

## Bước 21: edit_distance(a, b)

<!-- module
ID: m030
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=1-8
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết edit_distance(a, b). Viết riêng edit_distance theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. edit_distance nhận (a, b); Levenshtein DP hai hàng, áp dụng được chuỗi ký tự và danh sách từ. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện edit_distance(a, b) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của edit_distance(a, b) tại đây.
```

## Bước 22: ctc_decode(path)

<!-- module
ID: m031
Depends: m012, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=10-17
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_text(text)](#context-m012)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết ctc_decode(path). Viết riêng ctc_decode theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. ctc_decode nhận (path); Gộp nhãn lặp liền, blank reset previous; bỏ blank0, UNK thành replacement char, normalize_text như gold. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — normalize_text(text):
[Dán tóm tắt m012]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện ctc_decode(path) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — normalize_text(text):
[Dán tóm tắt m012]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của ctc_decode(path) tại đây.
```

## Bước 23: CER/WER và loss — ctc_loss

<!-- module
ID: m032
Depends: m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=19-19
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết CER/WER và loss — ctc_loss. Khởi tạo CTC loss với blank=0, reduction mean và zero_infinity=False; kiểm độ dài hợp lệ. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m032"></a>

```text
Tóm tắt giao diện CER/WER và loss — ctc_loss từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của CER/WER và loss — ctc_loss tại đây.
```

## Bước 24: compute_loss(model, batch)

<!-- module
ID: m033
Depends: m022, m027, m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=20-24
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [Ghép giao diện class OCRModel](#context-m027)
3. [CER/WER và loss — ctc_loss](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết compute_loss(model, batch). Giữ CTCLoss blank0 với logp đổi sang T,B,V, input lengths đúng và target_lengths thật. Khai báo ctc_loss=CTCLoss(blank=0,reduction mean,zero_infinity=False); lengths không vượt T. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]

Context m032 — CER/WER và loss — ctc_loss:
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]

Context m032 — CER/WER và loss — ctc_loss:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 25: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m034
Depends: m008, m027, m030, m031
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=26-48
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Ghép giao diện class OCRModel](#context-m027)
3. [edit_distance(a, b)](#context-m030)
4. [ctc_decode(path)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, return_predictions=False). Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]

Context m030 — edit_distance(a, b):
[Dán tóm tắt m030]

Context m031 — ctc_decode(path):
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]

Context m030 — edit_distance(a, b):
[Dán tóm tắt m030]

Context m031 — ctc_decode(path):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 26: make_optimizer(model)

<!-- module
ID: m035
Depends: m027
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=3-11
Summary budget: input=1200; output=500; reserve=300
Config owns: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
Config uses: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class OCRModel](#context-m027)

### Prompt tạo hoặc sửa module

```text
Viết make_optimizer(model). Khởi tạo optimizer AdamW cho tham số có thể học. Tách nhóm backbone và task head theo cấu trúc model. Tự chọn BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện make_optimizer(model) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]
```

```python
# Dán code hiện tại của make_optimizer(model) tại đây.
```

## Bước 27: Huấn luyện và checkpoint — best_score, model, optimizer…

<!-- module
ID: m036
Depends: m008, m025, m028, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=1-6
Summary budget: input=1200; output=500; reserve=300
Config owns: DIRECTION, EPOCHS
Config uses: DEVICE, DIRECTION, EPOCHS, MODEL_CONFIG
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [OCRModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
3. [build_model(cfg, initialize_pretrained=True)](#context-m028)
4. [make_optimizer(model)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — best_score, model, optimizer…. Giữ ràng buộc: DIRECTION='min'. Tự chọn EPOCHS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m028 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m028]

Context m035 — make_optimizer(model):
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, model, optimizer… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m028 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m028]

Context m035 — make_optimizer(model):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — best_score, model, optimizer… tại đây.
```

## Bước 28: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m037
Depends: m008, m011, m015, m023, m033, m034, m036
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=7-44
Summary budget: input=1200; output=500; reserve=300
Config owns: CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DIRECTION, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Dữ liệu và split — raw](#context-m011)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)
4. [Tạo DataLoader cho các split](#context-m023)
5. [compute_loss(model, batch)](#context-m033)
6. [evaluate(model, loader, return_predictions=False)](#context-m034)
7. [Huấn luyện và checkpoint — best_score, model, optimizer…](#context-m036)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='cer'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, DIRECTION, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m033 — compute_loss(model, batch):
[Dán tóm tắt m033]

Context m034 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m034]

Context m036 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m036]
```

### Tóm tắt context để dùng tiếp

<a id="context-m037"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m033 — compute_loss(model, batch):
[Dán tóm tắt m033]

Context m034 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m034]

Context m036 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m036]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 29: Tải lại và đánh giá test — best_model, checkpoint, inference_config…

<!-- module
ID: m038
Depends: m008, m011, m014, m015, m016, m023, m025, m028, m034, m037
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=1-18
Summary budget: input=1200; output=500; reserve=300
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Dữ liệu và split — raw](#context-m011)
3. [Chuẩn hóa và chia dữ liệu](#context-m014)
4. [Bảng ký tự và trạng thái tokenizer](#context-m015)
5. [restore_tokenizer(state)](#context-m016)
6. [Tạo DataLoader cho các split](#context-m023)
7. [OCRModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
8. [build_model(cfg, initialize_pretrained=True)](#context-m028)
9. [evaluate(model, loader, return_predictions=False)](#context-m034)
10. [Vòng huấn luyện, validation và lưu checkpoint](#context-m037)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá test — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m016 — restore_tokenizer(state):
[Dán tóm tắt m016]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m028 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m028]

Context m034 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m034]

Context m037 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m037]
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá test — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m016 — restore_tokenizer(state):
[Dán tóm tắt m016]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m028 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m028]

Context m034 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m034]

Context m037 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m037]
```

```python
# Dán code hiện tại của Tải lại và đánh giá test — best_model, checkpoint, inference_config… tại đây.
```

## Bước 30: Tải lại và đánh giá test

<!-- module
ID: m039
Depends: m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=19-19
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá test — best_model, checkpoint, inference_config…](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá test. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m038 — Tải lại và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m039"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá test từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m038 — Tải lại và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của Tải lại và đánh giá test tại đây.
```

## Bước 31: recognize_line(image)

<!-- module
ID: m040
Depends: m008, m017, m031, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=1-7
Summary budget: input=1200; output=500; reserve=300
Config uses: DEVICE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [image_tensor(value)](#context-m017)
3. [ctc_decode(path)](#context-m031)
4. [Tải lại và đánh giá test — best_model, checkpoint, inference_config…](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết recognize_line(image). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Hàm recognize_line nhận (image), giữ chữ ký và output schema đã chốt, không cần gold. Dùng lại DEVICE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m031 — ctc_decode(path):
[Dán tóm tắt m031]

Context m038 — Tải lại và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m040"></a>

```text
Tóm tắt giao diện recognize_line(image) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m031 — ctc_decode(path):
[Dán tóm tắt m031]

Context m038 — Tải lại và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của recognize_line(image) tại đây.
```

## Bước 32: Đọc dòng mới

<!-- module
ID: m041
Depends: m014, m040
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=9-9
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [recognize_line(image)](#context-m040)

### Prompt tạo hoặc sửa module

```text
Viết Đọc dòng mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m040 — recognize_line(image):
[Dán tóm tắt m040]
```

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện Đọc dòng mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m040 — recognize_line(image):
[Dán tóm tắt m040]
```

```python
# Dán code hiện tại của Đọc dòng mới tại đây.
```

## Bước 33: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m039, m041
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá test](#context-m039)
2. [Đọc dòng mới](#context-m041)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m039 — Tải lại và đánh giá test:
[Dán tóm tắt m039]

Context m041 — Đọc dòng mới:
[Dán tóm tắt m041]
```
