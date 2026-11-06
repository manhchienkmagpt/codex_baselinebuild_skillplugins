# Prompt hardcore: COCO2014: ResNet50 và attention LSTM captioning

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

## Bước 2: coco_records(image_root, annotation_file, prefix)

<!-- module
ID: m011
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=2-17
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết coco_records(image_root, annotation_file, prefix). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. MS COCO Captions2014 local DATA_ROOT/train2014,val2014,annotations/captions_train2014.json và captions_val2014.json; CocoCaptions cần pycocotools. Record id,image_id,image,refs; train2014 là train; val2014 chia5000 validation và held-out còn lại theo image, seed42; không phải Karpathy/test server. Nguồn dữ liệu dataset: MS COCO Captions 2014. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m011"></a>

```text
Tóm tắt giao diện coco_records(image_root, annotation_file, prefix) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của coco_records(image_root, annotation_file, prefix) tại đây.
```

## Bước 3: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m012
Depends: m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=19-34
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_ROOT, SEED, VALIDATION_IMAGE_COUNT
Config uses: DATA_ROOT, SEED, VALIDATION_IMAGE_COUNT
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [coco_records(image_root, annotation_file, prefix)](#context-m011)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: MS COCO Captions 2014. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: DATA_ROOT=Path('data/coco2014'); SEED=42. Tự chọn VALIDATION_IMAGE_COUNT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — coco_records(image_root, annotation_file, prefix):
[Dán tóm tắt m011]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — coco_records(image_root, annotation_file, prefix):
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 4: Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…

<!-- module
ID: m013
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=1-1
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…. Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID… tại đây.
```

## Bước 5: tokenize(text)

<!-- module
ID: m014
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=2-3
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết tokenize(text). Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện tokenize(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của tokenize(text) tại đây.
```

## Bước 6: Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…

<!-- module
ID: m015
Depends: m012, m013, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=4-13
Summary budget: input=1200; output=500; reserve=300
Config owns: IMAGE_SIZE, MAX_TOKENS, MIN_WORD_FREQUENCY, VOCAB_SIZE
Config uses: IMAGE_SIZE, MAX_TOKENS, MIN_WORD_FREQUENCY, VOCAB_SIZE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m012)
2. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
3. [tokenize(text)](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…. Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Tự chọn IMAGE_SIZE, MAX_TOKENS, MIN_WORD_FREQUENCY, VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts… tại đây.
```

## Bước 7: restore_tokenizer(state)

<!-- module
ID: m016
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=14-17
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết restore_tokenizer(state). Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
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
Target: baseline.ipynb#cell=10; lines=19-27
Summary budget: input=1200; output=500; reserve=300
Config uses: IMAGE_SIZE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [open_rgb(value)](#context-m007)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết image_tensor(value). Nâng image_tensor thành letterbox RGB nền trắng 256 giữ toàn ảnh, rồi chuẩn ImageNet. Dùng lại IMAGE_SIZE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện image_tensor(value) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của image_tensor(value) tại đây.
```

## Bước 9: CaptionDataset.__init__(self, records, training=False)

<!-- module
ID: m018
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=30-34
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết CaptionDataset.__init__(self, records, training=False). Khởi tạo các thành phần và trạng thái cần cho class. Viết method trong class CaptionDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m018"></a>

```text
Tóm tắt giao diện CaptionDataset.__init__(self, records, training=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của CaptionDataset.__init__(self, records, training=False) tại đây.
```

## Bước 10: CaptionDataset.__len__(self)

<!-- module
ID: m019
Depends: m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=35-36
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionDataset.__init__(self, records, training=False)](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết CaptionDataset.__len__(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class CaptionDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện CaptionDataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của CaptionDataset.__len__(self) tại đây.
```

## Bước 11: CaptionDataset.__getitem__(self, index)

<!-- module
ID: m020
Depends: m013, m014, m015, m017, m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=37-42
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [tokenize(text)](#context-m014)
3. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
4. [image_tensor(value)](#context-m017)
5. [CaptionDataset.__init__(self, records, training=False)](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết CaptionDataset.__getitem__(self, index). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2, record. Dùng lại MAX_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Viết method trong class CaptionDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện CaptionDataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của CaptionDataset.__getitem__(self, index) tại đây.
```

## Bước 12: Ghép giao diện class CaptionDataset

<!-- module
ID: m021
Depends: m018, m019, m020
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=10; lines=29-42
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionDataset.__init__(self, records, training=False)](#context-m018)
2. [CaptionDataset.__len__(self)](#context-m019)
3. [CaptionDataset.__getitem__(self, index)](#context-m020)

<a id="context-m021"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class CaptionDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]

Context m019 — CaptionDataset.__len__(self):
[Dán tóm tắt m019]

Context m020 — CaptionDataset.__getitem__(self, index):
[Dán tóm tắt m020]
```

## Bước 13: collate_batch(items)

<!-- module
ID: m022
Depends: m013
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=44-49
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Giữ collate_batch với tokens PAD và image IDs khi đổi encoder; không lặp ảnh lúc đánh giá. Kết quả giữ các trường ids, images, records, tokens. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]
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
Depends: m012, m022
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

1. [Chuẩn hóa và chia dữ liệu](#context-m012)
2. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: EVAL_BATCH_SIZE được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 16: Tạo DataLoader cho các split

<!-- module
ID: m023
Depends: m009, m012, m014, m015, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=51-55
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m009)
2. [Chuẩn hóa và chia dữ liệu](#context-m012)
3. [tokenize(text)](#context-m014)
4. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
5. [Ghép giao diện class CaptionDataset](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Dùng lại MAX_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m014 — tokenize(text):
[Dán tóm tắt m014]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m021 — Ghép giao diện class CaptionDataset:
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m014 — tokenize(text):
[Dán tóm tắt m014]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m021 — Ghép giao diện class CaptionDataset:
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 17: CaptionModel.__init__(self, cfg, initialize_pretrained=True)

<!-- module
ID: m025
Depends: m013, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=3-20
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM, MODEL_CONFIG
Config uses: DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.__init__(self, cfg, initialize_pretrained=True). Thay CNN bằng ResNet50 ImageNet V2, chỉ fine-tune layer4 và giữ BatchNorm eval. Bổ sung embedding, additive attention, initial_h/c, LSTMCell và output head vào CaptionModel. cfg vocab_size,embed_dim256,hidden_dim512,memory_dim256,dropout0.3; decoder input embed+memory; output Linear(hidden+memory,V); memory_attention,hidden_attention,attention_score. Tự chọn DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa vocab_size, embed_dim, hidden_dim, memory_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện CaptionModel.__init__(self, cfg, initialize_pretrained=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của CaptionModel.__init__(self, cfg, initialize_pretrained=True) tại đây.
```

## Bước 18: CaptionModel.train(self, mode=True)

<!-- module
ID: m026
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=21-25
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.train(self, mode=True). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện CaptionModel.train(self, mode=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.train(self, mode=True) tại đây.
```

## Bước 19: CaptionModel.encode(self, images)

<!-- module
ID: m027
Depends: m022, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=26-30
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.encode(self, images). Viết encode lấy feature map ResNet thành spatial memory và start_state từ trung bình memory. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện CaptionModel.encode(self, images) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.encode(self, images) tại đây.
```

## Bước 20: CaptionModel.start_state(self, memory)

<!-- module
ID: m028
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=31-33
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.start_state(self, memory). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m028"></a>

```text
Tóm tắt giao diện CaptionModel.start_state(self, memory) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.start_state(self, memory) tại đây.
```

## Bước 21: CaptionModel.step(self, token, state, memory)

<!-- module
ID: m029
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=34-40
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.step(self, token, state, memory). Viết step chọn spatial context bằng additive attention rồi cập nhật LSTMCell và logits. Giữ thứ tự đầu ra: logits, kết quả 2. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m029"></a>

```text
Tóm tắt giao diện CaptionModel.step(self, token, state, memory) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.step(self, token, state, memory) tại đây.
```

## Bước 22: CaptionModel.forward(self, images, tokens)

<!-- module
ID: m030
Depends: m022, m025, m027, m028, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=41-45
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
3. [CaptionModel.encode(self, images)](#context-m027)
4. [CaptionModel.start_state(self, memory)](#context-m028)
5. [CaptionModel.step(self, token, state, memory)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.forward(self, images, tokens). Viết forward(images,tokens) dùng encode, start_state và step theo từng target token để teacher forcing. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện CaptionModel.forward(self, images, tokens) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của CaptionModel.forward(self, images, tokens) tại đây.
```

## Bước 23: CaptionModel.generate(self, images, max_tokens)

<!-- module
ID: m031
Depends: m013, m022, m025, m027, m028, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=46-57
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [collate_batch(items)](#context-m022)
3. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
4. [CaptionModel.encode(self, images)](#context-m027)
5. [CaptionModel.start_state(self, memory)](#context-m028)
6. [CaptionModel.step(self, token, state, memory)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.generate(self, images, max_tokens). Thay generate bằng greedy decoding dùng step, chặn PAD/BOS và dừng EOS độc lập từng ảnh. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện CaptionModel.generate(self, images, max_tokens) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của CaptionModel.generate(self, images, max_tokens) tại đây.
```

## Bước 24: Ghép giao diện class CaptionModel

<!-- module
ID: m032
Depends: m025, m026, m027, m028, m029, m030, m031
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=2-57
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
2. [CaptionModel.train(self, mode=True)](#context-m026)
3. [CaptionModel.encode(self, images)](#context-m027)
4. [CaptionModel.start_state(self, memory)](#context-m028)
5. [CaptionModel.step(self, token, state, memory)](#context-m029)
6. [CaptionModel.forward(self, images, tokens)](#context-m030)
7. [CaptionModel.generate(self, images, max_tokens)](#context-m031)

<a id="context-m032"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class CaptionModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m026 — CaptionModel.train(self, mode=True):
[Dán tóm tắt m026]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]

Context m030 — CaptionModel.forward(self, images, tokens):
[Dán tóm tắt m030]

Context m031 — CaptionModel.generate(self, images, max_tokens):
[Dán tóm tắt m031]
```

## Bước 25: build_model(cfg, initialize_pretrained=True)

<!-- module
ID: m033
Depends: m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=58-59
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class CaptionModel](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết build_model(cfg, initialize_pretrained=True). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện build_model(cfg, initialize_pretrained=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của build_model(cfg, initialize_pretrained=True) tại đây.
```

## Bước 26: decode_tokens(ids)

<!-- module
ID: m035
Depends: m013, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=1-6
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết decode_tokens(ids). Viết riêng decode_tokens theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. decode_tokens nhận (ids); Bỏ PAD/BOS, dừng EOS, map id_to_word và ghép từ theo baseline, không dùng references để sửa caption. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện decode_tokens(ids) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của decode_tokens(ids) tại đây.
```

## Bước 27: compute_loss(model, batch)

<!-- module
ID: m036
Depends: m013, m022, m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=7-10
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [collate_batch(items)](#context-m022)
3. [Ghép giao diện class CaptionModel](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết compute_loss(model, batch). Giữ cross-entropy teacher forcing, bỏ PAD và dịch caption một token cho attention decoder. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 28: ptb_tokenize(mapping)

<!-- module
ID: m038
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=5-22
Summary budget: input=1200; output=500; reserve=300
Config owns: JAVA_EXECUTABLE
Config uses: JAVA_EXECUTABLE
-->

### Prompt tạo hoặc sửa module

```text
Viết ptb_tokenize(mapping). Viết riêng ptb_tokenize theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. ptb_tokenize nhận (mapping); Dùng Stanford PTBTokenizer đi kèm pycocoevalcap và Java, preserveLines/lowerCase, bỏ PTB punctuation; giữ key/count, lỗi khi thiếu Java. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Tự chọn JAVA_EXECUTABLE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện ptb_tokenize(mapping) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của ptb_tokenize(mapping) tại đây.
```

## Bước 29: score_corpus(records, predictions)

<!-- module
ID: m039
Depends: m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=24-29
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [ptb_tokenize(mapping)](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết score_corpus(records, predictions). Viết riêng score_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. score_corpus nhận (records, predictions); PTB cả refs và predictions, Cider(n=4,sigma=6.0), trả cider raw không nhân100, image_count và tokenizer provenance. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Kết quả giữ các trường cider, image_count, scale, tokenizer. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m038 — ptb_tokenize(mapping):
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m039"></a>

```text
Tóm tắt giao diện score_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m038 — ptb_tokenize(mapping):
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của score_corpus(records, predictions) tại đây.
```

## Bước 30: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m040
Depends: m008, m015, m032, m035, m039
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=1-11
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
3. [Ghép giao diện class CaptionModel](#context-m032)
4. [decode_tokens(ids)](#context-m035)
5. [score_corpus(records, predictions)](#context-m039)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, return_predictions=False). Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Dùng lại MAX_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]

Context m035 — decode_tokens(ids):
[Dán tóm tắt m035]

Context m039 — score_corpus(records, predictions):
[Dán tóm tắt m039]
```

### Tóm tắt context để dùng tiếp

<a id="context-m040"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]

Context m035 — decode_tokens(ids):
[Dán tóm tắt m035]

Context m039 — score_corpus(records, predictions):
[Dán tóm tắt m039]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 31: make_optimizer(model)

<!-- module
ID: m041
Depends: m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=3-11
Summary budget: input=1200; output=500; reserve=300
Config owns: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
Config uses: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class CaptionModel](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết make_optimizer(model). Khởi tạo optimizer AdamW cho tham số có thể học. Tách nhóm backbone và task head theo cấu trúc model. Tự chọn BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện make_optimizer(model) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của make_optimizer(model) tại đây.
```

## Bước 32: Huấn luyện và checkpoint — best_score, model, optimizer…

<!-- module
ID: m042
Depends: m008, m025, m033, m041
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=1-6
Summary budget: input=1200; output=500; reserve=300
Config owns: DIRECTION, EPOCHS
Config uses: DEVICE, DIRECTION, EPOCHS, MODEL_CONFIG
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
3. [build_model(cfg, initialize_pretrained=True)](#context-m033)
4. [make_optimizer(model)](#context-m041)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — best_score, model, optimizer…. Giữ ràng buộc: DIRECTION='max'. Tự chọn EPOCHS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, model, optimizer… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — best_score, model, optimizer… tại đây.
```

## Bước 33: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m043
Depends: m008, m015, m023, m036, m040, m042
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=7-44
Summary budget: input=1200; output=500; reserve=300
Config owns: ARTIFACT_DIR, CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC, VARIANT
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DIRECTION, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC, VARIANT
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
3. [Tạo DataLoader cho các split](#context-m023)
4. [compute_loss(model, batch)](#context-m036)
5. [evaluate(model, loader, return_predictions=False)](#context-m040)
6. [Huấn luyện và checkpoint — best_score, model, optimizer…](#context-m042)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='cider'. Tự chọn GRAD_CLIP, PATIENCE, VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại DIRECTION, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m036 — compute_loss(model, batch):
[Dán tóm tắt m036]

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

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m036 — compute_loss(model, batch):
[Dán tóm tắt m036]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m042 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m042]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 34: Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…

<!-- module
ID: m044
Depends: m008, m012, m015, m016, m023, m025, m033, m040, m043
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=1-18
Summary budget: input=1200; output=500; reserve=300
Config owns: SOURCE
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Chuẩn hóa và chia dữ liệu](#context-m012)
3. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
4. [restore_tokenizer(state)](#context-m016)
5. [Tạo DataLoader cho các split](#context-m023)
6. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
7. [build_model(cfg, initialize_pretrained=True)](#context-m033)
8. [evaluate(model, loader, return_predictions=False)](#context-m040)
9. [Vòng huấn luyện, validation và lưu checkpoint](#context-m043)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Khai báo cùng phần này: SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m016 — restore_tokenizer(state):
[Dán tóm tắt m016]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m043 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m043]
```

### Tóm tắt context để dùng tiếp

<a id="context-m044"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m016 — restore_tokenizer(state):
[Dán tóm tắt m016]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m043 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m043]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… tại đây.
```

## Bước 35: Tải lại và đánh giá held-out

<!-- module
ID: m045
Depends: m044
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=19-19
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m044)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá held-out. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out tại đây.
```

## Bước 36: describe_image(image)

<!-- module
ID: m046
Depends: m008, m015, m017, m035, m044
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=24; lines=1-5
Summary budget: input=1200; output=500; reserve=300
Config uses: DEVICE, MAX_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
3. [image_tensor(value)](#context-m017)
4. [decode_tokens(ids)](#context-m035)
5. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m044)

### Prompt tạo hoặc sửa module

```text
Viết describe_image(image). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Hàm describe_image nhận (image), giữ chữ ký và output schema đã chốt, không cần gold. Dùng lại DEVICE, MAX_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m035 — decode_tokens(ids):
[Dán tóm tắt m035]

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện describe_image(image) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m035 — decode_tokens(ids):
[Dán tóm tắt m035]

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của describe_image(image) tại đây.
```

## Bước 37: Mô tả ảnh mới

<!-- module
ID: m047
Depends: m012, m046
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=24; lines=7-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m012)
2. [describe_image(image)](#context-m046)

### Prompt tạo hoặc sửa module

```text
Viết Mô tả ảnh mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m046 — describe_image(image):
[Dán tóm tắt m046]
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện Mô tả ảnh mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m046 — describe_image(image):
[Dán tóm tắt m046]
```

```python
# Dán code hiện tại của Mô tả ảnh mới tại đây.
```

## Bước 38: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m045, m047
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá held-out](#context-m045)
2. [Mô tả ảnh mới](#context-m047)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m045 — Tải lại và đánh giá held-out:
[Dán tóm tắt m045]

Context m047 — Mô tả ảnh mới:
[Dán tóm tắt m047]
```
