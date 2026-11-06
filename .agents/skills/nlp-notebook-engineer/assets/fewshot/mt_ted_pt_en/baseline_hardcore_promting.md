# Prompt hardcore: TED PT→EN: BPE và Transformer scratch

<!-- prompt-guide
Format: 3
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: clean_text(text)

<!-- module
ID: m007
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=5-6
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết clean_text(text). Chuẩn hóa Unicode NFC và khoảng trắng theo quy ước của dữ liệu; không làm mất nhãn hoặc references. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m007"></a>

```text
Tóm tắt giao diện clean_text(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của clean_text(text) tại đây.
```

## Bước 2: read_jsonl(path)

<!-- module
ID: m008
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=8-10
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết read_jsonl(path). Đọc UTF-8 JSONL, giữ thứ tự bản ghi và báo rõ lỗi thiếu file hoặc dữ liệu không hợp lệ. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m008"></a>

```text
Tóm tắt giao diện read_jsonl(path) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của read_jsonl(path) tại đây.
```

## Bước 3: tfds_rows(builder, splits, supervised=False) — phần 1/2

<!-- module
ID: m009
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=12-26
Summary budget: input=1200; output=500; reserve=300
Config owns: ARTIFACT_DIR, CACHE_DIR, VARIANT
Config uses: ARTIFACT_DIR, CACHE_DIR, VARIANT
-->

### Prompt tạo hoặc sửa module

```text
Viết tfds_rows(builder, splits, supervised=False) — phần 1/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện tfds_rows(builder, splits, supervised=False) — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của tfds_rows(builder, splits, supervised=False) — phần 1/2 tại đây.
```

## Bước 4: tfds_rows(builder, splits, supervised=False) — phần 2/2

<!-- module
ID: m010
Depends: m009
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=27-35
Summary budget: input=1200; output=500; reserve=300
Config uses: CACHE_DIR
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)

### Prompt tạo hoặc sửa module

```text
Viết tfds_rows(builder, splits, supervised=False) — phần 2/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 2/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

### Tóm tắt context để dùng tiếp

<a id="context-m010"></a>

```text
Tóm tắt giao diện tfds_rows(builder, splits, supervised=False) — phần 2/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

```python
# Dán code hiện tại của tfds_rows(builder, splits, supervised=False) — phần 2/2 tại đây.
```

## Bước 5: decode_utf8(value)

<!-- module
ID: m011
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=37-38
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết decode_utf8(value). Giải mã chuỗi hoặc bytes từ nguồn dữ liệu thành văn bản UTF-8. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m011"></a>

```text
Tóm tắt giao diện decode_utf8(value) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của decode_utf8(value) tại đây.
```

## Bước 6: load_native()

<!-- module
ID: m014
Depends: m008, m009, m010, m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=1-20
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_MODE, LOCAL_DATA_DIR, SOURCE
Config uses: CACHE_DIR, DATA_MODE, LOCAL_DATA_DIR, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m008)
2. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
3. [tfds_rows(builder, splits, supervised=False) — phần 2/2](#context-m010)
4. [decode_utf8(value)](#context-m011)

### Prompt tạo hoặc sửa module

```text
Viết load_native(). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. TED HRLR PT→EN TFDS ted_hrlr_translate/pt_to_en:1.0.0; local JSONL train/validation/test có pt,en. Nếu dùng cùng corpus HF neulab/ted_hrlr thì revision cf21722111c95aa73c30e13df430032db463736d. Nguồn dữ liệu dataset: TED HRLR Translate pt_to_en; tfds_builder: ted_hrlr_translate/pt_to_en:1.0.0. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: DATA_MODE='hf_parquet_recovery'. Khai báo cùng phần này: LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện load_native() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của load_native() tại đây.
```

## Bước 7: Dữ liệu và giao thức chia — native

<!-- module
ID: m015
Depends: m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=21-21
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [load_native()](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu và giao thức chia — native. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — load_native():
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — native từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — load_native():
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — native tại đây.
```

## Bước 8: Dữ liệu và giao thức chia — record, records, removed_train…

<!-- module
ID: m016
Depends: m007, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=2-11
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Dữ liệu và giao thức chia — native](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu và giao thức chia — record, records, removed_train…. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m015 — Dữ liệu và giao thức chia — native:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — record, records, removed_train… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m015 — Dữ liệu và giao thức chia — native:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — record, records, removed_train… tại đây.
```

## Bước 9: canonical_records(rows, split)

<!-- module
ID: m017
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=12-16
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết canonical_records(rows, split). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện canonical_records(rows, split) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của canonical_records(rows, split) tại đây.
```

## Bước 10: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m018
Depends: m016, m017
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=17-26
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — record, records, removed_train…](#context-m016)
2. [canonical_records(rows, split)](#context-m017)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: TED HRLR Translate pt_to_en; tfds_builder: ted_hrlr_translate/pt_to_en:1.0.0. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Dữ liệu và giao thức chia — record, records, removed_train…:
[Dán tóm tắt m016]

Context m017 — canonical_records(rows, split):
[Dán tóm tắt m017]
```

### Tóm tắt context để dùng tiếp

<a id="context-m018"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Dữ liệu và giao thức chia — record, records, removed_train…:
[Dán tóm tắt m016]

Context m017 — canonical_records(rows, split):
[Dán tóm tắt m017]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 11: Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…

<!-- module
ID: m020
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=2-3
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…. Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… tại đây.
```

## Bước 12: tokenizer_corpus()

<!-- module
ID: m021
Depends: m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=4-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết tokenizer_corpus(). Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện tokenizer_corpus() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của tokenizer_corpus() tại đây.
```

## Bước 13: Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp

<!-- module
ID: m022
Depends: m020, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=8-18
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [tokenizer_corpus()](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp. Fit SentencePiece BPE scratch với byte fallback chỉ trên train; giữ special IDs đã thống nhất, lưu tokenizer và trạng thái tiền xử lý để reload. PREPROCESS_CONFIG chỉ lưu thiết lập mã hóa/chuẩn hóa, không chứa cấu hình giải mã. Tự chọn MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m021 — tokenizer_corpus():
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m021 — tokenizer_corpus():
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp tại đây.
```

## Bước 14: restore_tokenizer(state)

<!-- module
ID: m023
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=19-21
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết restore_tokenizer(state). Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 15: PairDataset.__init__(self, records, training=False)

<!-- module
ID: m024
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=24-26
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết PairDataset.__init__(self, records, training=False). Khởi tạo các thành phần và trạng thái cần cho class. Viết method trong class PairDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện PairDataset.__init__(self, records, training=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của PairDataset.__init__(self, records, training=False) tại đây.
```

## Bước 16: PairDataset.__len__(self)

<!-- module
ID: m025
Depends: m024
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=27-27
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m024)

### Prompt tạo hoặc sửa module

```text
Viết PairDataset.__len__(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class PairDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện PairDataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]
```

```python
# Dán code hiện tại của PairDataset.__len__(self) tại đây.
```

## Bước 17: PairDataset.__getitem__(self, index)

<!-- module
ID: m026
Depends: m020, m022, m024
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=28-33
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
3. [PairDataset.__init__(self, records, training=False)](#context-m024)

### Prompt tạo hoặc sửa module

```text
Viết PairDataset.__getitem__(self, index). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2, record. Dùng lại MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Viết method trong class PairDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện PairDataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]
```

```python
# Dán code hiện tại của PairDataset.__getitem__(self, index) tại đây.
```

## Bước 18: Ghép giao diện class PairDataset

<!-- module
ID: m027
Depends: m024, m025, m026
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=11; lines=23-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m024)
2. [PairDataset.__len__(self)](#context-m025)
3. [PairDataset.__getitem__(self, index)](#context-m026)

<a id="context-m027"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class PairDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]

Context m025 — PairDataset.__len__(self):
[Dán tóm tắt m025]

Context m026 — PairDataset.__getitem__(self, index):
[Dán tóm tắt m026]
```

## Bước 19: collate_batch(items)

<!-- module
ID: m028
Depends: m020
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=35-41
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Giữ PairDataset và collate_batch khi chuyển BPE, target có BOS/EOS; chỉ rút ngắn input train, không rút gold metric. Kết quả giữ các trường ids, lengths, records, source, target. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]
```

### Tóm tắt context để dùng tiếp

<a id="context-m028"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 20: to_device(batch)

<!-- module
ID: m012
Depends: m028
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=40-41
Summary budget: input=1200; output=500; reserve=300
Config owns: DEVICE
Config uses: DEVICE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m028)

### Prompt tạo hoặc sửa module

```text
Viết to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 21: make_loader(dataset, train=False)

<!-- module
ID: m013
Depends: m028
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=43-46
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m028)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Giữ ràng buộc: SEED=42. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 22: Tạo DataLoader cho các split

<!-- module
ID: m029
Depends: m013, m018, m022, m027
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=43-50
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m013)
2. [Chuẩn hóa và chia dữ liệu](#context-m018)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
4. [Ghép giao diện class PairDataset](#context-m027)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Dùng lại MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m027 — Ghép giao diện class PairDataset:
[Dán tóm tắt m027]
```

### Tóm tắt context để dùng tiếp

<a id="context-m029"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m027 — Ghép giao diện class PairDataset:
[Dán tóm tắt m027]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 23: PositionEncoding.__init__(self, dim, dropout, max_length)

<!-- module
ID: m030
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=2-7
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết PositionEncoding.__init__(self, dim, dropout, max_length). Viết PositionEncoding sinusoidal với buffer và dropout cho tensor batch-first. Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class PositionEncoding kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện PositionEncoding.__init__(self, dim, dropout, max_length) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của PositionEncoding.__init__(self, dim, dropout, max_length) tại đây.
```

## Bước 24: PositionEncoding.forward(self, x)

<!-- module
ID: m031
Depends: m030
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=8-8
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PositionEncoding.__init__(self, dim, dropout, max_length)](#context-m030)

### Prompt tạo hoặc sửa module

```text
Viết PositionEncoding.forward(self, x). Viết PositionEncoding sinusoidal với buffer và dropout cho tensor batch-first. Viết method trong class PositionEncoding kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m030 — PositionEncoding.__init__(self, dim, dropout, max_length):
[Dán tóm tắt m030]
```

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện PositionEncoding.forward(self, x) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m030 — PositionEncoding.__init__(self, dim, dropout, max_length):
[Dán tóm tắt m030]
```

```python
# Dán code hiện tại của PositionEncoding.forward(self, x) tại đây.
```

## Bước 25: Ghép giao diện class PositionEncoding

<!-- module
ID: m032
Depends: m030, m031
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=1-8
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PositionEncoding.__init__(self, dim, dropout, max_length)](#context-m030)
2. [PositionEncoding.forward(self, x)](#context-m031)

<a id="context-m032"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class PositionEncoding kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m030 — PositionEncoding.__init__(self, dim, dropout, max_length):
[Dán tóm tắt m030]

Context m031 — PositionEncoding.forward(self, x):
[Dán tóm tắt m031]
```

## Bước 26: TextGenerator.__init__(self, cfg)

<!-- module
ID: m033
Depends: m020, m022, m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=11-20
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, EMBED_DIM, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, EMBED_DIM, HIDDEN_DIM
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
3. [Ghép giao diện class PositionEncoding](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.__init__(self, cfg). Thay constructor TextGenerator bằng Transformer encoder-decoder scratch, tied embedding/output và batch-first. Tự chọn DROPOUT, EMBED_DIM, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa vocab_size, embed_dim, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class PositionEncoding:
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện TextGenerator.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class PositionEncoding:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của TextGenerator.__init__(self, cfg) tại đây.
```

## Bước 27: TextGenerator.encode(self, source)

<!-- module
ID: m034
Depends: m020, m028, m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=21-23
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [TextGenerator.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.encode(self, source). Viết encode(source) dùng embedding scale căn embed_dim, positional encoding và source padding mask. Giữ thứ tự đầu ra: kết quả 1, mask. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện TextGenerator.encode(self, source) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của TextGenerator.encode(self, source) tại đây.
```

## Bước 28: TextGenerator.decode(self, target, memory, source_mask)

<!-- module
ID: m035
Depends: m020, m028, m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=24-28
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [TextGenerator.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.decode(self, target, memory, source_mask). Viết decode(target,memory,source_mask) có causal mask và padding masks nhất quán kiểu bool. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện TextGenerator.decode(self, target, memory, source_mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của TextGenerator.decode(self, target, memory, source_mask) tại đây.
```

## Bước 29: TextGenerator.forward(self, source, lengths, target)

<!-- module
ID: m036
Depends: m028, m033, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=29-31
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m028)
2. [TextGenerator.__init__(self, cfg)](#context-m033)
3. [TextGenerator.encode(self, source)](#context-m034)
4. [TextGenerator.decode(self, target, memory, source_mask)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.forward(self, source, lengths, target). Viết forward(source,lengths,target) nối encode và decode, trả log_softmax cho teacher forcing. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện TextGenerator.forward(self, source, lengths, target) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của TextGenerator.forward(self, source, lengths, target) tại đây.
```

## Bước 30: TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)

<!-- module
ID: m037
Depends: m020, m028, m033, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=32-52
Summary budget: input=1200; output=500; reserve=300
Config owns: BEAM_SIZE, LENGTH_PENALTY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [TextGenerator.__init__(self, cfg)](#context-m033)
4. [TextGenerator.encode(self, source)](#context-m034)
5. [TextGenerator.decode(self, target, memory, source_mask)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty). Viết phần khởi tạo beam cho generate, lặp memory/mask theo beam và bắt đầu BOS. Viết vòng beam chọn top-k theo log-prob, gom lại parent beam và đóng băng beam đã EOS. Chặn PAD/BOS; EOS của beam done có logp=0; gather seq/done/lens theo parent; dừng khi mọi done; không tính length thêm cho beam đã đóng; ghép sau phần khởi tạo generate. Tự chọn BEAM_SIZE, LENGTH_PENALTY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m037"></a>

```text
Tóm tắt giao diện TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) tại đây.
```

## Bước 31: Ghép giao diện class TextGenerator

<!-- module
ID: m038
Depends: m033, m034, m035, m036, m037
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=10-52
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [TextGenerator.__init__(self, cfg)](#context-m033)
2. [TextGenerator.encode(self, source)](#context-m034)
3. [TextGenerator.decode(self, target, memory, source_mask)](#context-m035)
4. [TextGenerator.forward(self, source, lengths, target)](#context-m036)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)](#context-m037)

<a id="context-m038"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class TextGenerator kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]

Context m036 — TextGenerator.forward(self, source, lengths, target):
[Dán tóm tắt m036]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]
```

## Bước 32: build_model(cfg)

<!-- module
ID: m039
Depends: m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=53-53
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class TextGenerator](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết build_model(cfg). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m039"></a>

```text
Tóm tắt giao diện build_model(cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của build_model(cfg) tại đây.
```

## Bước 33: Metric, loss và giải mã — bleu_metric, chrf_metric

<!-- module
ID: m042
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=2-3
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết Metric, loss và giải mã — bleu_metric, chrf_metric. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện Metric, loss và giải mã — bleu_metric, chrf_metric từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Metric, loss và giải mã — bleu_metric, chrf_metric tại đây.
```

## Bước 34: score_corpus(records, predictions)

<!-- module
ID: m043
Depends: m042
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=4-9
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Metric, loss và giải mã — bleu_metric, chrf_metric](#context-m042)

### Prompt tạo hoặc sửa module

```text
Viết score_corpus(records, predictions). Viết riêng score_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. score_corpus nhận (records, predictions); Dùng sacreBLEU BLEU/chrF corpus cho toàn bộ refs; giữ signature và primary bleu. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. Kết quả giữ các trường bleu, bleu_signature, chrf_signature, chrfpp. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m042 — Metric, loss và giải mã — bleu_metric, chrf_metric:
[Dán tóm tắt m042]
```

### Tóm tắt context để dùng tiếp

<a id="context-m043"></a>

```text
Tóm tắt giao diện score_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m042 — Metric, loss và giải mã — bleu_metric, chrf_metric:
[Dán tóm tắt m042]
```

```python
# Dán code hiện tại của score_corpus(records, predictions) tại đây.
```

## Bước 35: decode_tokens(ids)

<!-- module
ID: m044
Depends: m007, m020, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=1-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết decode_tokens(ids). Viết riêng decode_tokens theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. decode_tokens nhận (ids); Giải mã IDs bằng SentencePiece, bỏ PAD/BOS và dừng EOS; giữ clean_text NFC và whitespace. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m044"></a>

```text
Tóm tắt giao diện decode_tokens(ids) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của decode_tokens(ids) tại đây.
```

## Bước 36: compute_loss(model, batch)

<!-- module
ID: m045
Depends: m020, m028, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=8-14
Summary budget: input=1200; output=500; reserve=300
Config owns: LABEL_SMOOTHING
Config uses: LABEL_SMOOTHING
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [Ghép giao diện class TextGenerator](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết compute_loss(model, batch). Giữ loss teacher forcing dịch target một token, bỏ PAD và label smoothing dương được chọn phù hợp cho log-prob Transformer. Tự chọn LABEL_SMOOTHING phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 37: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m046
Depends: m012, m037, m038, m043, m044
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=16-26
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_DECODE_TOKENS
Config uses: BEAM_SIZE, LENGTH_PENALTY, MAX_DECODE_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)](#context-m037)
3. [Ghép giao diện class TextGenerator](#context-m038)
4. [score_corpus(records, predictions)](#context-m043)
5. [decode_tokens(ids)](#context-m044)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, return_predictions=False). Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Tự chọn MAX_DECODE_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại BEAM_SIZE, LENGTH_PENALTY từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]

Context m043 — score_corpus(records, predictions):
[Dán tóm tắt m043]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]

Context m043 — score_corpus(records, predictions):
[Dán tóm tắt m043]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 38: Metric, loss và giải mã

<!-- module
ID: m047
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=1-2
Summary budget: input=1200; output=500; reserve=300
Config owns: RUN_COMET
Config uses: RUN_COMET
-->

### Prompt tạo hoặc sửa module

```text
Viết Metric, loss và giải mã. Giữ ràng buộc: RUN_COMET=True. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện Metric, loss và giải mã từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Metric, loss và giải mã tại đây.
```

## Bước 39: Huấn luyện và checkpoint — model, optimizer, total_steps

<!-- module
ID: m048
Depends: m012, m029, m033, m039
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=1-4
Summary budget: input=1200; output=500; reserve=300
Config owns: EPOCHS, LEARNING_RATE, WEIGHT_DECAY
Config uses: DEVICE, EPOCHS, LEARNING_RATE, MODEL_CONFIG, WEIGHT_DECAY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Tạo DataLoader cho các split](#context-m029)
3. [TextGenerator.__init__(self, cfg)](#context-m033)
4. [build_model(cfg)](#context-m039)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — model, optimizer, total_steps. Khởi tạo optimizer AdamW cho tham số có thể học. Tự chọn EPOCHS, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — model, optimizer, total_steps từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — model, optimizer, total_steps tại đây.
```

## Bước 40: lr_multiplier(step)

<!-- module
ID: m049
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=5-7
Summary budget: input=1200; output=500; reserve=300
Config owns: WARMUP_STEPS
Config uses: WARMUP_STEPS
-->

### Prompt tạo hoặc sửa module

```text
Viết lr_multiplier(step). Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. AdamW lr theo cấu hình đã chọn, betas=(0.9,0.98), eps=1e-9, weight_decay theo cấu hình đã chọn; warmup=1000 rồi inverse-sqrt; epochs theo cấu hình đã chọn,batch theo cấu hình đã chọn,eval_batch=16,patience theo cấu hình đã chọn,clip theo cấu hình đã chọn; chọn BLEU validation. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. Tự chọn WARMUP_STEPS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m049"></a>

```text
Tóm tắt giao diện lr_multiplier(step) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của lr_multiplier(step) tại đây.
```

## Bước 41: Huấn luyện và checkpoint — best_score, scheduler, stale_epochs

<!-- module
ID: m050
Depends: m048, m049
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=8-9
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m048)
2. [lr_multiplier(step)](#context-m049)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — best_score, scheduler, stale_epochs. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m049 — lr_multiplier(step):
[Dán tóm tắt m049]
```

### Tóm tắt context để dùng tiếp

<a id="context-m050"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, scheduler, stale_epochs từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m049 — lr_multiplier(step):
[Dán tóm tắt m049]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — best_score, scheduler, stale_epochs tại đây.
```

## Bước 42: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m051
Depends: m009, m012, m022, m029, m045, m046, m048, m050
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=10-38
Summary budget: input=1200; output=500; reserve=300
Config owns: CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
4. [Tạo DataLoader cho các split](#context-m029)
5. [compute_loss(model, batch)](#context-m045)
6. [evaluate(model, loader, return_predictions=False)](#context-m046)
7. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m048)
8. [Huấn luyện và checkpoint — best_score, scheduler, stale_epochs](#context-m050)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='bleu'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m045 — compute_loss(model, batch):
[Dán tóm tắt m045]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m050 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m050]
```

### Tóm tắt context để dùng tiếp

<a id="context-m051"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m045 — compute_loss(model, batch):
[Dán tóm tắt m045]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m050 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m050]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 43: Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…

<!-- module
ID: m052
Depends: m009, m012, m014, m018, m022, m023, m029, m033, m039, m046, m051
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=1-17
Summary budget: input=1200; output=500; reserve=300
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [load_native()](#context-m014)
4. [Chuẩn hóa và chia dữ liệu](#context-m018)
5. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
6. [restore_tokenizer(state)](#context-m023)
7. [Tạo DataLoader cho các split](#context-m029)
8. [TextGenerator.__init__(self, cfg)](#context-m033)
9. [build_model(cfg)](#context-m039)
10. [evaluate(model, loader, return_predictions=False)](#context-m046)
11. [Vòng huấn luyện, validation và lưu checkpoint](#context-m051)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m014 — load_native():
[Dán tóm tắt m014]

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m023 — restore_tokenizer(state):
[Dán tóm tắt m023]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m051 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m051]
```

### Tóm tắt context để dùng tiếp

<a id="context-m052"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m014 — load_native():
[Dán tóm tắt m014]

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m023 — restore_tokenizer(state):
[Dán tóm tắt m023]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m051 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m051]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… tại đây.
```

## Bước 44: add_comet_metric(rows) — phần 1/2

<!-- module
ID: m053
Depends: m009, m012, m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=23; lines=1-17
Summary budget: input=1200; output=500; reserve=300
Config owns: ALLOW_PRETRAINED_EVALUATORS, COMET_BATCH_SIZE, METRICS_OFFLINE, METRIC_CACHE_DIR
Config uses: ALLOW_PRETRAINED_EVALUATORS, ARTIFACT_DIR, COMET_BATCH_SIZE, DEVICE, METRICS_OFFLINE, METRIC_CACHE_DIR, RUN_COMET
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Metric, loss và giải mã](#context-m047)

### Prompt tạo hoặc sửa module

```text
Viết add_comet_metric(rows) — phần 1/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường comet, comet_status. Giữ ràng buộc: ALLOW_PRETRAINED_EVALUATORS=True. Tự chọn COMET_BATCH_SIZE, METRICS_OFFLINE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: METRIC_CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, DEVICE, RUN_COMET từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]
```

### Tóm tắt context để dùng tiếp

<a id="context-m053"></a>

```text
Tóm tắt giao diện add_comet_metric(rows) — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của add_comet_metric(rows) — phần 1/2 tại đây.
```

## Bước 45: add_comet_metric(rows) — phần 2/2

<!-- module
ID: m054
Depends: m012, m047, m053
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=23; lines=18-24
Summary budget: input=1200; output=500; reserve=300
Config uses: ALLOW_PRETRAINED_EVALUATORS, COMET_BATCH_SIZE, DEVICE, METRICS_OFFLINE, METRIC_CACHE_DIR, RUN_COMET
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Metric, loss và giải mã](#context-m047)
3. [add_comet_metric(rows) — phần 1/2](#context-m053)

### Prompt tạo hoặc sửa module

```text
Viết add_comet_metric(rows) — phần 2/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường comet, comet_model, comet_status. Dùng lại ALLOW_PRETRAINED_EVALUATORS, COMET_BATCH_SIZE, DEVICE, METRICS_OFFLINE, METRIC_CACHE_DIR, RUN_COMET từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 2/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]

Context m053 — add_comet_metric(rows) — phần 1/2:
[Dán tóm tắt m053]
```

### Tóm tắt context để dùng tiếp

<a id="context-m054"></a>

```text
Tóm tắt giao diện add_comet_metric(rows) — phần 2/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]

Context m053 — add_comet_metric(rows) — phần 1/2:
[Dán tóm tắt m053]
```

```python
# Dán code hiện tại của add_comet_metric(rows) — phần 2/2 tại đây.
```

## Bước 46: Tải lại và đánh giá held-out — extra

<!-- module
ID: m055
Depends: m009, m052, m054
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=23; lines=27-30
Summary budget: input=1200; output=500; reserve=300
Config uses: ARTIFACT_DIR
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m052)
3. [add_comet_metric(rows) — phần 2/2](#context-m054)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá held-out — extra. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Dùng lại ARTIFACT_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m052 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m052]

Context m054 — add_comet_metric(rows) — phần 2/2:
[Dán tóm tắt m054]
```

### Tóm tắt context để dùng tiếp

<a id="context-m055"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — extra từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m052 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m052]

Context m054 — add_comet_metric(rows) — phần 2/2:
[Dán tóm tắt m054]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out — extra tại đây.
```

## Bước 47: rewrite_text(text)

<!-- module
ID: m056
Depends: m007, m012, m020, m022, m037, m044, m046, m052
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=25; lines=1-9
Summary budget: input=1200; output=500; reserve=300
Config uses: BEAM_SIZE, DEVICE, LENGTH_PENALTY, MAX_DECODE_TOKENS, MAX_SOURCE_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [to_device(batch)](#context-m012)
3. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
4. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)](#context-m037)
6. [decode_tokens(ids)](#context-m044)
7. [evaluate(model, loader, return_predictions=False)](#context-m046)
8. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m052)

### Prompt tạo hoặc sửa module

```text
Viết rewrite_text(text). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Dùng lại BEAM_SIZE, DEVICE, LENGTH_PENALTY, MAX_DECODE_TOKENS, MAX_SOURCE_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m052 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m052]
```

### Tóm tắt context để dùng tiếp

<a id="context-m056"></a>

```text
Tóm tắt giao diện rewrite_text(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m052 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m052]
```

```python
# Dán code hiện tại của rewrite_text(text) tại đây.
```

## Bước 48: Suy luận văn bản mới

<!-- module
ID: m057
Depends: m018, m056
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=25; lines=11-11
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m018)
2. [rewrite_text(text)](#context-m056)

### Prompt tạo hoặc sửa module

```text
Viết Suy luận văn bản mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m056 — rewrite_text(text):
[Dán tóm tắt m056]
```

### Tóm tắt context để dùng tiếp

<a id="context-m057"></a>

```text
Tóm tắt giao diện Suy luận văn bản mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m056 — rewrite_text(text):
[Dán tóm tắt m056]
```

```python
# Dán code hiện tại của Suy luận văn bản mới tại đây.
```

## Bước 49: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m055, m057
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá held-out — extra](#context-m055)
2. [Suy luận văn bản mới](#context-m057)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m055 — Tải lại và đánh giá held-out — extra:
[Dán tóm tắt m055]

Context m057 — Suy luận văn bản mới:
[Dán tóm tắt m057]
```
