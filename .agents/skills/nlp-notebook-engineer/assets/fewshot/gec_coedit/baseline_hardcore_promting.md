# Prompt hardcore: CoEdIT GEC: BPE, BiGRU attention và pointer-copy

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

## Bước 6: strip_instruction(text)

<!-- module
ID: m016
Depends: m007
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=3-6
Summary budget: input=1200; output=500; reserve=300
Config owns: KNOWN_GEC_PREFIXES
Config uses: KNOWN_GEC_PREFIXES
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)

### Prompt tạo hoặc sửa module

```text
Viết strip_instruction(text). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. grammarly/coedit revision e9a255c33ef910bc33a9d2b522653fa87521583e; chỉ task=gec, schema _id,task,src,tgt. Bỏ đúng instruction prefix đã kiểm tra, không split mọi dấu hai chấm; nhóm src thành refs/source_ids. Tự chọn KNOWN_GEC_PREFIXES phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện strip_instruction(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]
```

```python
# Dán code hiện tại của strip_instruction(text) tại đây.
```

## Bước 7: Dữ liệu và giao thức chia — raw

<!-- module
ID: m017
Depends: m008, m009
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=8-12
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_MODE, LOCAL_DATA_DIR, SOURCE
Config uses: CACHE_DIR, DATA_MODE, LOCAL_DATA_DIR, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m008)
2. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu và giao thức chia — raw. Giữ ràng buộc: DATA_MODE='huggingface'. Khai báo cùng phần này: LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — raw từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — raw tại đây.
```

## Bước 8: group_gec(rows)

<!-- module
ID: m018
Depends: m007, m016
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=14-24
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [strip_instruction(text)](#context-m016)

### Prompt tạo hoặc sửa module

```text
Viết group_gec(rows). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m016 — strip_instruction(text):
[Dán tóm tắt m016]
```

### Tóm tắt context để dùng tiếp

<a id="context-m018"></a>

```text
Tóm tắt giao diện group_gec(rows) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m016 — strip_instruction(text):
[Dán tóm tắt m016]
```

```python
# Dán code hiện tại của group_gec(rows) tại đây.
```

## Bước 9: Dữ liệu và giao thức chia — held, held_sources, native…

<!-- module
ID: m019
Depends: m017, m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=26-34
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, VAL_FRACTION
Config uses: SEED, VAL_FRACTION
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — raw](#context-m017)
2. [group_gec(rows)](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu và giao thức chia — held, held_sources, native…. Giữ ràng buộc: SEED=42; VAL_FRACTION=0.1. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m018 — group_gec(rows):
[Dán tóm tắt m018]
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — held, held_sources, native… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m018 — group_gec(rows):
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — held, held_sources, native… tại đây.
```

## Bước 10: Dữ liệu và giao thức chia — record, records, removed_train…

<!-- module
ID: m020
Depends: m007, m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=2-11
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Dữ liệu và giao thức chia — held, held_sources, native…](#context-m019)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu và giao thức chia — record, records, removed_train…. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — record, records, removed_train… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — record, records, removed_train… tại đây.
```

## Bước 11: canonical_records(rows, split)

<!-- module
ID: m021
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

<a id="context-m021"></a>

```text
Tóm tắt giao diện canonical_records(rows, split) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của canonical_records(rows, split) tại đây.
```

## Bước 12: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m022
Depends: m020, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=17-26
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — record, records, removed_train…](#context-m020)
2. [canonical_records(rows, split)](#context-m021)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: grammarly/coedit; revision: e9a255c33ef910bc33a9d2b522653fa87521583e. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Dữ liệu và giao thức chia — record, records, removed_train…:
[Dán tóm tắt m020]

Context m021 — canonical_records(rows, split):
[Dán tóm tắt m021]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Dữ liệu và giao thức chia — record, records, removed_train…:
[Dán tóm tắt m020]

Context m021 — canonical_records(rows, split):
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 13: Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…

<!-- module
ID: m024
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=2-3
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…. Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… tại đây.
```

## Bước 14: tokenizer_corpus()

<!-- module
ID: m025
Depends: m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=4-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết tokenizer_corpus(). Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện tokenizer_corpus() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của tokenizer_corpus() tại đây.
```

## Bước 15: Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp

<!-- module
ID: m026
Depends: m024, m025
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

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [tokenizer_corpus()](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp. Fit SentencePiece BPE scratch với byte fallback chỉ trên train; giữ special IDs đã thống nhất, lưu tokenizer và trạng thái tiền xử lý để reload. PREPROCESS_CONFIG chỉ lưu thiết lập mã hóa/chuẩn hóa, không chứa cấu hình giải mã. Tự chọn MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m025 — tokenizer_corpus():
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m025 — tokenizer_corpus():
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp tại đây.
```

## Bước 16: restore_tokenizer(state)

<!-- module
ID: m027
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=19-21
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết restore_tokenizer(state). Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 17: PairDataset.__init__(self, records, training=False)

<!-- module
ID: m028
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

<a id="context-m028"></a>

```text
Tóm tắt giao diện PairDataset.__init__(self, records, training=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của PairDataset.__init__(self, records, training=False) tại đây.
```

## Bước 18: PairDataset.__len__(self)

<!-- module
ID: m029
Depends: m028
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=27-27
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m028)

### Prompt tạo hoặc sửa module

```text
Viết PairDataset.__len__(self). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class PairDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]
```

### Tóm tắt context để dùng tiếp

<a id="context-m029"></a>

```text
Tóm tắt giao diện PairDataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của PairDataset.__len__(self) tại đây.
```

## Bước 19: PairDataset.__getitem__(self, index)

<!-- module
ID: m030
Depends: m024, m026, m028
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=28-33
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
3. [PairDataset.__init__(self, records, training=False)](#context-m028)

### Prompt tạo hoặc sửa module

```text
Viết PairDataset.__getitem__(self, index). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2, record. Dùng lại MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Viết method trong class PairDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện PairDataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của PairDataset.__getitem__(self, index) tại đây.
```

## Bước 20: Ghép giao diện class PairDataset

<!-- module
ID: m031
Depends: m028, m029, m030
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=11; lines=23-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m028)
2. [PairDataset.__len__(self)](#context-m029)
3. [PairDataset.__getitem__(self, index)](#context-m030)

<a id="context-m031"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class PairDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]

Context m029 — PairDataset.__len__(self):
[Dán tóm tắt m029]

Context m030 — PairDataset.__getitem__(self, index):
[Dán tóm tắt m030]
```

## Bước 21: collate_batch(items)

<!-- module
ID: m032
Depends: m024
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=35-41
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Giữ collate_batch và PairDataset khi đổi BPE; padding 0, BOS/EOS và lengths tương thích pointer-copy. PairDataset(records,training=False); source/target long, lengths CPU, ids,records; dùng mọi refs train, giữ refs đầy đủ để chấm. Kết quả giữ các trường ids, lengths, records, source, target. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]
```

### Tóm tắt context để dùng tiếp

<a id="context-m032"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 22: to_device(batch)

<!-- module
ID: m012
Depends: m032
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

1. [collate_batch(items)](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 23: make_loader(dataset, train=False)

<!-- module
ID: m013
Depends: m019, m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=43-46
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — held, held_sources, native…](#context-m019)
2. [collate_batch(items)](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 24: Tạo DataLoader cho các split

<!-- module
ID: m033
Depends: m013, m022, m026, m031
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
2. [Chuẩn hóa và chia dữ liệu](#context-m022)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
4. [Ghép giao diện class PairDataset](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Dùng lại MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m031 — Ghép giao diện class PairDataset:
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m031 — Ghép giao diện class PairDataset:
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 25: TextGenerator.__init__(self, cfg)

<!-- module
ID: m034
Depends: m024, m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=2-14
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, EMBED_DIM, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, EMBED_DIM, HIDDEN_DIM
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.__init__(self, cfg). Thay constructor TextGenerator bằng BiGRU encoder, additive attention, GRUCell decoder và gate copy. cfg vocab_size,embed_dim theo cấu hình đã chọn,hidden_dim theo cấu hình đã chọn,dropout theo cấu hình đã chọn; memory/state dim=384; embedding chung, bridge, memory_projection,hidden_projection,attention_score,output,gate,dropout. Tự chọn DROPOUT, EMBED_DIM, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa vocab_size, embed_dim, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện TextGenerator.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của TextGenerator.__init__(self, cfg) tại đây.
```

## Bước 26: TextGenerator.encode(self, source, lengths)

<!-- module
ID: m035
Depends: m024, m032, m034
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=15-20
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [TextGenerator.__init__(self, cfg)](#context-m034)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.encode(self, source, lengths). Viết encode(source,lengths) dùng packed BiGRU và bridge trạng thái hai chiều. Giữ thứ tự đầu ra: memory, kết quả 2, state, kết quả 4. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện TextGenerator.encode(self, source, lengths) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

```python
# Dán code hiện tại của TextGenerator.encode(self, source, lengths) tại đây.
```

## Bước 27: TextGenerator.step(self, token, hidden, memory, projected, mask, source)

<!-- module
ID: m036
Depends: m032, m034
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=21-32
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m032)
2. [TextGenerator.__init__(self, cfg)](#context-m034)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.step(self, token, hidden, memory, projected, mask, source). Viết step dùng masked additive attention, GRUCell và gate trộn vocabulary với copy bằng scatter_add. Giữ thứ tự đầu ra: kết quả 1, hidden. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện TextGenerator.step(self, token, hidden, memory, projected, mask, source) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

```python
# Dán code hiện tại của TextGenerator.step(self, token, hidden, memory, projected, mask, source) tại đây.
```

## Bước 28: TextGenerator.forward(self, source, lengths, target)

<!-- module
ID: m037
Depends: m032, m034, m035, m036
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=33-38
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m032)
2. [TextGenerator.__init__(self, cfg)](#context-m034)
3. [TextGenerator.encode(self, source, lengths)](#context-m035)
4. [TextGenerator.step(self, token, hidden, memory, projected, mask, source)](#context-m036)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.forward(self, source, lengths, target). Viết forward(source,lengths,target) gọi step theo token target và trả chuỗi log-prob. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]
```

### Tóm tắt context để dùng tiếp

<a id="context-m037"></a>

```text
Tóm tắt giao diện TextGenerator.forward(self, source, lengths, target) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]
```

```python
# Dán code hiện tại của TextGenerator.forward(self, source, lengths, target) tại đây.
```

## Bước 29: TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2

<!-- module
ID: m038
Depends: m024, m032, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=39-48
Summary budget: input=1200; output=500; reserve=300
Config owns: BEAM_SIZE, LENGTH_PENALTY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [TextGenerator.__init__(self, cfg)](#context-m034)
4. [TextGenerator.encode(self, source, lengths)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2. Viết phần đầu generate, mở rộng memory, projected, mask, source và hidden cho beam search. Tự chọn BEAM_SIZE, LENGTH_PENALTY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2 tại đây.
```

## Bước 30: TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2

<!-- module
ID: m039
Depends: m024, m032, m034, m036, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=49-63
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [TextGenerator.__init__(self, cfg)](#context-m034)
4. [TextGenerator.step(self, token, hidden, memory, projected, mask, source)](#context-m036)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2. Viết phần đầu generate, mở rộng memory, projected, mask, source và hidden cho beam search. Viết vòng top-k beam, gather hidden theo parent, chặn PAD/BOS và giữ nguyên beam đã EOS. Dùng step trả logp,next_hidden; gather seq/done/lens/next_hidden theo parent; EOS done có logp=0; dừng khi mọi done; trả phần vòng lặp để ghép. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Đây là phần 2/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m039"></a>

```text
Tóm tắt giao diện TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2 tại đây.
```

## Bước 31: Ghép giao diện class TextGenerator

<!-- module
ID: m040
Depends: m034, m035, m036, m037, m038, m039
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=1-63
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [TextGenerator.__init__(self, cfg)](#context-m034)
2. [TextGenerator.encode(self, source, lengths)](#context-m035)
3. [TextGenerator.step(self, token, hidden, memory, projected, mask, source)](#context-m036)
4. [TextGenerator.forward(self, source, lengths, target)](#context-m037)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)
6. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2](#context-m039)

<a id="context-m040"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class TextGenerator kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]

Context m037 — TextGenerator.forward(self, source, lengths, target):
[Dán tóm tắt m037]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m039 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2:
[Dán tóm tắt m039]
```

## Bước 32: build_model(cfg)

<!-- module
ID: m041
Depends: m040
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=64-64
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class TextGenerator](#context-m040)

### Prompt tạo hoặc sửa module

```text
Viết build_model(cfg). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện build_model(cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

```python
# Dán code hiện tại của build_model(cfg) tại đây.
```

## Bước 33: errant_annotator() — phần 1/2

<!-- module
ID: m045
Depends: m009
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=8-29
Summary budget: input=1200; output=500; reserve=300
Config owns: ALLOW_PRETRAINED_EVALUATORS, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL
Config uses: ALLOW_PRETRAINED_EVALUATORS, ARTIFACT_DIR, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)

### Prompt tạo hoặc sửa module

```text
Viết errant_annotator() — phần 1/2. Viết riêng errant_annotator theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. errant_annotator nhận (); Nạp errant3.0.2 và spaCy en_core_web_sm3.8.0 đã cài, chỉ evaluator; thiếu dependency báo lỗi rõ, không tự thay model. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Giữ ràng buộc: ALLOW_PRETRAINED_EVALUATORS=True. Tự chọn SPACY_MODEL_SHA256, SPACY_MODEL_URL phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: METRIC_CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện errant_annotator() — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

```python
# Dán code hiện tại của errant_annotator() — phần 1/2 tại đây.
```

## Bước 34: errant_annotator() — phần 2/2

<!-- module
ID: m046
Depends: m045
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=30-32
Summary budget: input=1200; output=500; reserve=300
Config uses: ALLOW_PRETRAINED_EVALUATORS, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_annotator() — phần 1/2](#context-m045)

### Prompt tạo hoặc sửa module

```text
Viết errant_annotator() — phần 2/2. Viết riêng errant_annotator theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. errant_annotator nhận (); Nạp errant3.0.2 và spaCy en_core_web_sm3.8.0 đã cài, chỉ evaluator; thiếu dependency báo lỗi rõ, không tự thay model. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Dùng lại ALLOW_PRETRAINED_EVALUATORS, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 2/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m045 — errant_annotator() — phần 1/2:
[Dán tóm tắt m045]
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện errant_annotator() — phần 2/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m045 — errant_annotator() — phần 1/2:
[Dán tóm tắt m045]
```

```python
# Dán code hiện tại của errant_annotator() — phần 2/2 tại đây.
```

## Bước 35: edits_as_rows(annotator, original, corrected, coder)

<!-- module
ID: m047
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=34-39
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết edits_as_rows(annotator, original, corrected, coder). Viết riêng edits_as_rows theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. edits_as_rows nhận (annotator, original, corrected, coder); Dùng annotator.parse/annotate lev=False,merging rules, trả edit offsets/type/correction/coder; noop khi không edits. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện edits_as_rows(annotator, original, corrected, coder) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của edits_as_rows(annotator, original, corrected, coder) tại đây.
```

## Bước 36: errant_corpus(records, predictions)

<!-- module
ID: m048
Depends: m046, m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=41-60
Summary budget: input=1200; output=500; reserve=300
Config owns: _REFERENCE_EDITS
Config uses: _REFERENCE_EDITS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_annotator() — phần 2/2](#context-m046)
2. [edits_as_rows(annotator, original, corrected, coder)](#context-m047)

### Prompt tạo hoặc sửa module

```text
Viết errant_corpus(records, predictions). Viết riêng errant_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. errant_corpus nhận (records, predictions); So edits theo API compare_m2 chính thức, beta0.5, nhiều references/coders; cộng TP/FP/FN corpus rồi F0.5, không average sentence F0.5. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Kết quả giữ các trường edit_counts, errant_f05, errant_mode, errant_precision, errant_recall. Khai báo cùng phần này: _REFERENCE_EDITS được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]

Context m047 — edits_as_rows(annotator, original, corrected, coder):
[Dán tóm tắt m047]
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện errant_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]

Context m047 — edits_as_rows(annotator, original, corrected, coder):
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của errant_corpus(records, predictions) tại đây.
```

## Bước 37: gleu_corpus(records, predictions)

<!-- module
ID: m049
Depends: m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=62-83
Summary budget: input=1200; output=500; reserve=300
Config owns: GLEU_ITERATIONS
Config uses: GLEU_ITERATIONS, SEED
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — held, held_sources, native…](#context-m019)

### Prompt tạo hoặc sửa module

```text
Viết gleu_corpus(records, predictions). Viết riêng gleu_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. gleu_corpus nhận (records, predictions); GLEU2016 source-aware ngram1..4, trừ giữ sai từ source, bootstrap reference theo seed42 và500iterations; không nltk corpus_gleu. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Kết quả giữ các trường gleu, gleu_iterations, gleu_variant. Tự chọn GLEU_ITERATIONS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

### Tóm tắt context để dùng tiếp

<a id="context-m049"></a>

```text
Tóm tắt giao diện gleu_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của gleu_corpus(records, predictions) tại đây.
```

## Bước 38: score_corpus(records, predictions)

<!-- module
ID: m050
Depends: m048, m049
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=85-86
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_corpus(records, predictions)](#context-m048)
2. [gleu_corpus(records, predictions)](#context-m049)

### Prompt tạo hoặc sửa module

```text
Viết score_corpus(records, predictions). Viết riêng score_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. score_corpus nhận (records, predictions); Gộp errant_corpus và gleu_corpus; giữ keys errant_f05,gleu và provenance evaluator. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m048 — errant_corpus(records, predictions):
[Dán tóm tắt m048]

Context m049 — gleu_corpus(records, predictions):
[Dán tóm tắt m049]
```

### Tóm tắt context để dùng tiếp

<a id="context-m050"></a>

```text
Tóm tắt giao diện score_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m048 — errant_corpus(records, predictions):
[Dán tóm tắt m048]

Context m049 — gleu_corpus(records, predictions):
[Dán tóm tắt m049]
```

```python
# Dán code hiện tại của score_corpus(records, predictions) tại đây.
```

## Bước 39: decode_tokens(ids)

<!-- module
ID: m051
Depends: m007, m024, m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=1-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)

### Prompt tạo hoặc sửa module

```text
Viết decode_tokens(ids). Viết riêng decode_tokens theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. decode_tokens nhận (ids); Giải mã BPE, bỏ PAD/BOS và dừng EOS; không sửa nội dung dự đoán theo gold. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

### Tóm tắt context để dùng tiếp

<a id="context-m051"></a>

```text
Tóm tắt giao diện decode_tokens(ids) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của decode_tokens(ids) tại đây.
```

## Bước 40: compute_loss(model, batch)

<!-- module
ID: m052
Depends: m024, m032, m040
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

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [Ghép giao diện class TextGenerator](#context-m040)

### Prompt tạo hoặc sửa module

```text
Viết compute_loss(model, batch). Viết loss scalar nối model và batch theo hợp đồng, bỏ padding và giữ gradient; kiểm loss hữu hạn. AdamW lr theo cấu hình đã chọn,betas=(0.9,0.98),eps=1e-9,weight_decay theo cấu hình đã chọn; warmup=0; epochs theo cấu hình đã chọn,batch theo cấu hình đã chọn,eval_batch=16,patience theo cấu hình đã chọn,clip theo cấu hình đã chọn; chọn errant_f05 validation tăng. Dùng các API forward/loss đã dựng, không dùng gold trong đường suy luận. Tự chọn LABEL_SMOOTHING phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

### Tóm tắt context để dùng tiếp

<a id="context-m052"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 41: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m053
Depends: m012, m038, m040, m050, m051
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
2. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)
3. [Ghép giao diện class TextGenerator](#context-m040)
4. [score_corpus(records, predictions)](#context-m050)
5. [decode_tokens(ids)](#context-m051)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, return_predictions=False). Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Tự chọn MAX_DECODE_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại BEAM_SIZE, LENGTH_PENALTY từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]

Context m050 — score_corpus(records, predictions):
[Dán tóm tắt m050]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]
```

### Tóm tắt context để dùng tiếp

<a id="context-m053"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]

Context m050 — score_corpus(records, predictions):
[Dán tóm tắt m050]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 42: Metric, loss và giải mã

<!-- module
ID: m054
Depends: m046
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=1-1
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_annotator() — phần 2/2](#context-m046)

### Prompt tạo hoặc sửa module

```text
Viết Metric, loss và giải mã. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]
```

### Tóm tắt context để dùng tiếp

<a id="context-m054"></a>

```text
Tóm tắt giao diện Metric, loss và giải mã từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]
```

```python
# Dán code hiện tại của Metric, loss và giải mã tại đây.
```

## Bước 43: Huấn luyện và checkpoint — model, optimizer, total_steps

<!-- module
ID: m055
Depends: m012, m033, m034, m041
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
2. [Tạo DataLoader cho các split](#context-m033)
3. [TextGenerator.__init__(self, cfg)](#context-m034)
4. [build_model(cfg)](#context-m041)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — model, optimizer, total_steps. Khởi tạo optimizer AdamW cho tham số có thể học. Tự chọn EPOCHS, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]
```

### Tóm tắt context để dùng tiếp

<a id="context-m055"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — model, optimizer, total_steps từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — model, optimizer, total_steps tại đây.
```

## Bước 44: lr_multiplier(step)

<!-- module
ID: m056
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
Viết lr_multiplier(step). Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. AdamW lr theo cấu hình đã chọn,betas=(0.9,0.98),eps=1e-9,weight_decay theo cấu hình đã chọn; warmup=0; epochs theo cấu hình đã chọn,batch theo cấu hình đã chọn,eval_batch=16,patience theo cấu hình đã chọn,clip theo cấu hình đã chọn; chọn errant_f05 validation tăng. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. Tự chọn WARMUP_STEPS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m056"></a>

```text
Tóm tắt giao diện lr_multiplier(step) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của lr_multiplier(step) tại đây.
```

## Bước 45: Huấn luyện và checkpoint — best_score, scheduler, stale_epochs

<!-- module
ID: m057
Depends: m055, m056
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=8-9
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m055)
2. [lr_multiplier(step)](#context-m056)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện và checkpoint — best_score, scheduler, stale_epochs. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m056 — lr_multiplier(step):
[Dán tóm tắt m056]
```

### Tóm tắt context để dùng tiếp

<a id="context-m057"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, scheduler, stale_epochs từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m056 — lr_multiplier(step):
[Dán tóm tắt m056]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — best_score, scheduler, stale_epochs tại đây.
```

## Bước 46: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m058
Depends: m009, m012, m026, m033, m052, m053, m055, m057
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
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
4. [Tạo DataLoader cho các split](#context-m033)
5. [compute_loss(model, batch)](#context-m052)
6. [evaluate(model, loader, return_predictions=False)](#context-m053)
7. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m055)
8. [Huấn luyện và checkpoint — best_score, scheduler, stale_epochs](#context-m057)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='errant_f05'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m052 — compute_loss(model, batch):
[Dán tóm tắt m052]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m057 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m057]
```

### Tóm tắt context để dùng tiếp

<a id="context-m058"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m052 — compute_loss(model, batch):
[Dán tóm tắt m052]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m057 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m057]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 47: Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…

<!-- module
ID: m059
Depends: m009, m012, m017, m022, m026, m027, m033, m034, m041, m053, m058
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
3. [Dữ liệu và giao thức chia — raw](#context-m017)
4. [Chuẩn hóa và chia dữ liệu](#context-m022)
5. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
6. [restore_tokenizer(state)](#context-m027)
7. [Tạo DataLoader cho các split](#context-m033)
8. [TextGenerator.__init__(self, cfg)](#context-m034)
9. [build_model(cfg)](#context-m041)
10. [evaluate(model, loader, return_predictions=False)](#context-m053)
11. [Vòng huấn luyện, validation và lưu checkpoint](#context-m058)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m027 — restore_tokenizer(state):
[Dán tóm tắt m027]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m058 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m058]
```

### Tóm tắt context để dùng tiếp

<a id="context-m059"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m027 — restore_tokenizer(state):
[Dán tóm tắt m027]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m058 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m058]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… tại đây.
```

## Bước 48: rewrite_text(text)

<!-- module
ID: m060
Depends: m007, m012, m024, m026, m038, m051, m053, m059
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=24; lines=1-9
Summary budget: input=1200; output=500; reserve=300
Config uses: BEAM_SIZE, DEVICE, LENGTH_PENALTY, MAX_DECODE_TOKENS, MAX_SOURCE_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [to_device(batch)](#context-m012)
3. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
4. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)
6. [decode_tokens(ids)](#context-m051)
7. [evaluate(model, loader, return_predictions=False)](#context-m053)
8. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m059)

### Prompt tạo hoặc sửa module

```text
Viết rewrite_text(text). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Dùng lại BEAM_SIZE, DEVICE, LENGTH_PENALTY, MAX_DECODE_TOKENS, MAX_SOURCE_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m059 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m059]
```

### Tóm tắt context để dùng tiếp

<a id="context-m060"></a>

```text
Tóm tắt giao diện rewrite_text(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m059 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m059]
```

```python
# Dán code hiện tại của rewrite_text(text) tại đây.
```

## Bước 49: Suy luận văn bản mới

<!-- module
ID: m061
Depends: m022, m060
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=24; lines=11-11
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m022)
2. [rewrite_text(text)](#context-m060)

### Prompt tạo hoặc sửa module

```text
Viết Suy luận văn bản mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m060 — rewrite_text(text):
[Dán tóm tắt m060]
```

### Tóm tắt context để dùng tiếp

<a id="context-m061"></a>

```text
Tóm tắt giao diện Suy luận văn bản mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m060 — rewrite_text(text):
[Dán tóm tắt m060]
```

```python
# Dán code hiện tại của Suy luận văn bản mới tại đây.
```

## Bước 50: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m010, m011, m054, m061
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 2/2](#context-m010)
2. [decode_utf8(value)](#context-m011)
3. [Metric, loss và giải mã](#context-m054)
4. [Suy luận văn bản mới](#context-m061)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]

Context m054 — Metric, loss và giải mã:
[Dán tóm tắt m054]

Context m061 — Suy luận văn bản mới:
[Dán tóm tắt m061]
```
