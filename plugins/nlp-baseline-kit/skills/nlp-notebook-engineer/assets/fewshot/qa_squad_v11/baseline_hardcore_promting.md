# Prompt hardcore: SQuAD v1.1: word/char, attention và span decoding

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

## Bước 6: adapt(row)

<!-- module
ID: m014
Depends: m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=1-13
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [decode_utf8(value)](#context-m011)

### Prompt tạo hoặc sửa module

```text
Viết adapt(row). Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. SQuAD v1.1 TFDS squad/v1.1:3.0.0; schema id,title,context,question,answers[{answer_start,text}]. Context giữ nguyên để kiểm offsets; official validation là held-out, 10% train titles làm validation với seed=42; assert context/ID không trùng split. Nguồn dữ liệu dataset: SQuAD v1.1; tfds_builder: squad/v1.1:3.0.0. Giữ split, ID, nhãn và references của nguồn. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện adapt(row) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của adapt(row) tại đây.
```

## Bước 7: Dữ liệu, offset và split — article, document, file…

<!-- module
ID: m015
Depends: m008, m009, m010, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=15-35
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_MODE, LOCAL_DATA_DIR, SOURCE
Config uses: CACHE_DIR, DATA_MODE, LOCAL_DATA_DIR, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m008)
2. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
3. [tfds_rows(builder, splits, supervised=False) — phần 2/2](#context-m010)
4. [adapt(row)](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu, offset và split — article, document, file…. Giữ ràng buộc: DATA_MODE='tfds'. Khai báo cùng phần này: LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m014 — adapt(row):
[Dán tóm tắt m014]
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Dữ liệu, offset và split — article, document, file… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m014 — adapt(row):
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Dữ liệu, offset và split — article, document, file… tại đây.
```

## Bước 8: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m016
Depends: m014, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=36-50
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, VAL_FRACTION
Config uses: SEED, VAL_FRACTION
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [adapt(row)](#context-m014)
2. [Dữ liệu, offset và split — article, document, file…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Viết Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: SQuAD v1.1; tfds_builder: squad/v1.1:3.0.0. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: SEED=42; VAL_FRACTION=0.1. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — adapt(row):
[Dán tóm tắt m014]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — adapt(row):
[Dán tóm tắt m014]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 9: Dữ liệu, offset và split

<!-- module
ID: m017
Depends: m016
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=51-52
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu, offset và split. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Dữ liệu, offset và split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]
```

```python
# Dán code hiện tại của Dữ liệu, offset và split tại đây.
```

## Bước 10: Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…

<!-- module
ID: m019
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=2-4
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…. Bổ sung character vocabulary train-only cho QA, giữ word vocabulary và offsets context gốc. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN… tại đây.
```

## Bước 11: context_tokens(text)

<!-- module
ID: m020
Depends: m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=5-9
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)

### Prompt tạo hoặc sửa module

```text
Viết context_tokens(text). Bổ sung character vocabulary train-only cho QA, giữ word vocabulary và offsets context gốc. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện context_tokens(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của context_tokens(text) tại đây.
```

## Bước 12: question_words(text)

<!-- module
ID: m021
Depends: m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=10-10
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_QUESTION_TOKENS
Config uses: MAX_QUESTION_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)

### Prompt tạo hoặc sửa module

```text
Viết question_words(text). Bổ sung character vocabulary train-only cho QA, giữ word vocabulary và offsets context gốc. Tự chọn MAX_QUESTION_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện question_words(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của question_words(text) tại đây.
```

## Bước 13: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m022
Depends: m016, m019, m020
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=12-27
Summary budget: input=1200; output=500; reserve=300
Config owns: VOCAB_SIZE
Config uses: VOCAB_SIZE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)
2. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)
3. [context_tokens(text)](#context-m020)

### Prompt tạo hoặc sửa module

```text
Viết Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Tự chọn VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

```python
# Dán code hiện tại của Bảng ký tự và trạng thái tokenizer tại đây.
```

## Bước 14: restore_tokenizer(state)

<!-- module
ID: m024
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=32-34
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết restore_tokenizer(state). Bổ sung character vocabulary train-only cho QA, giữ word vocabulary và offsets context gốc. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 15: answer_token_span(context, answer)

<!-- module
ID: m025
Depends: m020
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=36-43
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [context_tokens(text)](#context-m020)

### Prompt tạo hoặc sửa module

```text
Viết answer_token_span(context, answer). Viết riêng answer_token_span theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. answer_token_span nhận (context, answer); Căn token span bao phủ tối thiểu qua offsets context gốc bằng bisect, báo exact alignment; không sửa gold text. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. Giữ thứ tự đầu ra: left, right, exact. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện answer_token_span(context, answer) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

```python
# Dán code hiện tại của answer_token_span(context, answer) tại đây.
```

## Bước 16: QADataset.__init__(self, records, training=False)

<!-- module
ID: m026
Depends: m020, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=46-59
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_CONTEXT_TOKENS, WINDOW_STEP
Config uses: MAX_CONTEXT_TOKENS, WINDOW_STEP
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [context_tokens(text)](#context-m020)
2. [answer_token_span(context, answer)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết QADataset.__init__(self, records, training=False). Giữ windows 384 bước 256, căn gold theo offsets và câu hỏi tối đa64; không mất đáp án ở biên. Tự chọn MAX_CONTEXT_TOKENS, WINDOW_STEP phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Viết method trong class QADataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện QADataset.__init__(self, records, training=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của QADataset.__init__(self, records, training=False) tại đây.
```

## Bước 17: QADataset.__len__(self)

<!-- module
ID: m027
Depends: m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=60-60
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QADataset.__init__(self, records, training=False)](#context-m026)

### Prompt tạo hoặc sửa module

```text
Viết QADataset.__len__(self). Giữ windows 384 bước 256, căn gold theo offsets và câu hỏi tối đa64; không mất đáp án ở biên. Viết method trong class QADataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]
```

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện QADataset.__len__(self) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của QADataset.__len__(self) tại đây.
```

## Bước 18: QADataset.__getitem__(self, index)

<!-- module
ID: m028
Depends: m020, m021, m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=61-67
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [context_tokens(text)](#context-m020)
2. [question_words(text)](#context-m021)
3. [QADataset.__init__(self, records, training=False)](#context-m026)

### Prompt tạo hoặc sửa module

```text
Viết QADataset.__getitem__(self, index). Giữ windows 384 bước 256, căn gold theo offsets và câu hỏi tối đa64; không mất đáp án ở biên. Kết quả giữ các trường id, offsets, question, record, targets, words. Viết method trong class QADataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m021 — question_words(text):
[Dán tóm tắt m021]

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]
```

### Tóm tắt context để dùng tiếp

<a id="context-m028"></a>

```text
Tóm tắt giao diện QADataset.__getitem__(self, index) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m021 — question_words(text):
[Dán tóm tắt m021]

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của QADataset.__getitem__(self, index) tại đây.
```

## Bước 19: Ghép giao diện class QADataset

<!-- module
ID: m029
Depends: m026, m027, m028
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=10; lines=45-67
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QADataset.__init__(self, records, training=False)](#context-m026)
2. [QADataset.__len__(self)](#context-m027)
3. [QADataset.__getitem__(self, index)](#context-m028)

<a id="context-m029"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class QADataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]

Context m027 — QADataset.__len__(self):
[Dán tóm tắt m027]

Context m028 — QADataset.__getitem__(self, index):
[Dán tóm tắt m028]
```

## Bước 20: encode_words(rows, width)

<!-- module
ID: m030
Depends: m019, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=69-77
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_WORD_CHARS
Config uses: MAX_WORD_CHARS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)
2. [Bảng ký tự và trạng thái tokenizer](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết encode_words(rows, width). Bổ sung char IDs trong encode_words và cập nhật batch context/question để model dùng char-CNN. Giữ thứ tự đầu ra: words, chars. Tự chọn MAX_WORD_CHARS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện encode_words(rows, width) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của encode_words(rows, width) tại đây.
```

## Bước 21: collate_batch(items)

<!-- module
ID: m031
Depends: m030
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=79-89
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [encode_words(rows, width)](#context-m030)

### Prompt tạo hoặc sửa module

```text
Viết collate_batch(items). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường context, context_chars, context_mask, ids, items, lengths, question, question_chars, question_lengths, question_mask, targets. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m030 — encode_words(rows, width):
[Dán tóm tắt m030]
```

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m030 — encode_words(rows, width):
[Dán tóm tắt m030]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 22: to_device(batch)

<!-- module
ID: m012
Depends: m031
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

1. [collate_batch(items)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 23: make_loader(dataset, train=False)

<!-- module
ID: m013
Depends: m016, m031
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

1. [Chuẩn hóa và chia dữ liệu](#context-m016)
2. [collate_batch(items)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Viết make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 24: Tạo DataLoader cho các split

<!-- module
ID: m032
Depends: m013, m016, m022, m025, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=91-100
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_ANSWER_TOKENS
Config uses: MAX_ANSWER_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m013)
2. [Chuẩn hóa và chia dữ liệu](#context-m016)
3. [Bảng ký tự và trạng thái tokenizer](#context-m022)
4. [answer_token_span(context, answer)](#context-m025)
5. [Ghép giao diện class QADataset](#context-m029)

### Prompt tạo hoặc sửa module

```text
Viết Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Tự chọn MAX_ANSWER_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]
```

### Tóm tắt context để dùng tiếp

<a id="context-m032"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 25: QAModel.__init__(self, cfg)

<!-- module
ID: m033
Depends: m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=2-16
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, HIDDEN_DIM
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết QAModel.__init__(self, cfg). Thay constructor QAModel bằng word/char-CNN, BiLSTM và attention hai chiều với start/end heads. cfg n_words,n_chars,hidden_dim theo cấu hình đã chọn,dropout theo cấu hình đã chọn; word128,char16,charCNN32 kernel3; encoder input160, dim256; modeling input4*dim, end_encoder dim; heads input5*dim. Tự chọn DROPOUT, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa n_words, n_chars, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện QAModel.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của QAModel.__init__(self, cfg) tại đây.
```

## Bước 26: QAModel.char_features(self, chars)

<!-- module
ID: m034
Depends: m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=17-22
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QAModel.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết QAModel.char_features(self, chars). Viết char_features bỏ PAD khi max-pooling, từ toàn PAD trả vector không. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện QAModel.char_features(self, chars) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của QAModel.char_features(self, chars) tại đây.
```

## Bước 27: QAModel.sequence(self, module, x, lengths)

<!-- module
ID: m035
Depends: m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=23-26
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QAModel.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết QAModel.sequence(self, module, x, lengths). Viết sequence(module,x,lengths) đóng gói BiLSTM, dropout và khôi phục đúng chiều dài tensor. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện QAModel.sequence(self, module, x, lengths) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của QAModel.sequence(self, module, x, lengths) tại đây.
```

## Bước 28: QAModel.forward(self, batch)

<!-- module
ID: m036
Depends: m031, m033, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=27-41
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)
2. [QAModel.__init__(self, cfg)](#context-m033)
3. [QAModel.char_features(self, chars)](#context-m034)
4. [QAModel.sequence(self, module, x, lengths)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Viết QAModel.forward(self, batch). Viết phần đầu forward: word/char features, encoder chung và similarity trilinear có masks đúng. Hoàn tất forward bằng fusion bốn thành phần, modeling/end BiLSTM và hai span heads. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — QAModel.char_features(self, chars):
[Dán tóm tắt m034]

Context m035 — QAModel.sequence(self, module, x, lengths):
[Dán tóm tắt m035]
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện QAModel.forward(self, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — QAModel.char_features(self, chars):
[Dán tóm tắt m034]

Context m035 — QAModel.sequence(self, module, x, lengths):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của QAModel.forward(self, batch) tại đây.
```

## Bước 29: Ghép giao diện class QAModel

<!-- module
ID: m037
Depends: m033, m034, m035, m036
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=1-41
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QAModel.__init__(self, cfg)](#context-m033)
2. [QAModel.char_features(self, chars)](#context-m034)
3. [QAModel.sequence(self, module, x, lengths)](#context-m035)
4. [QAModel.forward(self, batch)](#context-m036)

<a id="context-m037"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class QAModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — QAModel.char_features(self, chars):
[Dán tóm tắt m034]

Context m035 — QAModel.sequence(self, module, x, lengths):
[Dán tóm tắt m035]

Context m036 — QAModel.forward(self, batch):
[Dán tóm tắt m036]
```

## Bước 30: build_model(cfg)

<!-- module
ID: m038
Depends: m037
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=42-42
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class QAModel](#context-m037)

### Prompt tạo hoặc sửa module

```text
Viết build_model(cfg). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện build_model(cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

```python
# Dán code hiện tại của build_model(cfg) tại đây.
```

## Bước 31: normalize_answer(text)

<!-- module
ID: m040
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=1-5
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết normalize_answer(text). Viết riêng normalize_answer theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. normalize_answer nhận (text); Chuẩn SQuAD: lowercase, bỏ ASCII punctuation và articles a/an/the, collapse whitespace. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m040"></a>

```text
Tóm tắt giao diện normalize_answer(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của normalize_answer(text) tại đây.
```

## Bước 32: answer_metrics(prediction, references)

<!-- module
ID: m041
Depends: m040
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=6-15
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_answer(text)](#context-m040)

### Prompt tạo hoặc sửa module

```text
Viết answer_metrics(prediction, references). Viết riêng answer_metrics theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. answer_metrics nhận (prediction, references); Tính exact match và bag-token F1, max qua mọi refs; common0 trả F1=0 theo evaluator v1.1. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m040 — normalize_answer(text):
[Dán tóm tắt m040]
```

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện answer_metrics(prediction, references) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m040 — normalize_answer(text):
[Dán tóm tắt m040]
```

```python
# Dán code hiện tại của answer_metrics(prediction, references) tại đây.
```

## Bước 33: compute_loss(model, batch)

<!-- module
ID: m042
Depends: m031, m037
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=17-21
Summary budget: input=1200; output=500; reserve=300
Config owns: NEGATIVE_WINDOW_WEIGHT
Config uses: NEGATIVE_WINDOW_WEIGHT
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)
2. [Ghép giao diện class QAModel](#context-m037)

### Prompt tạo hoặc sửa module

```text
Viết compute_loss(model, batch). Giữ cross-entropy start/end có window weights và gold alignment khi đổi kiến trúc. Tự chọn NEGATIVE_WINDOW_WEIGHT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 34: best_spans(start, end, batch)

<!-- module
ID: m043
Depends: m031, m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=23-30
Summary budget: input=1200; output=500; reserve=300
Config uses: MAX_ANSWER_TOKENS
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)
2. [Tạo DataLoader cho các split](#context-m032)

### Prompt tạo hoặc sửa module

```text
Viết best_spans(start, end, batch). Viết riêng best_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. best_spans nhận (start, end, batch); Chọn joint logits với start>=1,end>=start,length<=64 và context mask; trả starts,ends,margin so CLS scores để gộp windows. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. Giữ thứ tự đầu ra: kết quả 1, kết quả 2, kết quả 3. Dùng lại MAX_ANSWER_TOKENS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]
```

### Tóm tắt context để dùng tiếp

<a id="context-m043"></a>

```text
Tóm tắt giao diện best_spans(start, end, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của best_spans(start, end, batch) tại đây.
```

## Bước 35: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m044
Depends: m012, m037, m041, m043
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=32-50
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Ghép giao diện class QAModel](#context-m037)
3. [answer_metrics(prediction, references)](#context-m041)
4. [best_spans(start, end, batch)](#context-m043)

### Prompt tạo hoặc sửa module

```text
Viết evaluate(model, loader, return_predictions=False). Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]

Context m041 — answer_metrics(prediction, references):
[Dán tóm tắt m041]

Context m043 — best_spans(start, end, batch):
[Dán tóm tắt m043]
```

### Tóm tắt context để dùng tiếp

<a id="context-m044"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]

Context m041 — answer_metrics(prediction, references):
[Dán tóm tắt m041]

Context m043 — best_spans(start, end, batch):
[Dán tóm tắt m043]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 36: Huấn luyện/checkpoint — model, optimizer, total_steps

<!-- module
ID: m045
Depends: m012, m032, m033, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=1-4
Summary budget: input=1200; output=500; reserve=300
Config owns: EPOCHS, LEARNING_RATE, WEIGHT_DECAY
Config uses: DEVICE, EPOCHS, LEARNING_RATE, MODEL_CONFIG, WEIGHT_DECAY
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Tạo DataLoader cho các split](#context-m032)
3. [QAModel.__init__(self, cfg)](#context-m033)
4. [build_model(cfg)](#context-m038)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện/checkpoint — model, optimizer, total_steps. Khởi tạo optimizer AdamW cho tham số có thể học. Tự chọn EPOCHS, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện Huấn luyện/checkpoint — model, optimizer, total_steps từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của Huấn luyện/checkpoint — model, optimizer, total_steps tại đây.
```

## Bước 37: lr_multiplier(step)

<!-- module
ID: m046
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=5-7
Summary budget: input=1200; output=500; reserve=300
Config owns: WARMUP_STEPS
Config uses: WARMUP_STEPS
-->

### Prompt tạo hoặc sửa module

```text
Viết lr_multiplier(step). Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. AdamW lr theo cấu hình đã chọn,weight_decay theo cấu hình đã chọn,warmup=0; epochs theo cấu hình đã chọn,batch theo cấu hình đã chọn,patience theo cấu hình đã chọn,clip theo cấu hình đã chọn; chọn exact_match validation tăng; vocabulary chỉ train. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. Tự chọn WARMUP_STEPS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện lr_multiplier(step) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của lr_multiplier(step) tại đây.
```

## Bước 38: Huấn luyện/checkpoint — best_score, scheduler, stale_epochs

<!-- module
ID: m047
Depends: m045, m046
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=8-9
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Huấn luyện/checkpoint — model, optimizer, total_steps](#context-m045)
2. [lr_multiplier(step)](#context-m046)

### Prompt tạo hoặc sửa module

```text
Viết Huấn luyện/checkpoint — best_score, scheduler, stale_epochs. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m046 — lr_multiplier(step):
[Dán tóm tắt m046]
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện Huấn luyện/checkpoint — best_score, scheduler, stale_epochs từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m046 — lr_multiplier(step):
[Dán tóm tắt m046]
```

```python
# Dán code hiện tại của Huấn luyện/checkpoint — best_score, scheduler, stale_epochs tại đây.
```

## Bước 39: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m048
Depends: m009, m012, m022, m032, m042, m044, m045, m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=10-38
Summary budget: input=1200; output=500; reserve=300
Config owns: CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Bảng ký tự và trạng thái tokenizer](#context-m022)
4. [Tạo DataLoader cho các split](#context-m032)
5. [compute_loss(model, batch)](#context-m042)
6. [evaluate(model, loader, return_predictions=False)](#context-m044)
7. [Huấn luyện/checkpoint — model, optimizer, total_steps](#context-m045)
8. [Huấn luyện/checkpoint — best_score, scheduler, stale_epochs](#context-m047)

### Prompt tạo hoặc sửa module

```text
Viết Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='exact_match'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m042 — compute_loss(model, batch):
[Dán tóm tắt m042]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m047 — Huấn luyện/checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m047]
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m042 — compute_loss(model, batch):
[Dán tóm tắt m042]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m047 — Huấn luyện/checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 40: Reload và held-out official validation — best_model, checkpoint, inference_config…

<!-- module
ID: m049
Depends: m009, m012, m015, m016, m022, m024, m032, m033, m038, m044, m048
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=1-17
Summary budget: input=1200; output=500; reserve=300
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Dữ liệu, offset và split — article, document, file…](#context-m015)
4. [Chuẩn hóa và chia dữ liệu](#context-m016)
5. [Bảng ký tự và trạng thái tokenizer](#context-m022)
6. [restore_tokenizer(state)](#context-m024)
7. [Tạo DataLoader cho các split](#context-m032)
8. [QAModel.__init__(self, cfg)](#context-m033)
9. [build_model(cfg)](#context-m038)
10. [evaluate(model, loader, return_predictions=False)](#context-m044)
11. [Vòng huấn luyện, validation và lưu checkpoint](#context-m048)

### Prompt tạo hoặc sửa module

```text
Viết Reload và held-out official validation — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m024 — restore_tokenizer(state):
[Dán tóm tắt m024]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m048 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m048]
```

### Tóm tắt context để dùng tiếp

<a id="context-m049"></a>

```text
Tóm tắt giao diện Reload và held-out official validation — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m024 — restore_tokenizer(state):
[Dán tóm tắt m024]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m048 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m048]
```

```python
# Dán code hiện tại của Reload và held-out official validation — best_model, checkpoint, inference_config… tại đây.
```

## Bước 41: answer_question(context, question)

<!-- module
ID: m050
Depends: m012, m013, m029, m043, m049
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=21; lines=1-13
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [make_loader(dataset, train=False)](#context-m013)
3. [Ghép giao diện class QADataset](#context-m029)
4. [best_spans(start, end, batch)](#context-m043)
5. [Reload và held-out official validation — best_model, checkpoint, inference_config…](#context-m049)

### Prompt tạo hoặc sửa module

```text
Viết answer_question(context, question). Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. Kết quả giữ các trường end, start, text. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]

Context m043 — best_spans(start, end, batch):
[Dán tóm tắt m043]

Context m049 — Reload và held-out official validation — best_model, checkpoint, inference_config…:
[Dán tóm tắt m049]
```

### Tóm tắt context để dùng tiếp

<a id="context-m050"></a>

```text
Tóm tắt giao diện answer_question(context, question) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]

Context m043 — best_spans(start, end, batch):
[Dán tóm tắt m043]

Context m049 — Reload và held-out official validation — best_model, checkpoint, inference_config…:
[Dán tóm tắt m049]
```

```python
# Dán code hiện tại của answer_question(context, question) tại đây.
```

## Bước 42: Suy luận câu hỏi mới — sample

<!-- module
ID: m051
Depends: m016, m050
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=21; lines=15-16
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)
2. [answer_question(context, question)](#context-m050)

### Prompt tạo hoặc sửa module

```text
Viết Suy luận câu hỏi mới — sample. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m050 — answer_question(context, question):
[Dán tóm tắt m050]
```

### Tóm tắt context để dùng tiếp

<a id="context-m051"></a>

```text
Tóm tắt giao diện Suy luận câu hỏi mới — sample từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m050 — answer_question(context, question):
[Dán tóm tắt m050]
```

```python
# Dán code hiện tại của Suy luận câu hỏi mới — sample tại đây.
```

## Bước 43: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m007, m017, m051
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Dữ liệu, offset và split](#context-m017)
3. [Suy luận câu hỏi mới — sample](#context-m051)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m017 — Dữ liệu, offset và split:
[Dán tóm tắt m017]

Context m051 — Suy luận câu hỏi mới — sample:
[Dán tóm tắt m051]
```
