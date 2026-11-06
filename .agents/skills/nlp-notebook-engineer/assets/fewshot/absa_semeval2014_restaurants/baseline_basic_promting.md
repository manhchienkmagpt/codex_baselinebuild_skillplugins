# Prompt basic: SemEval restaurants: character BIO, word features, BiLSTM-CRF và polarity

<!-- prompt-guide
Format: 3
Variant: basic
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
Source: baseline_basic.ipynb#cell=5; lines=1-41
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
SEED = 42
DATASET_ID = "tomaarsen/setfit-absa-semeval-restaurants"
DATASET_REVISION = "8885372fe73256f96bb60f65b550538ac5c26047"
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
VARIANT = "basic"
USE_LOCAL_DATA = False
LOCAL_DATA_DIR = Path("data/setfit_absa_restaurants")
ARTIFACT_DIR = Path("artifacts") / VARIANT
CACHE_DIR = ARTIFACT_DIR / "dataset_cache"
CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pt"
VAL_FRACTION = 0.20
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
POLARITY_LOSS_WEIGHT = 0.25
CHAR_EMBED_DIM = 32
WORD_EMBED_DIM = 0
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2
WORD_MIN_FREQ = 1
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Thiết bị: {DEVICE}; biến thể: {VARIANT}; batch: {BATCH_SIZE}; tối đa {EPOCHS} epoch")
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=7; lines=1-22
Source: baseline_basic.ipynb#cell=7; lines=1-22
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

<a id="context-m006"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện load_raw_data() từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def load_raw_data():
    try:
        if USE_LOCAL_DATA:
            paths = {split: LOCAL_DATA_DIR / f"{split}.parquet" for split in ("train", "test")}
            expected = {
                "train": "d82e7c7b8b78bb7280ccfee2ce36f5de04d9fd3d82c8259d852f7d031d5840ac",
                "test": "c91a8ebd0cdbec6999bd6cff844a1081efbc9039afc950a10dc1dc8cbf70083e",
            }
            for split, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Thiếu {path}; tải đúng Parquet ở revision đã ghi.")
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected[split]:
                    raise ValueError(f"{path} không khớp SHA-256 của revision đã kiểm tra.")
            return load_dataset("parquet", data_files={k: str(v) for k, v in paths.items()},
                                cache_dir=str(CACHE_DIR))
        return load_dataset(DATASET_ID, revision=DATASET_REVISION, cache_dir=str(CACHE_DIR))
    except Exception as exc:
        raise RuntimeError(
            "Không đọc được bộ dữ liệu yêu cầu. Kiểm tra mạng/quyền Hugging Face, hoặc đặt "
            "USE_LOCAL_DATA=True và cung cấp train.parquet, test.parquet đúng revision tại "
            f"{LOCAL_DATA_DIR}. Không thay bằng bộ dữ liệu khác. Chi tiết: {exc}"
        ) from exc
```

## Bước 3: group_rows(rows, split) — phần 1/3

<!-- module
ID: m007
Depends: m004
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=7; lines=25-27
Source: baseline_basic.ipynb#cell=7; lines=25-27
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

<a id="context-m007"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện group_rows(rows, split) — phần 1/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def group_rows(rows, split):
    grouped = {}
    duplicates = 0
```

## Bước 4: group_rows(rows, split) — phần 2/3

<!-- module
ID: m008
Depends: m004, m007
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=7; lines=28-52
Source: baseline_basic.ipynb#cell=7; lines=28-52
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [group_rows(rows, split) — phần 1/3](#context-m007)

<a id="context-m008"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện group_rows(rows, split) — phần 2/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m007 — group_rows(rows, split) — phần 1/3:
[Dán tóm tắt m007]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    for row in rows:
        if not {"text", "span", "label", "ordinal"}.issubset(row):
            raise ValueError("Schema phải có text, span, label, ordinal.")
        text, term, ordinal = row["text"], row["span"], row["ordinal"]
        if not isinstance(text, str) or not isinstance(term, str) or not text or not term:
            raise ValueError(f"Văn bản/aspect rỗng hoặc sai kiểu ở {split}: {row}")
        if not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(f"ordinal phải là số nguyên không âm: {row}")
        matches = list(re.finditer(re.escape(term), text))
        if ordinal >= len(matches):
            raise ValueError(f"Không tìm được lần xuất hiện ordinal={ordinal}: {row}")
        start, end = matches[ordinal].span()  # Chính xác, phân biệt hoa/thường; end loại trừ.
        label = row["label"]
        if label not in LABEL_TO_ID and not (split == "test" and label == ""):
            raise ValueError(f"Polarity không hợp lệ ở {split}: {label!r}")
        label = label if label else None
        record = grouped.setdefault(text, {"sid": hashlib.sha256(text.encode()).hexdigest(),
                                           "text": text, "spans": {}})
        key = (start, end)
        if key in record["spans"]:
            if record["spans"][key] != label:
                raise ValueError(f"Nhãn mâu thuẫn cho cùng vị trí: {row}")
            duplicates += 1
        record["spans"][key] = label
    records = []
```

## Bước 5: group_rows(rows, split) — phần 3/3

<!-- module
ID: m009
Depends: m004, m008
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=7; lines=53-60
Source: baseline_basic.ipynb#cell=7; lines=53-60
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [group_rows(rows, split) — phần 2/3](#context-m008)

<a id="context-m009"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện group_rows(rows, split) — phần 3/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m008 — group_rows(rows, split) — phần 2/3:
[Dán tóm tắt m008]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    for text in sorted(grouped):
        record = grouped[text]
        record["spans"] = [(a, b, p) for (a, b), p in sorted(record["spans"].items())]
        if any(left[1] > right[0] for left, right in zip(record["spans"], record["spans"][1:])):
            raise ValueError("Có span chồng lấn; cần mô hình span thay cho BIO một lớp.")
        records.append(record)
    print(f"{split}: {len(rows)} hàng → {len(records)} nhóm câu; gộp {duplicates} hàng lặp.")
    return records
```

## Bước 6: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m010
Depends: m004, m006, m009
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=7; lines=63-81
Source: baseline_basic.ipynb#cell=7; lines=63-81
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [load_raw_data()](#context-m006)
3. [group_rows(rows, split) — phần 3/3](#context-m009)

<a id="context-m010"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m006 — load_raw_data():
[Dán tóm tắt m006]

Context m009 — group_rows(rows, split) — phần 3/3:
[Dán tóm tắt m009]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
raw = load_raw_data()
if set(raw) != {"train", "test"}:
    raise ValueError(f"Split khác revision đã kiểm tra: {list(raw)}")
all_train_records = group_rows(raw["train"], "train")
test_records = group_rows(raw["test"], "test")
# Chỉ dùng danh tính câu test để kiểm tra trùng; không dùng nhãn test để học/chọn mô hình.
test_texts = {record["text"] for record in test_records}
eligible_records = [record for record in all_train_records if record["text"] not in test_texts]
removed_overlap = len(all_train_records) - len(eligible_records)
indices = list(range(len(eligible_records)))
random.Random(SEED).shuffle(indices)
val_size = max(1, math.ceil(VAL_FRACTION * len(indices)))
val_indices = set(indices[:val_size])
train_records = [r for i, r in enumerate(eligible_records) if i not in val_indices]
val_records = [r for i, r in enumerate(eligible_records) if i in val_indices]
if not train_records or not val_records:
    raise ValueError("Không đủ nhóm câu để chia train/validation.")
assert not ({r["sid"] for r in train_records} & {r["sid"] for r in val_records})
assert not ({r["sid"] for r in eligible_records} & {r["sid"] for r in test_records})
```

## Bước 7: Tải dữ liệu, nhóm câu và chia validation

<!-- module
ID: m012
Depends: m010
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=7; lines=87-94
Source: baseline_basic.ipynb#cell=7; lines=87-94
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m010)

<a id="context-m012"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tải dữ liệu, nhóm câu và chia validation từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
SPLIT_FINGERPRINTS = {name: fingerprint(records) for name, records in
                      (("train", train_records), ("validation", val_records), ("test", test_records))}
print(f"Loại khỏi train gốc {removed_overlap} nhóm câu trùng test.")
for name, records in (("train", train_records), ("validation", val_records), ("test", test_records)):
    counts = Counter(p if p is not None else "chưa có nhãn" for r in records for _, _, p in r["spans"])
    print(f"{name}: {len(records)} câu; polarity={dict(counts)}; SHA-256={SPLIT_FINGERPRINTS[name]}")
print("Ví dụ train:", train_records[0])
print("Lưu ý: nguồn SetFit chỉ giữ câu có aspect; không phải toàn bộ câu của SemEval gốc.")
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
Source: baseline_basic.ipynb#cell=9; lines=3-10
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Chuẩn hóa và chia dữ liệu](#context-m010)
3. [words_with_offsets(text)](#context-m014)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m014 — words_with_offsets(text):
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
WORD_PATTERN = r"\w+|[^\w\s]"



char_counts = Counter(ch for record in train_records for ch in record["text"])
char_vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
char_vocab.update({ch: i + 2 for i, ch in enumerate(sorted(char_counts))})
word_vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
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
Source: baseline_basic.ipynb#cell=9; lines=13-22
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [words_with_offsets(text)](#context-m014)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung encode_record(record). Nâng encode_record để word ID phủ đúng khoảng ký tự, giữ BIO và spans gốc. Kết quả giữ các trường char_ids, record, tags, word_ids. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m014 — words_with_offsets(text):
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def encode_record(record):
    text=record["text"]
    char_ids=[char_vocab.get(ch,UNK_ID) for ch in text]
    tags=[0]*len(text)
    for start,end,_ in record["spans"]:
        assert 0<=start<end<=len(text)
        assert all(tag==0 for tag in tags[start:end])
        tags[start]=1
        tags[start+1:end]=[2]*(end-start-1)
    return {"char_ids":char_ids,"tags":tags,"record":record}
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=35-36
Source: baseline_basic.ipynb#cell=9; lines=26-27
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [encode_record(record)](#context-m016)

<a id="context-m017"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện AspectDataset.__init__(self, records) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m016 — encode_record(record):
[Dán tóm tắt m016]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self, records):
        self.items = [encode_record(record) for record in records]
```

## Bước 12: AspectDataset.__len__(self)

<!-- module
ID: m018
Depends: m017
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=38-39
Source: baseline_basic.ipynb#cell=9; lines=29-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectDataset.__init__(self, records)](#context-m017)

<a id="context-m018"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện AspectDataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):
        return len(self.items)
```

## Bước 13: AspectDataset.__getitem__(self, index)

<!-- module
ID: m019
Depends: m017
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=41-42
Source: baseline_basic.ipynb#cell=9; lines=32-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectDataset.__init__(self, records)](#context-m017)

<a id="context-m019"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện AspectDataset.__getitem__(self, index) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m017 — AspectDataset.__init__(self, records):
[Dán tóm tắt m017]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __getitem__(self, index):
        return self.items[index]
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
Source: baseline_basic.ipynb#cell=9; lines=36-53
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung collate_batch(items). Cập nhật collate_batch nhận word IDs, masks và gold polarity spans hợp lệ. lengths CPU, mask bool; gold_spans[M,4] chỉ polarity có nhãn; records giữ sid,offsets; AspectDataset/make_loader dùng encode_record mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def collate_batch(items):
    lengths = torch.tensor([len(item["char_ids"]) for item in items], dtype=torch.long)
    width = int(lengths.max())
    batch = {name: torch.zeros(len(items), width, dtype=torch.long)
             for name in ("char_ids", "tags")}
    spans = []
    for i, item in enumerate(items):
        length = len(item["char_ids"])
        for name in batch:
            batch[name][i, :length] = torch.tensor(item[name])
        for start, end, polarity in item["record"]["spans"]:
            if polarity is not None:
                spans.append((i, start, end, LABEL_TO_ID[polarity]))
    batch["lengths"] = lengths  # pack_padded_sequence nhận lengths trên CPU.
    batch["mask"] = torch.arange(width)[None, :] < lengths[:, None]
    batch["gold_spans"] = torch.tensor(spans, dtype=torch.long).reshape(-1, 4)
    batch["records"] = [item["record"] for item in items]
    return batch
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=65-68
Source: baseline_basic.ipynb#cell=9; lines=56-59
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class AspectDataset](#context-m020)
3. [collate_batch(items)](#context-m021)

<a id="context-m022"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện make_loader(records, shuffle=False, batch_size) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m020 — Ghép giao diện class AspectDataset:
[Dán tóm tắt m020]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def make_loader(records, shuffle=False, batch_size=EVAL_BATCH_SIZE):
    return DataLoader(AspectDataset(records), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))
```

## Bước 17: Tạo DataLoader cho các split

<!-- module
ID: m023
Depends: m004, m010, m015, m022
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=70-74
Source: baseline_basic.ipynb#cell=9; lines=61-65
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Chuẩn hóa và chia dữ liệu](#context-m010)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)
4. [make_loader(records, shuffle=False, batch_size)](#context-m022)

<a id="context-m023"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m010 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m010]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m022 — make_loader(records, shuffle=False, batch_size):
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_loader = make_loader(train_records, shuffle=True, batch_size=BATCH_SIZE)
val_loader = make_loader(val_records)
test_loader = make_loader(test_records)
print(f"Từ vựng chỉ fit train: {len(char_vocab)} ký tự; {len(word_vocab)} từ.")
print(f"Độ dài ký tự lớn nhất trong train: {max(len(r['text']) for r in train_records)}; không truncation.")
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
Source: baseline_basic.ipynb#cell=11; lines=2-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class LinearChainCRF](#context-m028)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung AspectModel.__init__(self, cfg, n_chars, n_words). Thay constructor AspectModel bằng char/word embedding, BiLSTM số lớp được chọn, tag head, CRF và polarity head. Khai báo cùng phần này: MODEL_CONFIG có các khóa char_dim, word_dim, hidden_dim, num_layers, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m028 — Ghép giao diện class LinearChainCRF:
[Dán tóm tắt m028]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self, cfg, n_chars, n_words):
        super().__init__()
        self.char_embedding = nn.Embedding(n_chars, cfg["char_dim"], padding_idx=PAD_ID)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.encoder = nn.RNN(cfg["char_dim"], cfg["hidden_dim"], batch_first=True,
                              nonlinearity="tanh", bidirectional=False)
        self.tag_head = nn.Linear(cfg["hidden_dim"], 3)
        self.polarity_head = nn.Linear(cfg["hidden_dim"], len(LABELS))
        # Trọng số được tính chỉ từ BIO train; được lưu trong state_dict để tải lại chính xác.
        self.register_buffer("tag_weights", torch.ones(3))
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
Source: baseline_basic.ipynb#cell=11; lines=13-21
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m021)
2. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung AspectModel.forward(self, batch). Viết forward ghép char/word embedding theo ký tự, packed BiLSTM và BIO emissions. Giữ thứ tự đầu ra: hidden, kết quả 2. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def forward(self, batch):
        embedded = self.dropout(self.char_embedding(batch["char_ids"]))
        packed = pack_padded_sequence(embedded, batch["lengths"],
                                       batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True,
                                         total_length=batch["char_ids"].size(1))
        hidden = self.dropout(hidden)
        return hidden, self.tag_head(hidden)
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
Source: baseline_basic.ipynb#cell=11; lines=23-25
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m021)
2. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung AspectModel.extraction_loss(self, emissions, batch). Viết extraction_loss và decode của AspectModel để gọi CRF đã dựng, giữ chữ ký caller hiện tại. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m021 — collate_batch(items):
[Dán tóm tắt m021]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def extraction_loss(self, emissions, batch):
        return nn.functional.cross_entropy(emissions[batch["mask"]],
                                            batch["tags"][batch["mask"]], weight=self.tag_weights)
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
Source: baseline_basic.ipynb#cell=11; lines=27-29
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung AspectModel.decode(self, emissions, mask). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def decode(self, emissions, mask):
        ids = emissions.argmax(-1).tolist()
        return [seq[:length] for seq, length in zip(ids, mask.sum(1).tolist())]
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
Source: baseline_basic.ipynb#cell=11; lines=31-38
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung AspectModel.classify_spans(self, hidden, spans, mask). Nâng classify_spans dùng hai biên, mean span và mean câu có mask cho polarity. Viết method trong class AspectModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def classify_spans(self, hidden, spans, mask):
        if spans.numel() == 0:
            return hidden.new_empty((0, len(LABELS)))
        row, start, end = spans[:, 0], spans[:, 1], spans[:, 2]
        prefix = torch.cat([hidden.new_zeros(hidden.size(0), 1, hidden.size(2)),
                            hidden.cumsum(dim=1)], dim=1)
        mean = (prefix[row, end] - prefix[row, start]) / (end - start).unsqueeze(1)
        return self.polarity_head(mean)
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
Source: baseline_basic.ipynb#cell=12; lines=1-7
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)
3. [AspectModel.__init__(self, cfg, n_chars, n_words)](#context-m029)
4. [Ghép giao diện class AspectModel](#context-m034)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Cấu hình kiến trúc mô hình. Dùng lại MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m029 — AspectModel.__init__(self, cfg, n_chars, n_words):
[Dán tóm tắt m029]

Context m034 — Ghép giao diện class AspectModel:
[Dán tóm tắt m034]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
MODEL_CONFIG = {"char_dim": CHAR_EMBED_DIM, "word_dim": WORD_EMBED_DIM,
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}
model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE)
tag_counts = torch.bincount(torch.tensor([tag for item in train_loader.dataset.items for tag in item["tags"]]), minlength=3).float()
tag_weights = tag_counts.clamp_min(1).rsqrt()
model.tag_weights.copy_((tag_weights / tag_weights.mean()).to(DEVICE))
print(f"Số tham số học từ đầu: {sum(p.numel() for p in model.parameters()):,}")
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=1-3
Source: baseline_basic.ipynb#cell=14; lines=1-3
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [collate_batch(items)](#context-m021)

<a id="context-m036"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện move_batch(batch) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m021 — collate_batch(items):
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def move_batch(batch):
    return {name: (value.to(DEVICE) if torch.is_tensor(value) and name != "lengths" else value)
            for name, value in batch.items()}
```

## Bước 31: bio_to_spans(tags)

<!-- module
ID: m037
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=6-15
Source: baseline_basic.ipynb#cell=14; lines=6-15
-->

<a id="context-m037"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện bio_to_spans(tags) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def bio_to_spans(tags):
    # I sau O được sửa thành B; cùng quy tắc ở cả hai biến thể. Không trim khoảng trắng.
    spans, start = [], None
    for index, tag in enumerate(list(tags) + [0]):
        if tag in (0, 1) and start is not None:
            spans.append((start, index))
            start = None
        if tag == 1 or (tag == 2 and start is None):
            start = index
    return spans
```

## Bước 32: set_prf(gold, predicted)

<!-- module
ID: m038
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=18-24
Source: baseline_basic.ipynb#cell=14; lines=18-24
-->

<a id="context-m038"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện set_prf(gold, predicted) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def set_prf(gold, predicted):
    tp = len(gold & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * tp / (len(gold) + len(predicted)) if gold or predicted else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "predicted": len(predicted), "gold": len(gold)}
```

## Bước 33: classification_metrics(gold, predicted)

<!-- module
ID: m039
Depends: m004
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=27-45
Source: baseline_basic.ipynb#cell=14; lines=27-45
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)

<a id="context-m039"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện classification_metrics(gold, predicted) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def classification_metrics(gold, predicted):
    if not gold:
        return {"macro_f1": None, "micro_f1": None, "accuracy": None, "count": 0,
                "per_class": None, "confusion_matrix": None}
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for truth, guess in zip(gold, predicted):
        confusion[truth, guess] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted_count = int(confusion[:, i].sum())
        denominator = support + predicted_count
        per_class[label] = {"f1": 2 * tp / denominator if denominator else 0.0,
                            "support": support}
    accuracy = float(np.trace(confusion) / confusion.sum())
    return {"macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "micro_f1": accuracy, "accuracy": accuracy, "count": len(gold),
            "per_class": per_class, "confusion_matrix": confusion.tolist()}
```

## Bước 34: evaluate(model, loader, include_predictions=False) — phần 1/3

<!-- module
ID: m040
Depends: m004, m034, m036, m037, m038, m039
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=48-53
Source: baseline_basic.ipynb#cell=14; lines=48-53
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Ghép giao diện class AspectModel](#context-m034)
3. [move_batch(batch)](#context-m036)
4. [bio_to_spans(tags)](#context-m037)
5. [set_prf(gold, predicted)](#context-m038)
6. [classification_metrics(gold, predicted)](#context-m039)

<a id="context-m040"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, include_predictions=False) — phần 1/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def evaluate(model, loader, include_predictions=False):
    model.eval()
    gold_aspects, predicted_aspects, gold_joint, predicted_joint = set(), set(), set(), set()
    polarity_gold, polarity_predicted, examples = [], [], []
    joint_sentence_count = 0
```

## Bước 35: evaluate(model, loader, include_predictions=False) — phần 2/3

<!-- module
ID: m041
Depends: m004, m034, m036, m037, m038, m039, m040
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=54-82
Source: baseline_basic.ipynb#cell=14; lines=54-82
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

<a id="context-m041"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, include_predictions=False) — phần 2/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    for cpu_batch in loader:
        batch = move_batch(cpu_batch)
        hidden, emissions = model(batch)
        paths = model.decode(emissions, batch["mask"])
        predicted_spans = [bio_to_spans(path) for path in paths]
        candidates = [(i, a, b) for i, spans in enumerate(predicted_spans) for a, b in spans]
        span_tensor = torch.tensor(candidates, dtype=torch.long, device=DEVICE).reshape(-1, 3)
        logits = model.classify_spans(hidden, span_tensor, batch["mask"])
        guesses = logits.argmax(-1).tolist() if len(candidates) else []
        by_sentence = defaultdict(list)
        for (i, a, b), label in zip(candidates, guesses):
            by_sentence[i].append((a, b, LABELS[label]))
        gold_span_tensor = batch["gold_spans"]
        gold_logits = model.classify_spans(hidden, gold_span_tensor[:, :3], batch["mask"])
        if len(gold_span_tensor):
            polarity_gold.extend(gold_span_tensor[:, 3].tolist())
            polarity_predicted.extend(gold_logits.argmax(-1).tolist())
        for i, record in enumerate(batch["records"]):
            sid = record["sid"]
            gold_aspects.update((sid, a, b) for a, b, _ in record["spans"])
            predicted_aspects.update((sid, a, b) for a, b, _ in by_sentence[i])
            if all(p is not None for _, _, p in record["spans"]):
                joint_sentence_count += 1
                gold_joint.update((sid, a, b, p) for a, b, p in record["spans"])
                predicted_joint.update((sid, a, b, p) for a, b, p in by_sentence[i])
            if include_predictions:
                examples.append({"text": record["text"], "aspects": [
                    {"start": a, "end": b, "span": record["text"][a:b], "polarity": p}
                    for a, b, p in by_sentence[i]]})
```

## Bước 36: evaluate(model, loader, include_predictions=False) — phần 3/3

<!-- module
ID: m042
Depends: m004, m034, m036, m037, m038, m039, m041
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=83-87
Source: baseline_basic.ipynb#cell=14; lines=83-87
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

<a id="context-m042"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, include_predictions=False) — phần 3/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    scores = {"aspect_span": set_prf(gold_aspects, predicted_aspects),
              "polarity_gold_span": classification_metrics(polarity_gold, polarity_predicted),
              "joint_span_polarity": set_prf(gold_joint, predicted_joint) if joint_sentence_count else None,
              "joint_labeled_sentence_count": joint_sentence_count}
    return (scores, examples) if include_predictions else scores
```

## Bước 37: Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs

<!-- module
ID: m043
Depends: m004, m035
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=17; lines=1-2
Source: baseline_basic.ipynb#cell=16; lines=1-2
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Cấu hình kiến trúc mô hình](#context-m035)

<a id="context-m043"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Huấn luyện và lưu checkpoint — best_f1, optimizer, stale_epochs từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m035 — Cấu hình kiến trúc mô hình:
[Dán tóm tắt m035]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
best_f1, stale_epochs = -1.0, 0
```

## Bước 38: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m044
Depends: m004, m015, m023, m035, m036, m042, m043
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=17; lines=3-47
Source: baseline_basic.ipynb#cell=16; lines=3-47
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

<a id="context-m044"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
for epoch in range(1, EPOCHS + 1):
    model.train()
    totals, sentence_count = np.zeros(3), 0
    for cpu_batch in train_loader:
        batch = move_batch(cpu_batch)
        optimizer.zero_grad(set_to_none=True)
        hidden, emissions = model(batch)
        extraction_loss = model.extraction_loss(emissions, batch)
        spans = batch["gold_spans"]
        if len(spans):
            polarity_logits = model.classify_spans(hidden, spans[:, :3], batch["mask"])
            polarity_loss = nn.functional.cross_entropy(polarity_logits, spans[:, 3])
        else:
            polarity_loss = hidden.sum() * 0.0
        loss = extraction_loss + POLARITY_LOSS_WEIGHT * polarity_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss không hữu hạn; kiểm tra dữ liệu và cấu hình cố định.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        count = len(batch["records"])
        totals += np.array([loss.item(), extraction_loss.item(), polarity_loss.item()]) * count
        sentence_count += count
    losses = dict(zip(("loss", "extraction_loss", "polarity_loss"), (totals / sentence_count).tolist()))
    metrics = evaluate(model, val_loader)
    score = metrics["aspect_span"]["f1"]
    polarity_f1 = metrics["polarity_gold_span"]["macro_f1"]
    print(f"Epoch {epoch:02d}: loss={losses['loss']:.4f}; "
          f"validation aspect F1={score:.4f}; polarity macro-F1={polarity_f1:.4f}")
    if score > best_f1:  # Hòa điểm: giữ checkpoint sớm hơn, không nhìn test.
        best_f1, stale_epochs = score, 0
        torch.save({
            "model_state": model.state_dict(),
            "char_vocab": char_vocab,
            "word_vocab": word_vocab,
            "labels": list(LABELS),
            "tags": list(TAGS),
            "epoch": epoch,
        }, CHECKPOINT_PATH)
        print("Đã lưu checkpoint tốt nhất theo aspect span F1 trên validation.")
    else:
        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"Dừng sớm sau {PATIENCE} epoch không cải thiện.")
            break
```

## Bước 39: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m045
Depends: m004, m010, m022, m034, m042
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=19; lines=2-18
Source: baseline_basic.ipynb#cell=18; lines=2-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Chuẩn hóa và chia dữ liệu](#context-m010)
3. [make_loader(records, shuffle=False, batch_size)](#context-m022)
4. [Ghép giao diện class AspectModel](#context-m034)
5. [evaluate(model, loader, include_predictions=False) — phần 3/3](#context-m042)

<a id="context-m045"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
assert checkpoint["labels"] == list(LABELS) and checkpoint["tags"] == list(TAGS)
char_vocab, word_vocab = checkpoint["char_vocab"], checkpoint["word_vocab"]
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"char_dim": CHAR_EMBED_DIM, "word_dim": WORD_EMBED_DIM,
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}
best_model = AspectModel(inference_config, len(char_vocab), len(word_vocab)).to(DEVICE)
best_model.load_state_dict(checkpoint["model_state"])
best_model.eval()
# Tạo lại loader từ vocabulary đã tải; không phụ thuộc state của mô hình cũ.
final_val_metrics = evaluate(best_model, make_loader(val_records))
test_metrics, test_predictions = evaluate(best_model, make_loader(test_records), include_predictions=True)
print(f"Đã nạp checkpoint epoch {checkpoint['epoch']}; mọi metric ở thang 0–1.")
print("Validation:", json.dumps(final_val_metrics, ensure_ascii=False, indent=2))
print("Test:", json.dumps(test_metrics, ensure_ascii=False, indent=2))
if test_metrics["polarity_gold_span"]["count"] == 0:
    print("Test không có nhãn polarity: không tính polarity F1/accuracy hay joint F1 trên test.")
```

## Bước 40: Tải lại và đánh giá cuối — results

<!-- module
ID: m046
Depends: m004, m012, m045
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=19; lines=19-25
Source: baseline_basic.ipynb#cell=18; lines=19-25
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [Tải dữ liệu, nhóm câu và chia validation](#context-m012)
3. [Bảng ký tự và trạng thái tokenizer](#context-m045)

<a id="context-m046"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tải lại và đánh giá cuối — results từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m004 — Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…:
[Dán tóm tắt m004]

Context m012 — Tải dữ liệu, nhóm câu và chia validation:
[Dán tóm tắt m012]

Context m045 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m045]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
results = {"variant": VARIANT, "epoch": checkpoint["epoch"],
           "dataset_revision": DATASET_REVISION, "split_fingerprints": SPLIT_FINGERPRINTS,
           "validation": final_val_metrics, "test": test_metrics}
(ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "test_predictions.json").write_text(
    json.dumps(test_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print("Dự đoán test đầu tiên:", test_predictions[0])
```

## Bước 41: predict_text(text)

<!-- module
ID: m047
Depends: m004, m016, m021, m036, m037, m045
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=21; lines=1-14
Source: baseline_basic.ipynb#cell=20; lines=1-14
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Cấu hình cố định — ARTIFACT_DIR, BATCH_SIZE, CACHE_DIR…](#context-m004)
2. [encode_record(record)](#context-m016)
3. [collate_batch(items)](#context-m021)
4. [move_batch(batch)](#context-m036)
5. [bio_to_spans(tags)](#context-m037)
6. [Bảng ký tự và trạng thái tokenizer](#context-m045)

<a id="context-m047"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện predict_text(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def predict_text(text):
    if not isinstance(text, str) or not text:
        raise ValueError("Cần một câu không rỗng.")
    record = {"sid": hashlib.sha256(text.encode()).hexdigest(), "text": text, "spans": []}
    batch = move_batch(collate_batch([encode_record(record)]))
    best_model.eval()
    hidden, emissions = best_model(batch)
    spans = bio_to_spans(best_model.decode(emissions, batch["mask"])[0])
    candidates = torch.tensor([(0, a, b) for a, b in spans], device=DEVICE, dtype=torch.long).reshape(-1, 3)
    logits = best_model.classify_spans(hidden, candidates, batch["mask"])
    polarities = logits.argmax(-1).tolist() if spans else []
    return [{"start": a, "end": b, "span": text[a:b], "polarity": LABELS[p]}
            for (a, b), p in zip(spans, polarities)]
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
