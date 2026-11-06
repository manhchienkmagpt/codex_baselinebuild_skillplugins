# Prompt basic: FUNSD: word/char/layout/ROI, BiLSTM và CRF

<!-- prompt-guide
Format: 3
Variant: basic
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=6; lines=5-15
Source: baseline_basic.ipynb#cell=6; lines=5-15
-->

<a id="context-m007"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện open_rgb(value) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def open_rgb(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return ImageOps.exif_transpose(Image.open(io.BytesIO(value["bytes"]))).convert("RGB")
        value = value.get("path")
    if value is None:
        raise ValueError("Thiếu ảnh; cần PIL.Image, bytes ảnh hoặc đường dẫn thật.")
    with Image.open(value) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
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
Source: baseline_basic.ipynb#cell=5; lines=1-32
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…. Giữ ràng buộc: DATASET_ID=SOURCE['dataset']; DATASET_REVISION=SOURCE['revision']. Tự chọn USE_LOCAL_DATA, VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
SEED = 42
VARIANT = "basic"
ARTIFACT_DIR = Path("artifacts/document_funsd") / VARIANT
CACHE_DIR = ARTIFACT_DIR / "data_cache"
WEIGHTS_CACHE_DIR = ARTIFACT_DIR / "weights"
CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pt"
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(WEIGHTS_CACHE_DIR))  # Chỉ đặt nơi cache, không dùng torch.hub.load.
print(f"Thiết bị: {DEVICE}; biến thể: {VARIANT}; PyTorch {torch.__version__}; torchvision {torchvision.__version__}")


SOURCE = {'row': 4, 'task': 'FUNSD conditional entity recognition, không có gold K-V links', 'dataset': 'nielsr/funsd', 'revision': '7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5', 'url': 'https://huggingface.co/datasets/nielsr/funsd'}
DATASET_ID = SOURCE["dataset"]
DATASET_REVISION = SOURCE["revision"]
USE_LOCAL_DATA = False
LOCAL_DATA_DIR = Path("data/funsd_saved")
VAL_FRACTION = 0.20
IMAGE_SIZE = 256
MAX_WORD_CHARS = 32
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
HIDDEN_DIM = 64
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=16-24
Source: baseline_basic.ipynb#cell=8; lines=16-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)

<a id="context-m012"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện canonical_bio(tags) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def canonical_bio(tags):
    result, previous = [], "O"
    for index in tags:
        name = TAG_NAMES[index]
        if name.startswith("I-") and previous not in ("B-" + name[2:], name):
            name = "B-" + name[2:]
        result.append(TAG_NAMES.index(name))
        previous = name
    return result
```

## Bước 4: adapt(dataset, split)

<!-- module
ID: m013
Depends: m012
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=26-44
Source: baseline_basic.ipynb#cell=8; lines=26-44
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_bio(tags)](#context-m012)

<a id="context-m013"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện adapt(dataset, split) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m012 — canonical_bio(tags):
[Dán tóm tắt m012]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def adapt(dataset, split):
    required = {"id", "image", "words", "bboxes", "ner_tags"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Thiếu trường FUNSD: {required - set(dataset.column_names)}")
    dataset = dataset.cast_column("image", HFImage(decode=False))
    rows, repairs = [], 0
    for row in dataset:
        if not row["words"] or not (len(row["words"]) == len(row["bboxes"]) == len(row["ner_tags"])):
            raise ValueError("words/bboxes/ner_tags không cùng độ dài hoặc rỗng.")
        for box in row["bboxes"]:
            if len(box) != 4 or not all(0 <= value <= 1000 for value in box) or box[0] > box[2] or box[1] > box[3]:
                raise ValueError(f"BBox không hợp lệ: {box}")
        tags = canonical_bio(row["ner_tags"])
        repairs += sum(a != b for a, b in zip(tags, row["ner_tags"]))
        # ID HF chỉ có ý nghĩa trong split. Không sắp lại words theo bbox vì làm đổi chuỗi nhãn.
        rows.append({"id": f"{split}:{row['id']}", "image": row["image"],
                     "words": row["words"], "boxes": row["bboxes"], "tags": tags})
    print(f"{split}: {len(rows)} tài liệu; chuẩn hóa {repairs} I-tag không hợp lệ thành B cùng loại.")
    return rows
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
Source: baseline_basic.ipynb#cell=8; lines=46-54
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [adapt(dataset, split)](#context-m013)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: nielsr/funsd; revision: 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: SEED=42; VAL_FRACTION=0.2. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m013 — adapt(dataset, split):
[Dán tóm tắt m013]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
all_train = sorted(adapt(raw["train"], "train"), key=lambda row: row["id"])
test_records = adapt(raw["test"], "test")
order = list(range(len(all_train)))
random.Random(SEED).shuffle(order)
valid_ids = set(order[:max(1, math.ceil(VAL_FRACTION * len(order)))])
train_records = [row for i, row in enumerate(all_train) if i not in valid_ids]
val_records = [row for i, row in enumerate(all_train) if i in valid_ids]
if not train_records or not val_records:
    raise ValueError("Không đủ tài liệu để chia train/validation.")
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=56-57
Source: baseline_basic.ipynb#cell=8; lines=56-57
-->

<a id="context-m015"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện document_key(row) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def document_key(row):
    return fingerprint({"words": row["words"], "boxes": row["boxes"]})
```

## Bước 7: Đọc FUNSD và chia theo tài liệu

<!-- module
ID: m016
Depends: m014, m015
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=58-65
Source: baseline_basic.ipynb#cell=8; lines=58-65
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [document_key(row)](#context-m015)

<a id="context-m016"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Đọc FUNSD và chia theo tài liệu từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — document_key(row):
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
keys = [{document_key(r) for r in records} for records in (train_records, val_records, test_records)]
if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
    raise ValueError("Có tài liệu trùng nội dung giữa các split; cần xử lý trước huấn luyện.")
SPLIT_FINGERPRINTS = {name: fingerprint([{k: r[k] for k in ("id", "words", "boxes", "tags")} for r in records])
                      for name, records in (("train", train_records), ("validation", val_records), ("test", test_records))}
print("Số tài liệu train/validation/test:", len(train_records), len(val_records), len(test_records))
print("Fingerprint:", SPLIT_FINGERPRINTS)
print("Không có nhãn linking trong schema này: K-V linking F1 sẽ là null, không phải 0.")
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
Source: baseline_basic.ipynb#cell=10; lines=1-7
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [Chuẩn hóa và chia dữ liệu](#context-m014)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Tự chọn IMAGE_SIZE, MAX_WORD_CHARS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
word_vocab = {"<PAD>": 0, "<UNK>": 1}
word_vocab.update({word: i + 2 for i, word in enumerate(sorted({w.lower() for r in train_records for w in r["words"]}))})

TOKENIZER_STATE = {"word_vocab": word_vocab, "tag_names": TAG_NAMES}
PREPROCESS_CONFIG = {"image_size": IMAGE_SIZE, "max_word_chars": MAX_WORD_CHARS,
                     "word_order": "giữ thứ tự HF", "boxes_scale": 1000,
                     "image_norm": "RGB 0..1", "bio_policy": "I sai ngữ cảnh thành B cùng loại"}
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
Source: baseline_basic.ipynb#cell=10; lines=8-10
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung restore_tokenizer(state). Bổ sung char vocabulary train-only và lưu cùng word_vocab, TAG_NAMES để reload. PAD0 UNK1; TOKENIZER_STATE word_vocab,char_vocab,tag_names; MAX_WORD_CHARS32,IMAGE_SIZE384; không học vocabulary từ validation/test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def restore_tokenizer(state):
    global word_vocab, TAG_NAMES
    word_vocab, TAG_NAMES = state["word_vocab"], state["tag_names"]
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=14-15
Source: baseline_basic.ipynb#cell=10; lines=13-14
-->

<a id="context-m019"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện FormDataset.__init__(self, records) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self, records):
        self.records = records
```

## Bước 11: FormDataset.__len__(self)

<!-- module
ID: m020
Depends: m019
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=16-17
Source: baseline_basic.ipynb#cell=10; lines=15-16
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [FormDataset.__init__(self, records)](#context-m019)

<a id="context-m020"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện FormDataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):
        return len(self.records)
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
Source: baseline_basic.ipynb#cell=10; lines=17-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [open_rgb(value)](#context-m007)
2. [Bảng ký tự và trạng thái tokenizer](#context-m017)
3. [FormDataset.__init__(self, records)](#context-m019)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung FormDataset.__getitem__(self, index). Bổ sung RGB image tensor và chars vào FormDataset/collate_batch, giữ box/tag/word alignment. Giữ thứ tự đầu ra: image, record. Dùng lại IMAGE_SIZE từ context, không chọn hoặc khai báo lại giá trị. Viết method trong class FormDataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m019 — FormDataset.__init__(self, records):
[Dán tóm tắt m019]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __getitem__(self, index):
        return self.records[index]
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
Source: baseline_basic.ipynb#cell=10; lines=20-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m017)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung collate_batch(items). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường boxes, chars, ids, images, lengths, mask, records, tags, words. Dùng lại MAX_WORD_CHARS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def collate_batch(items):
    lengths=torch.tensor([len(record["words"]) for record in items])
    width=int(lengths.max())
    words=torch.zeros(len(items),width,dtype=torch.long)
    boxes=torch.zeros(len(items),width,4)
    tags=torch.zeros(len(items),width,dtype=torch.long)
    for i,record in enumerate(items):
        n=len(record["words"])
        words[i,:n]=torch.tensor([word_vocab.get(w.lower(),1) for w in record["words"]])
        boxes[i,:n]=torch.tensor(record["boxes"],dtype=torch.float32)/1000
        tags[i,:n]=torch.tensor(record["tags"])
    return {"words":words,"boxes":boxes,"tags":tags,"lengths":lengths,
            "mask":torch.arange(width)[None]<lengths[:,None],
            "ids":[r["id"] for r in items],"records":items}
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
Source: baseline_basic.ipynb#cell=6; lines=17-19
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def to_device(batch):
    return {key: value.to(DEVICE) if torch.is_tensor(value) and key not in ("lengths", "widths", "target_lengths") else value
            for key, value in batch.items()}
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
Source: baseline_basic.ipynb#cell=6; lines=21-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [collate_batch(items)](#context-m023)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m023 — collate_batch(items):
[Dán tóm tắt m023]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def make_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train, num_workers=0, collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
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
Source: baseline_basic.ipynb#cell=10; lines=35-38
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m009)
2. [Chuẩn hóa và chia dữ liệu](#context-m014)
3. [Bảng ký tự và trạng thái tokenizer](#context-m017)
4. [Ghép giao diện class FormDataset](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m022 — Ghép giao diện class FormDataset:
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_loader = make_loader(FormDataset(train_records), train=True)
val_loader = make_loader(FormDataset(val_records))
test_loader = make_loader(FormDataset(test_records))
print("Vocabulary train:", len(word_vocab), "từ;", 0, "đặc trưng ký tự; không đọc ảnh vì model basic không dùng.")
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
Source: baseline_basic.ipynb#cell=12; lines=2-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [Bảng ký tự và trạng thái tokenizer](#context-m017)
3. [Ghép giao diện class EntityCRF](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung FormModel.__init__(self, cfg). Thay constructor FormModel bằng word/char/layout/visual CNN, BiLSTM và EntityCRF. Tự chọn DROPOUT, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa n_words, n_chars, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m017 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m017]

Context m029 — Ghép giao diện class EntityCRF:
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self, cfg):
        super().__init__()
        self.word_embedding = nn.Embedding(cfg["n_words"], 64, padding_idx=0)
        self.layout = nn.Sequential(nn.Linear(6,16),nn.ReLU())
        self.encoder = nn.RNN(80,cfg["hidden_dim"],batch_first=True)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["hidden_dim"],len(TAG_NAMES))
        self.register_buffer("tag_weights", torch.ones(len(TAG_NAMES)))
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
Source: baseline_basic.ipynb#cell=12; lines=10-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [FormModel.__init__(self, cfg)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung FormModel.forward(self, batch). Viết phần đầu forward trích char-CNN masked pooling và layout từ tọa độ cùng kích thước box. Viết phần ROI Align theo boxes và ghép visual features về đúng word của từng tài liệu. Hoàn tất forward: ghép word,char,layout,ROI, chạy packed BiLSTM và tag head, nối loss/decode CRF. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def forward(self, batch):
        box = batch["boxes"]
        layout = self.layout(torch.cat([box,box[:,:,2:]-box[:,:,:2]],-1))
        x = self.dropout(torch.cat([self.word_embedding(batch["words"]),layout],-1))
        packed = pack_padded_sequence(x,batch["lengths"],batch_first=True,enforce_sorted=False)
        hidden,_ = self.encoder(packed)
        hidden,_ = pad_packed_sequence(hidden,batch_first=True,total_length=batch["words"].size(1))
        return self.head(self.dropout(hidden))
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
Source: baseline_basic.ipynb#cell=12; lines=18-19
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [FormModel.__init__(self, cfg)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung FormModel.loss(self, emissions, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def loss(self, emissions, batch):
        return nn.functional.cross_entropy(emissions[batch["mask"]],batch["tags"][batch["mask"]],weight=self.tag_weights)
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
Source: baseline_basic.ipynb#cell=12; lines=20-21
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [FormModel.__init__(self, cfg)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung FormModel.decode(self, emissions, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class FormModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def decode(self, emissions, batch):
        return [canonical_bio(ids[:length]) for ids,length in zip(emissions.argmax(-1).tolist(),batch["lengths"].tolist())]
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
Source: baseline_basic.ipynb#cell=12; lines=22-27
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class FormModel](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung build_model(cfg, initialize_pretrained=True). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_model(cfg, initialize_pretrained=True):
    model = FormModel(cfg)
    counts = torch.bincount(torch.tensor([t for r in train_records for t in r["tags"]]),minlength=len(TAG_NAMES)).float()
    weights = counts.clamp_min(1).rsqrt()
    model.tag_weights.copy_(weights / weights.mean())
    return model
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=1-10
Source: baseline_basic.ipynb#cell=15; lines=1-10
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…](#context-m011)
2. [canonical_bio(tags)](#context-m012)

<a id="context-m038"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện entity_spans(tags) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m011 — Đọc FUNSD và chia theo tài liệu — TAG_NAMES, actual_tags, expected_tags…:
[Dán tóm tắt m011]

Context m012 — canonical_bio(tags):
[Dán tóm tắt m012]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def entity_spans(tags):
    tags = canonical_bio(tags)
    output, start, kind = [], None, None
    for i, tag_id in enumerate(tags + [0]):
        tag = TAG_NAMES[tag_id]
        if start is not None and (tag == "O" or tag.startswith("B-") or tag[2:] != kind):
            output.append((start,i,kind)); start,kind = None,None
        if tag.startswith("B-"):
            start,kind = i,tag[2:]
    return output
```

## Bước 30: compute_loss(model, batch)

<!-- module
ID: m039
Depends: m023, m035
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=12-13
Source: baseline_basic.ipynb#cell=15; lines=12-13
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m023)
2. [Ghép giao diện class FormModel](#context-m035)

<a id="context-m039"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện compute_loss(model, batch) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m023 — collate_batch(items):
[Dán tóm tắt m023]

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def compute_loss(model,batch):
    return model.loss(model(batch),batch)
```

## Bước 31: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m040
Depends: m008, m035, m038
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=15-37
Source: baseline_basic.ipynb#cell=15; lines=15-37
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Ghép giao diện class FormModel](#context-m035)
3. [entity_spans(tags)](#context-m038)

<a id="context-m040"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]

Context m038 — entity_spans(tags):
[Dán tóm tắt m038]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def evaluate(model,loader,return_predictions=False):
    model.eval()
    gold,predicted=set(),set()
    examples=[]
    for cpu_batch in loader:
        batch=to_device(cpu_batch)
        paths=model.decode(model(batch),batch)
        for sid,record,path in zip(batch["ids"],batch["records"],paths):
            gold.update((sid,a,b,k) for a,b,k in entity_spans(record["tags"]))
            guessed=entity_spans(path)
            predicted.update((sid,a,b,k) for a,b,k in guessed)
            if return_predictions:
                examples.append({"id":sid,"entities":[{"start_word":a,"end_word":b,"type":k,
                    "text":" ".join(record["words"][a:b]),"word_boxes":record["boxes"][a:b]} for a,b,k in guessed],
                    "key_value_links":None,"linking_status":"Nguồn HF không có gold linking"})
    tp=len(gold & predicted)
    precision=tp/len(predicted) if predicted else 0.0
    recall=tp/len(gold) if gold else 0.0
    score=2*tp/(len(gold)+len(predicted)) if gold or predicted else 0.0
    metrics={"entity_f1":score,"precision":precision,"recall":recall,"tp":tp,
             "gold_entities":len(gold),"predicted_entities":len(predicted),"key_value_linking_f1":None}
    return (metrics,examples) if return_predictions else metrics
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
Source: baseline_basic.ipynb#cell=17; lines=3-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class FormModel](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_optimizer(model). Khởi tạo optimizer AdamW cho tham số có thể học. Tách nhóm backbone và task head theo cấu trúc model. Tự chọn BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m035 — Ghép giao diện class FormModel:
[Dán tóm tắt m035]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def make_optimizer(model):
    visual, remaining = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (visual if name.startswith("backbone.") else remaining).append(parameter)
    groups = [{"params": remaining, "lr": LEARNING_RATE}]
    if visual:
        groups.append({"params": visual, "lr": BACKBONE_LEARNING_RATE})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
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
Source: baseline_basic.ipynb#cell=18; lines=1-6
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [FormModel.__init__(self, cfg)](#context-m031)
3. [build_model(cfg, initialize_pretrained=True)](#context-m036)
4. [make_optimizer(model)](#context-m041)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Huấn luyện và checkpoint — best_score, model, optimizer…. Giữ ràng buộc: DIRECTION='max'. Tự chọn EPOCHS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m031 — FormModel.__init__(self, cfg):
[Dán tóm tắt m031]

Context m036 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m036]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
model = build_model(MODEL_CONFIG, initialize_pretrained=True).to(DEVICE)
print(f"Tham số học: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
optimizer = make_optimizer(model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_score = math.inf if DIRECTION == "min" else -math.inf
stale_epochs = 0
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
Source: baseline_basic.ipynb#cell=18; lines=7-44
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
Sửa hoặc bổ sung Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='entity_f1'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, DIRECTION, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, seen = 0.0, 0
    for cpu_batch in tqdm(train_loader, desc="Huấn luyện", leave=False):
        batch = to_device(cpu_batch)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss không hữu hạn; kiểm tra nhãn, mask và cấu hình cố định.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        count = len(batch["ids"])
        total += loss.item() * count
        seen += count
    if not seen:
        raise ValueError("Tập huấn luyện rỗng.")
    loss = total / seen
    metrics = evaluate(model, val_loader)
    score = float(metrics[PRIMARY_METRIC])
    if not math.isfinite(score):
        raise ValueError("Metric validation không hữu hạn.")
    scheduler.step()
    print(f"Epoch {epoch:02d}: loss={loss:.4f}; validation={metrics}")
    improved = score < best_score if DIRECTION == "min" else score > best_score
    if improved:
        best_score, stale_epochs = score, 0
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "tokenizer": TOKENIZER_STATE,
        }, CHECKPOINT_PATH)
        print("Đã lưu checkpoint tốt nhất theo primary metric validation.")
    else:
        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"Dừng sớm sau {PATIENCE} epoch không cải thiện.")
            break
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
Source: baseline_basic.ipynb#cell=20; lines=1-18
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
Sửa hoặc bổ sung Reload và đánh giá test — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
assert checkpoint["tokenizer"] == TOKENIZER_STATE
restore_tokenizer(checkpoint["tokenizer"])
# Không tải backbone lần nữa: state_dict chứa đầy đủ trọng số visual và head.
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"n_words":len(word_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
best_model = build_model(inference_config, initialize_pretrained=False).to(DEVICE)
best_model.load_state_dict(checkpoint["model_state"])
best_model.eval()
validation_metrics = evaluate(best_model, val_loader)
test_metrics, predictions = evaluate(best_model, test_loader, return_predictions=True)
results = {"epoch": checkpoint["epoch"], "source": SOURCE, "split_fingerprints": SPLIT_FINGERPRINTS,
           "validation": validation_metrics, "held_out_test": test_metrics}
(ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Đã nạp epoch {checkpoint['epoch']}.")
print("Validation:", validation_metrics)
print("Held-out:", test_metrics)
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=23; lines=1-14
Source: baseline_basic.ipynb#cell=22; lines=1-14
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Ghép giao diện class FormDataset](#context-m022)
3. [collate_batch(items)](#context-m023)
4. [entity_spans(tags)](#context-m038)
5. [Reload và đánh giá test — best_model, checkpoint, inference_config…](#context-m044)

<a id="context-m046"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện parse_form(image, words, bboxes) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def parse_form(image,words,bboxes):
    if not words or len(words)!=len(bboxes):
        raise ValueError("Cần words và bboxes chuẩn hóa 0..1000, cùng độ dài; ảnh đơn lẻ chưa đủ cho pipeline này.")
    for box in bboxes:
        if len(box)!=4 or not all(0<=v<=1000 for v in box) or box[0]>box[2] or box[1]>box[3]:
            raise ValueError("BBox sai định dạng/miền giá trị.")
    record={"id":"new","image":image,"words":words,"boxes":bboxes,"tags":[0]*len(words)}
    dataset=FormDataset([record])
    batch=to_device(collate_batch([dataset[0]]))
    best_model.eval()
    tags=best_model.decode(best_model(batch),batch)[0]
    return [{"start_word":a,"end_word":b,"type":kind,"text":" ".join(words[a:b])}
            for a,b,kind in entity_spans(tags)]
```

## Bước 38: Phân tích tài liệu mới — sample

<!-- module
ID: m047
Depends: m014, m046
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=23; lines=16-17
Source: baseline_basic.ipynb#cell=22; lines=16-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)
2. [parse_form(image, words, bboxes)](#context-m046)

<a id="context-m047"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Phân tích tài liệu mới — sample từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m046 — parse_form(image, words, bboxes):
[Dán tóm tắt m046]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
sample=test_records[0]
print(parse_form(sample["image"],sample["words"],sample["boxes"]))
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
