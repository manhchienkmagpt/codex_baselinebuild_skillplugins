# Prompt basic: IAM-line: ResNet18, BiLSTM và CTC

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
Source: baseline_basic.ipynb#cell=5; lines=1-34
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Dữ liệu và split — raw. Giữ ràng buộc: DATASET_ID=SOURCE['dataset']; DATASET_REVISION=SOURCE['revision']. Tự chọn USE_LOCAL_DATA, VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
SEED = 42
VARIANT = "basic"
ARTIFACT_DIR = Path("artifacts/ocr_iam_line") / VARIANT
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


SOURCE = {'row': 1, 'task': 'OCR dòng chữ viết tay tiếng Anh', 'dataset': 'Teklia/IAM-line', 'revision': 'fbdad97500ce54635c0d1ba306bf535cb40656cf', 'url': 'https://huggingface.co/datasets/Teklia/IAM-line'}
DATASET_ID = SOURCE["dataset"]
DATASET_REVISION = SOURCE["revision"]
USE_LOCAL_DATA = False
LOCAL_DATA_DIR = Path("data/iam_line_saved")
IMAGE_HEIGHT = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = BATCH_SIZE
HIDDEN_DIM = 128
DROPOUT = 0.0
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.001
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=12-13
Source: baseline_basic.ipynb#cell=8; lines=12-13
-->

<a id="context-m012"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện normalize_text(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())
```

## Bước 4: adapt_split(dataset, split)

<!-- module
ID: m013
Depends: m012
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=15-25
Source: baseline_basic.ipynb#cell=8; lines=15-25
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_text(text)](#context-m012)

<a id="context-m013"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện adapt_split(dataset, split) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m012 — normalize_text(text):
[Dán tóm tắt m012]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def adapt_split(dataset, split):
    if not {"image", "text"}.issubset(dataset.column_names):
        raise ValueError("Schema IAM cần image và text.")
    dataset = dataset.cast_column("image", HFImage(decode=False))
    rows = []
    for index, row in enumerate(dataset):
        text = normalize_text(row["text"])
        if not text:
            raise ValueError(f"Transcription rỗng tại {split}:{index}; kiểm tra nguồn.")
        rows.append({"id": f"{split}:{index}", "image": row["image"], "text": text})
    return rows
```

## Bước 5: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m014
Depends: m011, m013
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=27-35
Source: baseline_basic.ipynb#cell=8; lines=27-35
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và split — raw](#context-m011)
2. [adapt_split(dataset, split)](#context-m013)

<a id="context-m014"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m011 — Dữ liệu và split — raw:
[Dán tóm tắt m011]

Context m013 — adapt_split(dataset, split):
[Dán tóm tắt m013]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_records = adapt_split(raw["train"], "train")
val_records = adapt_split(raw["validation"], "validation")
test_records = adapt_split(raw["test"], "test")
SPLIT_FINGERPRINTS = {name: fingerprint([(r["id"], r["text"]) for r in records])
                      for name, records in (("train", train_records), ("validation", val_records), ("test", test_records))}
print("Số dòng theo split:", len(train_records), len(val_records), len(test_records))
print("Fingerprint:", SPLIT_FINGERPRINTS)
print("Ví dụ nhãn train:", train_records[0]["text"])
print("Giữ split có sẵn; HF không có writer_id nên chưa chứng minh được writer-disjoint.")
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
Source: baseline_basic.ipynb#cell=10; lines=1-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m014)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Tự chọn IMAGE_HEIGHT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
BLANK_ID, UNK_ID = 0, 1
characters = sorted({ch for record in train_records for ch in record["text"]})
char_vocab = {"<BLANK>": BLANK_ID, "<UNK>": UNK_ID}
char_vocab.update({ch: index + 2 for index, ch in enumerate(characters)})
id_to_char = {index: ch for ch, index in char_vocab.items()}
TOKENIZER_STATE = {"char_vocab": char_vocab, "blank_id": BLANK_ID, "unk_id": UNK_ID}
PREPROCESS_CONFIG = {"image_height": IMAGE_HEIGHT, "width_multiple": 4,
                     "normalization": "NFC, gộp khoảng trắng; giữ hoa/thường và dấu câu",
                     "image_normalization": "grayscale 0..1; white right padding required by variable-width CTC", "decoding": "CTC greedy"}
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=11-14
Source: baseline_basic.ipynb#cell=10; lines=11-14
-->

<a id="context-m016"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện restore_tokenizer(state) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def restore_tokenizer(state):
    global char_vocab, id_to_char
    char_vocab = state["char_vocab"]
    id_to_char = {index: ch for ch, index in char_vocab.items()}
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
Source: baseline_basic.ipynb#cell=10; lines=16-22
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [open_rgb(value)](#context-m007)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung image_tensor(value). Nâng image_tensor sang RGB chiều cao theo cấu hình giữ tỷ lệ, chuẩn ImageNet; không cắt chiều rộng làm mất chữ. Dùng lại IMAGE_HEIGHT từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def image_tensor(value):
    image = open_rgb(value)
    width = max(32, 4 * math.ceil(image.width * IMAGE_HEIGHT / image.height / 4))
    # Giữ trọn dòng và tỷ lệ gần nguyên gốc, chỉ làm tròn width lên bội số 4.
    image = image.resize((width, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image.convert("L"))
    return tensor
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=26-27
Source: baseline_basic.ipynb#cell=10; lines=25-26
-->

<a id="context-m018"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện OCRDataset.__init__(self, records) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self, records):
        self.records = records
```

## Bước 10: OCRDataset.__len__(self)

<!-- module
ID: m019
Depends: m018
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=28-29
Source: baseline_basic.ipynb#cell=10; lines=27-28
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [OCRDataset.__init__(self, records)](#context-m018)

<a id="context-m019"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện OCRDataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):
        return len(self.records)
```

## Bước 11: OCRDataset.__getitem__(self, index)

<!-- module
ID: m020
Depends: m015, m017, m018
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=30-33
Source: baseline_basic.ipynb#cell=10; lines=29-32
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)
2. [image_tensor(value)](#context-m017)
3. [OCRDataset.__init__(self, records)](#context-m018)

<a id="context-m020"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện OCRDataset.__getitem__(self, index) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m018 — OCRDataset.__init__(self, records):
[Dán tóm tắt m018]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __getitem__(self, index):
        record = self.records[index]
        target = torch.tensor([char_vocab.get(ch, UNK_ID) for ch in record["text"]], dtype=torch.long)
        return image_tensor(record["image"]), target, record
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
Source: baseline_basic.ipynb#cell=10; lines=34-47
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung collate_batch(items). Cập nhật padding nền trắng đã chuẩn hóa và kiểm tra số bước CTC kể cả ký tự lặp. Kết quả giữ các trường ids, images, target_lengths, targets, texts, widths. Dùng lại IMAGE_HEIGHT từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def collate_batch(items):
    widths = torch.tensor([image.shape[-1] for image, _, _ in items], dtype=torch.long)
    images = torch.ones(len(items), 1, IMAGE_HEIGHT, int(widths.max()))
    targets, lengths = [], []
    for i, (image, target, record) in enumerate(items):
        images[i, :, :, :image.shape[-1]] = image
        required = len(target) + int((target[1:] == target[:-1]).sum())
        if image.shape[-1] // 4 < required:
            raise ValueError(f"Dòng {record['id']} không đủ bước CTC. Tăng IMAGE_HEIGHT thủ công rồi Run All.")
        targets.append(target)
        lengths.append(len(target))
    return {"images": images, "widths": widths, "targets": torch.cat(targets),
            "target_lengths": torch.tensor(lengths), "ids": [r["id"] for _, _, r in items],
            "texts": [r["text"] for _, _, r in items]}
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
Source: baseline_basic.ipynb#cell=6; lines=17-19
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
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
Source: baseline_basic.ipynb#cell=6; lines=21-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Giữ ràng buộc: SEED=42. Tự chọn BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: EVAL_BATCH_SIZE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=50-56
Source: baseline_basic.ipynb#cell=10; lines=49-55
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m009)
2. [Chuẩn hóa và chia dữ liệu](#context-m014)
3. [Bảng ký tự và trạng thái tokenizer](#context-m015)
4. [Ghép giao diện class OCRDataset](#context-m021)

<a id="context-m023"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m014 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m014]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]

Context m021 — Ghép giao diện class OCRDataset:
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_loader = make_loader(OCRDataset(train_records), train=True)
val_loader = make_loader(OCRDataset(val_records))
test_loader = make_loader(OCRDataset(test_records))
print("Số ký tự train (kể cả special):", len(char_vocab))
for name, records in (("validation", val_records), ("test", test_records)):
    unseen = sorted({ch for r in records for ch in r["text"] if ch not in char_vocab})
    print(f"Ký tự chưa gặp ở {name}: {unseen}; giữ gold nguyên văn, không bỏ ký tự khi chấm.")
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
Source: baseline_basic.ipynb#cell=12; lines=2-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung OCRModel.__init__(self, cfg, initialize_pretrained=True). Thay CNN scratch bằng ResNet18 ImageNet V1, chỉnh stride để chiều ngang giảm4 và chiều cao giảm32. OCRModel(cfg,initialize_pretrained=True); conv1(2,1), maxpool(2,2), layer2(2,1),layer3(2,2),layer4(2,1), cả conv1/downsample; tải sai weights phải lỗi, fc=Identity. Bổ sung BiLSTM số lớp được chọn và CTC head vào constructor OCRModel, giữ cfg và build_model. Tự chọn DROPOUT, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa hidden_dim, dropout, num_classes; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class OCRModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.sequence = nn.RNN(128, cfg["hidden_dim"], nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(cfg["hidden_dim"], cfg["num_classes"])
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
Source: baseline_basic.ipynb#cell=12; lines=10-16
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [OCRModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung OCRModel.forward(self, images, widths). Thay forward để lấy feature map ResNet, trung bình chiều cao và chạy packed BiLSTM theo widths//4. Giữ thứ tự đầu ra: kết quả 1, lengths. Viết method trong class OCRModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def forward(self, images, widths):
        x = self.visual(images).mean(dim=2).transpose(1, 2)
        lengths = widths // 4
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.sequence(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=x.size(1))
        return self.head(hidden).log_softmax(-1), lengths
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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=12; lines=35-36
Source: baseline_basic.ipynb#cell=12; lines=18-19
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class OCRModel](#context-m027)

<a id="context-m028"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện build_model(cfg, initialize_pretrained=True) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_model(cfg, initialize_pretrained=True):
    return OCRModel(cfg, initialize_pretrained)
```

## Bước 21: edit_distance(a, b)

<!-- module
ID: m030
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=1-8
Source: baseline_basic.ipynb#cell=15; lines=1-8
-->

<a id="context-m030"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện edit_distance(a, b) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]
```

## Bước 22: ctc_decode(path)

<!-- module
ID: m031
Depends: m012, m015
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=10-17
Source: baseline_basic.ipynb#cell=15; lines=10-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_text(text)](#context-m012)
2. [Bảng ký tự và trạng thái tokenizer](#context-m015)

<a id="context-m031"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện ctc_decode(path) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m012 — normalize_text(text):
[Dán tóm tắt m012]

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def ctc_decode(path):
    output, previous = [], None
    for index in path:
        index = int(index)
        if index != BLANK_ID and index != previous:
            output.append(id_to_char[index] if index != UNK_ID else "�")
        previous = index
    return normalize_text("".join(output))
```

## Bước 23: CER/WER và loss — ctc_loss

<!-- module
ID: m032
Depends: m015
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=19-19
Source: baseline_basic.ipynb#cell=15; lines=19-19
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m015)

<a id="context-m032"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện CER/WER và loss — ctc_loss từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m015 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
ctc_loss = nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=False)
```

## Bước 24: compute_loss(model, batch)

<!-- module
ID: m033
Depends: m022, m027, m032
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=20-24
Source: baseline_basic.ipynb#cell=15; lines=20-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [Ghép giao diện class OCRModel](#context-m027)
3. [CER/WER và loss — ctc_loss](#context-m032)

<a id="context-m033"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện compute_loss(model, batch) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]

Context m032 — CER/WER và loss — ctc_loss:
[Dán tóm tắt m032]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def compute_loss(model, batch):
    log_probs, lengths = model(batch["images"], batch["widths"])
    if int(lengths.max()) > log_probs.size(1):
        raise ValueError("Độ dài CTC không khớp feature map.")
    return ctc_loss(log_probs.transpose(0, 1), batch["targets"], lengths, batch["target_lengths"])
```

## Bước 25: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m034
Depends: m008, m027, m030, m031
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=26-48
Source: baseline_basic.ipynb#cell=15; lines=26-48
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Ghép giao diện class OCRModel](#context-m027)
3. [edit_distance(a, b)](#context-m030)
4. [ctc_decode(path)](#context-m031)

<a id="context-m034"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]

Context m030 — edit_distance(a, b):
[Dán tóm tắt m030]

Context m031 — ctc_decode(path):
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def evaluate(model, loader, return_predictions=False):
    model.eval()
    char_errors = word_errors = char_total = word_total = 0
    predictions = []
    for cpu_batch in loader:
        batch = to_device(cpu_batch)
        log_probs, lengths = model(batch["images"], batch["widths"])
        paths = log_probs.argmax(-1).cpu().tolist()
        for sid, gold, path, length in zip(batch["ids"], batch["texts"], paths, lengths.tolist()):
            prediction = ctc_decode(path[:length])
            char_errors += edit_distance(gold, prediction)
            word_errors += edit_distance(gold.split(), prediction.split())
            char_total += len(gold)
            word_total += len(gold.split())
            if return_predictions:
                predictions.append({"id": sid, "reference": gold, "prediction": prediction})
    if not char_total or not word_total:
        raise ValueError("Không có gold để tính CER/WER.")
    metrics = {"cer": char_errors / char_total, "wer": word_errors / word_total,
               "character_errors": char_errors, "reference_characters": char_total,
               "word_errors": word_errors, "reference_words": word_total}
    return (metrics, predictions) if return_predictions else metrics
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
Source: baseline_basic.ipynb#cell=17; lines=3-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class OCRModel](#context-m027)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_optimizer(model). Khởi tạo optimizer AdamW cho tham số có thể học. Tách nhóm backbone và task head theo cấu trúc model. Tự chọn BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m027 — Ghép giao diện class OCRModel:
[Dán tóm tắt m027]
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
Source: baseline_basic.ipynb#cell=18; lines=1-6
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [OCRModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
3. [build_model(cfg, initialize_pretrained=True)](#context-m028)
4. [make_optimizer(model)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Huấn luyện và checkpoint — best_score, model, optimizer…. Giữ ràng buộc: DIRECTION='min'. Tự chọn EPOCHS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — OCRModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m028 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m028]

Context m035 — make_optimizer(model):
[Dán tóm tắt m035]
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
Source: baseline_basic.ipynb#cell=18; lines=7-44
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
Sửa hoặc bổ sung Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='cer'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, DIRECTION, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=20; lines=1-18
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE
Source: baseline_basic.ipynb#cell=20; lines=1-18
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

<a id="context-m038"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tải lại và đánh giá test — best_model, checkpoint, inference_config… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

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

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
assert checkpoint["tokenizer"] == TOKENIZER_STATE
restore_tokenizer(checkpoint["tokenizer"])
# Không tải backbone lần nữa: state_dict chứa đầy đủ trọng số visual và head.
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "num_classes": len(char_vocab)}
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=22; lines=1-7
Config uses: DEVICE
Source: baseline_basic.ipynb#cell=22; lines=1-7
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [image_tensor(value)](#context-m017)
3. [ctc_decode(path)](#context-m031)
4. [Tải lại và đánh giá test — best_model, checkpoint, inference_config…](#context-m038)

<a id="context-m040"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện recognize_line(image) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m031 — ctc_decode(path):
[Dán tóm tắt m031]

Context m038 — Tải lại và đánh giá test — best_model, checkpoint, inference_config…:
[Dán tóm tắt m038]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def recognize_line(image):
    tensor = image_tensor(image)
    width = torch.tensor([tensor.shape[-1]])
    best_model.eval()
    log_probs, lengths = best_model(tensor.unsqueeze(0).to(DEVICE), width)
    return ctc_decode(log_probs[0, :int(lengths[0])].argmax(-1).tolist())
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

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

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
