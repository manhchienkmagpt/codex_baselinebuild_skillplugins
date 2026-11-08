# Prompt basic: OCR IAM-line — bản tối ưu

<!-- prompt-guide
Format: 4
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
Constraint workflow: extract-before-edit
Reference workflow: report-first
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Lượt phân tích dùng Code cần sửa, Code liên quan, Nguồn tham khảo và context. Lượt sửa chỉ cần Code cần sửa, kết quả phân tích và context; gửi thêm nguồn khi còn điều chưa rõ. Module mới dùng ý tưởng, nguồn tham khảo và context trực tiếp. Tham số trong nguồn là gợi ý để chọn phù hợp. Chỉ tóm tắt ở bước có liên kết dùng tiếp; không giới hạn cứng số dòng/token của bản tóm tắt. Nếu cả phiên không đủ chỗ, chia theo thành phần hoàn chỉnh.

## Bước 1: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh

<!-- module
ID: s01
Group: preprocessing
Kind: code
Depends: none
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=8; lines=12-13
Source: baseline_basic.ipynb#cell=10; lines=2-5
Source: baseline_basic.ipynb#cell=10; lines=11-22
Source: baseline_basic.ipynb#cell=10; lines=52-55
Related source: baseline_basic.ipynb#cell=5; lines=27-27
Related source: baseline_basic.ipynb#cell=6; lines=5-15
Related source: baseline_basic.ipynb#cell=8; lines=15-29
Related source: baseline_basic.ipynb#cell=10; lines=1-1
Related source: baseline_basic.ipynb#cell=10; lines=6-6
Related source: baseline_basic.ipynb#cell=13; lines=1-1
Target: baseline.ipynb#cell=8; lines=12-13
Target: baseline.ipynb#cell=10; lines=2-5
Target: baseline.ipynb#cell=10; lines=11-23
Target: baseline.ipynb#cell=10; lines=53-56
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=1-1
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=22-24
Reference code: baseline.ipynb#cell=5; lines=38-39
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn.
```

Trích báo cáo:

```markdown
2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học.
```

Trích báo cáo:

```markdown
3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference.
```

Trích báo cáo:

```markdown
4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
```

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
SOURCE = {'row': 1, 'task': 'OCR dòng chữ viết tay tiếng Anh', 'dataset': 'Teklia/IAM-line', 'revision': 'fbdad97500ce54635c0d1ba306bf535cb40656cf', 'url': 'https://huggingface.co/datasets/Teklia/IAM-line'}
DATASET_ID = SOURCE["dataset"]
DATASET_REVISION = SOURCE["revision"]
```

```python
PRIMARY_METRIC = "cer"
DIRECTION = "min"
```

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt trích ràng buộc

<a id="constraints-s01"></a>

Gửi cùng [nguồn tham khảo](#references-s01), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s01) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())
```

```python
characters = sorted({ch for record in train_records for ch in record["text"]})
char_vocab = {"<BLANK>": BLANK_ID, "<UNK>": UNK_ID}
char_vocab.update({ch: index + 2 for index, ch in enumerate(characters)})
id_to_char = {index: ch for ch, index in char_vocab.items()}
```

```python
def restore_tokenizer(state):
    global char_vocab, id_to_char
    char_vocab = state["char_vocab"]
    id_to_char = {index: ch for ch, index in char_vocab.items()}

def image_tensor(value):
    image = open_rgb(value)
    width = max(32, 4 * math.ceil(image.width * IMAGE_HEIGHT / image.height / 4))
    # Giữ trọn dòng và tỷ lệ gần nguyên gốc, chỉ làm tròn width lên bội số 4.
    image = image.resize((width, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image.convert("L"))
    return tensor
```

```python
print("Số ký tự train (kể cả special):", len(char_vocab))
for name, records in (("validation", val_records), ("test", test_records)):
    unseen = sorted({ch for r in records for ch in r["text"] if ch not in char_vocab})
    print(f"Ký tự chưa gặp ở {name}: {unseen}; giữ gold nguyên văn, không bỏ ký tự khi chấm.")
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
IMAGE_HEIGHT = 64
```

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

train_records = adapt_split(raw["train"], "train")
val_records = adapt_split(raw["validation"], "validation")
test_records = adapt_split(raw["test"], "test")
```

```python
BLANK_ID, UNK_ID = 0, 1
```

```python
TOKENIZER_STATE = {"char_vocab": char_vocab, "blank_id": BLANK_ID, "unk_id": UNK_ID}
```

```python
MODEL_CONFIG = {"hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "num_classes": len(char_vocab)}
```

Đối chiếu thêm code hiện tại của: OCRDataset; ctc_decode(path); cell 20, dòng 3–3; cell 20, dòng 6–6; recognize_line(image). Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, target_lengths, targets, texts, widths}; OCRModel: __init__(cfg, initialize_pretrained=True); forward(images, widths).

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 2: Tiền xử lí dữ liệu — Dataset và batch — phần việc 1

<!-- module
ID: s02
Group: preprocessing
Kind: code
Depends: s01
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=6; lines=21-24
Source: baseline_basic.ipynb#cell=10; lines=24-32
Related source: baseline_basic.ipynb#cell=5; lines=1-1
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=28-29
Related source: baseline_basic.ipynb#cell=10; lines=1-3
Related source: baseline_basic.ipynb#cell=10; lines=16-22
Related source: baseline_basic.ipynb#cell=10; lines=34-47
Related source: baseline_basic.ipynb#cell=10; lines=50-50
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=25-52
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)

### Nguồn tham khảo

<a id="references-s02"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn.
```

Trích báo cáo:

```markdown
2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học.
```

Trích báo cáo:

```markdown
3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference.
```

Trích báo cáo:

```markdown
4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt trích ràng buộc

<a id="constraints-s02"></a>

Gửi cùng [nguồn tham khảo](#references-s02), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s02) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def make_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train, num_workers=0, collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

```python
class OCRDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        target = torch.tensor([char_vocab.get(ch, UNK_ID) for ch in record["text"]], dtype=torch.long)
        return image_tensor(record["image"]), target, record
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
SEED = 42
```

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
BATCH_SIZE = 16
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
BLANK_ID, UNK_ID = 0, 1
characters = sorted({ch for record in train_records for ch in record["text"]})
char_vocab = {"<BLANK>": BLANK_ID, "<UNK>": UNK_ID}
```

```python
def image_tensor(value):
    image = open_rgb(value)
    width = max(32, 4 * math.ceil(image.width * IMAGE_HEIGHT / image.height / 4))
    # Giữ trọn dòng và tỷ lệ gần nguyên gốc, chỉ làm tròn width lên bội số 4.
    image = image.resize((width, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image.convert("L"))
    return tensor
```

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

```python
val_loader = make_loader(OCRDataset(val_records))
```

Đối chiếu thêm code hiện tại của: cell 10, dòng 49–49; cell 10, dòng 51–51. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, target_lengths, targets, texts, widths}; OCRModel: __init__(cfg, initialize_pretrained=True); forward(images, widths).

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 3: Tiền xử lí dữ liệu — Dataset và batch — phần việc 2

<!-- module
ID: s03
Group: preprocessing
Kind: code
Depends: s02
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=34-51
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=27-27
Related source: baseline_basic.ipynb#cell=6; lines=21-24
Related source: baseline_basic.ipynb#cell=8; lines=27-29
Related source: baseline_basic.ipynb#cell=10; lines=24-32
Related source: baseline_basic.ipynb#cell=20; lines=10-11
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=25-52
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: preprocess_3
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 1](#context-s02)

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn.
```

Trích báo cáo:

```markdown
2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học.
```

Trích báo cáo:

```markdown
3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference.
```

Trích báo cáo:

```markdown
4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt trích ràng buộc

<a id="constraints-s03"></a>

Gửi cùng [nguồn tham khảo](#references-s03), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s03) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s02 — Dataset và batch — phần việc 1:
[Dán tóm tắt s02]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

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

train_loader = make_loader(OCRDataset(train_records), train=True)
val_loader = make_loader(OCRDataset(val_records))
test_loader = make_loader(OCRDataset(test_records))
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
IMAGE_HEIGHT = 64
```

```python
def make_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train, num_workers=0, collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

```python
train_records = adapt_split(raw["train"], "train")
val_records = adapt_split(raw["validation"], "validation")
test_records = adapt_split(raw["test"], "test")
```

```python
class OCRDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        target = torch.tensor([char_vocab.get(ch, UNK_ID) for ch in record["text"]], dtype=torch.long)
        return image_tensor(record["image"]), target, record
```

```python
validation_metrics = evaluate(best_model, val_loader)
test_metrics, predictions = evaluate(best_model, test_loader, return_predictions=True)
```

Đối chiếu thêm code hiện tại của: cell 18, dòng 7–44. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, target_lengths, targets, texts, widths}; OCRModel: __init__(cfg, initialize_pretrained=True); forward(images, widths).

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 4: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=12; lines=1-16
Related source: baseline_basic.ipynb#cell=12; lines=18-19
Target: baseline.ipynb#cell=12; lines=3-33
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: architecture
Config uses: preprocess_3
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 2](#context-s03)

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC**. Encoder torchvision/resnet18 dùng `ResNet18_Weights.IMAGENET1K_V1`, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. CTC cộng xác suất nhiều alignment hợp lệ, không cần biết ký tự nằm ở pixel nào; greedy decoder gộp lặp và bỏ blank theo độ dài ảnh thật.
```

Trích báo cáo:

```markdown
Giữ tỷ lệ và stride nhỏ theo chiều ngang xử lý trực tiếp yêu cầu độ dài CTC. BiLSTM có thể hỗ trợ nét chữ mơ hồ dựa trên hai phía, encoder pretrained chỉ cung cấp feature ảnh; alphabet và recognizer học mới. Không dùng full pretrained OCR model hoặc language decoder. Viết tay khác miền ImageNet và không có language model nên vẫn có thể sai từ hiếm.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
BLANK_ID, UNK_ID = 0, 1
```

### Prompt trích ràng buộc

<a id="constraints-s04"></a>

Gửi cùng [nguồn tham khảo](#references-s04), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s04) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s03 — Dataset và batch — phần việc 2:
[Dán tóm tắt s03]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class OCRModel(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.sequence = nn.RNN(128, cfg["hidden_dim"], nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(cfg["hidden_dim"], cfg["num_classes"])
    def forward(self, images, widths):
        x = self.visual(images).mean(dim=2).transpose(1, 2)
        lengths = widths // 4
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.sequence(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=x.size(1))
        return self.head(hidden).log_softmax(-1), lengths
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
def build_model(cfg, initialize_pretrained=True):
    return OCRModel(cfg, initialize_pretrained)
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, target_lengths, targets, texts, widths}; OCRModel: __init__(cfg, initialize_pretrained=True); forward(images, widths).

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 5: Siêu tham số

<!-- module
ID: s05
Group: hyperparameters
Kind: code
Depends: s03, s04
Budget: input=1000; output=600; reserve=200
Summary: no
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=5; lines=28-29
Source: baseline_basic.ipynb#cell=5; lines=32-37
Related source: baseline_basic.ipynb#cell=6; lines=21-24
Related source: baseline_basic.ipynb#cell=17; lines=3-11
Related source: baseline_basic.ipynb#cell=18; lines=4-4
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=64-64
Reference code: baseline.ipynb#cell=5; lines=28-29
Reference code: baseline.ipynb#cell=5; lines=32-39
Reference missing: no
Config owns: training_config
Config uses: preprocess_3, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 2](#context-s03)
- [Model và giao diện sử dụng](#context-s04)

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Các giá trị dưới đây lấy từ `baseline.ipynb` của bản optimized và là **đề xuất có cơ sở**, chưa phải bộ siêu tham số tốt nhất đã đo trên validation. Lý do là giả thuyết về độ phủ, capacity, điều chuẩn và chi phí cần được kiểm bằng dữ liệu thực; không suy ra tối ưu chỉ vì giá trị thường dùng.
```

Trích báo cáo:

```markdown
Khi kiểm độ nhạy, giữ cố định split/metric và chỉ thay một yếu tố quan trọng ở mỗi so sánh. Nếu validation không cải thiện, không kết luận cấu hình lớn hơn hoặc training lâu hơn là tối ưu. Không dùng held-out để chọn lại các giá trị.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
BATCH_SIZE = 8
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 5.0
PRIMARY_METRIC = "cer"
DIRECTION = "min"
```

### Prompt trích ràng buộc

<a id="constraints-s05"></a>

Gửi cùng [nguồn tham khảo](#references-s05), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s05) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s03 — Dataset và batch — phần việc 2:
[Dán tóm tắt s03]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 16
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 5.0
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
def make_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train, num_workers=0, collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

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

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

Đối chiếu thêm code hiện tại của: cell 18, dòng 7–44. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, target_lengths, targets, texts, widths}; OCRModel: __init__(cfg, initialize_pretrained=True); forward(images, widths).
