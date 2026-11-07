# Prompt basic: OCR IAM-line — bản tối ưu

<!-- prompt-guide
Format: 4
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Chỉ tóm tắt ở bước có liên kết dùng tiếp; web không đặt trần output riêng nên giới hạn độ dài tóm tắt là yêu cầu mềm.

## Bước 1: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh

<!-- module
ID: s01
Group: preprocessing
Kind: code
Depends: none
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=8; lines=12-13
Source: baseline_basic.ipynb#cell=10; lines=2-5
Source: baseline_basic.ipynb#cell=10; lines=11-22
Source: baseline_basic.ipynb#cell=10; lines=52-55
Target: baseline.ipynb#cell=8; lines=12-13
Target: baseline.ipynb#cell=10; lines=2-5
Target: baseline.ipynb#cell=10; lines=11-23
Target: baseline.ipynb#cell=10; lines=53-56
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: normalize_text, restore_tokenizer, image_tensor.
Ý tưởng tiền xử lí: 1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn. 2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học. 3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference. 4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"BLANK_ID":0,"UNK_ID":1}.
Nguồn: {"dataset":"Teklia/IAM-line","revision":"fbdad97500ce54635c0d1ba306bf535cb40656cf"}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=6; lines=21-24
Source: baseline_basic.ipynb#cell=10; lines=24-32
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=25-52
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: make_loader, OCRDataset.
Ý tưởng tiền xử lí: 1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn. 2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học. 3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference. 4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"BLANK_ID":0,"UNK_ID":1}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=34-51
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=25-52
Config owns: preprocess_3
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 1](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: collate_batch.
Ý tưởng tiền xử lí: 1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn. 2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học. 3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference. 4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"BLANK_ID":0,"UNK_ID":1}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch — phần việc 1:
[Dán tóm tắt s02]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=12; lines=1-16
Target: baseline.ipynb#cell=12; lines=3-33
Config owns: architecture
Config uses: preprocess_3
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 2](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC. Encoder torchvision/resnet18 dùng ResNet18_Weights.IMAGENET1K_V1, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. CTC cộng xác suất nhiều alignment hợp lệ, không cần biết ký tự nằm ở pixel nào; greedy decoder gộp lặp và bỏ blank theo độ dài ảnh thật. Chỉ cho pretrained backbone ảnh resnet18/ResNet18_Weights.IMAGENET1K_V1; phần ngôn ngữ/head học mới.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch — phần việc 2:
[Dán tóm tắt s03]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Source: baseline_basic.ipynb#cell=5; lines=28-29
Source: baseline_basic.ipynb#cell=5; lines=32-37
Config owns: training_config
Config uses: preprocess_3, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 2](#context-s03)
- [Model và giao diện sử dụng](#context-s04)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng.
Phương pháp: Chọn ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC. Encoder torchvision/resnet18 dùng ResNet18_Weights.IMAGENET1K_V1, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. Metric: cer (càng thấp càng tốt). Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch — phần việc 2:
[Dán tóm tắt s03]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Code gửi kèm

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
