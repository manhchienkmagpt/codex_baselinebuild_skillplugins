# Prompt basic: FUNSD — nhận dạng thực thể tài liệu — bản tối ưu

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
Source: baseline_basic.ipynb#cell=10; lines=1-2
Source: baseline_basic.ipynb#cell=10; lines=8-10
Source: baseline_basic.ipynb#cell=10; lines=38-38
Target: baseline.ipynb#cell=10; lines=1-4
Target: baseline.ipynb#cell=10; lines=9-11
Target: baseline.ipynb#cell=10; lines=47-47
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: restore_tokenizer.
Ý tưởng tiền xử lí: 1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn. 2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu. 3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác. 4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
Nguồn: {"dataset":"nielsr/funsd","revision":"7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5"}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
word_vocab = {"<PAD>": 0, "<UNK>": 1}
word_vocab.update({word: i + 2 for i, word in enumerate(sorted({w.lower() for r in train_records for w in r["words"]}))})
```

```python
def restore_tokenizer(state):
    global word_vocab, TAG_NAMES
    word_vocab, TAG_NAMES = state["word_vocab"], state["tag_names"]
```

```python
print("Vocabulary train:", len(word_vocab), "từ;", 0, "đặc trưng ký tự; không đọc ảnh vì model basic không dùng.")
```

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 2: Tiền xử lí dữ liệu — Dataset và batch

<!-- module
ID: s02
Group: preprocessing
Kind: code
Depends: s01
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=6; lines=21-24
Source: baseline_basic.ipynb#cell=10; lines=12-37
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=13-46
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: make_loader, FormDataset, collate_batch.
Ý tưởng tiền xử lí: 1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn. 2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu. 3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác. 4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
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
class FormDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        return self.records[index]

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

train_loader = make_loader(FormDataset(train_records), train=True)
val_loader = make_loader(FormDataset(val_records))
test_loader = make_loader(FormDataset(test_records))
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 3: Xử lí mô hình — Thành phần EntityCRF

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=12; lines=1-45
Config owns: component_EntityCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)

### Prompt tạo hoặc sửa module

```text
Viết trọn thành phần EntityCRF phục vụ phương pháp bên dưới; định nghĩa API nhận tensor/mask và trả kết quả để model chính dùng. Với CRF: gồm NLL và Viterbi, transition hợp lệ, bỏ padding; không chỉ tạo khung lớp.
Ý tưởng mô hình: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans. Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: collate_batch(items) -> batch{boxes, ids, lengths, mask, records, tags, words}; FormModel: __init__(cfg); forward(batch); loss(emissions, batch); decode(emissions, batch).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]
```

### Code gửi kèm

Phần bổ sung mới; dùng giao diện hiện có và context ở trên.

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
Depends: s02, s03
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=12; lines=1-21
Target: baseline.ipynb#cell=13; lines=2-42
Config owns: architecture
Config uses: preprocess_2, component_EntityCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Thành phần EntityCRF](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans. Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; Không dùng pretrained cho tokenizer/embedding/model tác vụ.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s03 — Thành phần EntityCRF:
[Dán tóm tắt s03]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class FormModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.word_embedding = nn.Embedding(cfg["n_words"], 64, padding_idx=0)
        self.layout = nn.Sequential(nn.Linear(6,16),nn.ReLU())
        self.encoder = nn.RNN(80,cfg["hidden_dim"],batch_first=True)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["hidden_dim"],len(TAG_NAMES))
        self.register_buffer("tag_weights", torch.ones(len(TAG_NAMES)))
    def forward(self, batch):
        box = batch["boxes"]
        layout = self.layout(torch.cat([box,box[:,:,2:]-box[:,:,:2]],-1))
        x = self.dropout(torch.cat([self.word_embedding(batch["words"]),layout],-1))
        packed = pack_padded_sequence(x,batch["lengths"],batch_first=True,enforce_sorted=False)
        hidden,_ = self.encoder(packed)
        hidden,_ = pad_packed_sequence(hidden,batch_first=True,total_length=batch["words"].size(1))
        return self.head(self.dropout(hidden))
    def loss(self, emissions, batch):
        return nn.functional.cross_entropy(emissions[batch["mask"]],batch["tags"][batch["mask"]],weight=self.tag_weights)
    def decode(self, emissions, batch):
        return [canonical_bio(ids[:length]) for ids,length in zip(emissions.argmax(-1).tolist(),batch["lengths"].tolist())]
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
Depends: s02, s04
Budget: input=1000; output=600; reserve=200
Summary: no
Source: baseline_basic.ipynb#cell=5; lines=30-31
Source: baseline_basic.ipynb#cell=5; lines=34-39
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Model và giao diện sử dụng](#context-s04)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng.
Phương pháp: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. Metric: entity_f1. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
```

```python
EPOCHS = 60
PATIENCE = 12
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
```
