# Prompt basic: Trích xuất aspect và phân loại polarity — bản tối ưu

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
Source: baseline_basic.ipynb#cell=9; lines=7-10
Source: baseline_basic.ipynb#cell=9; lines=13-22
Source: baseline_basic.ipynb#cell=9; lines=64-65
Target: baseline.ipynb#cell=9; lines=5-31
Target: baseline.ipynb#cell=9; lines=73-74
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: encode_record.
Ý tưởng tiền xử lí: 1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc. 2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold. 3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó. 4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
char_counts = Counter(ch for record in train_records for ch in record["text"])
char_vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
char_vocab.update({ch: i + 2 for i, ch in enumerate(sorted(char_counts))})
word_vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
```

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

```python
print(f"Từ vựng chỉ fit train: {len(char_vocab)} ký tự; {len(word_vocab)} từ.")
print(f"Độ dài ký tự lớn nhất trong train: {max(len(r['text']) for r in train_records)}; không truncation.")
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
Source: baseline_basic.ipynb#cell=9; lines=25-33
Source: baseline_basic.ipynb#cell=9; lines=36-53
Source: baseline_basic.ipynb#cell=9; lines=56-63
Target: baseline.ipynb#cell=9; lines=34-42
Target: baseline.ipynb#cell=9; lines=45-62
Target: baseline.ipynb#cell=9; lines=65-72
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: AspectDataset, collate_batch, make_loader.
Ý tưởng tiền xử lí: 1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc. 2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold. 3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó. 4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s01]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class AspectDataset(Dataset):
    def __init__(self, records):
        self.items = [encode_record(record) for record in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
```

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

```python
def make_loader(records, shuffle=False, batch_size=EVAL_BATCH_SIZE):
    return DataLoader(AspectDataset(records), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))

train_loader = make_loader(train_records, shuffle=True, batch_size=BATCH_SIZE)
val_loader = make_loader(val_records)
test_loader = make_loader(test_records)
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 3: Xử lí mô hình — Thành phần LinearChainCRF

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=11; lines=1-53
Config owns: component_LinearChainCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)

### Prompt tạo hoặc sửa module

```text
Viết trọn thành phần LinearChainCRF phục vụ phương pháp bên dưới; định nghĩa API nhận tensor/mask và trả kết quả để model chính dùng. Với CRF: gồm NLL và Viterbi, transition hợp lệ, bỏ padding; không chỉ tạo khung lớp.
Ý tưởng mô hình: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: collate_batch(items); AspectModel: __init__(cfg, n_chars, n_words); forward(batch); extraction_loss(emissions, batch); decode(emissions, mask); classify_spans(hidden, spans, mask).
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
Source: baseline_basic.ipynb#cell=11; lines=1-38
Target: baseline.ipynb#cell=12; lines=1-43
Config owns: architecture
Config uses: preprocess_2, component_LinearChainCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Thành phần LinearChainCRF](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s03 — Thành phần LinearChainCRF:
[Dán tóm tắt s03]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class AspectModel(nn.Module):
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

    def forward(self, batch):
        embedded = self.dropout(self.char_embedding(batch["char_ids"]))
        packed = pack_padded_sequence(embedded, batch["lengths"],
                                       batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True,
                                         total_length=batch["char_ids"].size(1))
        hidden = self.dropout(hidden)
        return hidden, self.tag_head(hidden)

    def extraction_loss(self, emissions, batch):
        return nn.functional.cross_entropy(emissions[batch["mask"]],
                                            batch["tags"][batch["mask"]], weight=self.tag_weights)

    def decode(self, emissions, mask):
        ids = emissions.argmax(-1).tolist()
        return [seq[:length] for seq, length in zip(ids, mask.sum(1).tolist())]

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
Source: baseline_basic.ipynb#cell=5; lines=13-19
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Model và giao diện sử dụng](#context-s04)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng.
Phương pháp: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; Metric: metric đã khai báo trong notebook. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
```
