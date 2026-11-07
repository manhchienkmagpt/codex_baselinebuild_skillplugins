# Prompt basic: COCO Captions 2014 — bản tối ưu

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
Source: baseline_basic.ipynb#cell=10; lines=2-8
Source: baseline_basic.ipynb#cell=10; lines=14-21
Source: baseline_basic.ipynb#cell=10; lines=48-49
Target: baseline.ipynb#cell=10; lines=2-8
Target: baseline.ipynb#cell=10; lines=14-27
Target: baseline.ipynb#cell=10; lines=54-55
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: tokenize, restore_tokenizer, image_tensor.
Ý tưởng tiền xử lí: 1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh. 2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng. 3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp. 4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
Nguồn: {"dataset":"MS COCO Captions 2014","reader":"torchvision.datasets.CocoCaptions","split_protocol":"train2014; val2014 chia image-disjoint validation5000/held-out còn lại"}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def tokenize(text):
    return str(text).split()
counts=Counter(token for r in train_records for text in r["refs"] for token in tokenize(text))
words=sorted((w for w,c in counts.items() if c>=MIN_WORD_FREQUENCY),key=lambda w:(-counts[w],w))[:VOCAB_SIZE-4]
vocab={"<PAD>":PAD_ID,"<UNK>":UNK_ID,"<BOS>":BOS_ID,"<EOS>":EOS_ID}
vocab.update({word:i+4 for i,word in enumerate(words)})
id_to_word={i:word for word,i in vocab.items()}
```

```python
def restore_tokenizer(state):
    global vocab,id_to_word
    vocab=state["vocab"]
    id_to_word={i:word for word,i in vocab.items()}

def image_tensor(value):
    image=open_rgb(value).resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR)
    return TF.to_tensor(image)
```

```python
truncated=sum(len(tokenize(text))>MAX_TOKENS-1 for r in train_records for text in r["refs"])
print("Vocabulary train:",len(vocab),"; caption train bị rút ngắn:",truncated,"; gold đánh giá giữ đầy đủ.")
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
Source: baseline_basic.ipynb#cell=10; lines=23-47
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=29-53
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh](#context-s01)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: make_loader, CaptionDataset, collate_batch.
Ý tưởng tiền xử lí: 1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh. 2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng. 3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp. 4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
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
class CaptionDataset(Dataset):
    def __init__(self,records,training=False):
        self.records=records
        self.training=training
        # Mọi caption train được dùng; eval mỗi ảnh đúng một lần với đầy đủ references.
        self.index=[(i,j) for i,r in enumerate(records) for j in range(len(r["refs"]))] if training else [(i,0) for i in range(len(records))]
    def __len__(self):
        return len(self.index)
    def __getitem__(self,index):
        i,j=self.index[index]
        record=self.records[i]
        tokens=tokenize(record["refs"][j])[:MAX_TOKENS-1]
        target=[BOS_ID]+[vocab.get(w,UNK_ID) for w in tokens]+[EOS_ID]
        return image_tensor(record["image"]),torch.tensor(target),record

def collate_batch(items):
    width=max(len(target) for _,target,_ in items)
    tokens=torch.full((len(items),width),PAD_ID,dtype=torch.long)
    for i,(_,target,_) in enumerate(items):tokens[i,:len(target)]=target
    return {"images":torch.stack([image for image,_,_ in items]),"tokens":tokens,
            "ids":[r["id"] for _,_,r in items],"records":[r for _,_,r in items]}

train_loader=make_loader(CaptionDataset(train_records,training=True),train=True)
val_loader=make_loader(CaptionDataset(val_records))
test_loader=make_loader(CaptionDataset(test_records))
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 3: Xử lí mô hình — Encoder độc lập

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=12; lines=2-57
Config owns: encoder
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)

### Prompt tạo hoặc sửa module

```text
Viết một encoder hoàn chỉnh cho kiến trúc bên dưới, gồm embedding/backbone và mask/lengths. Xuất memory và trạng thái/shape cần cho attention/head; chưa viết decoder hay training loop.
Ý tưởng mô hình: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản. Chỉ cho pretrained backbone ảnh resnet50/ResNet50_Weights.IMAGENET1K_V2; phần ngôn ngữ/head học mới.
API hiện có: collate_batch(items) -> batch{ids, images, records, tokens}; CaptionModel: __init__(cfg, initialize_pretrained); forward(images, tokens); generate(images, max_tokens).
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
Source: baseline_basic.ipynb#cell=12; lines=1-29
Target: baseline.ipynb#cell=12; lines=2-57
Config owns: architecture
Config uses: preprocess_2, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Encoder độc lập](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản. Chỉ cho pretrained backbone ảnh resnet50/ResNet50_Weights.IMAGENET1K_V2; phần ngôn ngữ/head học mới.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s03 — Encoder độc lập:
[Dán tóm tắt s03]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class CaptionModel(nn.Module):
    def __init__(self,cfg,initialize_pretrained=True):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.initial=nn.Linear(128,cfg["hidden_dim"])
        self.embedding=nn.Embedding(cfg["vocab_size"],cfg["embed_dim"],padding_idx=PAD_ID)
        self.decoder=nn.RNNCell(cfg["embed_dim"],cfg["hidden_dim"])
        self.dropout=nn.Dropout(cfg["dropout"])
        self.output=nn.Linear(cfg["hidden_dim"],cfg["vocab_size"])
    def forward(self,images,tokens):
        hidden=self.initial(self.visual(images)).tanh(); outputs=[]
        for t in range(tokens.size(1)):
            hidden=self.decoder(self.embedding(tokens[:,t]),hidden)
            outputs.append(self.output(self.dropout(hidden)))
        return torch.stack(outputs,1)
    @torch.no_grad()
    def generate(self,images,max_tokens):
        hidden=self.initial(self.visual(images)).tanh()
        token=torch.full((len(images),),BOS_ID,device=images.device,dtype=torch.long)
        done=torch.zeros(len(images),device=images.device,dtype=torch.bool); outputs=[]
        for _ in range(max_tokens):
            hidden=self.decoder(self.embedding(token),hidden)
            logits=self.output(hidden); logits[:,[PAD_ID,BOS_ID]]=-torch.inf
            token=logits.argmax(-1); token=torch.where(done,torch.full_like(token,EOS_ID),token)
            outputs.append(token); done|=token==EOS_ID
            if bool(done.all()):break
        return torch.stack(outputs,1)
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
Source: baseline_basic.ipynb#cell=5; lines=24-25
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
Phương pháp: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Metric: cider. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 5.0
```
