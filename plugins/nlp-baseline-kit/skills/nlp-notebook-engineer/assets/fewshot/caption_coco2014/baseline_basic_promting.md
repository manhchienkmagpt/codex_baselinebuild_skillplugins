# Prompt basic: COCO Captions 2014 — bản tối ưu

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
Source: baseline_basic.ipynb#cell=10; lines=2-8
Source: baseline_basic.ipynb#cell=10; lines=14-21
Source: baseline_basic.ipynb#cell=10; lines=48-49
Related source: baseline_basic.ipynb#cell=5; lines=26-29
Related source: baseline_basic.ipynb#cell=6; lines=5-15
Related source: baseline_basic.ipynb#cell=8; lines=19-19
Related source: baseline_basic.ipynb#cell=10; lines=1-1
Related source: baseline_basic.ipynb#cell=10; lines=9-10
Related source: baseline_basic.ipynb#cell=13; lines=1-1
Related source: baseline_basic.ipynb#cell=15; lines=1-6
Related source: baseline_basic.ipynb#cell=22; lines=3-3
Related source: baseline_basic.ipynb#cell=22; lines=6-6
Target: baseline.ipynb#cell=10; lines=2-8
Target: baseline.ipynb#cell=10; lines=14-27
Target: baseline.ipynb#cell=10; lines=54-55
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=1-1
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=22-22
Reference code: baseline.ipynb#cell=5; lines=40-41
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh.
```

Trích báo cáo:

```markdown
2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng.
```

Trích báo cáo:

```markdown
3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp.
```

Trích báo cáo:

```markdown
4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold.
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
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
SOURCE = {'row': 2, 'task': 'Image Captioning', 'dataset': 'MS COCO Captions 2014', 'url': 'https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html', 'reader': 'torchvision.datasets.CocoCaptions', 'split_protocol': 'train2014; val2014 chia image-disjoint validation5000/held-out còn lại'}
```

```python
PRIMARY_METRIC = "cider"
DIRECTION = "max"
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
IMAGE_SIZE = 224
MAX_TOKENS = 40
VOCAB_SIZE = 10000
MIN_WORD_FREQUENCY = 1
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
train_records = coco_records(DATA_ROOT/"train2014",DATA_ROOT/"annotations/captions_train2014.json","train2014")
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

```python
TOKENIZER_STATE={"vocab":vocab,"tokenizer":"whitespace words, train-only vocabulary",
                 "pad":PAD_ID,"unk":UNK_ID,"bos":BOS_ID,"eos":EOS_ID}
```

```python
MODEL_CONFIG={"vocab_size":len(vocab),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"memory_dim":MEMORY_DIM,"dropout":DROPOUT}
```

```python
def decode_tokens(ids):
    result=[]
    for index in ids:
        if int(index)==EOS_ID:break
        if int(index) not in (PAD_ID,BOS_ID):result.append(id_to_word[int(index)])
    return " ".join(result)
```

```python
restore_tokenizer(checkpoint["tokenizer"])
```

```python
inference_config = {"vocab_size":len(vocab),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"memory_dim":MEMORY_DIM,"dropout":DROPOUT}
```

Đối chiếu thêm code hiện tại của: CaptionDataset; describe_image(image). Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, records, tokens}; CaptionModel: __init__(cfg, initialize_pretrained=True); forward(images, tokens); generate(images, max_tokens).

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

## Bước 2: Tiền xử lí dữ liệu — Dataset và batch

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
Source: baseline_basic.ipynb#cell=10; lines=23-47
Related source: baseline_basic.ipynb#cell=5; lines=1-1
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=24-27
Related source: baseline_basic.ipynb#cell=8; lines=19-19
Related source: baseline_basic.ipynb#cell=8; lines=28-29
Related source: baseline_basic.ipynb#cell=10; lines=1-3
Related source: baseline_basic.ipynb#cell=10; lines=6-6
Related source: baseline_basic.ipynb#cell=10; lines=19-21
Related source: baseline_basic.ipynb#cell=22; lines=10-11
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=29-53
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
1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh.
```

Trích báo cáo:

```markdown
2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng.
```

Trích báo cáo:

```markdown
3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp.
```

Trích báo cáo:

```markdown
4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
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
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE
IMAGE_SIZE = 224
MAX_TOKENS = 40
```

```python
train_records = coco_records(DATA_ROOT/"train2014",DATA_ROOT/"annotations/captions_train2014.json","train2014")
```

```python
val_records=[r for i,r in enumerate(val_pool) if i in val_ids]
test_records=[r for i,r in enumerate(val_pool) if i not in val_ids]
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
def tokenize(text):
    return str(text).split()
```

```python
vocab={"<PAD>":PAD_ID,"<UNK>":UNK_ID,"<BOS>":BOS_ID,"<EOS>":EOS_ID}
```

```python
def image_tensor(value):
    image=open_rgb(value).resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR)
    return TF.to_tensor(image)
```

```python
validation_metrics = evaluate(best_model, val_loader)
test_metrics, predictions = evaluate(best_model, test_loader, return_predictions=True)
```

Đối chiếu thêm code hiện tại của: cell 20, dòng 7–44. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, records, tokens}; CaptionModel: __init__(cfg, initialize_pretrained=True); forward(images, tokens); generate(images, max_tokens).

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

## Bước 3: Xử lí mô hình — Encoder độc lập

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=12; lines=2-57
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: encoder
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)

### Code liên quan

Giao diện hiện có để nối với module mới: collate_batch(items) -> batch{ids, images, records, tokens}; CaptionModel: __init__(cfg, initialize_pretrained=True); forward(images, tokens); generate(images, max_tokens).

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch**. Encoder dùng torchvision/resnet50 với `ResNet50_Weights.IMAGENET1K_V2`; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản.
```

Trích báo cáo:

```markdown
Ảnh có cấu trúc thị giác phong phú nên encoder đã học đặc trưng ảnh là điểm khởi đầu hợp lý, còn attention giúp tránh ép mọi vật vào một vector. Decoder và tokenizer đều học mới: ImageNet classifier không tự hoàn thành nhiệm vụ sinh caption, phù hợp chính sách pretrained theo thành phần. Giới hạn còn lại là caption chung chung, OOV và hallucination; CIDEr cao cũng không tự chứng minh mọi chi tiết đều đúng.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s03) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Encoder độc lập hoàn chỉnh để mô hình chính sử dụng.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s02 — Dataset và batch:
[Dán tóm tắt s02]
```

### Code cần sửa

Phần bổ sung mới; dùng giao diện hiện có và context ở trên.

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
Depends: s02, s03
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=12; lines=1-29
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=10; lines=1-1
Related source: baseline_basic.ipynb#cell=12; lines=30-31
Target: baseline.ipynb#cell=12; lines=2-57
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=10; lines=1-1
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Encoder độc lập](#context-s03)

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch**. Encoder dùng torchvision/resnet50 với `ResNet50_Weights.IMAGENET1K_V2`; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản.
```

Trích báo cáo:

```markdown
Ảnh có cấu trúc thị giác phong phú nên encoder đã học đặc trưng ảnh là điểm khởi đầu hợp lý, còn attention giúp tránh ép mọi vật vào một vector. Decoder và tokenizer đều học mới: ImageNet classifier không tự hoàn thành nhiệm vụ sinh caption, phù hợp chính sách pretrained theo thành phần. Giới hạn còn lại là caption chung chung, OOV và hallucination; CIDEr cao cũng không tự chứng minh mọi chi tiết đều đúng.
```

Trích báo cáo:

```markdown
[Tài liệu encoder/weights torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) là nguồn tham khảo cho thành phần thị giác; phần giải tác vụ còn lại được xây dựng độc lập như mô tả trên.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "image_components"
PRETRAINED_COMPONENTS = [{'source': 'torchvision', 'model_id': 'resnet50', 'revision': 'ResNet50_Weights.IMAGENET1K_V2', 'role': 'image_encoder', 'input': 'RGB image', 'output': 'spatial visual features, no text output', 'training': 'finetune'}]
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
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

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s03 — Encoder độc lập:
[Dán tóm tắt s03]
```

### Code cần sửa

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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

```python
def build_model(cfg,initialize_pretrained=True):
    return CaptionModel(cfg,initialize_pretrained)
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, records, tokens}; CaptionModel: __init__(cfg, initialize_pretrained=True); forward(images, tokens); generate(images, max_tokens).

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
Depends: s02, s04
Budget: input=1000; output=600; reserve=200
Summary: no
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=5; lines=24-25
Source: baseline_basic.ipynb#cell=5; lines=34-39
Related source: baseline_basic.ipynb#cell=6; lines=21-24
Related source: baseline_basic.ipynb#cell=19; lines=3-11
Related source: baseline_basic.ipynb#cell=20; lines=4-4
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=67-67
Reference code: baseline.ipynb#cell=5; lines=24-25
Reference code: baseline.ipynb#cell=5; lines=34-41
Reference missing: no
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
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
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE
```

```python
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.0003
BACKBONE_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 5.0
PRIMARY_METRIC = "cider"
DIRECTION = "max"
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

Context s02 — Dataset và batch:
[Dán tóm tắt s02]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Code cần sửa

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

Đối chiếu thêm code hiện tại của: cell 20, dòng 7–44. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, images, records, tokens}; CaptionModel: __init__(cfg, initialize_pretrained=True); forward(images, tokens); generate(images, max_tokens).
