# Prompt basic: FUNSD — nhận dạng thực thể tài liệu — bản tối ưu

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
Source: baseline_basic.ipynb#cell=10; lines=1-2
Source: baseline_basic.ipynb#cell=10; lines=8-10
Source: baseline_basic.ipynb#cell=10; lines=38-38
Related source: baseline_basic.ipynb#cell=8; lines=51-51
Related source: baseline_basic.ipynb#cell=10; lines=4-4
Related source: baseline_basic.ipynb#cell=10; lines=20-33
Related source: baseline_basic.ipynb#cell=13; lines=1-1
Related source: baseline_basic.ipynb#cell=20; lines=3-3
Related source: baseline_basic.ipynb#cell=20; lines=6-6
Target: baseline.ipynb#cell=10; lines=1-4
Target: baseline.ipynb#cell=10; lines=9-11
Target: baseline.ipynb#cell=10; lines=47-47
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=1-1
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference code: baseline.ipynb#cell=5; lines=22-24
Reference code: baseline.ipynb#cell=5; lines=27-27
Reference code: baseline.ipynb#cell=5; lines=40-41
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn.
```

Trích báo cáo:

```markdown
2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu.
```

Trích báo cáo:

```markdown
3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác.
```

Trích báo cáo:

```markdown
4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1.
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
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
SOURCE = {'row': 4, 'task': 'FUNSD conditional entity recognition, không có gold K-V links', 'dataset': 'nielsr/funsd', 'revision': '7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5', 'url': 'https://huggingface.co/datasets/nielsr/funsd'}
DATASET_ID = SOURCE["dataset"]
DATASET_REVISION = SOURCE["revision"]
```

```python
VAL_FRACTION = 0.20
```

```python
PRIMARY_METRIC = "entity_f1"
DIRECTION = "max"
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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
train_records = [row for i, row in enumerate(all_train) if i not in valid_ids]
```

```python
TOKENIZER_STATE = {"word_vocab": word_vocab, "tag_names": TAG_NAMES}
```

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

```python
MODEL_CONFIG = {"n_words":len(word_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
```

```python
restore_tokenizer(checkpoint["tokenizer"])
```

```python
inference_config = {"n_words":len(word_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{boxes, ids, lengths, mask, records, tags, words}; FormModel: __init__(cfg); forward(batch); loss(emissions, batch); decode(emissions, batch).

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
Source: baseline_basic.ipynb#cell=10; lines=12-37
Related source: baseline_basic.ipynb#cell=5; lines=1-1
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=30-31
Related source: baseline_basic.ipynb#cell=8; lines=47-47
Related source: baseline_basic.ipynb#cell=8; lines=51-52
Related source: baseline_basic.ipynb#cell=10; lines=1-1
Related source: baseline_basic.ipynb#cell=20; lines=10-11
Related source: baseline_basic.ipynb#cell=22; lines=1-14
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=13-46
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
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
1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn.
```

Trích báo cáo:

```markdown
2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu.
```

Trích báo cáo:

```markdown
3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác.
```

Trích báo cáo:

```markdown
4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
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
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
```

```python
test_records = adapt(raw["test"], "test")
```

```python
train_records = [row for i, row in enumerate(all_train) if i not in valid_ids]
val_records = [row for i, row in enumerate(all_train) if i in valid_ids]
```

```python
word_vocab = {"<PAD>": 0, "<UNK>": 1}
```

```python
validation_metrics = evaluate(best_model, val_loader)
test_metrics, predictions = evaluate(best_model, test_loader, return_predictions=True)
```

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

Đối chiếu thêm code hiện tại của: cell 18, dòng 7–44. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{boxes, ids, lengths, mask, records, tags, words}; FormModel: __init__(cfg); forward(batch); loss(emissions, batch); decode(emissions, batch).

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

## Bước 3: Xử lí mô hình — Thành phần EntityCRF

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=12; lines=1-45
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference missing: no
Config owns: component_EntityCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)

### Code liên quan

Giao diện hiện có để nối với module mới: collate_batch(items) -> batch{boxes, ids, lengths, mask, records, tags, words}; FormModel: __init__(cfg); forward(batch); loss(emissions, batch); decode(emissions, batch).

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF**. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans.
```

Trích báo cáo:

```markdown
Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; CRF giảm chuỗi nhãn sai cấu trúc. Tất cả khởi tạo mới, không pretrained. Mô hình không giải quyết OCR đầu vào hoặc linking; độ phức tạp là một rủi ro trên tập nhỏ nên validation theo trang rất quan trọng.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s03) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Thành phần EntityCRF hoàn chỉnh để mô hình chính sử dụng.

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
Source: baseline_basic.ipynb#cell=12; lines=1-21
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=8; lines=14-24
Related source: baseline_basic.ipynb#cell=12; lines=22-27
Target: baseline.ipynb#cell=13; lines=2-42
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=7-8
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, component_EntityCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Thành phần EntityCRF](#context-s03)

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF**. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans.
```

Trích báo cáo:

```markdown
Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; CRF giảm chuỗi nhãn sai cấu trúc. Tất cả khởi tạo mới, không pretrained. Mô hình không giải quyết OCR đầu vào hoặc linking; độ phức tạp là một rủi ro trên tập nhỏ nên validation theo trang rất quan trọng.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
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

Context s03 — Thành phần EntityCRF:
[Dán tóm tắt s03]
```

### Code cần sửa

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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
TAG_NAMES = expected_tags

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

```python
def build_model(cfg, initialize_pretrained=True):
    model = FormModel(cfg)
    counts = torch.bincount(torch.tensor([t for r in train_records for t in r["tags"]]),minlength=len(TAG_NAMES)).float()
    weights = counts.clamp_min(1).rsqrt()
    model.tag_weights.copy_(weights / weights.mean())
    return model
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{boxes, ids, lengths, mask, records, tags, words}; FormModel: __init__(cfg); forward(batch); loss(emissions, batch); decode(emissions, batch).

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
Source: baseline_basic.ipynb#cell=5; lines=30-31
Source: baseline_basic.ipynb#cell=5; lines=34-39
Related source: baseline_basic.ipynb#cell=6; lines=21-24
Related source: baseline_basic.ipynb#cell=17; lines=3-11
Related source: baseline_basic.ipynb#cell=18; lines=4-4
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=62-62
Reference code: baseline.ipynb#cell=5; lines=30-31
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
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
```

```python
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "entity_f1"
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

Giao diện dùng trong notebook: collate_batch(items) -> batch{boxes, ids, lengths, mask, records, tags, words}; FormModel: __init__(cfg); forward(batch); loss(emissions, batch); decode(emissions, batch).
