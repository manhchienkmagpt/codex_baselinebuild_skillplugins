# Prompt basic: Trích xuất aspect và phân loại polarity — bản tối ưu

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
Source: baseline_basic.ipynb#cell=9; lines=7-10
Source: baseline_basic.ipynb#cell=9; lines=13-22
Source: baseline_basic.ipynb#cell=9; lines=64-65
Related source: baseline_basic.ipynb#cell=5; lines=30-30
Related source: baseline_basic.ipynb#cell=7; lines=76-76
Related source: baseline_basic.ipynb#cell=9; lines=25-33
Related source: baseline_basic.ipynb#cell=12; lines=3-3
Related source: baseline_basic.ipynb#cell=18; lines=4-4
Related source: baseline_basic.ipynb#cell=18; lines=8-8
Related source: baseline_basic.ipynb#cell=20; lines=1-14
Target: baseline.ipynb#cell=9; lines=5-31
Target: baseline.ipynb#cell=9; lines=73-74
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=1-5
Reference code: baseline.ipynb#cell=5; lines=12-12
Reference code: baseline.ipynb#cell=5; lines=27-30
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc.
```

Trích báo cáo:

```markdown
2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold.
```

Trích báo cáo:

```markdown
3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó.
```

Trích báo cáo:

```markdown
4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
SEED = 42
DATASET_ID = "tomaarsen/setfit-absa-semeval-restaurants"
DATASET_REVISION = "8885372fe73256f96bb60f65b550538ac5c26047"
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
VAL_FRACTION = 0.20
```

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
PAD_ID, UNK_ID = 0, 1
```

```python
train_records = [r for i, r in enumerate(eligible_records) if i not in val_indices]
```

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
model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE)
```

```python
char_vocab, word_vocab = checkpoint["char_vocab"], checkpoint["word_vocab"]
```

```python
best_model = AspectModel(inference_config, len(char_vocab), len(word_vocab)).to(DEVICE)
```

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

Đối chiếu thêm code hiện tại của: cell 16, dòng 3–47. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items); AspectModel: __init__(cfg, n_chars, n_words); forward(batch); extraction_loss(emissions, batch); decode(emissions, mask); classify_spans(hidden, spans, mask).

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
Source: baseline_basic.ipynb#cell=9; lines=25-33
Source: baseline_basic.ipynb#cell=9; lines=36-53
Source: baseline_basic.ipynb#cell=9; lines=56-63
Related source: baseline_basic.ipynb#cell=5; lines=1-1
Related source: baseline_basic.ipynb#cell=5; lines=13-14
Related source: baseline_basic.ipynb#cell=5; lines=28-28
Related source: baseline_basic.ipynb#cell=5; lines=38-39
Related source: baseline_basic.ipynb#cell=7; lines=67-67
Related source: baseline_basic.ipynb#cell=7; lines=76-77
Related source: baseline_basic.ipynb#cell=9; lines=13-22
Related source: baseline_basic.ipynb#cell=12; lines=4-4
Related source: baseline_basic.ipynb#cell=18; lines=12-13
Target: baseline.ipynb#cell=9; lines=34-42
Target: baseline.ipynb#cell=9; lines=45-62
Target: baseline.ipynb#cell=9; lines=65-72
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=27-30
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
1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc.
```

Trích báo cáo:

```markdown
2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold.
```

Trích báo cáo:

```markdown
3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó.
```

Trích báo cáo:

```markdown
4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
SEED = 42
```

```python
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
```

```python
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
```

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
test_records = group_rows(raw["test"], "test")
```

```python
train_records = [r for i, r in enumerate(eligible_records) if i not in val_indices]
val_records = [r for i, r in enumerate(eligible_records) if i in val_indices]
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
tag_counts = torch.bincount(torch.tensor([tag for item in train_loader.dataset.items for tag in item["tags"]]), minlength=3).float()
```

```python
final_val_metrics = evaluate(best_model, make_loader(val_records))
test_metrics, test_predictions = evaluate(best_model, make_loader(test_records), include_predictions=True)
```

Đối chiếu thêm code hiện tại của: cell 16, dòng 3–47; predict_text(text). Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items); AspectModel: __init__(cfg, n_chars, n_words); forward(batch); extraction_loss(emissions, batch); decode(emissions, mask); classify_spans(hidden, spans, mask).

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

## Bước 3: Xử lí mô hình — Thành phần LinearChainCRF

<!-- module
ID: s03
Group: model
Kind: code
Depends: s02
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=11; lines=1-53
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=4-5
Reference code: baseline.ipynb#cell=5; lines=27-30
Reference missing: no
Config owns: component_LinearChainCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)

### Code liên quan

Giao diện hiện có để nối với module mới: collate_batch(items); AspectModel: __init__(cfg, n_chars, n_words); forward(batch); extraction_loss(emissions, batch); decode(emissions, mask); classify_spans(hidden, spans, mask).

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **embedding ký tự + embedding từ → BiLSTM hai lớp → CRF**, kèm head polarity cho mỗi span. BiLSTM đọc cả trái và phải để hiểu “slow” đang nói về “service”; CRF là bộ giải mã chọn cả chuỗi BIO hợp lệ thay vì chọn từng ký tự riêng. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán.
```

Trích báo cáo:

```markdown
Lựa chọn giải quyết đồng thời hai khó khăn đã thấy: offset không trùng biên từ và nhiều aspect/cảm xúc trong cùng câu. Trọng số loss BIO từ tần suất train giúp giảm áp đảo của O, nhưng lớp conflict vẫn thiếu dữ liệu. CRF không biểu diễn aspect lồng nhau; khảo sát nguồn chưa thấy span chồng lấn. Toàn bộ mô hình và vocabulary học từ đầu.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s03) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Thành phần LinearChainCRF hoàn chỉnh để mô hình chính sử dụng.

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
Source: baseline_basic.ipynb#cell=11; lines=1-38
Related source: baseline_basic.ipynb#cell=5; lines=27-27
Related source: baseline_basic.ipynb#cell=5; lines=30-30
Related source: baseline_basic.ipynb#cell=5; lines=38-39
Related source: baseline_basic.ipynb#cell=12; lines=3-3
Related source: baseline_basic.ipynb#cell=18; lines=8-8
Target: baseline.ipynb#cell=12; lines=1-43
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=4-5
Reference code: baseline.ipynb#cell=5; lines=27-30
Reference missing: no
Config owns: architecture
Config uses: preprocess_2, component_LinearChainCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s02)
- [Thành phần LinearChainCRF](#context-s03)

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **embedding ký tự + embedding từ → BiLSTM hai lớp → CRF**, kèm head polarity cho mỗi span. BiLSTM đọc cả trái và phải để hiểu “slow” đang nói về “service”; CRF là bộ giải mã chọn cả chuỗi BIO hợp lệ thay vì chọn từng ký tự riêng. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán.
```

Trích báo cáo:

```markdown
Lựa chọn giải quyết đồng thời hai khó khăn đã thấy: offset không trùng biên từ và nhiều aspect/cảm xúc trong cùng câu. Trọng số loss BIO từ tần suất train giúp giảm áp đảo của O, nhưng lớp conflict vẫn thiếu dữ liệu. CRF không biểu diễn aspect lồng nhau; khảo sát nguồn chưa thấy span chồng lấn. Toàn bộ mô hình và vocabulary học từ đầu.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PRETRAINING_POLICY = "scratch"
PRETRAINED_COMPONENTS = []
```

```python
LABELS = ("negative", "neutral", "positive", "conflict")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TAGS = ("O", "B", "I")
PAD_ID, UNK_ID = 0, 1
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

Context s03 — Thành phần LinearChainCRF:
[Dán tóm tắt s03]
```

### Code cần sửa

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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
LABELS = ("negative", "neutral", "positive", "conflict")
```

```python
PAD_ID, UNK_ID = 0, 1
```

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE)
```

```python
best_model = AspectModel(inference_config, len(char_vocab), len(word_vocab)).to(DEVICE)
```

Giao diện dùng trong notebook: collate_batch(items); AspectModel: __init__(cfg, n_chars, n_words); forward(batch); extraction_loss(emissions, batch); decode(emissions, mask); classify_spans(hidden, spans, mask).

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
Source: baseline_basic.ipynb#cell=5; lines=13-19
Related source: baseline_basic.ipynb#cell=5; lines=41-41
Related source: baseline_basic.ipynb#cell=9; lines=56-61
Related source: baseline_basic.ipynb#cell=16; lines=1-1
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=64-64
Reference code: baseline.ipynb#cell=5; lines=13-19
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
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
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
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
print(f"Thiết bị: {DEVICE}; biến thể: {VARIANT}; batch: {BATCH_SIZE}; tối đa {EPOCHS} epoch")
```

```python
def make_loader(records, shuffle=False, batch_size=EVAL_BATCH_SIZE):
    return DataLoader(AspectDataset(records), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0,
                      generator=torch.Generator().manual_seed(SEED))

train_loader = make_loader(train_records, shuffle=True, batch_size=BATCH_SIZE)
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
```

Đối chiếu thêm code hiện tại của: cell 16, dòng 3–47. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items); AspectModel: __init__(cfg, n_chars, n_words); forward(batch); extraction_loss(emissions, batch); decode(emissions, mask); classify_spans(hidden, spans, mask).
