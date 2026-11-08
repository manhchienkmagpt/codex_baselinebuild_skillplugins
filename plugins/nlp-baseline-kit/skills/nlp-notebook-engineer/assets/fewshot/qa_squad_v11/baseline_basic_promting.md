# Prompt basic: SQuAD v1.1 — QA trích xuất span — bản tối ưu

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

## Bước 1: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh — phần việc 1

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
Source: baseline_basic.ipynb#cell=6; lines=5-6
Source: baseline_basic.ipynb#cell=10; lines=5-23
Source: baseline_basic.ipynb#cell=10; lines=31-33
Related source: baseline_basic.ipynb#cell=5; lines=27-27
Related source: baseline_basic.ipynb#cell=5; lines=30-30
Related source: baseline_basic.ipynb#cell=8; lines=40-40
Related source: baseline_basic.ipynb#cell=10; lines=2-4
Related source: baseline_basic.ipynb#cell=10; lines=25-26
Related source: baseline_basic.ipynb#cell=10; lines=35-42
Related source: baseline_basic.ipynb#cell=10; lines=68-72
Related source: baseline_basic.ipynb#cell=10; lines=89-89
Related source: baseline_basic.ipynb#cell=13; lines=1-1
Related source: baseline_basic.ipynb#cell=19; lines=2-4
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=10; lines=5-25
Target: baseline.ipynb#cell=10; lines=32-43
Target: baseline.ipynb#cell=10; lines=69-77
Target: baseline.ipynb#cell=10; lines=94-100
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=5; lines=1-1
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=5; lines=21-23
Reference code: baseline.ipynb#cell=5; lines=26-26
Reference code: baseline.ipynb#cell=5; lines=44-44
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
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
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
SOURCE = {'row': 3, 'task': 'Extractive QA', 'dataset': 'SQuAD v1.1', 'tfds_builder': 'squad/v1.1:3.0.0', 'original_hint': 'squad/v1.1.0 (đã sửa config thành v1.1, builder version3.0.0)', 'url': 'https://www.tensorflow.org/datasets/catalog/squad'}
DATA_MODE = "tfds"
SOURCE["data_mode"] = DATA_MODE
```

```python
VAL_FRACTION = 0.10
```

```python
PRIMARY_METRIC = "exact_match"
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
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
def clean_text(text):
    return " ".join(unicodedata.normalize("NFC",text).split())
```

```python
def context_tokens(text):
    if text not in _CONTEXT_CACHE:
        matches=list(re.finditer(TOKEN_PATTERN,text))
        _CONTEXT_CACHE[text]={"words":[m.group() for m in matches],"offsets":[m.span() for m in matches]}
    return _CONTEXT_CACHE[text]
def question_words(text):return re.findall(TOKEN_PATTERN,text)[:MAX_QUESTION_TOKENS] or ["<UNK>"]

counts=Counter()

seen_context=set()
for record in train_records:
    if record["context"] not in seen_context:
        seen_context.add(record["context"])
        words=context_tokens(record["context"])["words"]
        counts.update(w.lower() for w in words)
    words=re.findall(TOKEN_PATTERN,record["question"])
    counts.update(w.lower() for w in words)
word_vocab={"<PAD>":PAD_ID,"<UNK>":UNK_ID,"<CLS>":CLS_ID}
word_vocab.update({word:i+3 for i,word in enumerate(sorted(counts,key=lambda w:(-counts[w],w))[:VOCAB_SIZE-3])})
```

```python
def restore_tokenizer(state):
    global word_vocab
    word_vocab=state["word_vocab"]
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
VOCAB_SIZE = 40000
```

```python
MAX_QUESTION_TOKENS = 64
```

```python
train_records=[r for r in pool if r["title"] not in val_titles]
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
TOKEN_PATTERN=r"\w+|[^\w\s]"
_CONTEXT_CACHE={}
```

```python
TOKENIZER_STATE={"word_vocab":word_vocab,"regex":TOKEN_PATTERN,
                 "pad":PAD_ID,"unk":UNK_ID,"cls":CLS_ID}
```

```python
def answer_token_span(context,answer):
    offsets=context_tokens(context)["offsets"]
    starts=[s for s,e in offsets];ends=[e for s,e in offsets]
    s=answer["answer_start"];e=s+len(answer["text"])
    left=bisect.bisect_right(ends,s);right=bisect.bisect_left(starts,e)-1
    if left>right or right>=len(offsets):raise ValueError("Answer không giao token nào.")
    exact=offsets[left][0]==s and offsets[right][1]==e
    return left,right,exact
```

```python
def encode_words(rows,width):
    words=torch.zeros(len(rows),width,dtype=torch.long)
    for i,tokens in enumerate(rows):
        words[i,:len(tokens)]=torch.tensor([CLS_ID if w=="<CLS>" else word_vocab.get(w.lower(),UNK_ID) for w in tokens])
    return words
```

```python
print("Word vocab chỉ train, không tạo char features:",len(word_vocab))
```

```python
MODEL_CONFIG={"n_words":len(word_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
```

```python
restore_tokenizer(checkpoint["tokenizer"])
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"n_words":len(word_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
```

Đối chiếu thêm code hiện tại của: QADataset. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).

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

## Bước 2: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh — phần việc 2

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
Source: baseline_basic.ipynb#cell=10; lines=35-42
Source: baseline_basic.ipynb#cell=10; lines=68-72
Source: baseline_basic.ipynb#cell=10; lines=89-95
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=31-31
Related source: baseline_basic.ipynb#cell=8; lines=36-36
Related source: baseline_basic.ipynb#cell=8; lines=40-41
Related source: baseline_basic.ipynb#cell=10; lines=2-2
Related source: baseline_basic.ipynb#cell=10; lines=5-9
Related source: baseline_basic.ipynb#cell=10; lines=22-22
Related source: baseline_basic.ipynb#cell=10; lines=74-84
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=10; lines=5-25
Target: baseline.ipynb#cell=10; lines=32-43
Target: baseline.ipynb#cell=10; lines=69-77
Target: baseline.ipynb#cell=10; lines=94-100
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh — phần việc 1](#context-s01)

### Nguồn tham khảo

<a id="references-s02"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
```

### Prompt trích ràng buộc

<a id="constraints-s02"></a>

Gửi cùng [nguồn tham khảo](#references-s02), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s02) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nâng cấp biểu diễn và tiền xử lí theo phương pháp tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s01 — Biểu diễn và căn chỉnh — phần việc 1:
[Dán tóm tắt s01]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def answer_token_span(context,answer):
    offsets=context_tokens(context)["offsets"]
    starts=[s for s,e in offsets];ends=[e for s,e in offsets]
    s=answer["answer_start"];e=s+len(answer["text"])
    left=bisect.bisect_right(ends,s);right=bisect.bisect_left(starts,e)-1
    if left>right or right>=len(offsets):raise ValueError("Answer không giao token nào.")
    exact=offsets[left][0]==s and offsets[right][1]==e
    return left,right,exact
```

```python
def encode_words(rows,width):
    words=torch.zeros(len(rows),width,dtype=torch.long)
    for i,tokens in enumerate(rows):
        words[i,:len(tokens)]=torch.tensor([CLS_ID if w=="<CLS>" else word_vocab.get(w.lower(),UNK_ID) for w in tokens])
    return words
```

```python
print("Word vocab chỉ train, không tạo char features:",len(word_vocab))
print("Số windows:",len(train_loader.dataset),len(val_loader.dataset),len(test_loader.dataset))
for name,records in (("train",train_records),("validation",val_records),("held_out",test_records)):
    alignment=[answer_token_span(r["context"],a) for r in records for a in r["answers"]]
    print(name,"gold answer boundaries giữa token:",sum(not exact for a,b,exact in alignment),
          "; gold dài hơn answer cap:",sum(b-a+1>MAX_ANSWER_TOKENS for a,b,exact in alignment))
print("Nhãn train là token span bao phủ tối thiểu; không sửa gold text khi chấm EM/F1. Giới hạn ranh giới được báo rõ.")
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
MAX_ANSWER_TOKENS = 64
```

```python
test_records=sorted([adapt(row) for row in raw["validation"]],key=lambda r:r["id"])
```

```python
train_records=[r for r in pool if r["title"] not in val_titles]
val_records=[r for r in pool if r["title"] in val_titles]
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
```

```python
def context_tokens(text):
    if text not in _CONTEXT_CACHE:
        matches=list(re.finditer(TOKEN_PATTERN,text))
        _CONTEXT_CACHE[text]={"words":[m.group() for m in matches],"offsets":[m.span() for m in matches]}
    return _CONTEXT_CACHE[text]
```

```python
word_vocab={"<PAD>":PAD_ID,"<UNK>":UNK_ID,"<CLS>":CLS_ID}
```

```python
def collate_batch(items):
    lengths=torch.tensor([len(item["words"]) for item in items])
    question_lengths=torch.tensor([len(item["question"]) for item in items])
    context=encode_words([item["words"] for item in items],int(lengths.max()))
    question=encode_words([item["question"] for item in items],int(question_lengths.max()))
    return {"context":context,"question":question,
            "lengths":lengths,"question_lengths":question_lengths,
            "context_mask":torch.arange(context.size(1))[None]<lengths[:,None],
            "question_mask":torch.arange(question.size(1))[None]<question_lengths[:,None],
            "targets":torch.tensor([item["targets"] for item in items]),
            "ids":[item["id"] for item in items],"items":items}
```

Đối chiếu thêm code hiện tại của: QADataset. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).

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

## Bước 3: Tiền xử lí dữ liệu — Dataset và batch — phần việc 1

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
Source: baseline_basic.ipynb#cell=6; lines=43-46
Related source: baseline_basic.ipynb#cell=5; lines=1-1
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=36-37
Related source: baseline_basic.ipynb#cell=10; lines=74-88
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: preprocess_3
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh — phần việc 2](#context-s02)

### Nguồn tham khảo

<a id="references-s03"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
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

Context s02 — Biểu diễn và căn chỉnh — phần việc 2:
[Dán tóm tắt s02]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def make_loader(dataset,train=False):
    return DataLoader(dataset,batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train,num_workers=0,collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
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
EVAL_BATCH_SIZE = 32
```

```python
def collate_batch(items):
    lengths=torch.tensor([len(item["words"]) for item in items])
    question_lengths=torch.tensor([len(item["question"]) for item in items])
    context=encode_words([item["words"] for item in items],int(lengths.max()))
    question=encode_words([item["question"] for item in items],int(question_lengths.max()))
    return {"context":context,"question":question,
            "lengths":lengths,"question_lengths":question_lengths,
            "context_mask":torch.arange(context.size(1))[None]<lengths[:,None],
            "question_mask":torch.arange(question.size(1))[None]<question_lengths[:,None],
            "targets":torch.tensor([item["targets"] for item in items]),
            "ids":[item["id"] for item in items],"items":items}

train_loader=make_loader(QADataset(train_records,True),True)
val_loader=make_loader(QADataset(val_records))
test_loader=make_loader(QADataset(test_records))
```

Đối chiếu thêm code hiện tại của: answer_question(context, question). Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).

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

## Bước 4: Tiền xử lí dữ liệu — Dataset và batch — phần việc 2

<!-- module
ID: s04
Group: preprocessing
Kind: code
Depends: s03
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=44-66
Related source: baseline_basic.ipynb#cell=5; lines=28-29
Related source: baseline_basic.ipynb#cell=10; lines=5-10
Related source: baseline_basic.ipynb#cell=10; lines=35-42
Related source: baseline_basic.ipynb#cell=10; lines=86-88
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: preprocess_4
Config uses: preprocess_3
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 1](#context-s03)

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
```

### Prompt trích ràng buộc

<a id="constraints-s04"></a>

Gửi cùng [nguồn tham khảo](#references-s04), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s04) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s03 — Dataset và batch — phần việc 1:
[Dán tóm tắt s03]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class QADataset(Dataset):
    def __init__(self,records,training=False):
        self.records=records;self.features=[]
        for record in records:
            n=len(context_tokens(record["context"])["words"])
            if not n:raise ValueError("Context không có token.")
            gold=[answer_token_span(record["context"],a)[:2] for a in record["answers"]] if training else []
            has_positive=False
            for start in range(0,n,WINDOW_STEP):
                stop=min(start+MAX_CONTEXT_TOKENS,n);targets=(0,0)
                for a,b in gold:
                    if start<=a<=b<stop:targets=(a-start+1,b-start+1);has_positive=True;break
                self.features.append({"record":record,"start":start,"stop":stop,"targets":targets})
                if stop==n:break
            if training and not has_positive:raise ValueError("Không có window chứa answer; cần tăng context cap/overlap thủ công.")
    def __len__(self):return len(self.features)
    def __getitem__(self,index):
        feature=self.features[index];record=feature["record"]
        cached=context_tokens(record["context"])
        words=["<CLS>"]+cached["words"][feature["start"]:feature["stop"]]
        offsets=[(-1,-1)]+cached["offsets"][feature["start"]:feature["stop"]]
        return {"words":words,"question":question_words(record["question"]),"offsets":offsets,
                "targets":feature["targets"],"record":record,"id":f"{record['id']}:{feature['start']}"}
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
MAX_CONTEXT_TOKENS = 384
WINDOW_STEP = 256
```

```python
def context_tokens(text):
    if text not in _CONTEXT_CACHE:
        matches=list(re.finditer(TOKEN_PATTERN,text))
        _CONTEXT_CACHE[text]={"words":[m.group() for m in matches],"offsets":[m.span() for m in matches]}
    return _CONTEXT_CACHE[text]
def question_words(text):return re.findall(TOKEN_PATTERN,text)[:MAX_QUESTION_TOKENS] or ["<UNK>"]
```

```python
def answer_token_span(context,answer):
    offsets=context_tokens(context)["offsets"]
    starts=[s for s,e in offsets];ends=[e for s,e in offsets]
    s=answer["answer_start"];e=s+len(answer["text"])
    left=bisect.bisect_right(ends,s);right=bisect.bisect_left(starts,e)-1
    if left>right or right>=len(offsets):raise ValueError("Answer không giao token nào.")
    exact=offsets[left][0]==s and offsets[right][1]==e
    return left,right,exact
```

```python
train_loader=make_loader(QADataset(train_records,True),True)
val_loader=make_loader(QADataset(val_records))
test_loader=make_loader(QADataset(test_records))
```

Đối chiếu thêm code hiện tại của: answer_question(context, question). Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).

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

## Bước 5: Tiền xử lí dữ liệu — Dataset và batch — phần việc 3

<!-- module
ID: s05
Group: preprocessing
Kind: code
Depends: s04
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=74-90
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=6; lines=43-46
Related source: baseline_basic.ipynb#cell=8; lines=36-36
Related source: baseline_basic.ipynb#cell=8; lines=40-41
Related source: baseline_basic.ipynb#cell=10; lines=68-72
Related source: baseline_basic.ipynb#cell=17; lines=4-4
Related source: baseline_basic.ipynb#cell=19; lines=8-9
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: preprocess_5
Config uses: preprocess_4
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 2](#context-s04)

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ.
```

Trích báo cáo:

```markdown
2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê.
```

Trích báo cáo:

```markdown
3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài.
```

Trích báo cáo:

```markdown
4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
```

Trích báo cáo:

```markdown
Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
```

### Prompt trích ràng buộc

<a id="constraints-s05"></a>

Gửi cùng [nguồn tham khảo](#references-s05), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s05) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Nối Dataset, batch và padding với biểu diễn đã chốt.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s04 — Dataset và batch — phần việc 2:
[Dán tóm tắt s04]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def collate_batch(items):
    lengths=torch.tensor([len(item["words"]) for item in items])
    question_lengths=torch.tensor([len(item["question"]) for item in items])
    context=encode_words([item["words"] for item in items],int(lengths.max()))
    question=encode_words([item["question"] for item in items],int(question_lengths.max()))
    return {"context":context,"question":question,
            "lengths":lengths,"question_lengths":question_lengths,
            "context_mask":torch.arange(context.size(1))[None]<lengths[:,None],
            "question_mask":torch.arange(question.size(1))[None]<question_lengths[:,None],
            "targets":torch.tensor([item["targets"] for item in items]),
            "ids":[item["id"] for item in items],"items":items}

train_loader=make_loader(QADataset(train_records,True),True)
val_loader=make_loader(QADataset(val_records))
test_loader=make_loader(QADataset(test_records))
print("Word vocab chỉ train, không tạo char features:",len(word_vocab))
print("Số windows:",len(train_loader.dataset),len(val_loader.dataset),len(test_loader.dataset))
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
def make_loader(dataset,train=False):
    return DataLoader(dataset,batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train,num_workers=0,collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

```python
test_records=sorted([adapt(row) for row in raw["validation"]],key=lambda r:r["id"])
```

```python
train_records=[r for r in pool if r["title"] not in val_titles]
val_records=[r for r in pool if r["title"] in val_titles]
```

```python
def encode_words(rows,width):
    words=torch.zeros(len(rows),width,dtype=torch.long)
    for i,tokens in enumerate(rows):
        words[i,:len(tokens)]=torch.tensor([CLS_ID if w=="<CLS>" else word_vocab.get(w.lower(),UNK_ID) for w in tokens])
    return words
```

```python
total_steps=max(1,EPOCHS*len(train_loader))
```

```python
validation_metrics=evaluate(best_model,val_loader)
test_metrics,predictions=evaluate(best_model,test_loader,return_predictions=True)
```

Đối chiếu thêm code hiện tại của: QADataset; cell 17, dòng 10–38. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).

### Tóm tắt context để dùng tiếp

<a id="context-s05"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 6: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s06
Group: model
Kind: code
Depends: s05
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=12; lines=1-17
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=12; lines=18-18
Target: baseline.ipynb#cell=12; lines=1-41
Reference report: huong_giai_quyet.md#lines=37-37
Reference report: huong_giai_quyet.md#lines=39-39
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=10; lines=2-2
Reference missing: no
Config owns: architecture
Config uses: preprocess_5
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 3](#context-s05)

### Nguồn tham khảo

<a id="references-s06"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads**. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; attention giúp từng token context đọc phần question liên quan. Toàn bộ mô hình học từ đầu, không dùng BERT/tokenizer pretrained.
```

Trích báo cáo:

```markdown
Câu hỏi và đoạn văn cần so khớp ở mức token nên tương tác hai chiều có căn cứ hơn chỉ một vector question. Char-CNN giảm phụ thuộc vào word vocabulary giới hạn. Bản hiện tại vẫn không thể chọn chính xác biên giữa một token; đây là trần biểu diễn trên một số gold EM. Lựa chọn là thiết kế scratch có cơ sở, chưa chứng minh tối ưu toàn bộ SQuAD.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
```

### Prompt trích ràng buộc

<a id="constraints-s06"></a>

Gửi cùng [nguồn tham khảo](#references-s06), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s06) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s05 — Dataset và batch — phần việc 3:
[Dán tóm tắt s05]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class QAModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        dim=cfg["hidden_dim"]
        self.embedding=nn.Embedding(cfg["n_words"],64,padding_idx=0)
        self.context_encoder=nn.RNN(64,dim,batch_first=True)
        self.question_encoder=nn.RNN(64,dim,batch_first=True)
        self.dropout=nn.Dropout(cfg["dropout"])
        self.fusion=nn.Sequential(nn.Linear(3*dim,dim),nn.Tanh())
        self.head=nn.Linear(dim,2)
    def forward(self,batch):
        packed=pack_padded_sequence(self.dropout(self.embedding(batch["context"])),batch["lengths"],batch_first=True,enforce_sorted=False)
        c,_=self.context_encoder(packed);c,_=pad_packed_sequence(c,batch_first=True,total_length=batch["context"].size(1))
        q=pack_padded_sequence(self.dropout(self.embedding(batch["question"])),batch["question_lengths"],batch_first=True,enforce_sorted=False)
        _,q=self.question_encoder(q);q=q[-1][:,None].expand(-1,c.size(1),-1)
        logits=self.head(self.dropout(self.fusion(torch.cat([c,q,c*q],-1))))
        return logits[:,:,0].masked_fill(~batch["context_mask"],-10000),logits[:,:,1].masked_fill(~batch["context_mask"],-10000)
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
def build_model(cfg):return QAModel(cfg)
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).

### Tóm tắt context để dùng tiếp

<a id="context-s06"></a>

```text
Tóm tắt code thật gọn để sinh module tương tự hoặc viết module gọi nó.
Giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái,
ràng buộc và logic xử lý chính.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 7: Siêu tham số

<!-- module
ID: s07
Group: hyperparameters
Kind: code
Depends: s05, s06
Budget: input=1000; output=600; reserve=200
Summary: no
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=5; lines=36-43
Related source: baseline_basic.ipynb#cell=6; lines=43-46
Related source: baseline_basic.ipynb#cell=17; lines=3-7
Reference report: huong_giai_quyet.md#lines=47-47
Reference report: huong_giai_quyet.md#lines=66-66
Reference code: baseline.ipynb#cell=5; lines=36-44
Reference missing: no
Config owns: training_config
Config uses: preprocess_5, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 3](#context-s05)
- [Model và giao diện sử dụng](#context-s06)

### Nguồn tham khảo

<a id="references-s07"></a>

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
BATCH_SIZE = 24
EVAL_BATCH_SIZE = 24
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "exact_match"
```

### Prompt trích ràng buộc

<a id="constraints-s07"></a>

Gửi cùng [nguồn tham khảo](#references-s07), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s07) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s05 — Dataset và batch — phần việc 3:
[Dán tóm tắt s05]

Context s06 — Model và giao diện sử dụng:
[Dán tóm tắt s06]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
def make_loader(dataset,train=False):
    return DataLoader(dataset,batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train,num_workers=0,collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

```python
optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,betas=(0.9,0.98),eps=1e-9,weight_decay=WEIGHT_DECAY)
total_steps=max(1,EPOCHS*len(train_loader))
def lr_multiplier(step):
    step+=1
    return min(step/max(1,WARMUP_STEPS),math.sqrt(max(1,WARMUP_STEPS)/max(step,WARMUP_STEPS))) if WARMUP_STEPS else 1.0
```

Đối chiếu thêm code hiện tại của: cell 17, dòng 10–38. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{context, context_mask, ids, items, lengths, question, question_lengths, question_mask, targets}; QAModel: __init__(cfg); forward(batch).
