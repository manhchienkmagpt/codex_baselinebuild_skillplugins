# Prompt basic: SQuAD v1.1 — QA trích xuất span — bản tối ưu

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

## Bước 1: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh — phần việc 1

<!-- module
ID: s01
Group: preprocessing
Kind: code
Depends: none
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=6; lines=5-6
Source: baseline_basic.ipynb#cell=10; lines=5-23
Source: baseline_basic.ipynb#cell=10; lines=31-33
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=10; lines=5-25
Target: baseline.ipynb#cell=10; lines=32-43
Target: baseline.ipynb#cell=10; lines=69-77
Target: baseline.ipynb#cell=10; lines=94-100
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: clean_text, context_tokens, question_words, restore_tokenizer.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Nguồn: {"dataset":"SQuAD v1.1","tfds_builder":"squad/v1.1:3.0.0","original_hint":"squad/v1.1.0 (đã sửa config thành v1.1, builder version3.0.0)"}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=35-42
Source: baseline_basic.ipynb#cell=10; lines=68-72
Source: baseline_basic.ipynb#cell=10; lines=89-95
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=10; lines=5-25
Target: baseline.ipynb#cell=10; lines=32-43
Target: baseline.ipynb#cell=10; lines=69-77
Target: baseline.ipynb#cell=10; lines=94-100
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh — phần việc 1](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: answer_token_span, encode_words.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Biểu diễn và căn chỉnh — phần việc 1:
[Dán tóm tắt s01]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Config owns: preprocess_3
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh — phần việc 2](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: make_loader.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s02 — Biểu diễn và căn chỉnh — phần việc 2:
[Dán tóm tắt s02]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def make_loader(dataset,train=False):
    return DataLoader(dataset,batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train,num_workers=0,collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=44-66
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Config owns: preprocess_4
Config uses: preprocess_3
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 1](#context-s03)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: QADataset.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch — phần việc 1:
[Dán tóm tắt s03]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=10; lines=74-90
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Config owns: preprocess_5
Config uses: preprocess_4
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 2](#context-s04)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: collate_batch.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s04 — Dataset và batch — phần việc 2:
[Dán tóm tắt s04]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s05"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=12; lines=1-17
Target: baseline.ipynb#cell=12; lines=1-41
Config owns: architecture
Config uses: preprocess_5
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 3](#context-s05)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; attention giúp từng token context đọc phần question liên quan. Toàn bộ mô hình học từ đầu, không dùng BERT/tokenizer pretrained. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s05 — Dataset và batch — phần việc 3:
[Dán tóm tắt s05]
```

### Code gửi kèm

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

### Tóm tắt context để dùng tiếp

<a id="context-s06"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
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
Source: baseline_basic.ipynb#cell=5; lines=36-43
Config owns: training_config
Config uses: preprocess_5, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch — phần việc 3](#context-s05)
- [Model và giao diện sử dụng](#context-s06)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng.
Phương pháp: Chọn word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; Metric: exact_match. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s05 — Dataset và batch — phần việc 3:
[Dán tóm tắt s05]

Context s06 — Model và giao diện sử dụng:
[Dán tóm tắt s06]
```

### Code gửi kèm

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
