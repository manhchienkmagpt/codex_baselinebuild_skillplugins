# Prompt basic: Dịch máy TED PT→EN — bản tối ưu

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
Source: baseline_basic.ipynb#cell=11; lines=4-22
Source: baseline_basic.ipynb#cell=11; lines=29-31
Source: baseline_basic.ipynb#cell=11; lines=56-56
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=11; lines=3-13
Target: baseline.ipynb#cell=11; lines=19-21
Target: baseline.ipynb#cell=11; lines=46-50
Config owns: preprocess_1
-->

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn. Phạm vi sửa: clean_text, BasicWordTokenizer, restore_tokenizer.
Ý tưởng tiền xử lí: 1. Giữ đúng cặp pt/en và split corpus, lọc cặp rỗng; NFC và gộp khoảng trắng nhưng giữ case/dấu câu. Loại trùng source chéo split theo giao thức đã nêu trước khi fit. 2. Fit chung BPE SentencePiece 8.000 cho source/target chỉ bằng train, có byte fallback. Chia đơn vị dưới từ giúp biểu diễn biến thể hình thái và từ hiếm mà không dùng tokenizer pretrained. 3. Cap source/target/decoding ở 128 token, thêm BOS/EOS/PAD đúng vị trí. Gold dùng chấm BLEU/chrF++ vẫn đầy đủ; theo dõi phần bị cap để hiểu giới hạn model. 4. Mask PAD ở source/target và dùng causal mask cho decoder để không nhìn token tương lai. Loss bỏ PAD; inference beam chỉ dựa source và xác suất model. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
Nguồn: {"dataset":"TED HRLR Translate pt_to_en","tfds_builder":"ted_hrlr_translate/pt_to_en:1.0.0","recovery_dataset":"neulab/ted_hrlr","recovery_revision":"cf21722111c95aa73c30e13df430032db463736d"}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
def clean_text(text):
    return " ".join(unicodedata.normalize("NFC",text).split())
```

```python
class BasicWordTokenizer:
    """Tách theo khoảng trắng; vocabulary chỉ từ train, không BPE hay pretrained."""
    def __init__(self,vocab):
        self.vocab=dict(vocab)
        self.words={index:word for word,index in self.vocab.items()}
    def encode(self,text,out_type=int):
        return [self.vocab.get(word,UNK_ID) for word in str(text).split()]
    def decode(self,ids):
        return " ".join(self.words.get(int(index),"<UNK>") for index in ids
                        if int(index) not in (PAD_ID,BOS_ID,EOS_ID))
    def get_piece_size(self):
        return len(self.vocab)

counts=Counter(word for record in train_records
               for text in [record["src"],*record["refs"]] for word in text.split())
words=sorted(counts,key=lambda word:(-counts[word],word))[:VOCAB_SIZE-4]
vocab={"<PAD>":PAD_ID,"<UNK>":UNK_ID,"<BOS>":BOS_ID,"<EOS>":EOS_ID}
vocab.update({word:index+4 for index,word in enumerate(words)})
sp=BasicWordTokenizer(vocab)
```

```python
def restore_tokenizer(state):
    global sp
    sp=BasicWordTokenizer(state["vocab"])
```

```python
print("Tokenizer vocab:",sp.get_piece_size(),"; fit chỉ train.")
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
Source: baseline_basic.ipynb#cell=11; lines=57-60
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=11; lines=3-13
Target: baseline.ipynb#cell=11; lines=19-21
Target: baseline.ipynb#cell=11; lines=46-50
Config owns: preprocess_2
Config uses: preprocess_1
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh — phần việc 1](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Giữ đúng cặp pt/en và split corpus, lọc cặp rỗng; NFC và gộp khoảng trắng nhưng giữ case/dấu câu. Loại trùng source chéo split theo giao thức đã nêu trước khi fit. 2. Fit chung BPE SentencePiece 8.000 cho source/target chỉ bằng train, có byte fallback. Chia đơn vị dưới từ giúp biểu diễn biến thể hình thái và từ hiếm mà không dùng tokenizer pretrained. 3. Cap source/target/decoding ở 128 token, thêm BOS/EOS/PAD đúng vị trí. Gold dùng chấm BLEU/chrF++ vẫn đầy đủ; theo dõi phần bị cap để hiểu giới hạn model. 4. Mask PAD ở source/target và dùng causal mask cho decoder để không nhìn token tương lai. Loss bỏ PAD; inference beam chỉ dựa source và xác suất model. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Biểu diễn và căn chỉnh — phần việc 1:
[Dán tóm tắt s01]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
for name,records in (("train",train_records),("validation",val_records),("test",test_records)):
    source_cut=sum(len(sp.encode(r["src"]))>MAX_SOURCE_TOKENS for r in records)
    target_cut=sum(len(sp.encode(ref))>MAX_TARGET_TOKENS-2 for r in records for ref in r["refs"])
    print(f"{name}: source quá cap={source_cut}; target quá cap train={target_cut}; gold metric giữ đầy đủ.")
```

### Tóm tắt context để dùng tiếp

<a id="context-s02"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 3: Tiền xử lí dữ liệu — Dataset và batch

<!-- module
ID: s03
Group: preprocessing
Kind: code
Depends: s02
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=6; lines=43-46
Source: baseline_basic.ipynb#cell=11; lines=33-55
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=11; lines=23-45
Config owns: preprocess_3
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Biểu diễn và căn chỉnh — phần việc 2](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train. Phạm vi sửa: make_loader, PairDataset, collate_batch.
Ý tưởng tiền xử lí: 1. Giữ đúng cặp pt/en và split corpus, lọc cặp rỗng; NFC và gộp khoảng trắng nhưng giữ case/dấu câu. Loại trùng source chéo split theo giao thức đã nêu trước khi fit. 2. Fit chung BPE SentencePiece 8.000 cho source/target chỉ bằng train, có byte fallback. Chia đơn vị dưới từ giúp biểu diễn biến thể hình thái và từ hiếm mà không dùng tokenizer pretrained. 3. Cap source/target/decoding ở 128 token, thêm BOS/EOS/PAD đúng vị trí. Gold dùng chấm BLEU/chrF++ vẫn đầy đủ; theo dõi phần bị cap để hiểu giới hạn model. 4. Mask PAD ở source/target và dùng causal mask cho decoder để không nhìn token tương lai. Loss bỏ PAD; inference beam chỉ dựa source và xác suất model. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
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

```python
class PairDataset(Dataset):
    def __init__(self,records,training=False):
        self.records=records
        self.index=[(i,j) for i,r in enumerate(records) for j in range(len(r["refs"]))] if training else [(i,0) for i in range(len(records))]
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        i,j=self.index[index];record=self.records[i]
        source=sp.encode(record["src"],out_type=int)[:MAX_SOURCE_TOKENS]
        if not source:source=[UNK_ID]
        target=[BOS_ID]+sp.encode(record["refs"][j],out_type=int)[:MAX_TARGET_TOKENS-2]+[EOS_ID]
        return torch.tensor(source),torch.tensor(target),record

def collate_batch(items):
    lengths=torch.tensor([len(src) for src,_,_ in items])
    source=torch.full((len(items),int(lengths.max())),PAD_ID,dtype=torch.long)
    target=torch.full((len(items),max(len(tgt) for _,tgt,_ in items)),PAD_ID,dtype=torch.long)
    for i,(src,tgt,_) in enumerate(items):source[i,:len(src)]=src;target[i,:len(tgt)]=tgt
    return {"source":source,"target":target,"lengths":lengths,
            "ids":[r["id"] for _,_,r in items],"records":[r for _,_,r in items]}

train_loader=make_loader(PairDataset(train_records,True),True)
val_loader=make_loader(PairDataset(val_records))
test_loader=make_loader(PairDataset(test_records))
```

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 4: Xử lí mô hình — Encoder độc lập

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=13; lines=1-52
Config owns: encoder
Config uses: preprocess_3
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết một encoder hoàn chỉnh cho kiến trúc bên dưới, gồm embedding/backbone và mask/lengths. Xuất memory và trạng thái/shape cần cho attention/head; chưa viết decoder hay training loop.
Ý tưởng mô hình: Chọn Transformer encoder–decoder scratch: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Chỉ dùng kiến trúc PyTorch khởi tạo mới; không text weights/embedding/tokenizer pretrained. Attention toàn chuỗi phù hợp việc đổi trật tự và nối các phần xa nhau; Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size, length_penalty).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]
```

### Code gửi kèm

Phần bổ sung mới; dùng giao diện hiện có và context ở trên.

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 5: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s05
Group: model
Kind: code
Depends: s03, s04
Budget: input=1000; output=600; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Source: baseline_basic.ipynb#cell=13; lines=1-30
Target: baseline.ipynb#cell=13; lines=1-52
Config owns: architecture
Config uses: preprocess_3, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Encoder độc lập](#context-s04)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn Transformer encoder–decoder scratch: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Chỉ dùng kiến trúc PyTorch khởi tạo mới; không text weights/embedding/tokenizer pretrained. Attention toàn chuỗi phù hợp việc đổi trật tự và nối các phần xa nhau; Không dùng pretrained cho tokenizer/embedding/model tác vụ.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Encoder độc lập:
[Dán tóm tắt s04]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
class TextGenerator(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.embedding=nn.Embedding(cfg["vocab_size"],cfg["embed_dim"],padding_idx=PAD_ID)
        self.encoder=nn.RNN(cfg["embed_dim"],cfg["hidden_dim"],batch_first=True)
        self.decoder=nn.RNNCell(cfg["embed_dim"],cfg["hidden_dim"])
        self.dropout=nn.Dropout(cfg["dropout"])
        self.output=nn.Linear(cfg["hidden_dim"],cfg["vocab_size"])
    def encode(self,source,lengths):
        packed=pack_padded_sequence(self.dropout(self.embedding(source)),lengths,batch_first=True,enforce_sorted=False)
        _,hidden=self.encoder(packed)
        return hidden[-1]
    def forward(self,source,lengths,target):
        hidden=self.encode(source,lengths);output=[]
        for t in range(target.size(1)):
            hidden=self.decoder(self.embedding(target[:,t]),hidden)
            output.append(self.output(self.dropout(hidden)).log_softmax(-1))
        return torch.stack(output,1)
    @torch.no_grad()
    def generate(self,source,lengths,max_tokens,beam_size=1,length_penalty=0.):
        hidden=self.encode(source,lengths)
        token=torch.full((len(source),),BOS_ID,device=source.device,dtype=torch.long)
        done=torch.zeros(len(source),device=source.device,dtype=torch.bool);output=[]
        for _ in range(max_tokens):
            hidden=self.decoder(self.embedding(token),hidden)
            logits=self.output(hidden);logits[:,[PAD_ID,BOS_ID]]=-torch.inf
            token=logits.argmax(-1);token=torch.where(done,torch.full_like(token,EOS_ID),token)
            output.append(token);done|=token==EOS_ID
            if bool(done.all()):break
        return torch.stack(output,1)
```

### Tóm tắt context để dùng tiếp

<a id="context-s05"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 6: Siêu tham số

<!-- module
ID: s06
Group: hyperparameters
Kind: code
Depends: s03, s05
Budget: input=1000; output=600; reserve=200
Summary: no
Source: baseline_basic.ipynb#cell=5; lines=27-28
Source: baseline_basic.ipynb#cell=5; lines=38-44
Source: baseline_basic.ipynb#cell=5; lines=47-47
Config owns: training_config
Config uses: preprocess_3, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s05)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Cập nhật constants và các điểm gọi training bị ảnh hưởng; tận dụng vòng train, metric và lưu/nạp hiện có, không viết lại chúng.
Phương pháp: Chọn Transformer encoder–decoder scratch: embedding 256, bốn lớp encoder và bốn lớp decoder, tám attention heads, feed-forward 1.024, positional encoding hình sin và tied input/output embedding. Self-attention kết hợp các token trong cùng câu, cross-attention cho decoder đọc source. Metric: bleu. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s05 — Model và giao diện sử dụng:
[Dán tóm tắt s05]
```

### Code gửi kèm

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32
```

```python
LABEL_SMOOTHING = 0.0
EPOCHS = 45
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
```

```python
COMET_BATCH_SIZE = 8
```
