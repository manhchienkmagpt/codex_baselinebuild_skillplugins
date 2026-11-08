# Prompt basic: Sửa ngữ pháp CoEdIT GEC — bản tối ưu

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
Source: baseline_basic.ipynb#cell=11; lines=4-22
Source: baseline_basic.ipynb#cell=11; lines=29-31
Source: baseline_basic.ipynb#cell=11; lines=56-56
Related source: baseline_basic.ipynb#cell=5; lines=29-29
Related source: baseline_basic.ipynb#cell=8; lines=3-6
Related source: baseline_basic.ipynb#cell=8; lines=14-24
Related source: baseline_basic.ipynb#cell=9; lines=2-5
Related source: baseline_basic.ipynb#cell=9; lines=17-17
Related source: baseline_basic.ipynb#cell=11; lines=2-2
Related source: baseline_basic.ipynb#cell=14; lines=1-1
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=11; lines=3-13
Target: baseline.ipynb#cell=11; lines=19-21
Target: baseline.ipynb#cell=11; lines=46-50
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=5; lines=1-1
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=5; lines=21-21
Reference code: baseline.ipynb#cell=5; lines=24-26
Reference code: baseline.ipynb#cell=5; lines=45-45
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference missing: no
Config owns: preprocess_1
-->

### Nguồn tham khảo

<a id="references-s01"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.
```

Trích báo cáo:

```markdown
1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu.
```

Trích báo cáo:

```markdown
2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer.
```

Trích báo cáo:

```markdown
3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm.
```

Trích báo cáo:

```markdown
4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu.
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
SOURCE = {'row': 4, 'task': 'Grammar Error Correction', 'dataset': 'grammarly/coedit', 'revision': 'e9a255c33ef910bc33a9d2b522653fa87521583e', 'url': 'https://huggingface.co/datasets/grammarly/coedit'}
```

```python
SOURCE["data_mode"] = DATA_MODE
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'spaCy', 'model_id': 'en_core_web_sm', 'revision': '3.8.0', 'training': 'frozen', 'sha256': '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85', 'purpose': 'ERRANT annotation, không tạo task output'}]
VAL_FRACTION = 0.1
```

```python
PRIMARY_METRIC = "errant_f05"
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

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
VOCAB_SIZE = 8000
```

```python
def strip_instruction(text):
    for prefix in sorted(KNOWN_GEC_PREFIXES,key=len,reverse=True):
        if text.startswith(prefix):return clean_text(text[len(prefix):])
    raise ValueError(f"Instruction chưa kiểm tra: {text[:120]!r}; bổ sung prefix đúng nghĩa sau khi kiểm tra, không split mọi dấu hai chấm.")
```

```python
def group_gec(rows):
    groups={};count=0
    for row in rows:
        if row["task"]!="gec":continue
        count+=1;source=strip_instruction(row["src"]);target=clean_text(row["tgt"])
        if not source or not target:raise ValueError("Câu GEC rỗng sau bỏ instruction.")
        record=groups.setdefault(source,{"src":source,"refs":[],"source_ids":[]})
        if target not in record["refs"]:record["refs"].append(target)
        record["source_ids"].append(str(row["_id"]))
    print("Số hàng gec:",count,"; nhóm câu gốc:",len(groups))
    return [groups[key] for key in sorted(groups)]
```

```python
for split,records in native.items():
    for record in records:
        record["src"]=clean_text(record["src"]);record["refs"]=[clean_text(ref) for ref in record["refs"]]
        if not record["src"] or not record["refs"] or any(not r for r in record["refs"]):raise ValueError(f"Mẫu rỗng ở{split}.")
```

```python
train_records=canonical_records(native["train"],"train")
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

```python
MODEL_CONFIG={"vocab_size":sp.get_piece_size(),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
```

Đối chiếu thêm code hiện tại của: cell 11, dòng 23–24; PairDataset; cell 11, dòng 57–60; errant_corpus(records, predictions); gleu_corpus(records, predictions); decode_tokens(ids); cell 22, dòng 2–2; cell 22, dòng 4–4; rewrite_text(text). Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size=1, length_penalty=0.0).

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
Source: baseline_basic.ipynb#cell=11; lines=57-60
Related source: baseline_basic.ipynb#cell=5; lines=30-31
Related source: baseline_basic.ipynb#cell=9; lines=17-19
Related source: baseline_basic.ipynb#cell=11; lines=22-22
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=11; lines=3-13
Target: baseline.ipynb#cell=11; lines=19-21
Target: baseline.ipynb#cell=11; lines=46-50
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=11; lines=2-2
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
Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.
```

Trích báo cáo:

```markdown
1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu.
```

Trích báo cáo:

```markdown
2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer.
```

Trích báo cáo:

```markdown
3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm.
```

Trích báo cáo:

```markdown
4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu.
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
for name,records in (("train",train_records),("validation",val_records),("test",test_records)):
    source_cut=sum(len(sp.encode(r["src"]))>MAX_SOURCE_TOKENS for r in records)
    target_cut=sum(len(sp.encode(ref))>MAX_TARGET_TOKENS-2 for r in records for ref in r["refs"])
    print(f"{name}: source quá cap={source_cut}; target quá cap train={target_cut}; gold metric giữ đầy đủ.")
```

### Code liên quan

Đính kèm cho lượt phân tích cùng nguồn tham khảo và context ở trên. Các đoạn dưới chỉ để đối chiếu, không thuộc phạm vi sửa. Lượt sửa dùng kết quả phân tích; gửi thêm code đối chiếu khi còn điều chưa rõ.

```python
MAX_SOURCE_TOKENS = 160
MAX_TARGET_TOKENS = 160
```

```python
train_records=canonical_records(native["train"],"train")
val_records=canonical_records(native["validation"],"validation")
test_records=canonical_records(native["test"],"test")
```

```python
sp=BasicWordTokenizer(vocab)
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size=1, length_penalty=0.0).

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

## Bước 3: Tiền xử lí dữ liệu — Dataset và batch

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
Source: baseline_basic.ipynb#cell=11; lines=33-55
Related source: baseline_basic.ipynb#cell=5; lines=1-1
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=5; lines=27-31
Related source: baseline_basic.ipynb#cell=9; lines=17-19
Related source: baseline_basic.ipynb#cell=11; lines=2-2
Related source: baseline_basic.ipynb#cell=11; lines=22-22
Related source: baseline_basic.ipynb#cell=20; lines=4-4
Related source: baseline_basic.ipynb#cell=22; lines=8-9
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=11; lines=23-45
Reference report: huong_giai_quyet.md#lines=25-25
Reference report: huong_giai_quyet.md#lines=27-27
Reference report: huong_giai_quyet.md#lines=29-29
Reference report: huong_giai_quyet.md#lines=31-31
Reference report: huong_giai_quyet.md#lines=33-33
Reference report: huong_giai_quyet.md#lines=35-35
Reference code: baseline.ipynb#cell=11; lines=2-2
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
Token là đơn vị model đọc hoặc sinh; BPE có thể chia một từ thành nhiều mảnh nhỏ. Vocabulary là danh sách token đã học từ train; byte fallback giúp mã hóa cả ký tự/từ chưa có dưới dạng token hoàn chỉnh.
```

Trích báo cáo:

```markdown
1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu.
```

Trích báo cáo:

```markdown
2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer.
```

Trích báo cáo:

```markdown
3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm.
```

Trích báo cáo:

```markdown
4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu.
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
VOCAB_SIZE = 8000
MAX_SOURCE_TOKENS = 160
MAX_TARGET_TOKENS = 160
```

```python
train_records=canonical_records(native["train"],"train")
val_records=canonical_records(native["validation"],"validation")
test_records=canonical_records(native["test"],"test")
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

```python
sp=BasicWordTokenizer(vocab)
```

```python
total_steps=max(1,EPOCHS*len(train_loader))
```

```python
validation_metrics=evaluate(best_model,val_loader)
test_metrics,predictions=evaluate(best_model,test_loader,return_predictions=True)
```

Đối chiếu thêm code hiện tại của: cell 20, dòng 10–38. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size=1, length_penalty=0.0).

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

## Bước 4: Xử lí mô hình — Encoder độc lập

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Constraints: no
Summary budget: input=1100; output=500; reserve=200
Target: baseline.ipynb#cell=13; lines=1-63
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference missing: no
Config owns: encoder
Config uses: preprocess_3
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Code liên quan

Giao diện hiện có để nối với module mới: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size=1, length_penalty=0.0).

### Nguồn tham khảo

<a id="references-s04"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **BiGRU encoder + attention cộng + GRU decoder với pointer-copy**, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold.
```

Trích báo cáo:

```markdown
Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Attention giảm nút thắt hidden cuối. Copy vẫn có thể giữ nguyên lỗi, còn decoder có thể sửa quá mức; ERRANT F0.5 validation là tín hiệu lựa chọn. spaCy en_core_web_sm 3.8.0 chỉ là evaluator ERRANT đã được cho phép riêng, không cung cấp feature hay sinh output cho task model.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Prompt tạo hoặc sửa module

Gửi cùng [nguồn tham khảo](#references-s04) và context ở trên.

```text
Tạo module theo ý tưởng:
Viết Encoder độc lập hoàn chỉnh để mô hình chính sử dụng.

Dùng nguồn tham khảo và context đính kèm; giữ tương thích với code hiện tại. Đánh dấu điều chưa rõ.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]
```

### Code cần sửa

Phần bổ sung mới; dùng giao diện hiện có và context ở trên.

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

## Bước 5: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s05
Group: model
Kind: code
Depends: s03, s04
Budget: input=1000; output=600; reserve=200
Summary: yes
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Summary budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=13; lines=1-30
Related source: baseline_basic.ipynb#cell=5; lines=15-16
Related source: baseline_basic.ipynb#cell=11; lines=2-2
Related source: baseline_basic.ipynb#cell=13; lines=31-31
Target: baseline.ipynb#cell=13; lines=1-63
Reference report: huong_giai_quyet.md#lines=39-39
Reference report: huong_giai_quyet.md#lines=41-41
Reference code: baseline.ipynb#cell=5; lines=7-9
Reference code: baseline.ipynb#cell=11; lines=2-2
Reference missing: no
Config owns: architecture
Config uses: preprocess_3, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Encoder độc lập](#context-s04)

### Nguồn tham khảo

<a id="references-s05"></a>

Đính kèm các đoạn nguồn dưới đây cho lượt phân tích hoặc tạo module mới; chatbot tự rút phương pháp và ràng buộc. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu chỉ minh họa phương pháp.

Trích báo cáo:

```markdown
Chọn **BiGRU encoder + attention cộng + GRU decoder với pointer-copy**, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold.
```

Trích báo cáo:

```markdown
Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Attention giảm nút thắt hidden cuối. Copy vẫn có thể giữ nguyên lỗi, còn decoder có thể sửa quá mức; ERRANT F0.5 validation là tín hiệu lựa chọn. spaCy en_core_web_sm 3.8.0 chỉ là evaluator ERRANT đã được cho phép riêng, không cung cấp feature hay sinh output cho task model.
```

Code/cấu hình tham khảo trích từ notebook tối ưu:

```python
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
```

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Prompt trích ràng buộc

<a id="constraints-s05"></a>

Gửi cùng [nguồn tham khảo](#references-s05), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s05) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Triển khai mô hình theo nguồn tham khảo và context hiện có.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Encoder độc lập:
[Dán tóm tắt s04]
```

### Code cần sửa

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
def build_model(cfg):return TextGenerator(cfg)
```

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size=1, length_penalty=0.0).

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

## Bước 6: Siêu tham số

<!-- module
ID: s06
Group: hyperparameters
Kind: code
Depends: s03, s05
Budget: input=1000; output=600; reserve=200
Summary: no
Constraints: yes
Constraints budget: input=1100; output=500; reserve=200
Source: baseline_basic.ipynb#cell=5; lines=27-28
Source: baseline_basic.ipynb#cell=5; lines=38-44
Related source: baseline_basic.ipynb#cell=6; lines=43-46
Related source: baseline_basic.ipynb#cell=17; lines=8-14
Related source: baseline_basic.ipynb#cell=20; lines=3-7
Reference report: huong_giai_quyet.md#lines=49-49
Reference report: huong_giai_quyet.md#lines=67-67
Reference code: baseline.ipynb#cell=5; lines=27-28
Reference code: baseline.ipynb#cell=5; lines=38-45
Reference missing: no
Config owns: training_config
Config uses: preprocess_3, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s05)

### Nguồn tham khảo

<a id="references-s06"></a>

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
EVAL_BATCH_SIZE = 16
```

```python
LABEL_SMOOTHING = 0.0
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "errant_f05"
```

### Prompt trích ràng buộc

<a id="constraints-s06"></a>

Gửi cùng [nguồn tham khảo](#references-s06), code hiện tại và context phụ thuộc.

```text
Đọc code hiện tại và nguồn tham khảo theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Rút ra tên/chữ ký cần giữ và các ràng buộc cần tuân thủ.
Đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Liệt kê ngắn gọn, chưa sửa code.
```

### Prompt tạo hoặc sửa module

Dán kết quả [phân tích code và nguồn](#constraints-s06) vào ô bên dưới; gửi cùng Code cần sửa và context.

```text
Sửa phần “Code cần sửa” theo ý tưởng:
Chọn siêu tham số phù hợp với phương pháp và dữ liệu tham khảo.

Dựa trên phân tích:
[Dán kết quả phân tích.]

Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s05 — Model và giao diện sử dụng:
[Dán tóm tắt s05]
```

### Code cần sửa

Đính kèm các đoạn dưới đây; thay bằng bản hiện tại nếu đã sửa:

```python
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
```

```python
LABEL_SMOOTHING = 0.0
EPOCHS = 40
PATIENCE = 8
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
def compute_loss(model,batch):
    logp=model(batch["source"],batch["lengths"],batch["target"][:,:-1])
    target=batch["target"][:,1:]
    losses=-logp.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    if LABEL_SMOOTHING:losses=(1-LABEL_SMOOTHING)*losses+LABEL_SMOOTHING*(-logp.mean(-1))
    mask=target!=PAD_ID
    return losses[mask].mean()
```

```python
optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,betas=(0.9,0.98),eps=1e-9,weight_decay=WEIGHT_DECAY)
total_steps=max(1,EPOCHS*len(train_loader))
def lr_multiplier(step):
    step+=1
    return min(step/max(1,WARMUP_STEPS),math.sqrt(max(1,WARMUP_STEPS)/max(step,WARMUP_STEPS))) if WARMUP_STEPS else 1.0
```

Đối chiếu thêm code hiện tại của: cell 20, dòng 10–38. Gửi thêm phần liên quan nếu cần, chia theo thành phần hoàn chỉnh khi vượt ngân sách phiên.

Giao diện dùng trong notebook: collate_batch(items) -> batch{ids, lengths, records, source, target}; TextGenerator: __init__(cfg); forward(source, lengths, target); generate(source, lengths, max_tokens, beam_size=1, length_penalty=0.0).
