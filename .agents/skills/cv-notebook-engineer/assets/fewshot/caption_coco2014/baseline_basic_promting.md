# Prompt basic: COCO2014: ResNet50 và attention LSTM captioning

<!-- prompt-guide
Format: 3
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: open_rgb(value)

<!-- module
ID: m007
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=6; lines=5-15
Source: baseline_basic.ipynb#cell=6; lines=5-15
-->

<a id="context-m007"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện open_rgb(value) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

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

## Bước 2: coco_records(image_root, annotation_file, prefix)

<!-- module
ID: m011
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=2-17
Source: baseline_basic.ipynb#cell=8; lines=2-17
-->

<a id="context-m011"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện coco_records(image_root, annotation_file, prefix) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def coco_records(image_root, annotation_file, prefix):
    if not image_root.is_dir() or not annotation_file.is_file():
        raise FileNotFoundError(
            f"Thiếu {image_root} hoặc {annotation_file}. CocoCaptions chỉ đọc dữ liệu. "
            "Tải train2014.zip, val2014.zip và annotations_trainval2014.zip từ cocodataset.org/#download, "
            "giải nén vào DATA_ROOT để có train2014/, val2014/, annotations/. Cần pycocotools.")
    dataset = CocoCaptions(root=str(image_root), annFile=str(annotation_file))
    rows = []
    for image_id in sorted(dataset.ids):
        info = dataset.coco.loadImgs(image_id)[0]
        path = image_root / info["file_name"]
        captions = [ann["caption"] for ann in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=image_id))]
        if not path.is_file() or not captions or any(not isinstance(c,str) or not c.strip() for c in captions):
            raise ValueError(f"Ảnh/caption không đầy đủ: {path}; không bỏ mẫu âm thầm.")
        rows.append({"id":f"{prefix}:{image_id}","image_id":int(image_id),"image":str(path),"refs":captions})
    return rows
```

## Bước 3: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m012
Depends: m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=19-34
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_ROOT, SEED, VALIDATION_IMAGE_COUNT
Config uses: DATA_ROOT, SEED, VALIDATION_IMAGE_COUNT
Source: baseline_basic.ipynb#cell=8; lines=19-34
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [coco_records(image_root, annotation_file, prefix)](#context-m011)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: MS COCO Captions 2014. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: DATA_ROOT=Path('data/coco2014'); SEED=42. Tự chọn VALIDATION_IMAGE_COUNT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m011 — coco_records(image_root, annotation_file, prefix):
[Dán tóm tắt m011]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_records = coco_records(DATA_ROOT/"train2014",DATA_ROOT/"annotations/captions_train2014.json","train2014")
val_pool = coco_records(DATA_ROOT/"val2014",DATA_ROOT/"annotations/captions_val2014.json","val2014")
if {r["image_id"] for r in train_records} & {r["image_id"] for r in val_pool}:
    raise ValueError("Image ID trùng giữa train2014 và val2014.")
if len(val_pool)<=VALIDATION_IMAGE_COUNT:
    raise ValueError("val2014 không đủ ảnh để tách validation và held-out; kiểm tra dữ liệu hoặc sửa VALIDATION_IMAGE_COUNT thủ công.")
order=list(range(len(val_pool)))
random.Random(SEED).shuffle(order)
val_ids=set(order[:VALIDATION_IMAGE_COUNT])
val_records=[r for i,r in enumerate(val_pool) if i in val_ids]
test_records=[r for i,r in enumerate(val_pool) if i not in val_ids]
SPLIT_FINGERPRINTS={name:fingerprint([(r["id"],r["refs"]) for r in records])
                    for name,records in (("train",train_records),("validation",val_records),("held_out_val2014",test_records))}
print("Số ảnh train/validation/held-out:",len(train_records),len(val_records),len(test_records))
print("Fingerprint:",SPLIT_FINGERPRINTS)
print("Held-out là một phần val2014 có nhãn, không phải COCO test server hoặc Karpathy split.")
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m011 — coco_records(image_root, annotation_file, prefix):
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 4: Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…

<!-- module
ID: m013
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=1-1
Source: baseline_basic.ipynb#cell=10; lines=1-1
-->

<a id="context-m013"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

## Bước 5: tokenize(text)

<!-- module
ID: m014
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=2-3
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=10; lines=2-3
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung tokenize(text). Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def tokenize(text):
    return str(text).split()
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện tokenize(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của tokenize(text) tại đây.
```

## Bước 6: Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…

<!-- module
ID: m015
Depends: m012, m013, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=4-13
Summary budget: input=1200; output=500; reserve=300
Config owns: IMAGE_SIZE, MAX_TOKENS, MIN_WORD_FREQUENCY, VOCAB_SIZE
Config uses: IMAGE_SIZE, MAX_TOKENS, MIN_WORD_FREQUENCY, VOCAB_SIZE
Source: baseline_basic.ipynb#cell=10; lines=4-13
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m012)
2. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
3. [tokenize(text)](#context-m014)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…. Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Tự chọn IMAGE_SIZE, MAX_TOKENS, MIN_WORD_FREQUENCY, VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
counts=Counter(token for r in train_records for text in r["refs"] for token in tokenize(text))
words=sorted((w for w,c in counts.items() if c>=MIN_WORD_FREQUENCY),key=lambda w:(-counts[w],w))[:VOCAB_SIZE-4]
vocab={"<PAD>":PAD_ID,"<UNK>":UNK_ID,"<BOS>":BOS_ID,"<EOS>":EOS_ID}
vocab.update({word:i+4 for i,word in enumerate(words)})
id_to_word={i:word for word,i in vocab.items()}
TOKENIZER_STATE={"vocab":vocab,"tokenizer":"whitespace words, train-only vocabulary",
                 "pad":PAD_ID,"unk":UNK_ID,"bos":BOS_ID,"eos":EOS_ID}
PREPROCESS_CONFIG={"image_size":IMAGE_SIZE,"image_geometry":"resize trực tiếp, RGB 0..1",
                    "image_norm":"RGB 0..1","max_train_tokens":MAX_TOKENS,
                    "max_decode_tokens":MAX_TOKENS,"decode":"greedy, dừng EOS"}
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts… tại đây.
```

## Bước 7: restore_tokenizer(state)

<!-- module
ID: m016
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=14-17
Source: baseline_basic.ipynb#cell=10; lines=14-17
-->

<a id="context-m016"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện restore_tokenizer(state) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def restore_tokenizer(state):
    global vocab,id_to_word
    vocab=state["vocab"]
    id_to_word={i:word for word,i in vocab.items()}
```

## Bước 8: image_tensor(value)

<!-- module
ID: m017
Depends: m007, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=19-27
Summary budget: input=1200; output=500; reserve=300
Config uses: IMAGE_SIZE
Source: baseline_basic.ipynb#cell=10; lines=19-21
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [open_rgb(value)](#context-m007)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung image_tensor(value). Nâng image_tensor thành letterbox RGB nền trắng 256 giữ toàn ảnh, rồi chuẩn ImageNet. Dùng lại IMAGE_SIZE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def image_tensor(value):
    image=open_rgb(value).resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR)
    return TF.to_tensor(image)
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện image_tensor(value) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — open_rgb(value):
[Dán tóm tắt m007]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của image_tensor(value) tại đây.
```

## Bước 9: CaptionDataset.__init__(self, records, training=False)

<!-- module
ID: m018
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=30-34
Source: baseline_basic.ipynb#cell=10; lines=24-28
-->

<a id="context-m018"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện CaptionDataset.__init__(self, records, training=False) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self,records,training=False):
        self.records=records
        self.training=training
        # Mọi caption train được dùng; eval mỗi ảnh đúng một lần với đầy đủ references.
        self.index=[(i,j) for i,r in enumerate(records) for j in range(len(r["refs"]))] if training else [(i,0) for i in range(len(records))]
```

## Bước 10: CaptionDataset.__len__(self)

<!-- module
ID: m019
Depends: m018
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=35-36
Source: baseline_basic.ipynb#cell=10; lines=29-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionDataset.__init__(self, records, training=False)](#context-m018)

<a id="context-m019"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện CaptionDataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):
        return len(self.index)
```

## Bước 11: CaptionDataset.__getitem__(self, index)

<!-- module
ID: m020
Depends: m013, m014, m015, m017, m018
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=37-42
Config uses: MAX_TOKENS
Source: baseline_basic.ipynb#cell=10; lines=31-36
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [tokenize(text)](#context-m014)
3. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
4. [image_tensor(value)](#context-m017)
5. [CaptionDataset.__init__(self, records, training=False)](#context-m018)

<a id="context-m020"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện CaptionDataset.__getitem__(self, index) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m014 — tokenize(text):
[Dán tóm tắt m014]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __getitem__(self,index):
        i,j=self.index[index]
        record=self.records[i]
        tokens=tokenize(record["refs"][j])[:MAX_TOKENS-1]
        target=[BOS_ID]+[vocab.get(w,UNK_ID) for w in tokens]+[EOS_ID]
        return image_tensor(record["image"]),torch.tensor(target),record
```

## Bước 12: Ghép giao diện class CaptionDataset

<!-- module
ID: m021
Depends: m018, m019, m020
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=10; lines=29-42
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionDataset.__init__(self, records, training=False)](#context-m018)
2. [CaptionDataset.__len__(self)](#context-m019)
3. [CaptionDataset.__getitem__(self, index)](#context-m020)

<a id="context-m021"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class CaptionDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m018 — CaptionDataset.__init__(self, records, training=False):
[Dán tóm tắt m018]

Context m019 — CaptionDataset.__len__(self):
[Dán tóm tắt m019]

Context m020 — CaptionDataset.__getitem__(self, index):
[Dán tóm tắt m020]
```

## Bước 13: collate_batch(items)

<!-- module
ID: m022
Depends: m013
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=44-49
Source: baseline_basic.ipynb#cell=10; lines=38-43
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)

<a id="context-m022"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện collate_batch(items) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def collate_batch(items):
    width=max(len(target) for _,target,_ in items)
    tokens=torch.full((len(items),width),PAD_ID,dtype=torch.long)
    for i,(_,target,_) in enumerate(items):tokens[i,:len(target)]=target
    return {"images":torch.stack([image for image,_,_ in items]),"tokens":tokens,
            "ids":[r["id"] for _,_,r in items],"records":[r for _,_,r in items]}
```

## Bước 14: to_device(batch)

<!-- module
ID: m008
Depends: m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=17-19
Summary budget: input=1200; output=500; reserve=300
Config owns: DEVICE
Config uses: DEVICE
Source: baseline_basic.ipynb#cell=6; lines=17-19
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def to_device(batch):
    return {key: value.to(DEVICE) if torch.is_tensor(value) and key not in ("lengths", "widths", "target_lengths") else value
            for key, value in batch.items()}
```

### Tóm tắt context để dùng tiếp

<a id="context-m008"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 15: make_loader(dataset, train=False)

<!-- module
ID: m009
Depends: m012, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=21-24
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
Source: baseline_basic.ipynb#cell=6; lines=21-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m012)
2. [collate_batch(items)](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: EVAL_BATCH_SIZE được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def make_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train, num_workers=0, collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 16: Tạo DataLoader cho các split

<!-- module
ID: m023
Depends: m009, m012, m014, m015, m021
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=51-55
Config uses: MAX_TOKENS
Source: baseline_basic.ipynb#cell=10; lines=45-49
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m009)
2. [Chuẩn hóa và chia dữ liệu](#context-m012)
3. [tokenize(text)](#context-m014)
4. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
5. [Ghép giao diện class CaptionDataset](#context-m021)

<a id="context-m023"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m009 — make_loader(dataset, train=False):
[Dán tóm tắt m009]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m014 — tokenize(text):
[Dán tóm tắt m014]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m021 — Ghép giao diện class CaptionDataset:
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_loader=make_loader(CaptionDataset(train_records,training=True),train=True)
val_loader=make_loader(CaptionDataset(val_records))
test_loader=make_loader(CaptionDataset(test_records))
truncated=sum(len(tokenize(text))>MAX_TOKENS-1 for r in train_records for text in r["refs"])
print("Vocabulary train:",len(vocab),"; caption train bị rút ngắn:",truncated,"; gold đánh giá giữ đầy đủ.")
```

## Bước 17: CaptionModel.__init__(self, cfg, initialize_pretrained=True)

<!-- module
ID: m025
Depends: m013, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=3-20
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM, MODEL_CONFIG
Config uses: DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM
Source: baseline_basic.ipynb#cell=12; lines=2-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung CaptionModel.__init__(self, cfg, initialize_pretrained=True). Thay CNN bằng ResNet50 ImageNet V2, chỉ fine-tune layer4 và giữ BatchNorm eval. Bổ sung embedding, additive attention, initial_h/c, LSTMCell và output head vào CaptionModel. cfg vocab_size,embed_dim256,hidden_dim512,memory_dim256,dropout0.3; decoder input embed+memory; output Linear(hidden+memory,V); memory_attention,hidden_attention,attention_score. Tự chọn DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa vocab_size, embed_dim, hidden_dim, memory_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
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
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện CaptionModel.__init__(self, cfg, initialize_pretrained=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của CaptionModel.__init__(self, cfg, initialize_pretrained=True) tại đây.
```

## Bước 18: CaptionModel.train(self, mode=True)

<!-- module
ID: m026
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=21-25
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.train(self, mode=True). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện CaptionModel.train(self, mode=True) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.train(self, mode=True) tại đây.
```

## Bước 19: CaptionModel.encode(self, images)

<!-- module
ID: m027
Depends: m022, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=26-30
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.encode(self, images). Viết encode lấy feature map ResNet thành spatial memory và start_state từ trung bình memory. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện CaptionModel.encode(self, images) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.encode(self, images) tại đây.
```

## Bước 20: CaptionModel.start_state(self, memory)

<!-- module
ID: m028
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=31-33
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.start_state(self, memory). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m028"></a>

```text
Tóm tắt giao diện CaptionModel.start_state(self, memory) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.start_state(self, memory) tại đây.
```

## Bước 21: CaptionModel.step(self, token, state, memory)

<!-- module
ID: m029
Depends: m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=34-40
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Viết CaptionModel.step(self, token, state, memory). Viết step chọn spatial context bằng additive attention rồi cập nhật LSTMCell và logits. Giữ thứ tự đầu ra: logits, kết quả 2. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m029"></a>

```text
Tóm tắt giao diện CaptionModel.step(self, token, state, memory) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của CaptionModel.step(self, token, state, memory) tại đây.
```

## Bước 22: CaptionModel.forward(self, images, tokens)

<!-- module
ID: m030
Depends: m022, m025, m027, m028, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=41-45
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=12; lines=12-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m022)
2. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
3. [CaptionModel.encode(self, images)](#context-m027)
4. [CaptionModel.start_state(self, memory)](#context-m028)
5. [CaptionModel.step(self, token, state, memory)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung CaptionModel.forward(self, images, tokens). Viết forward(images,tokens) dùng encode, start_state và step theo từng target token để teacher forcing. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def forward(self,images,tokens):
        hidden=self.initial(self.visual(images)).tanh(); outputs=[]
        for t in range(tokens.size(1)):
            hidden=self.decoder(self.embedding(tokens[:,t]),hidden)
            outputs.append(self.output(self.dropout(hidden)))
        return torch.stack(outputs,1)
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện CaptionModel.forward(self, images, tokens) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của CaptionModel.forward(self, images, tokens) tại đây.
```

## Bước 23: CaptionModel.generate(self, images, max_tokens)

<!-- module
ID: m031
Depends: m013, m022, m025, m027, m028, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=46-57
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=12; lines=18-29
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [collate_batch(items)](#context-m022)
3. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
4. [CaptionModel.encode(self, images)](#context-m027)
5. [CaptionModel.start_state(self, memory)](#context-m028)
6. [CaptionModel.step(self, token, state, memory)](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung CaptionModel.generate(self, images, max_tokens). Thay generate bằng greedy decoding dùng step, chặn PAD/BOS và dừng EOS độc lập từng ảnh. Viết method trong class CaptionModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
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

<a id="context-m031"></a>

```text
Tóm tắt giao diện CaptionModel.generate(self, images, max_tokens) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của CaptionModel.generate(self, images, max_tokens) tại đây.
```

## Bước 24: Ghép giao diện class CaptionModel

<!-- module
ID: m032
Depends: m025, m026, m027, m028, m029, m030, m031
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=2-57
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
2. [CaptionModel.train(self, mode=True)](#context-m026)
3. [CaptionModel.encode(self, images)](#context-m027)
4. [CaptionModel.start_state(self, memory)](#context-m028)
5. [CaptionModel.step(self, token, state, memory)](#context-m029)
6. [CaptionModel.forward(self, images, tokens)](#context-m030)
7. [CaptionModel.generate(self, images, max_tokens)](#context-m031)

<a id="context-m032"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class CaptionModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m026 — CaptionModel.train(self, mode=True):
[Dán tóm tắt m026]

Context m027 — CaptionModel.encode(self, images):
[Dán tóm tắt m027]

Context m028 — CaptionModel.start_state(self, memory):
[Dán tóm tắt m028]

Context m029 — CaptionModel.step(self, token, state, memory):
[Dán tóm tắt m029]

Context m030 — CaptionModel.forward(self, images, tokens):
[Dán tóm tắt m030]

Context m031 — CaptionModel.generate(self, images, max_tokens):
[Dán tóm tắt m031]
```

## Bước 25: build_model(cfg, initialize_pretrained=True)

<!-- module
ID: m033
Depends: m032
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=12; lines=58-59
Source: baseline_basic.ipynb#cell=12; lines=30-31
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class CaptionModel](#context-m032)

<a id="context-m033"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện build_model(cfg, initialize_pretrained=True) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_model(cfg,initialize_pretrained=True):
    return CaptionModel(cfg,initialize_pretrained)
```

## Bước 26: decode_tokens(ids)

<!-- module
ID: m035
Depends: m013, m015
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=1-6
Source: baseline_basic.ipynb#cell=15; lines=1-6
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)

<a id="context-m035"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện decode_tokens(ids) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def decode_tokens(ids):
    result=[]
    for index in ids:
        if int(index)==EOS_ID:break
        if int(index) not in (PAD_ID,BOS_ID):result.append(id_to_word[int(index)])
    return " ".join(result)
```

## Bước 27: compute_loss(model, batch)

<!-- module
ID: m036
Depends: m013, m022, m032
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=7-10
Source: baseline_basic.ipynb#cell=15; lines=7-10
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…](#context-m013)
2. [collate_batch(items)](#context-m022)
3. [Ghép giao diện class CaptionModel](#context-m032)

<a id="context-m036"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện compute_loss(model, batch) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m013 — Tokenizer, ảnh và Dataset — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m013]

Context m022 — collate_batch(items):
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def compute_loss(model,batch):
    tokens=batch["tokens"]
    logits=model(batch["images"],tokens[:,:-1])
    return nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),tokens[:,1:].reshape(-1),ignore_index=PAD_ID)
```

## Bước 28: ptb_tokenize(mapping)

<!-- module
ID: m038
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=5-22
Summary budget: input=1200; output=500; reserve=300
Config owns: JAVA_EXECUTABLE
Config uses: JAVA_EXECUTABLE
Source: baseline_basic.ipynb#cell=16; lines=5-22
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung ptb_tokenize(mapping). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn JAVA_EXECUTABLE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def ptb_tokenize(mapping):
    executable=shutil.which(JAVA_EXECUTABLE)
    if executable is None and Path(JAVA_EXECUTABLE).is_file():executable=JAVA_EXECUTABLE
    if executable is None:
        raise RuntimeError("CIDEr chuẩn COCO cần Java cho PTBTokenizer. Cài JRE/JDK và đặt JAVA_EXECUTABLE "
                           "thành java hoặc đường dẫn java.exe; không đổi tokenizer/metric tự động.")
    keys=[key for key,captions in mapping.items() for _ in captions]
    sentences="\n".join(text.replace("\n"," ") for captions in mapping.values() for text in captions)
    jar=Path(ptb_module.__file__).parent/ptb_module.STANFORD_CORENLP_3_4_1_JAR
    completed=subprocess.run([str(executable),"-cp",str(jar),"edu.stanford.nlp.process.PTBTokenizer",
                              "-preserveLines","-lowerCase"],input=sentences.encode("utf-8"),
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    lines=completed.stdout.decode("utf-8").splitlines()
    if len(lines)!=len(keys):raise ValueError("PTBTokenizer trả sai số dòng.")
    result=defaultdict(list)
    for key,line in zip(keys,lines):
        result[key].append(" ".join(w for w in line.split() if w not in ptb_module.PUNCTUATIONS))
    return dict(result)
```

### Tóm tắt context để dùng tiếp

<a id="context-m038"></a>

```text
Tóm tắt giao diện ptb_tokenize(mapping) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của ptb_tokenize(mapping) tại đây.
```

## Bước 29: score_corpus(records, predictions)

<!-- module
ID: m039
Depends: m038
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=24-29
Source: baseline_basic.ipynb#cell=16; lines=24-29
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [ptb_tokenize(mapping)](#context-m038)

<a id="context-m039"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện score_corpus(records, predictions) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m038 — ptb_tokenize(mapping):
[Dán tóm tắt m038]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def score_corpus(records,predictions):
    gold=ptb_tokenize({r["id"]:r["refs"] for r in records})
    guessed=ptb_tokenize({r["id"]:[text] for r,text in zip(records,predictions)})
    cider,per_image=Cider(n=4,sigma=6.0).compute_score(gold,guessed)
    return {"cider":float(cider),"image_count":len(records),"tokenizer":"COCO Stanford PTB lowercase",
            "scale":"pycocoevalcap raw, không nhân 100"}
```

## Bước 30: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m040
Depends: m008, m015, m032, m035, m039
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=17; lines=1-11
Config uses: MAX_TOKENS
Source: baseline_basic.ipynb#cell=17; lines=1-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
3. [Ghép giao diện class CaptionModel](#context-m032)
4. [decode_tokens(ids)](#context-m035)
5. [score_corpus(records, predictions)](#context-m039)

<a id="context-m040"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]

Context m035 — decode_tokens(ids):
[Dán tóm tắt m035]

Context m039 — score_corpus(records, predictions):
[Dán tóm tắt m039]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def evaluate(model,loader,return_predictions=False):
    model.eval(); records=[]; predictions=[]
    for cpu_batch in loader:
        batch=to_device(cpu_batch)
        output=model.generate(batch["images"],MAX_TOKENS).cpu().tolist()
        records.extend(batch["records"]); predictions.extend(decode_tokens(ids) for ids in output)
    if not records:raise ValueError("Không có ảnh đánh giá.")
    metrics=score_corpus(records,predictions)
    rows=[{"id":r["id"],"prediction":p,"references":r["refs"]} for r,p in zip(records,predictions)]
    return (metrics,rows) if return_predictions else metrics
```

## Bước 31: make_optimizer(model)

<!-- module
ID: m041
Depends: m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=3-11
Summary budget: input=1200; output=500; reserve=300
Config owns: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
Config uses: BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY
Source: baseline_basic.ipynb#cell=19; lines=3-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class CaptionModel](#context-m032)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_optimizer(model). Khởi tạo optimizer AdamW cho tham số có thể học. Tách nhóm backbone và task head theo cấu trúc model. Tự chọn BACKBONE_LEARNING_RATE, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

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

### Tóm tắt context để dùng tiếp

<a id="context-m041"></a>

```text
Tóm tắt giao diện make_optimizer(model) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — Ghép giao diện class CaptionModel:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của make_optimizer(model) tại đây.
```

## Bước 32: Huấn luyện và checkpoint — best_score, model, optimizer…

<!-- module
ID: m042
Depends: m008, m025, m033, m041
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=1-6
Summary budget: input=1200; output=500; reserve=300
Config owns: DIRECTION, EPOCHS
Config uses: DEVICE, DIRECTION, EPOCHS, MODEL_CONFIG
Source: baseline_basic.ipynb#cell=20; lines=1-6
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
3. [build_model(cfg, initialize_pretrained=True)](#context-m033)
4. [make_optimizer(model)](#context-m041)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Huấn luyện và checkpoint — best_score, model, optimizer…. Giữ ràng buộc: DIRECTION='max'. Tự chọn EPOCHS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
model = build_model(MODEL_CONFIG, initialize_pretrained=True).to(DEVICE)
print(f"Tham số học: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
optimizer = make_optimizer(model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_score = math.inf if DIRECTION == "min" else -math.inf
stale_epochs = 0
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, model, optimizer… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m041 — make_optimizer(model):
[Dán tóm tắt m041]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — best_score, model, optimizer… tại đây.
```

## Bước 33: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m043
Depends: m008, m015, m023, m036, m040, m042
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=7-44
Summary budget: input=1200; output=500; reserve=300
Config owns: ARTIFACT_DIR, CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC, VARIANT
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DIRECTION, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC, VARIANT
Source: baseline_basic.ipynb#cell=20; lines=7-44
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
3. [Tạo DataLoader cho các split](#context-m023)
4. [compute_loss(model, batch)](#context-m036)
5. [evaluate(model, loader, return_predictions=False)](#context-m040)
6. [Huấn luyện và checkpoint — best_score, model, optimizer…](#context-m042)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='cider'. Tự chọn GRAD_CLIP, PATIENCE, VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại DIRECTION, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m036 — compute_loss(model, batch):
[Dán tóm tắt m036]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m042 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m042]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, seen = 0.0, 0
    for cpu_batch in tqdm(train_loader, desc="Huấn luyện", leave=False):
        batch = to_device(cpu_batch)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss không hữu hạn; kiểm tra nhãn, mask và cấu hình cố định.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        count = len(batch["ids"])
        total += loss.item() * count
        seen += count
    if not seen:
        raise ValueError("Tập huấn luyện rỗng.")
    loss = total / seen
    metrics = evaluate(model, val_loader)
    score = float(metrics[PRIMARY_METRIC])
    if not math.isfinite(score):
        raise ValueError("Metric validation không hữu hạn.")
    scheduler.step()
    print(f"Epoch {epoch:02d}: loss={loss:.4f}; validation={metrics}")
    improved = score < best_score if DIRECTION == "min" else score > best_score
    if improved:
        best_score, stale_epochs = score, 0
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "tokenizer": TOKENIZER_STATE,
        }, CHECKPOINT_PATH)
        print("Đã lưu checkpoint tốt nhất theo primary metric validation.")
    else:
        stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"Dừng sớm sau {PATIENCE} epoch không cải thiện.")
            break
```

### Tóm tắt context để dùng tiếp

<a id="context-m043"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m036 — compute_loss(model, batch):
[Dán tóm tắt m036]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m042 — Huấn luyện và checkpoint — best_score, model, optimizer…:
[Dán tóm tắt m042]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 34: Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…

<!-- module
ID: m044
Depends: m008, m012, m015, m016, m023, m025, m033, m040, m043
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=1-18
Summary budget: input=1200; output=500; reserve=300
Config owns: SOURCE
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM, SOURCE
Source: baseline_basic.ipynb#cell=22; lines=1-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Chuẩn hóa và chia dữ liệu](#context-m012)
3. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
4. [restore_tokenizer(state)](#context-m016)
5. [Tạo DataLoader cho các split](#context-m023)
6. [CaptionModel.__init__(self, cfg, initialize_pretrained=True)](#context-m025)
7. [build_model(cfg, initialize_pretrained=True)](#context-m033)
8. [evaluate(model, loader, return_predictions=False)](#context-m040)
9. [Vòng huấn luyện, validation và lưu checkpoint](#context-m043)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Khai báo cùng phần này: SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, MEMORY_DIM từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m016 — restore_tokenizer(state):
[Dán tóm tắt m016]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m043 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m043]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
assert checkpoint["tokenizer"] == TOKENIZER_STATE
restore_tokenizer(checkpoint["tokenizer"])
# Không tải backbone lần nữa: state_dict chứa đầy đủ trọng số visual và head.
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"vocab_size":len(vocab),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"memory_dim":MEMORY_DIM,"dropout":DROPOUT}
best_model = build_model(inference_config, initialize_pretrained=False).to(DEVICE)
best_model.load_state_dict(checkpoint["model_state"])
best_model.eval()
validation_metrics = evaluate(best_model, val_loader)
test_metrics, predictions = evaluate(best_model, test_loader, return_predictions=True)
results = {"epoch": checkpoint["epoch"], "source": SOURCE, "split_fingerprints": SPLIT_FINGERPRINTS,
           "validation": validation_metrics, "held_out_test": test_metrics}
(ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Đã nạp epoch {checkpoint['epoch']}.")
print("Validation:", validation_metrics)
print("Held-out:", test_metrics)
```

### Tóm tắt context để dùng tiếp

<a id="context-m044"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m016 — restore_tokenizer(state):
[Dán tóm tắt m016]

Context m023 — Tạo DataLoader cho các split:
[Dán tóm tắt m023]

Context m025 — CaptionModel.__init__(self, cfg, initialize_pretrained=True):
[Dán tóm tắt m025]

Context m033 — build_model(cfg, initialize_pretrained=True):
[Dán tóm tắt m033]

Context m040 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m040]

Context m043 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m043]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… tại đây.
```

## Bước 35: Tải lại và đánh giá held-out

<!-- module
ID: m045
Depends: m044
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=22; lines=19-19
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m044)

### Prompt tạo hoặc sửa module

```text
Viết Tải lại và đánh giá held-out. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện Tải lại và đánh giá held-out từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của Tải lại và đánh giá held-out tại đây.
```

## Bước 36: describe_image(image)

<!-- module
ID: m046
Depends: m008, m015, m017, m035, m044
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=24; lines=1-5
Config uses: DEVICE, MAX_TOKENS
Source: baseline_basic.ipynb#cell=24; lines=1-5
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m008)
2. [Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…](#context-m015)
3. [image_tensor(value)](#context-m017)
4. [decode_tokens(ids)](#context-m035)
5. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m044)

<a id="context-m046"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện describe_image(image) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m008 — to_device(batch):
[Dán tóm tắt m008]

Context m015 — Tokenizer, ảnh và Dataset — PREPROCESS_CONFIG, TOKENIZER_STATE, counts…:
[Dán tóm tắt m015]

Context m017 — image_tensor(value):
[Dán tóm tắt m017]

Context m035 — decode_tokens(ids):
[Dán tóm tắt m035]

Context m044 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m044]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def describe_image(image):
    best_model.eval()
    output=best_model.generate(image_tensor(image).unsqueeze(0).to(DEVICE),MAX_TOKENS)
    return decode_tokens(output[0].tolist())
```

## Bước 37: Mô tả ảnh mới

<!-- module
ID: m047
Depends: m012, m046
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=24; lines=7-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m012)
2. [describe_image(image)](#context-m046)

### Prompt tạo hoặc sửa module

```text
Viết Mô tả ảnh mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m046 — describe_image(image):
[Dán tóm tắt m046]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện Mô tả ảnh mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m012]

Context m046 — describe_image(image):
[Dán tóm tắt m046]
```

```python
# Dán code hiện tại của Mô tả ảnh mới tại đây.
```

## Bước 38: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m045, m047
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá held-out](#context-m045)
2. [Mô tả ảnh mới](#context-m047)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m045 — Tải lại và đánh giá held-out:
[Dán tóm tắt m045]

Context m047 — Mô tả ảnh mới:
[Dán tóm tắt m047]
```
