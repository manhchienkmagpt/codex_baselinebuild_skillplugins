# Prompt basic: SQuAD v1.1: word/char, attention và span decoding

<!-- prompt-guide
Format: 3
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: clean_text(text)

<!-- module
ID: m007
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=6; lines=5-6
Source: baseline_basic.ipynb#cell=6; lines=5-6
-->

<a id="context-m007"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện clean_text(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def clean_text(text):
    return " ".join(unicodedata.normalize("NFC",text).split())
```

## Bước 2: read_jsonl(path)

<!-- module
ID: m008
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=6; lines=8-10
Source: baseline_basic.ipynb#cell=6; lines=8-10
-->

<a id="context-m008"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện read_jsonl(path) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def read_jsonl(path):
    if not path.is_file():raise FileNotFoundError(f"Thiếu {path}; cần JSONL xuất từ đúng nguồn/schema đã ghi.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

## Bước 3: tfds_rows(builder, splits, supervised=False) — phần 1/2

<!-- module
ID: m009
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=12-26
Summary budget: input=1200; output=500; reserve=300
Config owns: ARTIFACT_DIR, CACHE_DIR, VARIANT
Config uses: ARTIFACT_DIR, CACHE_DIR, VARIANT
Source: baseline_basic.ipynb#cell=6; lines=12-26
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung tfds_rows(builder, splits, supervised=False) — phần 1/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn VARIANT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: ARTIFACT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def tfds_rows(builder,splits,supervised=False):
    # TensorFlow chỉ là reader CPU; PyTorch vẫn sử dụng DEVICE CUDA khi có.
    if importlib.util.find_spec("tensorflow") is None:
        if sys.version_info >= (3,13):
            raise RuntimeError("TFDS reader cần TensorFlow. Host Python này chưa có wheel phù hợp. "
                "Dùng kernel Python3.11/3.12 + tensorflow==2.18.1 tensorflow-datasets==4.9.10, "
                "hoặc chọn DATA_MODE='local_jsonl' với dữ liệu xuất đúng nguồn. Không có dữ liệu thay thế tự động.")
        subprocess.check_call([sys.executable,"-m","pip","install","tensorflow==2.18.1","tensorflow-datasets==4.9.10"])
    elif importlib.util.find_spec("tensorflow_datasets") is None:
        subprocess.check_call([sys.executable,"-m","pip","install","tensorflow-datasets==4.9.10"])
    import tensorflow as tf
    try:tf.config.set_visible_devices([],"GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow đã khởi tạo GPU. Restart Kernel rồi Run All để reader chạy CPU.") from exc
    import tensorflow_datasets as tfds
```

### Tóm tắt context để dùng tiếp

<a id="context-m009"></a>

```text
Tóm tắt giao diện tfds_rows(builder, splits, supervised=False) — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của tfds_rows(builder, splits, supervised=False) — phần 1/2 tại đây.
```

## Bước 4: tfds_rows(builder, splits, supervised=False) — phần 2/2

<!-- module
ID: m010
Depends: m009
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=6; lines=27-35
Config uses: CACHE_DIR
Source: baseline_basic.ipynb#cell=6; lines=27-35
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)

<a id="context-m010"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện tfds_rows(builder, splits, supervised=False) — phần 2/2 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    try:
        data,info=tfds.load(builder,split=splits,as_supervised=supervised,with_info=True,
                            data_dir=str(CACHE_DIR),shuffle_files=False,try_gcs=True)
        print("TFDS:",builder,"; phiên bản thư viện",tfds.__version__,"; split counts",{k:v.num_examples for k,v in info.splits.items()})
        return {split:list(tfds.as_numpy(dataset)) for split,dataset in zip(splits,data)}
    except Exception as exc:
        raise RuntimeError(f"TFDS không đọc được {builder}. Kiểm tra cache/mạng/nguồn download; "
            "có thể dùng local_jsonl đã xuất từ đúng builder. Xem mục dữ liệu trong đề xuất, không đổi dataset âm thầm. "
            f"Chi tiết: {exc}") from exc
```

## Bước 5: decode_utf8(value)

<!-- module
ID: m011
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=6; lines=37-38
Source: baseline_basic.ipynb#cell=6; lines=37-38
-->

<a id="context-m011"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện decode_utf8(value) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def decode_utf8(value):
    return value.decode("utf-8") if isinstance(value,(bytes,np.bytes_)) else str(value)
```

## Bước 6: adapt(row)

<!-- module
ID: m014
Depends: m011
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=1-13
Source: baseline_basic.ipynb#cell=8; lines=1-13
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [decode_utf8(value)](#context-m011)

<a id="context-m014"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện adapt(row) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def adapt(row):
    answers=row["answers"]
    if isinstance(answers,dict):
        starts=answers["answer_start"];texts=answers["text"]
        answers=[{"answer_start":int(start),"text":decode_utf8(text)} for start,text in zip(starts,texts)]
    else:answers=[{"answer_start":int(a["answer_start"]),"text":decode_utf8(a["text"])} for a in answers]
    record={"id":decode_utf8(row["id"]),"title":decode_utf8(row["title"]),
            "context":decode_utf8(row["context"]),"question":decode_utf8(row["question"]),"answers":answers}
    if not answers:raise ValueError("SQuAD v1.1 phải có answer; không dùng v2 như v1.1.")
    for answer in answers:
        s=answer["answer_start"];e=s+len(answer["text"])
        if s<0 or record["context"][s:e]!=answer["text"]:raise ValueError(f"Offset gold không khớp ở{record['id']}; không strip/normalize context.")
    return record
```

## Bước 7: Dữ liệu, offset và split — article, document, file…

<!-- module
ID: m015
Depends: m008, m009, m010, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=15-35
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_MODE, LOCAL_DATA_DIR, SOURCE
Config uses: CACHE_DIR, DATA_MODE, LOCAL_DATA_DIR, SOURCE
Source: baseline_basic.ipynb#cell=8; lines=15-35
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m008)
2. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
3. [tfds_rows(builder, splits, supervised=False) — phần 2/2](#context-m010)
4. [adapt(row)](#context-m014)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Dữ liệu, offset và split — article, document, file…. Giữ ràng buộc: DATA_MODE='tfds'. Khai báo cùng phần này: LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m014 — adapt(row):
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
if DATA_MODE=="tfds":
    raw=tfds_rows(SOURCE["tfds_builder"],["train","validation"])
elif DATA_MODE=="original_json":
    import urllib.request
    raw={}
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    for split,file in (("train","train-v1.1.json"),("validation","dev-v1.1.json")):
        path=CACHE_DIR/file
        if not path.is_file():
            try:path.write_bytes(urllib.request.urlopen("https://rajpurkar.github.io/SQuAD-explorer/dataset/"+file,timeout=120).read())
            except Exception as exc:raise RuntimeError(f"Không tải được JSON SQuAD gốc. Đặt đúng file tại{path}. {exc}") from exc
        document=json.loads(path.read_text(encoding="utf-8"));rows=[]
        if document["version"]!="1.1":raise ValueError("Không đúng SQuAD1.1.")
        for article in document["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:rows.append({**qa,"title":article["title"],"context":paragraph["context"]})
        raw[split]=rows
elif DATA_MODE=="local_jsonl":raw={split:read_jsonl(LOCAL_DATA_DIR/f"{split}.jsonl") for split in ("train","validation")}
else:raise ValueError("DATA_MODE cần tfds, original_json hoặc local_jsonl.")

pool=sorted([adapt(row) for row in raw["train"]],key=lambda r:r["id"])
```

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Dữ liệu, offset và split — article, document, file… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m014 — adapt(row):
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Dữ liệu, offset và split — article, document, file… tại đây.
```

## Bước 8: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m016
Depends: m014, m015
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=36-50
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, VAL_FRACTION
Config uses: SEED, VAL_FRACTION
Source: baseline_basic.ipynb#cell=8; lines=36-50
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [adapt(row)](#context-m014)
2. [Dữ liệu, offset và split — article, document, file…](#context-m015)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Chuẩn hóa và chia dữ liệu. Chuẩn hóa các record và split theo giao thức nguồn, giữ ID/nhãn/references, kiểm nhóm trùng; chỉ fit biểu diễn trên train. Nguồn dữ liệu dataset: SQuAD v1.1; tfds_builder: squad/v1.1:3.0.0. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: SEED=42; VAL_FRACTION=0.1. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — adapt(row):
[Dán tóm tắt m014]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
test_records=sorted([adapt(row) for row in raw["validation"]],key=lambda r:r["id"])
titles=sorted({r["title"] for r in pool})
random.Random(SEED).shuffle(titles)
val_titles=set(titles[:max(1,math.ceil(VAL_FRACTION*len(titles)))])
train_records=[r for r in pool if r["title"] not in val_titles]
val_records=[r for r in pool if r["title"] in val_titles]
if not train_records or not val_records:raise ValueError("Không đủ Wikipedia titles để chia validation.")
context_sets=[{r["context"] for r in records} for records in (train_records,val_records,test_records)]
if context_sets[0]&context_sets[1] or context_sets[0]&context_sets[2] or context_sets[1]&context_sets[2]:
    raise ValueError("Context trùng giữa split; xử lý nhóm context trước khi học.")
if len({r["id"] for r in train_records+val_records+test_records})!=len(train_records+val_records+test_records):
    raise ValueError("Question ID trùng xuyên split.")
SPLIT_FINGERPRINTS={name:fingerprint(records) for name,records in (("train",train_records),("validation",val_records),("held_out_official_validation",test_records))}
print("Question counts:",len(train_records),len(val_records),len(test_records))
print("Title counts:",len({r["title"] for r in train_records}),len(val_titles),len({r["title"] for r in test_records}))
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — adapt(row):
[Dán tóm tắt m014]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]
```

```python
# Dán code hiện tại của Chuẩn hóa và chia dữ liệu tại đây.
```

## Bước 9: Dữ liệu, offset và split

<!-- module
ID: m017
Depends: m016
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=51-52
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)

### Prompt tạo hoặc sửa module

```text
Viết Dữ liệu, offset và split. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Dữ liệu, offset và split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]
```

```python
# Dán code hiện tại của Dữ liệu, offset và split tại đây.
```

## Bước 10: Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…

<!-- module
ID: m019
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=2-4
Source: baseline_basic.ipynb#cell=10; lines=2-4
-->

<a id="context-m019"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
PAD_ID,UNK_ID,CLS_ID=0,1,2
TOKEN_PATTERN=r"\w+|[^\w\s]"
_CONTEXT_CACHE={}
```

## Bước 11: context_tokens(text)

<!-- module
ID: m020
Depends: m019
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=5-9
Source: baseline_basic.ipynb#cell=10; lines=5-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)

<a id="context-m020"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện context_tokens(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def context_tokens(text):
    if text not in _CONTEXT_CACHE:
        matches=list(re.finditer(TOKEN_PATTERN,text))
        _CONTEXT_CACHE[text]={"words":[m.group() for m in matches],"offsets":[m.span() for m in matches]}
    return _CONTEXT_CACHE[text]
```

## Bước 12: question_words(text)

<!-- module
ID: m021
Depends: m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=10-10
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_QUESTION_TOKENS
Config uses: MAX_QUESTION_TOKENS
Source: baseline_basic.ipynb#cell=10; lines=10-10
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung question_words(text). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn MAX_QUESTION_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def question_words(text):return re.findall(TOKEN_PATTERN,text)[:MAX_QUESTION_TOKENS] or ["<UNK>"]
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện question_words(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của question_words(text) tại đây.
```

## Bước 13: Bảng ký tự và trạng thái tokenizer

<!-- module
ID: m022
Depends: m016, m019, m020
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=12-27
Summary budget: input=1200; output=500; reserve=300
Config owns: VOCAB_SIZE
Config uses: VOCAB_SIZE
Source: baseline_basic.ipynb#cell=10; lines=12-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)
2. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)
3. [context_tokens(text)](#context-m020)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Bảng ký tự và trạng thái tokenizer. Xây vocabulary ký tự chỉ từ train và lưu mapping/special IDs để reload; giữ nguyên gold chưa gặp khi đánh giá. Tự chọn VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
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

TOKENIZER_STATE={"word_vocab":word_vocab,"regex":TOKEN_PATTERN,
                 "pad":PAD_ID,"unk":UNK_ID,"cls":CLS_ID}
PREPROCESS_CONFIG={"max_context_tokens":MAX_CONTEXT_TOKENS,"window_step":WINDOW_STEP,
                    "max_question_tokens":MAX_QUESTION_TOKENS,"max_answer_tokens":MAX_ANSWER_TOKENS,
                    "max_word_chars":MAX_WORD_CHARS,"context_normalization":"không thay đổi chuỗi gốc",
                    "train_alignment":"minimal covering token span","decode":"best joint span minus CLS score across all windows"}
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện Bảng ký tự và trạng thái tokenizer từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

```python
# Dán code hiện tại của Bảng ký tự và trạng thái tokenizer tại đây.
```

## Bước 14: restore_tokenizer(state)

<!-- module
ID: m024
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=32-34
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=10; lines=31-33
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung restore_tokenizer(state). Bổ sung character vocabulary train-only cho QA, giữ word vocabulary và offsets context gốc. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def restore_tokenizer(state):
    global word_vocab
    word_vocab=state["word_vocab"]
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 15: answer_token_span(context, answer)

<!-- module
ID: m025
Depends: m020
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=36-43
Source: baseline_basic.ipynb#cell=10; lines=35-42
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [context_tokens(text)](#context-m020)

<a id="context-m025"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện answer_token_span(context, answer) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

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

## Bước 16: QADataset.__init__(self, records, training=False)

<!-- module
ID: m026
Depends: m020, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=46-59
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_CONTEXT_TOKENS, WINDOW_STEP
Config uses: MAX_CONTEXT_TOKENS, WINDOW_STEP
Source: baseline_basic.ipynb#cell=10; lines=45-58
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [context_tokens(text)](#context-m020)
2. [answer_token_span(context, answer)](#context-m025)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung QADataset.__init__(self, records, training=False). Khởi tạo các thành phần và trạng thái cần cho class. Tự chọn MAX_CONTEXT_TOKENS, WINDOW_STEP phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Viết method trong class QADataset kế thừa Dataset; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
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
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện QADataset.__init__(self, records, training=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của QADataset.__init__(self, records, training=False) tại đây.
```

## Bước 17: QADataset.__len__(self)

<!-- module
ID: m027
Depends: m026
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=60-60
Source: baseline_basic.ipynb#cell=10; lines=59-59
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QADataset.__init__(self, records, training=False)](#context-m026)

<a id="context-m027"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện QADataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):return len(self.features)
```

## Bước 18: QADataset.__getitem__(self, index)

<!-- module
ID: m028
Depends: m020, m021, m026
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=10; lines=61-67
Source: baseline_basic.ipynb#cell=10; lines=60-66
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [context_tokens(text)](#context-m020)
2. [question_words(text)](#context-m021)
3. [QADataset.__init__(self, records, training=False)](#context-m026)

<a id="context-m028"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện QADataset.__getitem__(self, index) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m020 — context_tokens(text):
[Dán tóm tắt m020]

Context m021 — question_words(text):
[Dán tóm tắt m021]

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __getitem__(self,index):
        feature=self.features[index];record=feature["record"]
        cached=context_tokens(record["context"])
        words=["<CLS>"]+cached["words"][feature["start"]:feature["stop"]]
        offsets=[(-1,-1)]+cached["offsets"][feature["start"]:feature["stop"]]
        return {"words":words,"question":question_words(record["question"]),"offsets":offsets,
                "targets":feature["targets"],"record":record,"id":f"{record['id']}:{feature['start']}"}
```

## Bước 19: Ghép giao diện class QADataset

<!-- module
ID: m029
Depends: m026, m027, m028
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=10; lines=45-67
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QADataset.__init__(self, records, training=False)](#context-m026)
2. [QADataset.__len__(self)](#context-m027)
3. [QADataset.__getitem__(self, index)](#context-m028)

<a id="context-m029"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class QADataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m026 — QADataset.__init__(self, records, training=False):
[Dán tóm tắt m026]

Context m027 — QADataset.__len__(self):
[Dán tóm tắt m027]

Context m028 — QADataset.__getitem__(self, index):
[Dán tóm tắt m028]
```

## Bước 20: encode_words(rows, width)

<!-- module
ID: m030
Depends: m019, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=69-77
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_WORD_CHARS
Config uses: MAX_WORD_CHARS
Source: baseline_basic.ipynb#cell=10; lines=68-72
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…](#context-m019)
2. [Bảng ký tự và trạng thái tokenizer](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung encode_words(rows, width). Bổ sung char IDs trong encode_words và cập nhật batch context/question để model dùng char-CNN. Giữ thứ tự đầu ra: words, chars. Tự chọn MAX_WORD_CHARS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def encode_words(rows,width):
    words=torch.zeros(len(rows),width,dtype=torch.long)
    for i,tokens in enumerate(rows):
        words[i,:len(tokens)]=torch.tensor([CLS_ID if w=="<CLS>" else word_vocab.get(w.lower(),UNK_ID) for w in tokens])
    return words
```

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện encode_words(rows, width) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Vocabulary, windows và Dataset — CLS_ID, PAD_ID, TOKEN_PATTERN…:
[Dán tóm tắt m019]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của encode_words(rows, width) tại đây.
```

## Bước 21: collate_batch(items)

<!-- module
ID: m031
Depends: m030
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=79-89
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=10; lines=74-84
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [encode_words(rows, width)](#context-m030)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung collate_batch(items). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường context, context_chars, context_mask, ids, items, lengths, question, question_chars, question_lengths, question_mask, targets. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m030 — encode_words(rows, width):
[Dán tóm tắt m030]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

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

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện collate_batch(items) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m030 — encode_words(rows, width):
[Dán tóm tắt m030]
```

```python
# Dán code hiện tại của collate_batch(items) tại đây.
```

## Bước 22: to_device(batch)

<!-- module
ID: m012
Depends: m031
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=40-41
Summary budget: input=1200; output=500; reserve=300
Config owns: DEVICE
Config uses: DEVICE
Source: baseline_basic.ipynb#cell=6; lines=40-41
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def to_device(batch):
    return {k:v.to(DEVICE) if torch.is_tensor(v) and k not in ("lengths","question_lengths") else v for k,v in batch.items()}
```

### Tóm tắt context để dùng tiếp

<a id="context-m012"></a>

```text
Tóm tắt giao diện to_device(batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 23: make_loader(dataset, train=False)

<!-- module
ID: m013
Depends: m016, m031
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=43-46
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
Source: baseline_basic.ipynb#cell=6; lines=43-46
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)
2. [collate_batch(items)](#context-m031)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def make_loader(dataset,train=False):
    return DataLoader(dataset,batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
                      shuffle=train,num_workers=0,collate_fn=collate_batch,
                      generator=torch.Generator().manual_seed(SEED))
```

### Tóm tắt context để dùng tiếp

<a id="context-m013"></a>

```text
Tóm tắt giao diện make_loader(dataset, train=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m031 — collate_batch(items):
[Dán tóm tắt m031]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 24: Tạo DataLoader cho các split

<!-- module
ID: m032
Depends: m013, m016, m022, m025, m029
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=10; lines=91-100
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_ANSWER_TOKENS
Config uses: MAX_ANSWER_TOKENS
Source: baseline_basic.ipynb#cell=10; lines=86-95
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m013)
2. [Chuẩn hóa và chia dữ liệu](#context-m016)
3. [Bảng ký tự và trạng thái tokenizer](#context-m022)
4. [answer_token_span(context, answer)](#context-m025)
5. [Ghép giao diện class QADataset](#context-m029)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tạo DataLoader cho các split. Tạo DataLoader cho train, validation và test từ Dataset và collate đã thống nhất; giữ membership và thứ tự đánh giá. Tự chọn MAX_ANSWER_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_loader=make_loader(QADataset(train_records,True),True)
val_loader=make_loader(QADataset(val_records))
test_loader=make_loader(QADataset(test_records))
print("Word vocab chỉ train, không tạo char features:",len(word_vocab))
print("Số windows:",len(train_loader.dataset),len(val_loader.dataset),len(test_loader.dataset))
for name,records in (("train",train_records),("validation",val_records),("held_out",test_records)):
    alignment=[answer_token_span(r["context"],a) for r in records for a in r["answers"]]
    print(name,"gold answer boundaries giữa token:",sum(not exact for a,b,exact in alignment),
          "; gold dài hơn answer cap:",sum(b-a+1>MAX_ANSWER_TOKENS for a,b,exact in alignment))
print("Nhãn train là token span bao phủ tối thiểu; không sửa gold text khi chấm EM/F1. Giới hạn ranh giới được báo rõ.")
```

### Tóm tắt context để dùng tiếp

<a id="context-m032"></a>

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m025 — answer_token_span(context, answer):
[Dán tóm tắt m025]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]
```

```python
# Dán code hiện tại của Tạo DataLoader cho các split tại đây.
```

## Bước 25: QAModel.__init__(self, cfg)

<!-- module
ID: m033
Depends: m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=2-16
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, HIDDEN_DIM
Source: baseline_basic.ipynb#cell=12; lines=2-10
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Bảng ký tự và trạng thái tokenizer](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung QAModel.__init__(self, cfg). Thay constructor QAModel bằng word/char-CNN, BiLSTM và attention hai chiều với start/end heads. cfg n_words,n_chars,hidden_dim theo cấu hình đã chọn,dropout theo cấu hình đã chọn; word128,char16,charCNN32 kernel3; encoder input160, dim256; modeling input4*dim, end_encoder dim; heads input5*dim. Tự chọn DROPOUT, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa n_words, n_chars, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self,cfg):
        super().__init__()
        dim=cfg["hidden_dim"]
        self.embedding=nn.Embedding(cfg["n_words"],64,padding_idx=0)
        self.context_encoder=nn.RNN(64,dim,batch_first=True)
        self.question_encoder=nn.RNN(64,dim,batch_first=True)
        self.dropout=nn.Dropout(cfg["dropout"])
        self.fusion=nn.Sequential(nn.Linear(3*dim,dim),nn.Tanh())
        self.head=nn.Linear(dim,2)
```

### Tóm tắt context để dùng tiếp

<a id="context-m033"></a>

```text
Tóm tắt giao diện QAModel.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của QAModel.__init__(self, cfg) tại đây.
```

## Bước 26: QAModel.char_features(self, chars)

<!-- module
ID: m034
Depends: m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=17-22
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QAModel.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết QAModel.char_features(self, chars). Viết char_features bỏ PAD khi max-pooling, từ toàn PAD trả vector không. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện QAModel.char_features(self, chars) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của QAModel.char_features(self, chars) tại đây.
```

## Bước 27: QAModel.sequence(self, module, x, lengths)

<!-- module
ID: m035
Depends: m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=23-26
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QAModel.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết QAModel.sequence(self, module, x, lengths). Viết sequence(module,x,lengths) đóng gói BiLSTM, dropout và khôi phục đúng chiều dài tensor. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện QAModel.sequence(self, module, x, lengths) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của QAModel.sequence(self, module, x, lengths) tại đây.
```

## Bước 28: QAModel.forward(self, batch)

<!-- module
ID: m036
Depends: m031, m033, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=12; lines=27-41
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=12; lines=11-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)
2. [QAModel.__init__(self, cfg)](#context-m033)
3. [QAModel.char_features(self, chars)](#context-m034)
4. [QAModel.sequence(self, module, x, lengths)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung QAModel.forward(self, batch). Viết phần đầu forward: word/char features, encoder chung và similarity trilinear có masks đúng. Hoàn tất forward bằng fusion bốn thành phần, modeling/end BiLSTM và hai span heads. Giữ thứ tự đầu ra: kết quả 1, kết quả 2. Viết method trong class QAModel kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — QAModel.char_features(self, chars):
[Dán tóm tắt m034]

Context m035 — QAModel.sequence(self, module, x, lengths):
[Dán tóm tắt m035]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def forward(self,batch):
        packed=pack_padded_sequence(self.dropout(self.embedding(batch["context"])),batch["lengths"],batch_first=True,enforce_sorted=False)
        c,_=self.context_encoder(packed);c,_=pad_packed_sequence(c,batch_first=True,total_length=batch["context"].size(1))
        q=pack_padded_sequence(self.dropout(self.embedding(batch["question"])),batch["question_lengths"],batch_first=True,enforce_sorted=False)
        _,q=self.question_encoder(q);q=q[-1][:,None].expand(-1,c.size(1),-1)
        logits=self.head(self.dropout(self.fusion(torch.cat([c,q,c*q],-1))))
        return logits[:,:,0].masked_fill(~batch["context_mask"],-10000),logits[:,:,1].masked_fill(~batch["context_mask"],-10000)
```

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện QAModel.forward(self, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — QAModel.char_features(self, chars):
[Dán tóm tắt m034]

Context m035 — QAModel.sequence(self, module, x, lengths):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của QAModel.forward(self, batch) tại đây.
```

## Bước 29: Ghép giao diện class QAModel

<!-- module
ID: m037
Depends: m033, m034, m035, m036
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=12; lines=1-41
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [QAModel.__init__(self, cfg)](#context-m033)
2. [QAModel.char_features(self, chars)](#context-m034)
3. [QAModel.sequence(self, module, x, lengths)](#context-m035)
4. [QAModel.forward(self, batch)](#context-m036)

<a id="context-m037"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class QAModel kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — QAModel.char_features(self, chars):
[Dán tóm tắt m034]

Context m035 — QAModel.sequence(self, module, x, lengths):
[Dán tóm tắt m035]

Context m036 — QAModel.forward(self, batch):
[Dán tóm tắt m036]
```

## Bước 30: build_model(cfg)

<!-- module
ID: m038
Depends: m037
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=12; lines=42-42
Source: baseline_basic.ipynb#cell=12; lines=18-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class QAModel](#context-m037)

<a id="context-m038"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện build_model(cfg) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_model(cfg):return QAModel(cfg)
```

## Bước 31: normalize_answer(text)

<!-- module
ID: m040
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=1-5
Source: baseline_basic.ipynb#cell=15; lines=1-5
-->

<a id="context-m040"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện normalize_answer(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def normalize_answer(text):
    text=text.lower()
    text="".join(ch for ch in text if ch not in string.punctuation)
    text=re.sub(r"\b(a|an|the)\b"," ",text)
    return " ".join(text.split())
```

## Bước 32: answer_metrics(prediction, references)

<!-- module
ID: m041
Depends: m040
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=6-15
Source: baseline_basic.ipynb#cell=15; lines=6-15
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_answer(text)](#context-m040)

<a id="context-m041"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện answer_metrics(prediction, references) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m040 — normalize_answer(text):
[Dán tóm tắt m040]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def answer_metrics(prediction,references):
    em=[];f1=[]
    for gold in references:
        p=normalize_answer(prediction);g=normalize_answer(gold)
        em.append(float(p==g))
        p_tokens=p.split();g_tokens=g.split();common=sum((Counter(p_tokens)&Counter(g_tokens)).values())
        if not common:f1.append(0.);continue  # Quy tắc evaluator SQuAD v1.1.
        precision=common/len(p_tokens);recall=common/len(g_tokens)
        f1.append(2*precision*recall/(precision+recall))
    return max(em),max(f1)
```

## Bước 33: compute_loss(model, batch)

<!-- module
ID: m042
Depends: m031, m037
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=15; lines=17-21
Summary budget: input=1200; output=500; reserve=300
Config owns: NEGATIVE_WINDOW_WEIGHT
Config uses: NEGATIVE_WINDOW_WEIGHT
Source: baseline_basic.ipynb#cell=15; lines=17-21
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)
2. [Ghép giao diện class QAModel](#context-m037)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung compute_loss(model, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn NEGATIVE_WINDOW_WEIGHT phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def compute_loss(model,batch):
    start,end=model(batch);targets=batch["targets"]
    loss=nn.functional.cross_entropy(start,targets[:,0],reduction="none")+nn.functional.cross_entropy(end,targets[:,1],reduction="none")
    weights=torch.where(targets[:,0]==0,torch.full_like(loss,NEGATIVE_WINDOW_WEIGHT),torch.ones_like(loss))
    return (loss*weights).sum()/weights.sum()
```

### Tóm tắt context để dùng tiếp

<a id="context-m042"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 34: best_spans(start, end, batch)

<!-- module
ID: m043
Depends: m031, m032
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=23-30
Config uses: MAX_ANSWER_TOKENS
Source: baseline_basic.ipynb#cell=15; lines=23-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m031)
2. [Tạo DataLoader cho các split](#context-m032)

<a id="context-m043"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện best_spans(start, end, batch) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m031 — collate_batch(items):
[Dán tóm tắt m031]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def best_spans(start,end,batch):
    n=start.size(1);indices=torch.arange(n,device=start.device)
    allowed=(indices[None,:]>=indices[:,None])&(indices[None,:]-indices[:,None]<MAX_ANSWER_TOKENS)&(indices[:,None]>=1)
    allowed=allowed[None]&batch["context_mask"][:,:,None]&batch["context_mask"][:,None,:]
    joint=(start[:,:,None]+end[:,None,:]).masked_fill(~allowed,-torch.inf)
    scores,flat=joint.flatten(1).max(-1)
    margin=scores-start[:,0]-end[:,0]
    return (flat//n).tolist(),(flat%n).tolist(),margin.tolist()
```

## Bước 35: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m044
Depends: m012, m037, m041, m043
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=15; lines=32-50
Source: baseline_basic.ipynb#cell=15; lines=32-50
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Ghép giao diện class QAModel](#context-m037)
3. [answer_metrics(prediction, references)](#context-m041)
4. [best_spans(start, end, batch)](#context-m043)

<a id="context-m044"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — Ghép giao diện class QAModel:
[Dán tóm tắt m037]

Context m041 — answer_metrics(prediction, references):
[Dán tóm tắt m041]

Context m043 — best_spans(start, end, batch):
[Dán tóm tắt m043]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def evaluate(model,loader,return_predictions=False):
    model.eval();best={}
    for cpu_batch in loader:
        batch=to_device(cpu_batch);start,end=model(batch)
        starts,ends,scores=best_spans(start,end,batch)
        for item,a,b,score in zip(batch["items"],starts,ends,scores):
            record=item["record"];qid=record["id"]
            s=item["offsets"][a][0];e=item["offsets"][b][1]
            if qid not in best or score>best[qid][0]:best[qid]=(score,record["context"][s:e],s,e)
    if set(best)!={r["id"] for r in loader.dataset.records}:raise ValueError("Mất question khi gộp windows; không chấm theo feature.")
    totals=np.zeros(2);predictions=[]
    for record in loader.dataset.records:
        score,text,s,e=best[record["id"]]
        refs=[a["text"] for a in record["answers"]]
        totals+=answer_metrics(text,refs)
        if return_predictions:predictions.append({"id":record["id"],"prediction":text,"start":s,"end":e,"references":refs})
    metrics={"exact_match":100*float(totals[0]/len(best)),"token_f1":100*float(totals[1]/len(best)),"question_count":len(best)}
    return (metrics,predictions) if return_predictions else metrics
```

## Bước 36: Huấn luyện/checkpoint — model, optimizer, total_steps

<!-- module
ID: m045
Depends: m012, m032, m033, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=1-4
Summary budget: input=1200; output=500; reserve=300
Config owns: EPOCHS, LEARNING_RATE, WEIGHT_DECAY
Config uses: DEVICE, EPOCHS, LEARNING_RATE, MODEL_CONFIG, WEIGHT_DECAY
Source: baseline_basic.ipynb#cell=5; lines=31-44
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Tạo DataLoader cho các split](#context-m032)
3. [QAModel.__init__(self, cfg)](#context-m033)
4. [build_model(cfg)](#context-m038)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Huấn luyện/checkpoint — model, optimizer, total_steps. Khởi tạo optimizer AdamW cho tham số có thể học. Tự chọn EPOCHS, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
MAX_ANSWER_TOKENS = 64
MAX_WORD_CHARS = 32
NEGATIVE_WINDOW_WEIGHT = 0.5
HIDDEN_DIM = 64
DROPOUT = 0.2
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "exact_match"
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện Huấn luyện/checkpoint — model, optimizer, total_steps từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của Huấn luyện/checkpoint — model, optimizer, total_steps tại đây.
```

## Bước 37: lr_multiplier(step)

<!-- module
ID: m046
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=5-7
Summary budget: input=1200; output=500; reserve=300
Config owns: WARMUP_STEPS
Config uses: WARMUP_STEPS
Source: baseline_basic.ipynb#cell=17; lines=5-7
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung lr_multiplier(step). Tạo hệ số lịch learning rate theo số bước và warmup đã chọn trong context huấn luyện. Tự chọn WARMUP_STEPS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def lr_multiplier(step):
    step+=1
    return min(step/max(1,WARMUP_STEPS),math.sqrt(max(1,WARMUP_STEPS)/max(step,WARMUP_STEPS))) if WARMUP_STEPS else 1.0
```

### Tóm tắt context để dùng tiếp

<a id="context-m046"></a>

```text
Tóm tắt giao diện lr_multiplier(step) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của lr_multiplier(step) tại đây.
```

## Bước 38: Huấn luyện/checkpoint — best_score, scheduler, stale_epochs

<!-- module
ID: m047
Depends: m045, m046
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=17; lines=8-9
Source: baseline_basic.ipynb#cell=17; lines=8-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Huấn luyện/checkpoint — model, optimizer, total_steps](#context-m045)
2. [lr_multiplier(step)](#context-m046)

<a id="context-m047"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Huấn luyện/checkpoint — best_score, scheduler, stale_epochs từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m046 — lr_multiplier(step):
[Dán tóm tắt m046]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
best_score=-math.inf;stale_epochs=0
```

## Bước 39: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m048
Depends: m009, m012, m022, m032, m042, m044, m045, m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=10-38
Summary budget: input=1200; output=500; reserve=300
Config owns: CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Source: baseline_basic.ipynb#cell=17; lines=10-38
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Bảng ký tự và trạng thái tokenizer](#context-m022)
4. [Tạo DataLoader cho các split](#context-m032)
5. [compute_loss(model, batch)](#context-m042)
6. [evaluate(model, loader, return_predictions=False)](#context-m044)
7. [Huấn luyện/checkpoint — model, optimizer, total_steps](#context-m045)
8. [Huấn luyện/checkpoint — best_score, scheduler, stale_epochs](#context-m047)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='exact_match'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m042 — compute_loss(model, batch):
[Dán tóm tắt m042]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m047 — Huấn luyện/checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m047]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
for epoch in range(1,EPOCHS+1):
    model.train();total=0.;seen=0
    for cpu_batch in tqdm(train_loader,desc="Huấn luyện",leave=False):
        batch=to_device(cpu_batch)
        optimizer.zero_grad(set_to_none=True)
        loss=compute_loss(model,batch)
        if not torch.isfinite(loss):raise FloatingPointError("Loss không hữu hạn; kiểm tra mask/nhãn và cấu hình cố định.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        optimizer.step();scheduler.step()
        n=len(batch["ids"]);total+=loss.item()*n;seen+=n
    if not seen:raise ValueError("Train rỗng.")
    loss = total/seen
    metrics=evaluate(model,val_loader)
    score=float(metrics[PRIMARY_METRIC])
    if not math.isfinite(score):raise ValueError("Primary validation metric không hữu hạn.")
    print(f"Epoch{epoch:02d}: loss={loss:.4f}; validation={metrics}")
    if score>best_score:
        best_score=score;stale_epochs=0
        torch.save({
            "model_state": model.state_dict(),
            "tokenizer": TOKENIZER_STATE,
            "epoch": epoch,
        }, CHECKPOINT_PATH)
        print("Đã lưu checkpoint tốt nhất theo primary metric validation tăng.")
    else:
        stale_epochs+=1
        if stale_epochs>=PATIENCE:
            print(f"Dừng sớm sau {PATIENCE} epoch không cải thiện.");break
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m042 — compute_loss(model, batch):
[Dán tóm tắt m042]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m045 — Huấn luyện/checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m045]

Context m047 — Huấn luyện/checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 40: Reload và held-out official validation — best_model, checkpoint, inference_config…

<!-- module
ID: m049
Depends: m009, m012, m015, m016, m022, m024, m032, m033, m038, m044, m048
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=19; lines=1-17
Summary budget: input=1200; output=500; reserve=300
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE
Source: baseline_basic.ipynb#cell=19; lines=1-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Dữ liệu, offset và split — article, document, file…](#context-m015)
4. [Chuẩn hóa và chia dữ liệu](#context-m016)
5. [Bảng ký tự và trạng thái tokenizer](#context-m022)
6. [restore_tokenizer(state)](#context-m024)
7. [Tạo DataLoader cho các split](#context-m032)
8. [QAModel.__init__(self, cfg)](#context-m033)
9. [build_model(cfg)](#context-m038)
10. [evaluate(model, loader, return_predictions=False)](#context-m044)
11. [Vòng huấn luyện, validation và lưu checkpoint](#context-m048)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Reload và held-out official validation — best_model, checkpoint, inference_config…. Reload toàn bộ pipeline đã chọn, khôi phục cấu hình kiến trúc và tokenizer/vocabulary từ checkpoint hoặc context tương ứng; kiểm lại validation rồi đánh giá held-out đúng giao thức, không chọn lại tham số. Dùng lại ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, HIDDEN_DIM, SOURCE từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m024 — restore_tokenizer(state):
[Dán tóm tắt m024]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m048 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m048]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
checkpoint=torch.load(CHECKPOINT_PATH,map_location=DEVICE,weights_only=True)
restore_tokenizer(checkpoint["tokenizer"])
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"n_words":len(word_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
best_model=build_model(inference_config).to(DEVICE)
best_model.load_state_dict(checkpoint["model_state"])
best_model.eval()
validation_metrics=evaluate(best_model,val_loader)
test_metrics,predictions=evaluate(best_model,test_loader,return_predictions=True)
results={"source":SOURCE,"epoch":checkpoint["epoch"],"validation":validation_metrics,
         "held_out":test_metrics,"split_fingerprints":SPLIT_FINGERPRINTS}
(ARTIFACT_DIR/"metrics.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
(ARTIFACT_DIR/"predictions.json").write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding="utf-8")
print("Đã nạp checkpoint epoch",checkpoint["epoch"])
print("Validation:",validation_metrics)
print("Held-out:",test_metrics)
print("Ví dụ:",predictions[0] if predictions else "Không có mẫu")
```

### Tóm tắt context để dùng tiếp

<a id="context-m049"></a>

```text
Tóm tắt giao diện Reload và held-out official validation — best_model, checkpoint, inference_config… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m015 — Dữ liệu, offset và split — article, document, file…:
[Dán tóm tắt m015]

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m022 — Bảng ký tự và trạng thái tokenizer:
[Dán tóm tắt m022]

Context m024 — restore_tokenizer(state):
[Dán tóm tắt m024]

Context m032 — Tạo DataLoader cho các split:
[Dán tóm tắt m032]

Context m033 — QAModel.__init__(self, cfg):
[Dán tóm tắt m033]

Context m038 — build_model(cfg):
[Dán tóm tắt m038]

Context m044 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m044]

Context m048 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m048]
```

```python
# Dán code hiện tại của Reload và held-out official validation — best_model, checkpoint, inference_config… tại đây.
```

## Bước 41: answer_question(context, question)

<!-- module
ID: m050
Depends: m012, m013, m029, m043, m049
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=21; lines=1-13
Source: baseline_basic.ipynb#cell=21; lines=1-13
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [make_loader(dataset, train=False)](#context-m013)
3. [Ghép giao diện class QADataset](#context-m029)
4. [best_spans(start, end, batch)](#context-m043)
5. [Reload và held-out official validation — best_model, checkpoint, inference_config…](#context-m049)

<a id="context-m050"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện answer_question(context, question) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m029 — Ghép giao diện class QADataset:
[Dán tóm tắt m029]

Context m043 — best_spans(start, end, batch):
[Dán tóm tắt m043]

Context m049 — Reload và held-out official validation — best_model, checkpoint, inference_config…:
[Dán tóm tắt m049]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def answer_question(context,question):
    if not context.strip() or not question.strip():raise ValueError("Cần context/question không rỗng.")
    record={"id":"new-question","title":"new","context":context,"question":question,"answers":[]}
    dataset=QADataset([record]);best=None
    best_model.eval()
    for cpu_batch in make_loader(dataset):
        batch=to_device(cpu_batch);start,end=best_model(batch)
        starts,ends,scores=best_spans(start,end,batch)
        for item,a,b,score in zip(batch["items"],starts,ends,scores):
            s=item["offsets"][a][0];e=item["offsets"][b][1]
            if best is None or score>best[0]:best=(score,s,e)
    return {"text":context[best[1]:best[2]],"start":best[1],"end":best[2]}
```

## Bước 42: Suy luận câu hỏi mới — sample

<!-- module
ID: m051
Depends: m016, m050
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=21; lines=15-16
Source: baseline_basic.ipynb#cell=21; lines=15-16
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m016)
2. [answer_question(context, question)](#context-m050)

<a id="context-m051"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Suy luận câu hỏi mới — sample từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m016 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m016]

Context m050 — answer_question(context, question):
[Dán tóm tắt m050]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
sample=test_records[0]
print(answer_question(sample["context"],sample["question"]))
```

## Bước 43: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m007, m017, m051
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Dữ liệu, offset và split](#context-m017)
3. [Suy luận câu hỏi mới — sample](#context-m051)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m017 — Dữ liệu, offset và split:
[Dán tóm tắt m017]

Context m051 — Suy luận câu hỏi mới — sample:
[Dán tóm tắt m051]
```
