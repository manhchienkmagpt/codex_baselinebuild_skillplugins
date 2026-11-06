# Prompt basic: CoEdIT GEC: BPE, BiGRU attention và pointer-copy

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

## Bước 6: strip_instruction(text)

<!-- module
ID: m016
Depends: m007
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=3-6
Summary budget: input=1200; output=500; reserve=300
Config owns: KNOWN_GEC_PREFIXES
Config uses: KNOWN_GEC_PREFIXES
Source: baseline_basic.ipynb#cell=8; lines=3-6
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung strip_instruction(text). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn KNOWN_GEC_PREFIXES phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m007 — clean_text(text):
[Dán tóm tắt m007]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def strip_instruction(text):
    for prefix in sorted(KNOWN_GEC_PREFIXES,key=len,reverse=True):
        if text.startswith(prefix):return clean_text(text[len(prefix):])
    raise ValueError(f"Instruction chưa kiểm tra: {text[:120]!r}; bổ sung prefix đúng nghĩa sau khi kiểm tra, không split mọi dấu hai chấm.")
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện strip_instruction(text) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m007 — clean_text(text):
[Dán tóm tắt m007]
```

```python
# Dán code hiện tại của strip_instruction(text) tại đây.
```

## Bước 7: Dữ liệu và giao thức chia — raw

<!-- module
ID: m017
Depends: m008, m009
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=8-12
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_MODE, LOCAL_DATA_DIR, SOURCE
Config uses: CACHE_DIR, DATA_MODE, LOCAL_DATA_DIR, SOURCE
Source: baseline_basic.ipynb#cell=5; lines=1-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m008)
2. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Dữ liệu và giao thức chia — raw. Giữ ràng buộc: DATA_MODE='huggingface'. Khai báo cùng phần này: LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
SEED = 42
VARIANT = "basic"
ARTIFACT_DIR = Path("artifacts/gec_coedit") / VARIANT
CACHE_DIR = ARTIFACT_DIR / "data_cache"
METRIC_CACHE_DIR = ARTIFACT_DIR / "metric_cache"
CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pt"
TASK_PRETRAINING_POLICY = "scratch"
TASK_PRETRAINED_COMPONENTS = []
ALLOW_PRETRAINED_EVALUATORS = True  # Ngoại lệ người dùng cho phép chỉ dùng để chấm metric.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
print(f"Thiết bị: {DEVICE}; biến thể: {VARIANT}; PyTorch {torch.__version__}; Python {sys.version.split()[0]}")


SOURCE = {'row': 4, 'task': 'Grammar Error Correction', 'dataset': 'grammarly/coedit', 'revision': 'e9a255c33ef910bc33a9d2b522653fa87521583e', 'url': 'https://huggingface.co/datasets/grammarly/coedit'}
DATA_MODE = "huggingface"
LOCAL_DATA_DIR = Path("data/gec_coedit")
SOURCE["data_mode"] = DATA_MODE
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — raw từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — raw tại đây.
```

## Bước 8: group_gec(rows)

<!-- module
ID: m018
Depends: m007, m016
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=14-24
Source: baseline_basic.ipynb#cell=8; lines=14-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [strip_instruction(text)](#context-m016)

<a id="context-m018"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện group_gec(rows) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m016 — strip_instruction(text):
[Dán tóm tắt m016]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

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

## Bước 9: Dữ liệu và giao thức chia — held, held_sources, native…

<!-- module
ID: m019
Depends: m017, m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=26-34
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, VAL_FRACTION
Config uses: SEED, VAL_FRACTION
Source: baseline_basic.ipynb#cell=8; lines=26-34
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — raw](#context-m017)
2. [group_gec(rows)](#context-m018)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Dữ liệu và giao thức chia — held, held_sources, native…. Giữ ràng buộc: SEED=42; VAL_FRACTION=0.1. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m018 — group_gec(rows):
[Dán tóm tắt m018]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
pool=group_gec(raw["train"])
held=group_gec(raw["validation"])
held_sources={r["src"] for r in held}
pool=[r for r in pool if r["src"] not in held_sources]
order=list(range(len(pool)));random.Random(SEED).shuffle(order)
val_ids=set(order[:max(1,math.ceil(VAL_FRACTION*len(order)))])
native={"train":[r for i,r in enumerate(pool) if i not in val_ids],
        "validation":[r for i,r in enumerate(pool) if i in val_ids],"test":held}
print("Official CoEdIT validation dùng làm held-out; validation chọn checkpoint được tách từ train theo câu gốc.")
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — held, held_sources, native… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m018 — group_gec(rows):
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của Dữ liệu và giao thức chia — held, held_sources, native… tại đây.
```

## Bước 10: Dữ liệu và giao thức chia — record, records, removed_train…

<!-- module
ID: m020
Depends: m007, m019
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=2-11
Source: baseline_basic.ipynb#cell=9; lines=2-11
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Dữ liệu và giao thức chia — held, held_sources, native…](#context-m019)

<a id="context-m020"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — record, records, removed_train… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
for split,records in native.items():
    for record in records:
        record["src"]=clean_text(record["src"]);record["refs"]=[clean_text(ref) for ref in record["refs"]]
        if not record["src"] or not record["refs"] or any(not r for r in record["refs"]):raise ValueError(f"Mẫu rỗng ở{split}.")
test_sources={r["src"] for r in native["test"]}
removed_val=sum(r["src"] in test_sources for r in native["validation"])
native["validation"]=[r for r in native["validation"] if r["src"] not in test_sources]
unseen_sources=test_sources|{r["src"] for r in native["validation"]}
removed_train=sum(r["src"] in unseen_sources for r in native["train"])
native["train"]=[r for r in native["train"] if r["src"] not in unseen_sources]
```

## Bước 11: canonical_records(rows, split)

<!-- module
ID: m021
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=12-16
Source: baseline_basic.ipynb#cell=9; lines=12-16
-->

<a id="context-m021"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện canonical_records(rows, split) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def canonical_records(rows,split):
    output=[]
    for i,row in enumerate(sorted(rows,key=lambda r:(r["src"],tuple(r["refs"])))):
        output.append({**row,"id":f"{split}:{i}"})
    return output
```

## Bước 12: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m022
Depends: m020, m021
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=17-26
Source: baseline_basic.ipynb#cell=9; lines=17-26
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — record, records, removed_train…](#context-m020)
2. [canonical_records(rows, split)](#context-m021)

<a id="context-m022"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m020 — Dữ liệu và giao thức chia — record, records, removed_train…:
[Dán tóm tắt m020]

Context m021 — canonical_records(rows, split):
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_records=canonical_records(native["train"],"train")
val_records=canonical_records(native["validation"],"validation")
test_records=canonical_records(native["test"],"test")
if any(not r for r in (train_records,val_records,test_records)):raise ValueError("Một split rỗng sau xử lý grouping.")
SPLIT_FINGERPRINTS={name:fingerprint([{"src":r["src"],"refs":r["refs"]} for r in records])
                    for name,records in (("train",train_records),("validation",val_records),("test",test_records))}
print("Loại source trùng: train",removed_train,"; validation",removed_val)
print("Số mẫu sau grouping:",len(train_records),len(val_records),len(test_records))
print("Fingerprint:",SPLIT_FINGERPRINTS)
print("Ví dụ train:",train_records[0])
```

## Bước 13: Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…

<!-- module
ID: m024
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=2-3
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=11; lines=2-2
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…. Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Tóm tắt context để dùng tiếp

<a id="context-m024"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… tại đây.
```

## Bước 14: tokenizer_corpus()

<!-- module
ID: m025
Depends: m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=4-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m022)

### Prompt tạo hoặc sửa module

```text
Viết tokenizer_corpus(). Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện tokenizer_corpus() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của tokenizer_corpus() tại đây.
```

## Bước 15: Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp

<!-- module
ID: m026
Depends: m024, m025
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=8-18
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE
Source: baseline_basic.ipynb#cell=5; lines=25-46
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [tokenizer_corpus()](#context-m025)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp. Fit SentencePiece BPE scratch với byte fallback chỉ trên train; giữ special IDs đã thống nhất, lưu tokenizer và trạng thái tiền xử lý để reload. PREPROCESS_CONFIG chỉ lưu thiết lập mã hóa/chuẩn hóa, không chứa cấu hình giải mã. Tự chọn MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m025 — tokenizer_corpus():
[Dán tóm tắt m025]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'spaCy', 'model_id': 'en_core_web_sm', 'revision': '3.8.0', 'training': 'frozen', 'sha256': '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85', 'purpose': 'ERRANT annotation, không tạo task output'}]
VAL_FRACTION = 0.1
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
VOCAB_SIZE = 8000
MAX_SOURCE_TOKENS = 160
MAX_TARGET_TOKENS = 160
MAX_DECODE_TOKENS = 160
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BEAM_SIZE = 1
LENGTH_PENALTY = 0.0
LABEL_SMOOTHING = 0.0
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "errant_f05"
GLEU_ITERATIONS = 500
```

### Tóm tắt context để dùng tiếp

<a id="context-m026"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m025 — tokenizer_corpus():
[Dán tóm tắt m025]
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp tại đây.
```

## Bước 16: restore_tokenizer(state)

<!-- module
ID: m027
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=19-21
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=11; lines=29-31
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung restore_tokenizer(state). Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def restore_tokenizer(state):
    global sp
    sp=BasicWordTokenizer(state["vocab"])
```

### Tóm tắt context để dùng tiếp

<a id="context-m027"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 17: PairDataset.__init__(self, records, training=False)

<!-- module
ID: m028
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=24-26
Source: baseline_basic.ipynb#cell=11; lines=34-36
-->

<a id="context-m028"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện PairDataset.__init__(self, records, training=False) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self,records,training=False):
        self.records=records
        self.index=[(i,j) for i,r in enumerate(records) for j in range(len(r["refs"]))] if training else [(i,0) for i in range(len(records))]
```

## Bước 18: PairDataset.__len__(self)

<!-- module
ID: m029
Depends: m028
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=27-27
Source: baseline_basic.ipynb#cell=11; lines=37-37
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m028)

<a id="context-m029"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện PairDataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):return len(self.index)
```

## Bước 19: PairDataset.__getitem__(self, index)

<!-- module
ID: m030
Depends: m024, m026, m028
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=28-33
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS
Source: baseline_basic.ipynb#cell=11; lines=38-43
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
3. [PairDataset.__init__(self, records, training=False)](#context-m028)

<a id="context-m030"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện PairDataset.__getitem__(self, index) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __getitem__(self,index):
        i,j=self.index[index];record=self.records[i]
        source=sp.encode(record["src"],out_type=int)[:MAX_SOURCE_TOKENS]
        if not source:source=[UNK_ID]
        target=[BOS_ID]+sp.encode(record["refs"][j],out_type=int)[:MAX_TARGET_TOKENS-2]+[EOS_ID]
        return torch.tensor(source),torch.tensor(target),record
```

## Bước 20: Ghép giao diện class PairDataset

<!-- module
ID: m031
Depends: m028, m029, m030
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=11; lines=23-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m028)
2. [PairDataset.__len__(self)](#context-m029)
3. [PairDataset.__getitem__(self, index)](#context-m030)

<a id="context-m031"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class PairDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m028 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m028]

Context m029 — PairDataset.__len__(self):
[Dán tóm tắt m029]

Context m030 — PairDataset.__getitem__(self, index):
[Dán tóm tắt m030]
```

## Bước 21: collate_batch(items)

<!-- module
ID: m032
Depends: m024
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=35-41
Source: baseline_basic.ipynb#cell=11; lines=45-51
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)

<a id="context-m032"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện collate_batch(items) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def collate_batch(items):
    lengths=torch.tensor([len(src) for src,_,_ in items])
    source=torch.full((len(items),int(lengths.max())),PAD_ID,dtype=torch.long)
    target=torch.full((len(items),max(len(tgt) for _,tgt,_ in items)),PAD_ID,dtype=torch.long)
    for i,(src,tgt,_) in enumerate(items):source[i,:len(src)]=src;target[i,:len(tgt)]=tgt
    return {"source":source,"target":target,"lengths":lengths,
            "ids":[r["id"] for _,_,r in items],"records":[r for _,_,r in items]}
```

## Bước 22: to_device(batch)

<!-- module
ID: m012
Depends: m032
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

1. [collate_batch(items)](#context-m032)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
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

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 23: make_loader(dataset, train=False)

<!-- module
ID: m013
Depends: m019, m032
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

1. [Dữ liệu và giao thức chia — held, held_sources, native…](#context-m019)
2. [collate_batch(items)](#context-m032)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
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

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 24: Tạo DataLoader cho các split

<!-- module
ID: m033
Depends: m013, m022, m026, m031
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=43-50
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS
Source: baseline_basic.ipynb#cell=11; lines=53-60
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [make_loader(dataset, train=False)](#context-m013)
2. [Chuẩn hóa và chia dữ liệu](#context-m022)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
4. [Ghép giao diện class PairDataset](#context-m031)

<a id="context-m033"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m031 — Ghép giao diện class PairDataset:
[Dán tóm tắt m031]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
train_loader=make_loader(PairDataset(train_records,True),True)
val_loader=make_loader(PairDataset(val_records))
test_loader=make_loader(PairDataset(test_records))
print("Tokenizer vocab:",sp.get_piece_size(),"; fit chỉ train.")
for name,records in (("train",train_records),("validation",val_records),("test",test_records)):
    source_cut=sum(len(sp.encode(r["src"]))>MAX_SOURCE_TOKENS for r in records)
    target_cut=sum(len(sp.encode(ref))>MAX_TARGET_TOKENS-2 for r in records for ref in r["refs"])
    print(f"{name}: source quá cap={source_cut}; target quá cap train={target_cut}; gold metric giữ đầy đủ.")
```

## Bước 25: TextGenerator.__init__(self, cfg)

<!-- module
ID: m034
Depends: m024, m026
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=2-14
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, EMBED_DIM, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, EMBED_DIM, HIDDEN_DIM
Source: baseline_basic.ipynb#cell=13; lines=2-8
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.__init__(self, cfg). Thay constructor TextGenerator bằng BiGRU encoder, additive attention, GRUCell decoder và gate copy. cfg vocab_size,embed_dim theo cấu hình đã chọn,hidden_dim theo cấu hình đã chọn,dropout theo cấu hình đã chọn; memory/state dim=384; embedding chung, bridge, memory_projection,hidden_projection,attention_score,output,gate,dropout. Tự chọn DROPOUT, EMBED_DIM, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa vocab_size, embed_dim, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __init__(self,cfg):
        super().__init__()
        self.embedding=nn.Embedding(cfg["vocab_size"],cfg["embed_dim"],padding_idx=PAD_ID)
        self.encoder=nn.RNN(cfg["embed_dim"],cfg["hidden_dim"],batch_first=True)
        self.decoder=nn.RNNCell(cfg["embed_dim"],cfg["hidden_dim"])
        self.dropout=nn.Dropout(cfg["dropout"])
        self.output=nn.Linear(cfg["hidden_dim"],cfg["vocab_size"])
```

### Tóm tắt context để dùng tiếp

<a id="context-m034"></a>

```text
Tóm tắt giao diện TextGenerator.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

```python
# Dán code hiện tại của TextGenerator.__init__(self, cfg) tại đây.
```

## Bước 26: TextGenerator.encode(self, source, lengths)

<!-- module
ID: m035
Depends: m024, m032, m034
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=15-20
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=13; lines=9-12
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [TextGenerator.__init__(self, cfg)](#context-m034)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.encode(self, source, lengths). Viết encode(source,lengths) dùng packed BiGRU và bridge trạng thái hai chiều. Giữ thứ tự đầu ra: memory, kết quả 2, state, kết quả 4. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def encode(self,source,lengths):
        packed=pack_padded_sequence(self.dropout(self.embedding(source)),lengths,batch_first=True,enforce_sorted=False)
        _,hidden=self.encoder(packed)
        return hidden[-1]
```

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện TextGenerator.encode(self, source, lengths) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

```python
# Dán code hiện tại của TextGenerator.encode(self, source, lengths) tại đây.
```

## Bước 27: TextGenerator.step(self, token, hidden, memory, projected, mask, source)

<!-- module
ID: m036
Depends: m032, m034
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=21-32
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m032)
2. [TextGenerator.__init__(self, cfg)](#context-m034)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.step(self, token, hidden, memory, projected, mask, source). Viết step dùng masked additive attention, GRUCell và gate trộn vocabulary với copy bằng scatter_add. Giữ thứ tự đầu ra: kết quả 1, hidden. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m036"></a>

```text
Tóm tắt giao diện TextGenerator.step(self, token, hidden, memory, projected, mask, source) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]
```

```python
# Dán code hiện tại của TextGenerator.step(self, token, hidden, memory, projected, mask, source) tại đây.
```

## Bước 28: TextGenerator.forward(self, source, lengths, target)

<!-- module
ID: m037
Depends: m032, m034, m035, m036
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=33-38
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=13; lines=13-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m032)
2. [TextGenerator.__init__(self, cfg)](#context-m034)
3. [TextGenerator.encode(self, source, lengths)](#context-m035)
4. [TextGenerator.step(self, token, hidden, memory, projected, mask, source)](#context-m036)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.forward(self, source, lengths, target). Viết forward(source,lengths,target) gọi step theo token target và trả chuỗi log-prob. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def forward(self,source,lengths,target):
        hidden=self.encode(source,lengths);output=[]
        for t in range(target.size(1)):
            hidden=self.decoder(self.embedding(target[:,t]),hidden)
            output.append(self.output(self.dropout(hidden)).log_softmax(-1))
        return torch.stack(output,1)
```

### Tóm tắt context để dùng tiếp

<a id="context-m037"></a>

```text
Tóm tắt giao diện TextGenerator.forward(self, source, lengths, target) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]
```

```python
# Dán code hiện tại của TextGenerator.forward(self, source, lengths, target) tại đây.
```

## Bước 29: TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2

<!-- module
ID: m038
Depends: m024, m032, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=39-48
Summary budget: input=1200; output=500; reserve=300
Config owns: BEAM_SIZE, LENGTH_PENALTY
Source: baseline_basic.ipynb#cell=13; lines=19-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [TextGenerator.__init__(self, cfg)](#context-m034)
4. [TextGenerator.encode(self, source, lengths)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2. Viết phần đầu generate, mở rộng memory, projected, mask, source và hidden cho beam search. Tự chọn BEAM_SIZE, LENGTH_PENALTY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
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

<a id="context-m038"></a>

```text
Tóm tắt giao diện TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2 tại đây.
```

## Bước 30: TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2

<!-- module
ID: m039
Depends: m024, m032, m034, m036, m038
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=49-63
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=13; lines=19-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [TextGenerator.__init__(self, cfg)](#context-m034)
4. [TextGenerator.step(self, token, hidden, memory, projected, mask, source)](#context-m036)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2. Viết phần đầu generate, mở rộng memory, projected, mask, source và hidden cho beam search. Viết vòng top-k beam, gather hidden theo parent, chặn PAD/BOS và giữ nguyên beam đã EOS. Dùng step trả logp,next_hidden; gather seq/done/lens/next_hidden theo parent; EOS done có logp=0; dừng khi mọi done; trả phần vòng lặp để ghép. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Đây là phần 2/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
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

<a id="context-m039"></a>

```text
Tóm tắt giao diện TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2 tại đây.
```

## Bước 31: Ghép giao diện class TextGenerator

<!-- module
ID: m040
Depends: m034, m035, m036, m037, m038, m039
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=1-63
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [TextGenerator.__init__(self, cfg)](#context-m034)
2. [TextGenerator.encode(self, source, lengths)](#context-m035)
3. [TextGenerator.step(self, token, hidden, memory, projected, mask, source)](#context-m036)
4. [TextGenerator.forward(self, source, lengths, target)](#context-m037)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)
6. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2](#context-m039)

<a id="context-m040"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class TextGenerator kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m035 — TextGenerator.encode(self, source, lengths):
[Dán tóm tắt m035]

Context m036 — TextGenerator.step(self, token, hidden, memory, projected, mask, source):
[Dán tóm tắt m036]

Context m037 — TextGenerator.forward(self, source, lengths, target):
[Dán tóm tắt m037]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m039 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 2/2:
[Dán tóm tắt m039]
```

## Bước 32: build_model(cfg)

<!-- module
ID: m041
Depends: m040
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=13; lines=64-64
Source: baseline_basic.ipynb#cell=13; lines=31-31
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class TextGenerator](#context-m040)

<a id="context-m041"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện build_model(cfg) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_model(cfg):return TextGenerator(cfg)
```

## Bước 33: errant_annotator() — phần 1/2

<!-- module
ID: m045
Depends: m009
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=8-29
Summary budget: input=1200; output=500; reserve=300
Config owns: ALLOW_PRETRAINED_EVALUATORS, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL
Config uses: ALLOW_PRETRAINED_EVALUATORS, ARTIFACT_DIR, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL
Source: baseline_basic.ipynb#cell=16; lines=8-29
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung errant_annotator() — phần 1/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ ràng buộc: ALLOW_PRETRAINED_EVALUATORS=True. Tự chọn SPACY_MODEL_SHA256, SPACY_MODEL_URL phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: METRIC_CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def errant_annotator():
    global _ANNOTATOR
    if _ANNOTATOR is not None:return _ANNOTATOR
    if not ALLOW_PRETRAINED_EVALUATORS:raise RuntimeError("ERRANT chuẩn cần evaluator spaCy đã được cho phép riêng, không phải task model pretrained.")
    if importlib.util.find_spec("errant") is None:
        subprocess.check_call([sys.executable,"-m","pip","install","errant==3.0.2","spacy>=3.8,<3.9"])
    import spacy,errant
    if errant.__version__!="3.0.2":raise RuntimeError("Cần errant==3.0.2 cho cùng giao thức chấm.")
    model_dir=METRIC_CACHE_DIR/"spacy_en_core_web_sm_3.8.0"
    path=model_dir/"en_core_web_sm"/"en_core_web_sm-3.8.0"
    if not path.is_dir():
        model_dir.mkdir(parents=True,exist_ok=True)
        wheel=model_dir/"en_core_web_sm.whl"
        try:
            if not wheel.is_file():wheel.write_bytes(urllib.request.urlopen(SPACY_MODEL_URL,timeout=120).read())
        except Exception as exc:raise RuntimeError(f"Không tải được evaluator spaCy; đặt đúng wheel tại {wheel}. {exc}") from exc
        if hashlib.sha256(wheel.read_bytes()).hexdigest()!=SPACY_MODEL_SHA256:raise ValueError("SHA-256 wheel spaCy metric không đúng.")
        with zipfile.ZipFile(wheel) as archive:archive.extractall(model_dir)
    spacy.require_cpu()
    # Dùng API metric chính thức; package model được cô lập trong cache đã kiểm SHA.
    # errant.load tự nạp spaCy evaluator được cho phép, không nạp vào task model.
    if str(model_dir) not in sys.path:sys.path.insert(0,str(model_dir))
```

### Tóm tắt context để dùng tiếp

<a id="context-m045"></a>

```text
Tóm tắt giao diện errant_annotator() — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]
```

```python
# Dán code hiện tại của errant_annotator() — phần 1/2 tại đây.
```

## Bước 34: errant_annotator() — phần 2/2

<!-- module
ID: m046
Depends: m045
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=30-32
Config uses: ALLOW_PRETRAINED_EVALUATORS, METRIC_CACHE_DIR, SPACY_MODEL_SHA256, SPACY_MODEL_URL
Source: baseline_basic.ipynb#cell=16; lines=30-32
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_annotator() — phần 1/2](#context-m045)

<a id="context-m046"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện errant_annotator() — phần 2/2 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m045 — errant_annotator() — phần 1/2:
[Dán tóm tắt m045]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    _ANNOTATOR=errant.load("en")
    print("ERRANT evaluator:",errant.__version__,"spaCy",spacy.__version__,"model en_core_web_sm3.8.0; chỉ metric")
    return _ANNOTATOR
```

## Bước 35: edits_as_rows(annotator, original, corrected, coder)

<!-- module
ID: m047
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=34-39
Source: baseline_basic.ipynb#cell=16; lines=34-39
-->

<a id="context-m047"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện edits_as_rows(annotator, original, corrected, coder) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def edits_as_rows(annotator,original,corrected,coder):
    if original.text.strip()==corrected.strip():return [[-1,-1,"noop","-NONE-",coder]]
    target=annotator.parse(corrected,tokenise=True)
    edits=annotator.annotate(original,target,lev=False,merging="rules")
    if not edits:return [[-1,-1,"noop","-NONE-",coder]]
    return [[edit.o_start,edit.o_end,edit.type,edit.c_str,coder] for edit in edits]
```

## Bước 36: errant_corpus(records, predictions)

<!-- module
ID: m048
Depends: m046, m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=41-60
Summary budget: input=1200; output=500; reserve=300
Config owns: _REFERENCE_EDITS
Config uses: _REFERENCE_EDITS
Source: baseline_basic.ipynb#cell=16; lines=41-60
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_annotator() — phần 2/2](#context-m046)
2. [edits_as_rows(annotator, original, corrected, coder)](#context-m047)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung errant_corpus(records, predictions). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường edit_counts, errant_f05, errant_mode, errant_precision, errant_recall. Khai báo cùng phần này: _REFERENCE_EDITS được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]

Context m047 — edits_as_rows(annotator, original, corrected, coder):
[Dán tóm tắt m047]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def errant_corpus(records,predictions):
    from errant.commands.compare_m2 import process_edits,evaluate_edits,computeFScore
    annotator=errant_annotator()
    args=SimpleNamespace(dt=False,ds=False,cse=False,single=False,multi=False,filt=[],cat=None,beta=.5,verbose=False)
    best=Counter({"tp":0,"fp":0,"fn":0})
    for index,(record,prediction) in enumerate(zip(records,predictions)):
        key=fingerprint({"src":record["src"],"refs":record["refs"]})
        if key not in _REFERENCE_EDITS:
            original=annotator.parse(record["src"],tokenise=True)
            rows=[]
            for coder,reference in enumerate(record["refs"]):rows.extend(edits_as_rows(annotator,original,reference,coder))
            _REFERENCE_EDITS[key]=(original,rows)
        original,reference_rows=_REFERENCE_EDITS[key]
        hypothesis_rows=edits_as_rows(annotator,original,prediction,0)
        counts,_=evaluate_edits(process_edits(hypothesis_rows,args),process_edits(reference_rows,args),
                                best,index,original.text,args)
        best+=Counter(counts)
    precision,recall,score=computeFScore(best["tp"],best["fp"],best["fn"],.5)
    return {"errant_f05":100*score,"errant_precision":100*precision,"errant_recall":100*recall,
            "edit_counts":dict(best),"errant_mode":"3.0.2 correction-span, beta=.5; spaCy3.8.0 model; tokenise=True"}
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện errant_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]

Context m047 — edits_as_rows(annotator, original, corrected, coder):
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của errant_corpus(records, predictions) tại đây.
```

## Bước 37: gleu_corpus(records, predictions)

<!-- module
ID: m049
Depends: m019
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=16; lines=62-83
Summary budget: input=1200; output=500; reserve=300
Config owns: GLEU_ITERATIONS
Config uses: GLEU_ITERATIONS, SEED
Source: baseline_basic.ipynb#cell=16; lines=62-83
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — held, held_sources, native…](#context-m019)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung gleu_corpus(records, predictions). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường gleu, gleu_iterations, gleu_variant. Tự chọn GLEU_ITERATIONS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def gleu_corpus(records,predictions):
    # GLEU-2016 source-aware theo cnap/gec-ranking, không phải nltk corpus_gleu.
    prepared=[]
    for record,prediction in zip(records,predictions):
        source=record["src"].split();hyp=prediction.split();rows=[]
        for reference in record["refs"]:
            ref=reference.split();stats=[len(hyp),len(ref)]
            for n in range(1,5):
                def counter(words):return Counter(tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1)))
                s,h,r=counter(source),counter(hyp),counter(ref)
                incorrect=Counter({g:c for g,c in s.items() if g not in r})
                stats.extend([max(sum((h&r).values())-sum((h&incorrect).values()),0),max(len(hyp)+1-n,0)])
            rows.append(stats)
        prepared.append(rows)
    rng=random.Random(SEED);values=[]
    iterations=1 if all(len(rows)==1 for rows in prepared) else GLEU_ITERATIONS
    for _ in range(iterations):
        stats=np.array([rows[rng.randrange(len(rows))] for rows in prepared],dtype=np.float64).sum(0)
        if np.any(stats==0):values.append(0.);continue
        length,reference_length=stats[:2]
        values.append(math.exp(min(0.,1-reference_length/length)+sum(math.log(x/y) for x,y in zip(stats[2::2],stats[3::2]))/4))
    return {"gleu":100*float(np.mean(values)),"gleu_iterations":iterations,"gleu_variant":"GLEU2016 source-aware; whitespace tokens; seed42"}
```

### Tóm tắt context để dùng tiếp

<a id="context-m049"></a>

```text
Tóm tắt giao diện gleu_corpus(records, predictions) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m019 — Dữ liệu và giao thức chia — held, held_sources, native…:
[Dán tóm tắt m019]
```

```python
# Dán code hiện tại của gleu_corpus(records, predictions) tại đây.
```

## Bước 38: score_corpus(records, predictions)

<!-- module
ID: m050
Depends: m048, m049
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=85-86
Source: baseline_basic.ipynb#cell=16; lines=85-86
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_corpus(records, predictions)](#context-m048)
2. [gleu_corpus(records, predictions)](#context-m049)

<a id="context-m050"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện score_corpus(records, predictions) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m048 — errant_corpus(records, predictions):
[Dán tóm tắt m048]

Context m049 — gleu_corpus(records, predictions):
[Dán tóm tắt m049]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def score_corpus(records,predictions):
    return {**errant_corpus(records,predictions),**gleu_corpus(records,predictions)}
```

## Bước 39: decode_tokens(ids)

<!-- module
ID: m051
Depends: m007, m024, m026
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=17; lines=1-7
Source: baseline_basic.ipynb#cell=17; lines=1-7
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)

<a id="context-m051"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện decode_tokens(ids) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def decode_tokens(ids):
    output=[]
    for index in ids:
        index=int(index)
        if index==EOS_ID:break
        if index not in (PAD_ID,BOS_ID):output.append(index)
    return clean_text(sp.decode(output))
```

## Bước 40: compute_loss(model, batch)

<!-- module
ID: m052
Depends: m024, m032, m040
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=8-14
Summary budget: input=1200; output=500; reserve=300
Config owns: LABEL_SMOOTHING
Config uses: LABEL_SMOOTHING
Source: baseline_basic.ipynb#cell=17; lines=8-14
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
2. [collate_batch(items)](#context-m032)
3. [Ghép giao diện class TextGenerator](#context-m040)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung compute_loss(model, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn LABEL_SMOOTHING phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def compute_loss(model,batch):
    logp=model(batch["source"],batch["lengths"],batch["target"][:,:-1])
    target=batch["target"][:,1:]
    losses=-logp.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    if LABEL_SMOOTHING:losses=(1-LABEL_SMOOTHING)*losses+LABEL_SMOOTHING*(-logp.mean(-1))
    mask=target!=PAD_ID
    return losses[mask].mean()
```

### Tóm tắt context để dùng tiếp

<a id="context-m052"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m032 — collate_batch(items):
[Dán tóm tắt m032]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 41: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m053
Depends: m012, m038, m040, m050, m051
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=17; lines=16-26
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_DECODE_TOKENS
Config uses: BEAM_SIZE, LENGTH_PENALTY, MAX_DECODE_TOKENS
Source: baseline_basic.ipynb#cell=17; lines=16-26
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)
3. [Ghép giao diện class TextGenerator](#context-m040)
4. [score_corpus(records, predictions)](#context-m050)
5. [decode_tokens(ids)](#context-m051)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung evaluate(model, loader, return_predictions=False). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn MAX_DECODE_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại BEAM_SIZE, LENGTH_PENALTY từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]

Context m050 — score_corpus(records, predictions):
[Dán tóm tắt m050]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def evaluate(model,loader,return_predictions=False):
    model.eval();records=[];guesses=[]
    for cpu_batch in loader:
        batch=to_device(cpu_batch)
        outputs=model.generate(batch["source"],batch["lengths"],MAX_DECODE_TOKENS,BEAM_SIZE,LENGTH_PENALTY).cpu().tolist()
        records.extend(batch["records"]);guesses.extend(decode_tokens(ids) for ids in outputs)
    if not records:raise ValueError("Không có mẫu đánh giá.")
    metrics=score_corpus(records,guesses)
    rows=[{"id":r["id"],"source":r["src"],"prediction":prediction,"references":r["refs"]} for r,prediction in zip(records,guesses)]
    return (metrics,rows) if return_predictions else metrics
```

### Tóm tắt context để dùng tiếp

<a id="context-m053"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m040 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m040]

Context m050 — score_corpus(records, predictions):
[Dán tóm tắt m050]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 42: Metric, loss và giải mã

<!-- module
ID: m054
Depends: m046
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=1-1
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [errant_annotator() — phần 2/2](#context-m046)

### Prompt tạo hoặc sửa module

```text
Viết Metric, loss và giải mã. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m054"></a>

```text
Tóm tắt giao diện Metric, loss và giải mã từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m046 — errant_annotator() — phần 2/2:
[Dán tóm tắt m046]
```

```python
# Dán code hiện tại của Metric, loss và giải mã tại đây.
```

## Bước 43: Huấn luyện và checkpoint — model, optimizer, total_steps

<!-- module
ID: m055
Depends: m012, m033, m034, m041
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=1-4
Summary budget: input=1200; output=500; reserve=300
Config owns: EPOCHS, LEARNING_RATE, WEIGHT_DECAY
Config uses: DEVICE, EPOCHS, LEARNING_RATE, MODEL_CONFIG, WEIGHT_DECAY
Source: baseline_basic.ipynb#cell=5; lines=25-46
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Tạo DataLoader cho các split](#context-m033)
3. [TextGenerator.__init__(self, cfg)](#context-m034)
4. [build_model(cfg)](#context-m041)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Huấn luyện và checkpoint — model, optimizer, total_steps. Khởi tạo optimizer AdamW cho tham số có thể học. Tự chọn EPOCHS, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'spaCy', 'model_id': 'en_core_web_sm', 'revision': '3.8.0', 'training': 'frozen', 'sha256': '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85', 'purpose': 'ERRANT annotation, không tạo task output'}]
VAL_FRACTION = 0.1
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
VOCAB_SIZE = 8000
MAX_SOURCE_TOKENS = 160
MAX_TARGET_TOKENS = 160
MAX_DECODE_TOKENS = 160
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BEAM_SIZE = 1
LENGTH_PENALTY = 0.0
LABEL_SMOOTHING = 0.0
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "errant_f05"
GLEU_ITERATIONS = 500
```

### Tóm tắt context để dùng tiếp

<a id="context-m055"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — model, optimizer, total_steps từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — model, optimizer, total_steps tại đây.
```

## Bước 44: lr_multiplier(step)

<!-- module
ID: m056
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=5-7
Summary budget: input=1200; output=500; reserve=300
Config owns: WARMUP_STEPS
Config uses: WARMUP_STEPS
Source: baseline_basic.ipynb#cell=20; lines=5-7
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

<a id="context-m056"></a>

```text
Tóm tắt giao diện lr_multiplier(step) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của lr_multiplier(step) tại đây.
```

## Bước 45: Huấn luyện và checkpoint — best_score, scheduler, stale_epochs

<!-- module
ID: m057
Depends: m055, m056
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=20; lines=8-9
Source: baseline_basic.ipynb#cell=20; lines=8-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m055)
2. [lr_multiplier(step)](#context-m056)

<a id="context-m057"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, scheduler, stale_epochs từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m056 — lr_multiplier(step):
[Dán tóm tắt m056]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
best_score=-math.inf;stale_epochs=0
```

## Bước 46: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m058
Depends: m009, m012, m026, m033, m052, m053, m055, m057
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=10-38
Summary budget: input=1200; output=500; reserve=300
Config owns: CHECKPOINT_PATH, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, EPOCHS, GRAD_CLIP, PATIENCE, PRIMARY_METRIC
Source: baseline_basic.ipynb#cell=20; lines=10-38
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
4. [Tạo DataLoader cho các split](#context-m033)
5. [compute_loss(model, batch)](#context-m052)
6. [evaluate(model, loader, return_predictions=False)](#context-m053)
7. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m055)
8. [Huấn luyện và checkpoint — best_score, scheduler, stale_epochs](#context-m057)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='errant_f05'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m052 — compute_loss(model, batch):
[Dán tóm tắt m052]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m057 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m057]
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

<a id="context-m058"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m052 — compute_loss(model, batch):
[Dán tóm tắt m052]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m055 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m055]

Context m057 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m057]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 47: Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…

<!-- module
ID: m059
Depends: m009, m012, m017, m022, m026, m027, m033, m034, m041, m053, m058
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=22; lines=1-17
Config uses: ARTIFACT_DIR, CHECKPOINT_PATH, DEVICE, DROPOUT, EMBED_DIM, HIDDEN_DIM, SOURCE
Source: baseline_basic.ipynb#cell=22; lines=1-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Dữ liệu và giao thức chia — raw](#context-m017)
4. [Chuẩn hóa và chia dữ liệu](#context-m022)
5. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
6. [restore_tokenizer(state)](#context-m027)
7. [Tạo DataLoader cho các split](#context-m033)
8. [TextGenerator.__init__(self, cfg)](#context-m034)
9. [build_model(cfg)](#context-m041)
10. [evaluate(model, loader, return_predictions=False)](#context-m053)
11. [Vòng huấn luyện, validation và lưu checkpoint](#context-m058)

<a id="context-m059"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m017 — Dữ liệu và giao thức chia — raw:
[Dán tóm tắt m017]

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m027 — restore_tokenizer(state):
[Dán tóm tắt m027]

Context m033 — Tạo DataLoader cho các split:
[Dán tóm tắt m033]

Context m034 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m034]

Context m041 — build_model(cfg):
[Dán tóm tắt m041]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m058 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m058]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
checkpoint=torch.load(CHECKPOINT_PATH,map_location=DEVICE,weights_only=True)
restore_tokenizer(checkpoint["tokenizer"])
# Cấu hình kiến trúc ở notebook; kích thước vocab được lấy sau khi restore.
inference_config = {"vocab_size":sp.get_piece_size(),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}
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

## Bước 48: rewrite_text(text)

<!-- module
ID: m060
Depends: m007, m012, m024, m026, m038, m051, m053, m059
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=24; lines=1-9
Config uses: BEAM_SIZE, DEVICE, LENGTH_PENALTY, MAX_DECODE_TOKENS, MAX_SOURCE_TOKENS
Source: baseline_basic.ipynb#cell=24; lines=1-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [to_device(batch)](#context-m012)
3. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m024)
4. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m026)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2](#context-m038)
6. [decode_tokens(ids)](#context-m051)
7. [evaluate(model, loader, return_predictions=False)](#context-m053)
8. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m059)

<a id="context-m060"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện rewrite_text(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m024 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m024]

Context m026 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m026]

Context m038 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) — phần 1/2:
[Dán tóm tắt m038]

Context m051 — decode_tokens(ids):
[Dán tóm tắt m051]

Context m053 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m053]

Context m059 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m059]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
@torch.no_grad()
def rewrite_text(text):
    text=clean_text(text)
    if not text:raise ValueError("Cần văn bản không rỗng.")
    ids=sp.encode(text,out_type=int)[:MAX_SOURCE_TOKENS]
    source=torch.tensor([ids or [UNK_ID]],device=DEVICE)
    best_model.eval()
    output=best_model.generate(source,torch.tensor([source.size(1)]),MAX_DECODE_TOKENS,BEAM_SIZE,LENGTH_PENALTY)
    return decode_tokens(output[0].tolist())
```

## Bước 49: Suy luận văn bản mới

<!-- module
ID: m061
Depends: m022, m060
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=24; lines=11-11
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m022)
2. [rewrite_text(text)](#context-m060)

### Prompt tạo hoặc sửa module

```text
Viết Suy luận văn bản mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m060 — rewrite_text(text):
[Dán tóm tắt m060]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m061"></a>

```text
Tóm tắt giao diện Suy luận văn bản mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m022 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m022]

Context m060 — rewrite_text(text):
[Dán tóm tắt m060]
```

```python
# Dán code hiện tại của Suy luận văn bản mới tại đây.
```

## Bước 50: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m010, m011, m054, m061
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 2/2](#context-m010)
2. [decode_utf8(value)](#context-m011)
3. [Metric, loss và giải mã](#context-m054)
4. [Suy luận văn bản mới](#context-m061)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]

Context m054 — Metric, loss và giải mã:
[Dán tóm tắt m054]

Context m061 — Suy luận văn bản mới:
[Dán tóm tắt m061]
```
