# Prompt basic: TED PT→EN: BPE và Transformer scratch

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

## Bước 6: load_native()

<!-- module
ID: m014
Depends: m008, m009, m010, m011
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=8; lines=1-20
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_MODE, LOCAL_DATA_DIR, SOURCE
Config uses: CACHE_DIR, DATA_MODE, LOCAL_DATA_DIR, SOURCE
Source: baseline_basic.ipynb#cell=8; lines=1-20
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m008)
2. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
3. [tfds_rows(builder, splits, supervised=False) — phần 2/2](#context-m010)
4. [decode_utf8(value)](#context-m011)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung load_native(). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Nguồn dữ liệu dataset: TED HRLR Translate pt_to_en; tfds_builder: ted_hrlr_translate/pt_to_en:1.0.0. Giữ split, ID, nhãn và references của nguồn. Giữ ràng buộc: DATA_MODE='hf_parquet_recovery'. Khai báo cùng phần này: LOCAL_DATA_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại CACHE_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def load_native():
    splits=["train","validation","test"]
    if DATA_MODE=="tfds":
        raw=tfds_rows(SOURCE["tfds_builder"],splits,supervised=True)
        return {split:[{"src":decode_utf8(pt),"refs":[decode_utf8(en)]} for pt,en in raw[split]] for split in splits}
    if DATA_MODE=="hf_parquet_recovery":
        # Cùng corpus của tác giả neulab, không đổi dataset; branch Parquet cố định.
        from datasets import load_dataset
        files={split:f"https://huggingface.co/datasets/neulab/ted_hrlr/resolve/{SOURCE['recovery_revision']}/pt_to_en/{split}/0000.parquet" for split in splits}
        try:raw=load_dataset("parquet",data_files=files,cache_dir=str(CACHE_DIR))
        except Exception as exc:raise RuntimeError(f"Không đọc được recovery TED cùng corpus. Dùng cache TFDS hoặc local_jsonl đúng PT–EN; không có dataset thay thế. {exc}") from exc
        output={}
        for split in splits:
            rows=[r["translation"] for r in raw[split]]
            # HF conversion có một dòng rỗng cuối mỗi split; builder TFDS bỏ hàng nếu một bên rỗng.
            output[split]=[{"src":r["pt"],"refs":[r["en"]]} for r in rows if r["pt"] and r["en"]]
        return output
    if DATA_MODE=="local_jsonl":
        return {split:[{"src":r["pt"],"refs":[r["en"]]} for r in read_jsonl(LOCAL_DATA_DIR/f"{split}.jsonl") if r["pt"] and r["en"]] for split in splits}
    raise ValueError("DATA_MODE phải là tfds, hf_parquet_recovery hoặc local_jsonl.")
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện load_native() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m008 — read_jsonl(path):
[Dán tóm tắt m008]

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m010 — tfds_rows(builder, splits, supervised=False) — phần 2/2:
[Dán tóm tắt m010]

Context m011 — decode_utf8(value):
[Dán tóm tắt m011]
```

```python
# Dán code hiện tại của load_native() tại đây.
```

## Bước 7: Dữ liệu và giao thức chia — native

<!-- module
ID: m015
Depends: m014
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=8; lines=21-21
Source: baseline_basic.ipynb#cell=8; lines=21-21
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [load_native()](#context-m014)

<a id="context-m015"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — native từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m014 — load_native():
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
native=load_native()
```

## Bước 8: Dữ liệu và giao thức chia — record, records, removed_train…

<!-- module
ID: m016
Depends: m007, m015
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
2. [Dữ liệu và giao thức chia — native](#context-m015)

<a id="context-m016"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Dữ liệu và giao thức chia — record, records, removed_train… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m015 — Dữ liệu và giao thức chia — native:
[Dán tóm tắt m015]
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

## Bước 9: canonical_records(rows, split)

<!-- module
ID: m017
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=12-16
Source: baseline_basic.ipynb#cell=9; lines=12-16
-->

<a id="context-m017"></a>

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

## Bước 10: Chuẩn hóa và chia dữ liệu

<!-- module
ID: m018
Depends: m016, m017
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=17-26
Source: baseline_basic.ipynb#cell=9; lines=17-26
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Dữ liệu và giao thức chia — record, records, removed_train…](#context-m016)
2. [canonical_records(rows, split)](#context-m017)

<a id="context-m018"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Chuẩn hóa và chia dữ liệu từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m016 — Dữ liệu và giao thức chia — record, records, removed_train…:
[Dán tóm tắt m016]

Context m017 — canonical_records(rows, split):
[Dán tóm tắt m017]
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

## Bước 11: Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…

<!-- module
ID: m020
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
Sửa hoặc bổ sung Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…. Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
PAD_ID,UNK_ID,BOS_ID,EOS_ID=0,1,2,3
```

### Tóm tắt context để dùng tiếp

<a id="context-m020"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID… tại đây.
```

## Bước 12: tokenizer_corpus()

<!-- module
ID: m021
Depends: m018
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=4-7
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m018)

### Prompt tạo hoặc sửa module

```text
Viết tokenizer_corpus(). Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện tokenizer_corpus() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]
```

```python
# Dán code hiện tại của tokenizer_corpus() tại đây.
```

## Bước 13: Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp

<!-- module
ID: m022
Depends: m020, m021
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=8-18
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE
Config uses: MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE
Source: baseline_basic.ipynb#cell=5; lines=25-48
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [tokenizer_corpus()](#context-m021)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp. Fit SentencePiece BPE scratch với byte fallback chỉ trên train; giữ special IDs đã thống nhất, lưu tokenizer và trạng thái tiền xử lý để reload. PREPROCESS_CONFIG chỉ lưu thiết lập mã hóa/chuẩn hóa, không chứa cấu hình giải mã. Tự chọn MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS, VOCAB_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m021 — tokenizer_corpus():
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'Unbabel/wmt22-comet-da', 'revision': '2760a223ac957f30acfb18c8aa649b01cf1d75f2', 'training': 'frozen', 'purpose': 'COMET; chỉ final scoring'}, {'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'FacebookAI/xlm-roberta-large', 'revision': 'c23d21b0620b635a76227c604d44e43a9f0ee389', 'training': 'frozen', 'purpose': 'COMET encoder config/tokenizer; weights trong metric checkpoint'}]
VAL_FRACTION = 0.1
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32
VOCAB_SIZE = 8000
MAX_SOURCE_TOKENS = 128
MAX_TARGET_TOKENS = 128
MAX_DECODE_TOKENS = 128
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BEAM_SIZE = 1
LENGTH_PENALTY = 0.0
LABEL_SMOOTHING = 0.0
EPOCHS = 45
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "bleu"
RUN_COMET = True
COMET_BATCH_SIZE = 8
METRICS_OFFLINE = False
```

### Tóm tắt context để dùng tiếp

<a id="context-m022"></a>

```text
Tóm tắt giao diện Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m021 — tokenizer_corpus():
[Dán tóm tắt m021]
```

```python
# Dán code hiện tại của Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp tại đây.
```

## Bước 14: restore_tokenizer(state)

<!-- module
ID: m023
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
Sửa hoặc bổ sung restore_tokenizer(state). Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def restore_tokenizer(state):
    global sp
    sp=BasicWordTokenizer(state["vocab"])
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện restore_tokenizer(state) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của restore_tokenizer(state) tại đây.
```

## Bước 15: PairDataset.__init__(self, records, training=False)

<!-- module
ID: m024
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=24-26
Source: baseline_basic.ipynb#cell=11; lines=34-36
-->

<a id="context-m024"></a>

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

## Bước 16: PairDataset.__len__(self)

<!-- module
ID: m025
Depends: m024
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=27-27
Source: baseline_basic.ipynb#cell=11; lines=37-37
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m024)

<a id="context-m025"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện PairDataset.__len__(self) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    def __len__(self):return len(self.index)
```

## Bước 17: PairDataset.__getitem__(self, index)

<!-- module
ID: m026
Depends: m020, m022, m024
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

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
3. [PairDataset.__init__(self, records, training=False)](#context-m024)

<a id="context-m026"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện PairDataset.__getitem__(self, index) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]
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

## Bước 18: Ghép giao diện class PairDataset

<!-- module
ID: m027
Depends: m024, m025, m026
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=11; lines=23-33
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PairDataset.__init__(self, records, training=False)](#context-m024)
2. [PairDataset.__len__(self)](#context-m025)
3. [PairDataset.__getitem__(self, index)](#context-m026)

<a id="context-m027"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class PairDataset kế thừa Dataset. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m024 — PairDataset.__init__(self, records, training=False):
[Dán tóm tắt m024]

Context m025 — PairDataset.__len__(self):
[Dán tóm tắt m025]

Context m026 — PairDataset.__getitem__(self, index):
[Dán tóm tắt m026]
```

## Bước 19: collate_batch(items)

<!-- module
ID: m028
Depends: m020
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=35-41
Source: baseline_basic.ipynb#cell=11; lines=45-51
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)

<a id="context-m028"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện collate_batch(items) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]
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

## Bước 20: to_device(batch)

<!-- module
ID: m012
Depends: m028
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

1. [collate_batch(items)](#context-m028)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung to_device(batch). Chuyển các tensor cần thiết sang device; giữ tensor độ dài trên CPU nếu API đóng gói chuỗi hoặc CTC cần. Khai báo cùng phần này: DEVICE được suy ra từ dữ liệu/context hoặc API tương ứng. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
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

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của to_device(batch) tại đây.
```

## Bước 21: make_loader(dataset, train=False)

<!-- module
ID: m013
Depends: m028
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=6; lines=43-46
Summary budget: input=1200; output=500; reserve=300
Config owns: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
Config uses: BATCH_SIZE, EVAL_BATCH_SIZE, SEED
Source: baseline_basic.ipynb#cell=6; lines=43-46
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m028)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_loader(dataset, train=False). Tạo DataLoader cho train và đánh giá với collate đã thống nhất, shuffle chỉ cho train và num_workers=0. Giữ ràng buộc: SEED=42. Tự chọn BATCH_SIZE, EVAL_BATCH_SIZE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
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

Context m028 — collate_batch(items):
[Dán tóm tắt m028]
```

```python
# Dán code hiện tại của make_loader(dataset, train=False) tại đây.
```

## Bước 22: Tạo DataLoader cho các split

<!-- module
ID: m029
Depends: m013, m018, m022, m027
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
2. [Chuẩn hóa và chia dữ liệu](#context-m018)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
4. [Ghép giao diện class PairDataset](#context-m027)

<a id="context-m029"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tạo DataLoader cho các split từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m013 — make_loader(dataset, train=False):
[Dán tóm tắt m013]

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m027 — Ghép giao diện class PairDataset:
[Dán tóm tắt m027]
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

## Bước 23: PositionEncoding.__init__(self, dim, dropout, max_length)

<!-- module
ID: m030
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=2-7
Summary budget: input=1200; output=500; reserve=300
-->

### Prompt tạo hoặc sửa module

```text
Viết PositionEncoding.__init__(self, dim, dropout, max_length). Viết PositionEncoding sinusoidal với buffer và dropout cho tensor batch-first. Tự chọn kích thước và siêu tham số kiến trúc phù hợp, nêu ngắn lý do; đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class PositionEncoding kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m030"></a>

```text
Tóm tắt giao diện PositionEncoding.__init__(self, dim, dropout, max_length) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của PositionEncoding.__init__(self, dim, dropout, max_length) tại đây.
```

## Bước 24: PositionEncoding.forward(self, x)

<!-- module
ID: m031
Depends: m030
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=8-8
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PositionEncoding.__init__(self, dim, dropout, max_length)](#context-m030)

### Prompt tạo hoặc sửa module

```text
Viết PositionEncoding.forward(self, x). Viết PositionEncoding sinusoidal với buffer và dropout cho tensor batch-first. Viết method trong class PositionEncoding kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m030 — PositionEncoding.__init__(self, dim, dropout, max_length):
[Dán tóm tắt m030]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m031"></a>

```text
Tóm tắt giao diện PositionEncoding.forward(self, x) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m030 — PositionEncoding.__init__(self, dim, dropout, max_length):
[Dán tóm tắt m030]
```

```python
# Dán code hiện tại của PositionEncoding.forward(self, x) tại đây.
```

## Bước 25: Ghép giao diện class PositionEncoding

<!-- module
ID: m032
Depends: m030, m031
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=1-8
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [PositionEncoding.__init__(self, dim, dropout, max_length)](#context-m030)
2. [PositionEncoding.forward(self, x)](#context-m031)

<a id="context-m032"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class PositionEncoding kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m030 — PositionEncoding.__init__(self, dim, dropout, max_length):
[Dán tóm tắt m030]

Context m031 — PositionEncoding.forward(self, x):
[Dán tóm tắt m031]
```

## Bước 26: TextGenerator.__init__(self, cfg)

<!-- module
ID: m033
Depends: m020, m022, m032
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=11-20
Summary budget: input=1200; output=500; reserve=300
Config owns: DROPOUT, EMBED_DIM, HIDDEN_DIM, MODEL_CONFIG
Config uses: DROPOUT, EMBED_DIM, HIDDEN_DIM
Source: baseline_basic.ipynb#cell=13; lines=2-8
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
3. [Ghép giao diện class PositionEncoding](#context-m032)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.__init__(self, cfg). Thay constructor TextGenerator bằng Transformer encoder-decoder scratch, tied embedding/output và batch-first. Tự chọn DROPOUT, EMBED_DIM, HIDDEN_DIM phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: MODEL_CONFIG có các khóa vocab_size, embed_dim, hidden_dim, dropout; suy ra vocabulary/số lớp từ context dữ liệu. Đặt cấu hình dùng chung ngoài class và truyền qua cfg nếu API có cfg. Số lớp, vocabulary và kích thước đầu vào đã chốt phải lấy từ context/API, không tự đổi. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class PositionEncoding:
[Dán tóm tắt m032]
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

<a id="context-m033"></a>

```text
Tóm tắt giao diện TextGenerator.__init__(self, cfg) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m032 — Ghép giao diện class PositionEncoding:
[Dán tóm tắt m032]
```

```python
# Dán code hiện tại của TextGenerator.__init__(self, cfg) tại đây.
```

## Bước 27: TextGenerator.encode(self, source)

<!-- module
ID: m034
Depends: m020, m028, m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=21-23
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=13; lines=9-12
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [TextGenerator.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.encode(self, source). Viết encode(source) dùng embedding scale căn embed_dim, positional encoding và source padding mask. Giữ thứ tự đầu ra: kết quả 1, mask. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
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

<a id="context-m034"></a>

```text
Tóm tắt giao diện TextGenerator.encode(self, source) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của TextGenerator.encode(self, source) tại đây.
```

## Bước 28: TextGenerator.decode(self, target, memory, source_mask)

<!-- module
ID: m035
Depends: m020, m028, m033
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=24-28
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [TextGenerator.__init__(self, cfg)](#context-m033)

### Prompt tạo hoặc sửa module

```text
Viết TextGenerator.decode(self, target, memory, source_mask). Viết decode(target,memory,source_mask) có causal mask và padding masks nhất quán kiểu bool. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m035"></a>

```text
Tóm tắt giao diện TextGenerator.decode(self, target, memory, source_mask) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]
```

```python
# Dán code hiện tại của TextGenerator.decode(self, target, memory, source_mask) tại đây.
```

## Bước 29: TextGenerator.forward(self, source, lengths, target)

<!-- module
ID: m036
Depends: m028, m033, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=29-31
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=13; lines=13-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [collate_batch(items)](#context-m028)
2. [TextGenerator.__init__(self, cfg)](#context-m033)
3. [TextGenerator.encode(self, source)](#context-m034)
4. [TextGenerator.decode(self, target, memory, source_mask)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.forward(self, source, lengths, target). Viết forward(source,lengths,target) nối encode và decode, trả log_softmax cho teacher forcing. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
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

<a id="context-m036"></a>

```text
Tóm tắt giao diện TextGenerator.forward(self, source, lengths, target) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của TextGenerator.forward(self, source, lengths, target) tại đây.
```

## Bước 30: TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)

<!-- module
ID: m037
Depends: m020, m028, m033, m034, m035
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=13; lines=32-52
Summary budget: input=1200; output=500; reserve=300
Config owns: BEAM_SIZE, LENGTH_PENALTY
Source: baseline_basic.ipynb#cell=13; lines=19-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [TextGenerator.__init__(self, cfg)](#context-m033)
4. [TextGenerator.encode(self, source)](#context-m034)
5. [TextGenerator.decode(self, target, memory, source_mask)](#context-m035)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty). Viết phần khởi tạo beam cho generate, lặp memory/mask theo beam và bắt đầu BOS. Viết vòng beam chọn top-k theo log-prob, gom lại parent beam và đóng băng beam đã EOS. Chặn PAD/BOS; EOS của beam done có logp=0; gather seq/done/lens theo parent; dừng khi mọi done; không tính length thêm cho beam đã đóng; ghép sau phần khởi tạo generate. Tự chọn BEAM_SIZE, LENGTH_PENALTY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Viết method trong class TextGenerator kế thừa nn.Module; giữ phần đã có. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
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

<a id="context-m037"></a>

```text
Tóm tắt giao diện TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]
```

```python
# Dán code hiện tại của TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty) tại đây.
```

## Bước 31: Ghép giao diện class TextGenerator

<!-- module
ID: m038
Depends: m033, m034, m035, m036, m037
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Target: baseline.ipynb#cell=13; lines=10-52
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [TextGenerator.__init__(self, cfg)](#context-m033)
2. [TextGenerator.encode(self, source)](#context-m034)
3. [TextGenerator.decode(self, target, memory, source_mask)](#context-m035)
4. [TextGenerator.forward(self, source, lengths, target)](#context-m036)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)](#context-m037)

<a id="context-m038"></a>

### Prompt gộp context

```text
Gộp các context thành giao diện class TextGenerator kế thừa nn.Module. Giữ chữ ký methods, I/O và cấu hình đã chọn; chỉ trả chữ, không sinh lại class hoặc chọn lại tham số.

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m034 — TextGenerator.encode(self, source):
[Dán tóm tắt m034]

Context m035 — TextGenerator.decode(self, target, memory, source_mask):
[Dán tóm tắt m035]

Context m036 — TextGenerator.forward(self, source, lengths, target):
[Dán tóm tắt m036]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]
```

## Bước 32: build_model(cfg)

<!-- module
ID: m039
Depends: m038
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=13; lines=53-53
Source: baseline_basic.ipynb#cell=13; lines=31-31
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Ghép giao diện class TextGenerator](#context-m038)

<a id="context-m039"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện build_model(cfg) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_model(cfg):return TextGenerator(cfg)
```

## Bước 33: Metric, loss và giải mã — bleu_metric, chrf_metric

<!-- module
ID: m042
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=2-3
Source: baseline_basic.ipynb#cell=16; lines=2-3
-->

<a id="context-m042"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Metric, loss và giải mã — bleu_metric, chrf_metric từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
bleu_metric=BLEU(tokenize="13a",smooth_method="exp",effective_order=False)
chrf_metric=CHRF(word_order=2)
```

## Bước 34: score_corpus(records, predictions)

<!-- module
ID: m043
Depends: m042
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=16; lines=4-9
Source: baseline_basic.ipynb#cell=16; lines=4-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Metric, loss và giải mã — bleu_metric, chrf_metric](#context-m042)

<a id="context-m043"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện score_corpus(records, predictions) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m042 — Metric, loss và giải mã — bleu_metric, chrf_metric:
[Dán tóm tắt m042]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def score_corpus(records,predictions):
    refs=[[r["refs"][0] for r in records]]
    b=bleu_metric.corpus_score(predictions,refs)
    c=chrf_metric.corpus_score(predictions,refs)
    return {"bleu":float(b.score),"chrfpp":float(c.score),
            "bleu_signature":str(bleu_metric.get_signature()),"chrf_signature":str(chrf_metric.get_signature())}
```

## Bước 35: decode_tokens(ids)

<!-- module
ID: m044
Depends: m007, m020, m022
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
2. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)

<a id="context-m044"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện decode_tokens(ids) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]
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

## Bước 36: compute_loss(model, batch)

<!-- module
ID: m045
Depends: m020, m028, m038
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

1. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
2. [collate_batch(items)](#context-m028)
3. [Ghép giao diện class TextGenerator](#context-m038)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung compute_loss(model, batch). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn LABEL_SMOOTHING phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
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

<a id="context-m045"></a>

```text
Tóm tắt giao diện compute_loss(model, batch) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m028 — collate_batch(items):
[Dán tóm tắt m028]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]
```

```python
# Dán code hiện tại của compute_loss(model, batch) tại đây.
```

## Bước 37: evaluate(model, loader, return_predictions=False)

<!-- module
ID: m046
Depends: m012, m037, m038, m043, m044
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
2. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)](#context-m037)
3. [Ghép giao diện class TextGenerator](#context-m038)
4. [score_corpus(records, predictions)](#context-m043)
5. [decode_tokens(ids)](#context-m044)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung evaluate(model, loader, return_predictions=False). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Tự chọn MAX_DECODE_TOKENS phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại BEAM_SIZE, LENGTH_PENALTY từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]

Context m043 — score_corpus(records, predictions):
[Dán tóm tắt m043]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]
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

<a id="context-m046"></a>

```text
Tóm tắt giao diện evaluate(model, loader, return_predictions=False) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m038 — Ghép giao diện class TextGenerator:
[Dán tóm tắt m038]

Context m043 — score_corpus(records, predictions):
[Dán tóm tắt m043]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]
```

```python
# Dán code hiện tại của evaluate(model, loader, return_predictions=False) tại đây.
```

## Bước 38: Metric, loss và giải mã

<!-- module
ID: m047
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=18; lines=1-2
Summary budget: input=1200; output=500; reserve=300
Config owns: RUN_COMET
Config uses: RUN_COMET
Source: baseline_basic.ipynb#cell=5; lines=25-48
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Metric, loss và giải mã. Giữ ràng buộc: RUN_COMET=True. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'Unbabel/wmt22-comet-da', 'revision': '2760a223ac957f30acfb18c8aa649b01cf1d75f2', 'training': 'frozen', 'purpose': 'COMET; chỉ final scoring'}, {'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'FacebookAI/xlm-roberta-large', 'revision': 'c23d21b0620b635a76227c604d44e43a9f0ee389', 'training': 'frozen', 'purpose': 'COMET encoder config/tokenizer; weights trong metric checkpoint'}]
VAL_FRACTION = 0.1
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32
VOCAB_SIZE = 8000
MAX_SOURCE_TOKENS = 128
MAX_TARGET_TOKENS = 128
MAX_DECODE_TOKENS = 128
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BEAM_SIZE = 1
LENGTH_PENALTY = 0.0
LABEL_SMOOTHING = 0.0
EPOCHS = 45
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "bleu"
RUN_COMET = True
COMET_BATCH_SIZE = 8
METRICS_OFFLINE = False
```

### Tóm tắt context để dùng tiếp

<a id="context-m047"></a>

```text
Tóm tắt giao diện Metric, loss và giải mã từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của Metric, loss và giải mã tại đây.
```

## Bước 39: Huấn luyện và checkpoint — model, optimizer, total_steps

<!-- module
ID: m048
Depends: m012, m029, m033, m039
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=20; lines=1-4
Summary budget: input=1200; output=500; reserve=300
Config owns: EPOCHS, LEARNING_RATE, WEIGHT_DECAY
Config uses: DEVICE, EPOCHS, LEARNING_RATE, MODEL_CONFIG, WEIGHT_DECAY
Source: baseline_basic.ipynb#cell=5; lines=25-48
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Tạo DataLoader cho các split](#context-m029)
3. [TextGenerator.__init__(self, cfg)](#context-m033)
4. [build_model(cfg)](#context-m039)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Huấn luyện và checkpoint — model, optimizer, total_steps. Khởi tạo optimizer AdamW cho tham số có thể học. Tự chọn EPOCHS, LEARNING_RATE, WEIGHT_DECAY phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại DEVICE, MODEL_CONFIG từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
EVALUATION_COMPONENTS = [{'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'Unbabel/wmt22-comet-da', 'revision': '2760a223ac957f30acfb18c8aa649b01cf1d75f2', 'training': 'frozen', 'purpose': 'COMET; chỉ final scoring'}, {'role': 'evaluation_only', 'source': 'huggingface', 'model_id': 'FacebookAI/xlm-roberta-large', 'revision': 'c23d21b0620b635a76227c604d44e43a9f0ee389', 'training': 'frozen', 'purpose': 'COMET encoder config/tokenizer; weights trong metric checkpoint'}]
VAL_FRACTION = 0.1
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32
VOCAB_SIZE = 8000
MAX_SOURCE_TOKENS = 128
MAX_TARGET_TOKENS = 128
MAX_DECODE_TOKENS = 128
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BEAM_SIZE = 1
LENGTH_PENALTY = 0.0
LABEL_SMOOTHING = 0.0
EPOCHS = 45
PATIENCE = 8
LEARNING_RATE = 0.001
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0001
GRAD_CLIP = 1.0
PRIMARY_METRIC = "bleu"
RUN_COMET = True
COMET_BATCH_SIZE = 8
METRICS_OFFLINE = False
```

### Tóm tắt context để dùng tiếp

<a id="context-m048"></a>

```text
Tóm tắt giao diện Huấn luyện và checkpoint — model, optimizer, total_steps từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]
```

```python
# Dán code hiện tại của Huấn luyện và checkpoint — model, optimizer, total_steps tại đây.
```

## Bước 40: lr_multiplier(step)

<!-- module
ID: m049
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

<a id="context-m049"></a>

```text
Tóm tắt giao diện lr_multiplier(step) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của lr_multiplier(step) tại đây.
```

## Bước 41: Huấn luyện và checkpoint — best_score, scheduler, stale_epochs

<!-- module
ID: m050
Depends: m048, m049
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=20; lines=8-9
Source: baseline_basic.ipynb#cell=20; lines=8-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m048)
2. [lr_multiplier(step)](#context-m049)

<a id="context-m050"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Huấn luyện và checkpoint — best_score, scheduler, stale_epochs từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m049 — lr_multiplier(step):
[Dán tóm tắt m049]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
best_score=-math.inf;stale_epochs=0
```

## Bước 42: Vòng huấn luyện, validation và lưu checkpoint

<!-- module
ID: m051
Depends: m009, m012, m022, m029, m045, m046, m048, m050
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
3. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
4. [Tạo DataLoader cho các split](#context-m029)
5. [compute_loss(model, batch)](#context-m045)
6. [evaluate(model, loader, return_predictions=False)](#context-m046)
7. [Huấn luyện và checkpoint — model, optimizer, total_steps](#context-m048)
8. [Huấn luyện và checkpoint — best_score, scheduler, stale_epochs](#context-m050)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Vòng huấn luyện, validation và lưu checkpoint. Viết vòng huấn luyện dùng loss và optimizer đã chốt; đánh giá validation, early stopping và lưu checkpoint tốt nhất gồm weights, cấu hình model/giải mã đã chọn và tokenizer/vocabulary. Giữ metric chọn model từ context; không lưu optimizer/RNG nếu không yêu cầu resume. Giữ ràng buộc: PRIMARY_METRIC='bleu'. Tự chọn GRAD_CLIP, PATIENCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: CHECKPOINT_PATH được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, EPOCHS từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context. Nhắm GPU 32 GB với precision thường; không AMP, tự dò VRAM hoặc tự đổi batch.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m045 — compute_loss(model, batch):
[Dán tóm tắt m045]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m050 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m050]
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

<a id="context-m051"></a>

```text
Tóm tắt giao diện Vòng huấn luyện, validation và lưu checkpoint từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m045 — compute_loss(model, batch):
[Dán tóm tắt m045]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m048 — Huấn luyện và checkpoint — model, optimizer, total_steps:
[Dán tóm tắt m048]

Context m050 — Huấn luyện và checkpoint — best_score, scheduler, stale_epochs:
[Dán tóm tắt m050]
```

```python
# Dán code hiện tại của Vòng huấn luyện, validation và lưu checkpoint tại đây.
```

## Bước 43: Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…

<!-- module
ID: m052
Depends: m009, m012, m014, m018, m022, m023, m029, m033, m039, m046, m051
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
3. [load_native()](#context-m014)
4. [Chuẩn hóa và chia dữ liệu](#context-m018)
5. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
6. [restore_tokenizer(state)](#context-m023)
7. [Tạo DataLoader cho các split](#context-m029)
8. [TextGenerator.__init__(self, cfg)](#context-m033)
9. [build_model(cfg)](#context-m039)
10. [evaluate(model, loader, return_predictions=False)](#context-m046)
11. [Vòng huấn luyện, validation và lưu checkpoint](#context-m051)

<a id="context-m052"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — best_model, checkpoint, inference_config… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m014 — load_native():
[Dán tóm tắt m014]

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m023 — restore_tokenizer(state):
[Dán tóm tắt m023]

Context m029 — Tạo DataLoader cho các split:
[Dán tóm tắt m029]

Context m033 — TextGenerator.__init__(self, cfg):
[Dán tóm tắt m033]

Context m039 — build_model(cfg):
[Dán tóm tắt m039]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m051 — Vòng huấn luyện, validation và lưu checkpoint:
[Dán tóm tắt m051]
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

## Bước 44: add_comet_metric(rows) — phần 1/2

<!-- module
ID: m053
Depends: m009, m012, m047
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=23; lines=1-17
Summary budget: input=1200; output=500; reserve=300
Config owns: ALLOW_PRETRAINED_EVALUATORS, COMET_BATCH_SIZE, METRICS_OFFLINE, METRIC_CACHE_DIR
Config uses: ALLOW_PRETRAINED_EVALUATORS, ARTIFACT_DIR, COMET_BATCH_SIZE, DEVICE, METRICS_OFFLINE, METRIC_CACHE_DIR, RUN_COMET
Source: baseline_basic.ipynb#cell=23; lines=1-17
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [to_device(batch)](#context-m012)
3. [Metric, loss và giải mã](#context-m047)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung add_comet_metric(rows) — phần 1/2. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Kết quả giữ các trường comet, comet_status. Giữ ràng buộc: ALLOW_PRETRAINED_EVALUATORS=True. Tự chọn COMET_BATCH_SIZE, METRICS_OFFLINE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: METRIC_CACHE_DIR được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại ARTIFACT_DIR, DEVICE, RUN_COMET từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 1/2 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def add_comet_metric(rows):
    if not RUN_COMET:return {"comet":None,"comet_status":"Người chạy tắt RUN_COMET; chưa đo, không phải0."}
    if not ALLOW_PRETRAINED_EVALUATORS:raise RuntimeError("Chưa cho phép pretrained evaluator.")
    if importlib.util.find_spec("comet") is None:
        raise RuntimeError("Cần unbabel-comet==2.2.7 trong kernel Python3.11/3.12 phù hợp. "
                           "Cài trước Run All; nếu chủ động không chấm, đặt RUN_COMET=False và kết quả sẽ ghi chưa đo.")
    from huggingface_hub import snapshot_download
    from comet.models.regression.regression_metric import RegressionMetric
    model_path=snapshot_download("Unbabel/wmt22-comet-da",revision="2760a223ac957f30acfb18c8aa649b01cf1d75f2",
        cache_dir=str(METRIC_CACHE_DIR),local_files_only=METRICS_OFFLINE,
        allow_patterns=["checkpoints/model.ckpt","hparams.yaml","README.md"])
    encoder_path=snapshot_download("FacebookAI/xlm-roberta-large",revision="c23d21b0620b635a76227c604d44e43a9f0ee389",
        cache_dir=str(METRIC_CACHE_DIR),local_files_only=METRICS_OFFLINE,
        allow_patterns=["config.json","tokenizer.json","tokenizer_config.json","sentencepiece.bpe.model","special_tokens_map.json"])
    evaluator=RegressionMetric.load_from_checkpoint(str(Path(model_path)/"checkpoints/model.ckpt"),map_location="cpu",
        load_pretrained_weights=False,pretrained_model=str(encoder_path),local_files_only=True,strict=False)
    evaluator.eval()
```

### Tóm tắt context để dùng tiếp

<a id="context-m053"></a>

```text
Tóm tắt giao diện add_comet_metric(rows) — phần 1/2 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]
```

```python
# Dán code hiện tại của add_comet_metric(rows) — phần 1/2 tại đây.
```

## Bước 45: add_comet_metric(rows) — phần 2/2

<!-- module
ID: m054
Depends: m012, m047, m053
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=23; lines=18-24
Config uses: ALLOW_PRETRAINED_EVALUATORS, COMET_BATCH_SIZE, DEVICE, METRICS_OFFLINE, METRIC_CACHE_DIR, RUN_COMET
Source: baseline_basic.ipynb#cell=23; lines=18-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [to_device(batch)](#context-m012)
2. [Metric, loss và giải mã](#context-m047)
3. [add_comet_metric(rows) — phần 1/2](#context-m053)

<a id="context-m054"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện add_comet_metric(rows) — phần 2/2 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m047 — Metric, loss và giải mã:
[Dán tóm tắt m047]

Context m053 — add_comet_metric(rows) — phần 1/2:
[Dán tóm tắt m053]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    for p in evaluator.parameters():p.requires_grad=False
    data=[{"src":r["source"],"mt":r["prediction"],"ref":r["references"][0]} for r in rows]
    output=evaluator.predict(data,batch_size=COMET_BATCH_SIZE,gpus=1 if DEVICE.type=="cuda" else 0,num_workers=0)
    score=float(output.system_score)
    del evaluator;gc.collect()
    return {"comet":score,"comet_model":"Unbabel/wmt22-comet-da@2760a223ac957f30acfb18c8aa649b01cf1d75f2",
            "comet_status":"Đã đo; raw model score, không nhân100; không phải BLEU"}
```

## Bước 46: Tải lại và đánh giá held-out — extra

<!-- module
ID: m055
Depends: m009, m052, m054
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=23; lines=27-30
Config uses: ARTIFACT_DIR
Source: baseline_basic.ipynb#cell=23; lines=27-30
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [tfds_rows(builder, splits, supervised=False) — phần 1/2](#context-m009)
2. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m052)
3. [add_comet_metric(rows) — phần 2/2](#context-m054)

<a id="context-m055"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Tải lại và đánh giá held-out — extra từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m009 — tfds_rows(builder, splits, supervised=False) — phần 1/2:
[Dán tóm tắt m009]

Context m052 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m052]

Context m054 — add_comet_metric(rows) — phần 2/2:
[Dán tóm tắt m054]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
extra=add_comet_metric(predictions)
results["held_out"].update(extra)
(ARTIFACT_DIR/"metrics.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
print("COMET held-out:",extra)
```

## Bước 47: rewrite_text(text)

<!-- module
ID: m056
Depends: m007, m012, m020, m022, m037, m044, m046, m052
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=25; lines=1-9
Config uses: BEAM_SIZE, DEVICE, LENGTH_PENALTY, MAX_DECODE_TOKENS, MAX_SOURCE_TOKENS
Source: baseline_basic.ipynb#cell=25; lines=1-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [clean_text(text)](#context-m007)
2. [to_device(batch)](#context-m012)
3. [Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…](#context-m020)
4. [Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp](#context-m022)
5. [TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty)](#context-m037)
6. [decode_tokens(ids)](#context-m044)
7. [evaluate(model, loader, return_predictions=False)](#context-m046)
8. [Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…](#context-m052)

<a id="context-m056"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện rewrite_text(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m007 — clean_text(text):
[Dán tóm tắt m007]

Context m012 — to_device(batch):
[Dán tóm tắt m012]

Context m020 — Tokenizer train-only, Dataset và padding — BOS_ID, EOS_ID, PAD_ID…:
[Dán tóm tắt m020]

Context m022 — Tokenizer train-only, Dataset và padding — PREPROCESS_CONFIG, TOKENIZER_STATE, sp:
[Dán tóm tắt m022]

Context m037 — TextGenerator.generate(self, source, lengths, max_tokens, beam_size, length_penalty):
[Dán tóm tắt m037]

Context m044 — decode_tokens(ids):
[Dán tóm tắt m044]

Context m046 — evaluate(model, loader, return_predictions=False):
[Dán tóm tắt m046]

Context m052 — Tải lại và đánh giá held-out — best_model, checkpoint, inference_config…:
[Dán tóm tắt m052]
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

## Bước 48: Suy luận văn bản mới

<!-- module
ID: m057
Depends: m018, m056
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=25; lines=11-11
Summary budget: input=1200; output=500; reserve=300
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Chuẩn hóa và chia dữ liệu](#context-m018)
2. [rewrite_text(text)](#context-m056)

### Prompt tạo hoặc sửa module

```text
Viết Suy luận văn bản mới. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m056 — rewrite_text(text):
[Dán tóm tắt m056]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m057"></a>

```text
Tóm tắt giao diện Suy luận văn bản mới từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m018 — Chuẩn hóa và chia dữ liệu:
[Dán tóm tắt m018]

Context m056 — rewrite_text(text):
[Dán tóm tắt m056]
```

```python
# Dán code hiện tại của Suy luận văn bản mới tại đây.
```

## Bước 49: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m055, m057
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Tải lại và đánh giá held-out — extra](#context-m055)
2. [Suy luận văn bản mới](#context-m057)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m055 — Tải lại và đánh giá held-out — extra:
[Dán tóm tắt m055]

Context m057 — Suy luận văn bản mới:
[Dán tóm tắt m057]
```
