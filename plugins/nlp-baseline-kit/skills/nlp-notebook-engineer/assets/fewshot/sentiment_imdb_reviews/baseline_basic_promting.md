# Prompt basic: IMDb: TF-IDF từ/ký tự và NB-SVM

<!-- prompt-guide
Format: 3
Variant: basic
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Context merge budget: input=1200; output=500; reserve=300
-->

Mỗi lần mở phiên mới, gửi prompt và nội dung đính kèm được chỉ định. Lấy context ở các liên kết trước khi dán vào prompt; dùng lại tóm tắt nếu code chưa đổi.

## Bước 1: sha(text)

<!-- module
ID: m003
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=3; lines=25-26
Source: baseline_basic.ipynb#cell=3; lines=25-26
-->

<a id="context-m003"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện sha(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

## Bước 2: read_jsonl(path)

<!-- module
ID: m005
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=3; lines=31-36
Source: baseline_basic.ipynb#cell=3; lines=31-36
-->

<a id="context-m005"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện read_jsonl(path) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Tệp dữ liệu rỗng: {path}")
    return rows
```

## Bước 3: normalize_text(text)

<!-- module
ID: m009
Depends: none
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=5; lines=1-4
Source: baseline_basic.ipynb#cell=5; lines=1-4
-->

<a id="context-m009"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện normalize_text(text) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def normalize_text(text):
    text = html.unescape(unicodedata.normalize("NFKC", text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    return " ".join(text.split())
```

## Bước 4: canonical_rows(rows, split)

<!-- module
ID: m010
Depends: m003
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=6-18
Summary budget: input=1200; output=500; reserve=300
Config owns: LABEL_NAMES
Config uses: LABEL_NAMES
Source: baseline_basic.ipynb#cell=5; lines=6-18
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [sha(text)](#context-m003)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung canonical_rows(rows, split). Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ ràng buộc: LABEL_NAMES=['neg', 'pos']. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m003 — sha(text):
[Dán tóm tắt m003]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def canonical_rows(rows, split):
    result = []
    for row in rows:
        if not isinstance(row.get("text"), str):
            raise ValueError("Schema cần text dạng chuỗi và label số nguyên.")
        y = row.get("label")
        if isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < len(LABEL_NAMES):
            raise ValueError(f"Nhãn sai: {y!r}")
        text = row["text"]
        result.append({"text": text, "label": y, "key": sha(text + "\0" + str(y))})
    # Sắp theo nội dung để TFDS/JSONL/tar cùng nguồn có thứ tự chia giống nhau.
    result.sort(key=lambda x: (x["key"], x["text"]))
    return [{**r, "id": f"{split}:{i}:{r['key']}"} for i, r in enumerate(result)]
```

### Tóm tắt context để dùng tiếp

<a id="context-m010"></a>

```text
Tóm tắt giao diện canonical_rows(rows, split) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m003 — sha(text):
[Dán tóm tắt m003]
```

```python
# Dán code hiện tại của canonical_rows(rows, split) tại đây.
```

## Bước 5: load_raw(split) — phần 1/3

<!-- module
ID: m011
Depends: m005, m010
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=20-24
Summary budget: input=1200; output=500; reserve=300
Config owns: DATA_DIR, TASK
Config uses: DATA_DIR, LABEL_NAMES, TASK
Source: baseline_basic.ipynb#cell=5; lines=20-24
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m005)
2. [canonical_rows(rows, split)](#context-m010)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung load_raw(split) — phần 1/3. Triển khai đúng chức năng và hợp đồng đầu vào/đầu ra của module theo context. Giữ ràng buộc: DATA_DIR=Path('data/sentiment_imdb_reviews'). Tự chọn TASK phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Đây là phần 1/3 của cùng callable; giữ scope và biến trung gian từ context, chỉ chạy khi đủ phần. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def load_raw(split):
    local_path = DATA_DIR / f"{split}.jsonl"
    if local_path.exists():
        print(f"Đọc dữ liệu gốc cục bộ: {local_path}")
        return canonical_rows(read_jsonl(local_path), split)
```

### Tóm tắt context để dùng tiếp

<a id="context-m011"></a>

```text
Tóm tắt giao diện load_raw(split) — phần 1/3 từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

```python
# Dán code hiện tại của load_raw(split) — phần 1/3 tại đây.
```

## Bước 6: load_raw(split) — phần 2/3

<!-- module
ID: m012
Depends: m005, m010, m011
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=5; lines=25-71
Config uses: DATA_DIR, LABEL_NAMES, TASK
Source: baseline_basic.ipynb#cell=5; lines=25-71
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m005)
2. [canonical_rows(rows, split)](#context-m010)
3. [load_raw(split) — phần 1/3](#context-m011)

<a id="context-m012"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện load_raw(split) — phần 2/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m011 — load_raw(split) — phần 1/3:
[Dán tóm tắt m011]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    if TASK == "imdb_reviews":
        archive = DATA_DIR / "aclImdb_v1.tar.gz"
        if archive.exists():
            import tarfile
            rows = []
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    parts = member.name.split("/")
                    if (member.isfile() and len(parts) == 4 and parts[0] == "aclImdb"
                            and parts[1] == split and parts[2] in ("neg", "pos")
                            and parts[3].endswith(".txt")):
                        with tar.extractfile(member) as f:
                            # Cùng quy tắc với builder TFDS: đọc trọn review, strip hai đầu.
                            text = f.read().decode("utf-8").strip()
                        rows.append({"text": text, "label": int(parts[2] == "pos")})
            if not rows:
                raise ValueError(f"Không tìm thấy aclImdb/{split}/neg|pos/*.txt trong {archive}")
            print(f"Đọc archive Stanford gốc, cùng nguồn TFDS: {archive}")
            return canonical_rows(rows, split)
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")  # Chỉ dùng TensorFlow để đọc dữ liệu CPU.
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise RuntimeError("Cài tensorflow tensorflow-datasets trong Python được hỗ trợ; "
                "hoặc đặt archive Stanford aclImdb_v1.tar.gz trong DATA_DIR. "
                "Có thể dùng train.jsonl/test.jsonl export đúng IMDb, text:str, label:0|1.") from exc
        try:
            ds, info = tfds.load("imdb_reviews/plain_text:1.0.0", split=split,
                as_supervised=True, with_info=True, shuffle_files=False,
                data_dir=str(DATA_DIR / "tfds"))
            assert info.features["label"].names == ["neg", "pos"]
            rows = [{"text": t.decode("utf-8"), "label": int(y)} for t,y in tfds.as_numpy(ds)]
        except Exception as exc:
            raise RuntimeError("Không tải/đọc được IMDb từ TFDS. Kiểm tra mạng, thư mục cache "
                "hoặc đặt archive Stanford gốc trong DATA_DIR; không đổi dataset.") from exc
    else:
        from sklearn.datasets import fetch_20newsgroups
        try:
            data = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"),
                shuffle=False, data_home=str(DATA_DIR / "sklearn"), download_if_missing=True)
        except Exception as exc:
            raise RuntimeError("Không tải/đọc được 20 Newsgroups. Kiểm tra mạng/cache; "
                "hoặc cung cấp train.jsonl/test.jsonl đã remove headers, footers, quotes, "
                "nhãn theo LABEL_NAMES. Không dùng fetch_20newsgroups_vectorized.") from exc
        assert list(data.target_names) == LABEL_NAMES, "Thứ tự nhãn khác khai báo."
        rows = [{"text": text, "label": int(y)} for text,y in zip(data.data, data.target)]
```

## Bước 7: load_raw(split) — phần 3/3

<!-- module
ID: m013
Depends: m005, m010, m011, m012
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=5; lines=72-72
Config uses: DATA_DIR, LABEL_NAMES, TASK
Source: baseline_basic.ipynb#cell=5; lines=72-72
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [read_jsonl(path)](#context-m005)
2. [canonical_rows(rows, split)](#context-m010)
3. [load_raw(split) — phần 1/3](#context-m011)
4. [load_raw(split) — phần 2/3](#context-m012)

<a id="context-m013"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện load_raw(split) — phần 3/3 từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m005 — read_jsonl(path):
[Dán tóm tắt m005]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m011 — load_raw(split) — phần 1/3:
[Dán tóm tắt m011]

Context m012 — load_raw(split) — phần 2/3:
[Dán tóm tắt m012]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
    return canonical_rows(rows, split)
```

## Bước 8: Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…

<!-- module
ID: m014
Depends: m003, m009, m010, m013
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=74-94
Summary budget: input=1200; output=500; reserve=300
Config owns: SEED, SOURCE
Config uses: LABEL_NAMES, SEED, SOURCE
Source: baseline_basic.ipynb#cell=5; lines=74-94
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [sha(text)](#context-m003)
2. [normalize_text(text)](#context-m009)
3. [canonical_rows(rows, split)](#context-m010)
4. [load_raw(split) — phần 3/3](#context-m013)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…. Giữ ràng buộc: SEED=42. Khai báo cùng phần này: SOURCE mô tả đúng dataset và phiên bản được chỉ định trong phần dữ liệu. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
raw_train = load_raw("train")
texts_all = [normalize_text(r["text"]) for r in raw_train]
y_all = np.array([r["label"] for r in raw_train], dtype=np.int64)
# Bản sao văn bản ở cùng fold; văn bản rỗng vẫn là mẫu hợp lệ, nhóm riêng từng id.
groups = np.array([sha(t.casefold()) if t else sha(r["id"])
                   for r,t in zip(raw_train, texts_all)])
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(splitter.split(texts_all, y_all, groups))
assert not (set(groups[train_idx]) & set(groups[val_idx]))
assert set(y_all[train_idx]) == set(range(len(LABEL_NAMES)))
assert set(y_all[val_idx]) == set(range(len(LABEL_NAMES)))
train_text = [raw_train[i]["text"] for i in train_idx]
val_text = [raw_train[i]["text"] for i in val_idx]
y_train, y_val = y_all[train_idx], y_all[val_idx]
SPLIT_INFO = {
    "source": SOURCE, "seed": SEED, "splitter": "StratifiedGroupKFold, fold 0/5",
    "train": fingerprint([raw_train[i] for i in train_idx]),
    "validation": fingerprint([raw_train[i] for i in val_idx]),
    "counts": {"raw_train": len(raw_train), "train": len(train_idx), "validation": len(val_idx)},
}
print("Số mẫu và dấu vân tay split:", json.dumps(SPLIT_INFO, ensure_ascii=False, indent=2))
```

### Tóm tắt context để dùng tiếp

<a id="context-m014"></a>

```text
Tóm tắt giao diện Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]
```

```python
# Dán code hiện tại của Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train… tại đây.
```

## Bước 9: Đọc dữ liệu, chuẩn hóa và chia validation

<!-- module
ID: m015
Depends: m010, m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=5; lines=95-99
Summary budget: input=1200; output=500; reserve=300
Config uses: LABEL_NAMES
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)
2. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)

### Prompt tạo hoặc sửa module

```text
Viết Đọc dữ liệu, chuẩn hóa và chia validation. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

### Code gửi kèm

Module bổ sung mới; không cần gửi code cũ.

### Tóm tắt context để dùng tiếp

<a id="context-m015"></a>

```text
Tóm tắt giao diện Đọc dữ liệu, chuẩn hóa và chia validation từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của Đọc dữ liệu, chuẩn hóa và chia validation tại đây.
```

## Bước 10: build_features()

<!-- module
ID: m016
Depends: none
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=1-8
Summary budget: input=1200; output=500; reserve=300
Config owns: CHAR_MAX_FEATURES, MIN_DF, WORD_MAX_FEATURES
Config uses: CHAR_MAX_FEATURES, MIN_DF, WORD_MAX_FEATURES
Source: baseline_basic.ipynb#cell=7; lines=1-3
-->

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung build_features(). Kết hợp TF-IDF word và char_wb bằng FeatureUnion, chỉ fit train; tự chọn ngram_range và trọng số hai nhánh phù hợp. Tự chọn CHAR_MAX_FEATURES, MIN_DF, WORD_MAX_FEATURES phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def build_features():
    return CountVectorizer(tokenizer=str.split, token_pattern=None,
        lowercase=False, min_df=1, max_features=WORD_MAX_FEATURES, dtype=np.int64)
```

### Tóm tắt context để dùng tiếp

<a id="context-m016"></a>

```text
Tóm tắt giao diện build_features() từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.
```

```python
# Dán code hiện tại của build_features() tại đây.
```

## Bước 11: Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…

<!-- module
ID: m017
Depends: m014, m016
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=7; lines=10-23
Summary budget: input=1200; output=500; reserve=300
Config owns: NB_RATIO_ALPHA, USE_NB_RATIO
Config uses: NB_RATIO_ALPHA, USE_NB_RATIO
Source: baseline_basic.ipynb#cell=7; lines=5-9
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
2. [build_features()](#context-m016)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…. Tạo NB log-ratio nhị phân từ presence train với smoothing dương được chọn phù hợp; nhân cùng ratio vào CSR train và validation. Giữ ràng buộc: USE_NB_RATIO=True. Tự chọn NB_RATIO_ALPHA phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m016 — build_features():
[Dán tóm tắt m016]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
features = build_features()
X_train = features.fit_transform(train_text)
X_val = features.transform(val_text)
ratio = np.ones(X_train.shape[1], dtype=np.float32)
print("Kích thước đặc trưng train/validation:", X_train.shape, X_val.shape)
```

### Tóm tắt context để dùng tiếp

<a id="context-m017"></a>

```text
Tóm tắt giao diện Biểu diễn văn bản chỉ học trên train — X_train, X_val, features… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m016 — build_features():
[Dán tóm tắt m016]
```

```python
# Dán code hiện tại của Biểu diễn văn bản chỉ học trên train — X_train, X_val, features… tại đây.
```

## Bước 12: make_classifier(parameter)

<!-- module
ID: m019
Depends: m014
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=2-4
Summary budget: input=1200; output=500; reserve=300
Config owns: MAX_ITER, TOLERANCE
Config uses: MAX_ITER, SEED, TOLERANCE
Source: baseline_basic.ipynb#cell=9; lines=2-3
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung make_classifier(parameter). Tạo classifier LinearSVC theo API đã chốt; kiểm hội tụ và dùng cấu hình lựa chọn trên validation. Tự chọn MAX_ITER, TOLERANCE phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Dùng lại SEED từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def make_classifier(parameter):
    return MultinomialNB(alpha=parameter)
```

### Tóm tắt context để dùng tiếp

<a id="context-m019"></a>

```text
Tóm tắt giao diện make_classifier(parameter) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]
```

```python
# Dán code hiện tại của make_classifier(parameter) tại đây.
```

## Bước 13: classification_metrics(y, pred)

<!-- module
ID: m020
Depends: m010
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=9; lines=6-8
Config uses: LABEL_NAMES
Source: baseline_basic.ipynb#cell=9; lines=5-7
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)

<a id="context-m020"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện classification_metrics(y, pred) từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def classification_metrics(y, pred):
    return {"macro_f1": float(f1_score(y, pred, labels=list(range(len(LABEL_NAMES))),
        average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred))}
```

## Bước 14: Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…

<!-- module
ID: m021
Depends: m010, m014, m017, m019, m020
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=9; lines=10-35
Summary budget: input=1200; output=500; reserve=300
Config owns: C_VALUES, OUTPUT_DIR, PARAMETERS, PRIMARY_METRIC
Config uses: C_VALUES, LABEL_NAMES, OUTPUT_DIR, PARAMETERS, PRIMARY_METRIC
Source: baseline_basic.ipynb#cell=9; lines=9-34
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)
2. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
3. [Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…](#context-m017)
4. [make_classifier(parameter)](#context-m019)
5. [classification_metrics(y, pred)](#context-m020)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…. Fit pipeline trên train theo các ứng viên siêu tham số được chọn; chọn bằng metric validation và lưu đủ biểu diễn, classifier và nhãn để reload. Không fit lại trên test hoặc train+validation. Giữ ràng buộc: PRIMARY_METRIC='macro_f1'. Tự chọn C_VALUES phù hợp và khai báo ngay trong module này; nêu ngắn lý do. Khai báo cùng phần này: OUTPUT_DIR được suy ra từ dữ liệu/context hoặc API tương ứng; PARAMETERS được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại LABEL_NAMES từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m017 — Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…:
[Dán tóm tắt m017]

Context m019 — make_classifier(parameter):
[Dán tóm tắt m019]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
CHECKPOINT_PATH = OUTPUT_DIR / "best.joblib"
best_score = -float("inf")
history = []
for parameter in PARAMETERS:
    classifier = make_classifier(parameter)
    # Không chấp nhận nghiệm SVM chưa hội tụ rồi âm thầm báo kết quả.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(X_train, y_train)
    val_pred = classifier.predict(X_val)
    scores = classification_metrics(y_val, val_pred)
    history.append({"parameter": float(parameter), **scores})
    print("Tham số:", parameter, "| validation:", scores)
    if scores[PRIMARY_METRIC] > best_score:
        best_score = scores[PRIMARY_METRIC]
        # Chỉ lưu các thành phần đã fit cần để inference; cấu hình/provenance ở notebook.
        joblib.dump({
            "features": features,
            "ratio": ratio,
            "classifier": classifier,
            "label_names": LABEL_NAMES,
            "validation": scores,
        }, CHECKPOINT_PATH)
(OUTPUT_DIR / "validation_history.json").write_text(
    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
print("Đã lưu checkpoint tốt nhất theo Macro F1 validation:", CHECKPOINT_PATH)
```

### Tóm tắt context để dùng tiếp

<a id="context-m021"></a>

```text
Tóm tắt giao diện Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier… từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m017 — Biểu diễn văn bản chỉ học trên train — X_train, X_val, features…:
[Dán tóm tắt m017]

Context m019 — make_classifier(parameter):
[Dán tóm tắt m019]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]
```

```python
# Dán code hiện tại của Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier… tại đây.
```

## Bước 15: Nạp lại, đánh giá test một lần và dự đoán — best

<!-- module
ID: m022
Depends: m021
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=2-2
Source: baseline_basic.ipynb#cell=11; lines=2-2
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…](#context-m021)

<a id="context-m022"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Nạp lại, đánh giá test một lần và dự đoán — best từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
best = joblib.load(CHECKPOINT_PATH)
```

## Bước 16: predict_texts(texts)

<!-- module
ID: m023
Depends: m009, m022
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=4-7
Summary budget: input=1200; output=500; reserve=300
Source: baseline_basic.ipynb#cell=11; lines=4-7
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [normalize_text(text)](#context-m009)
2. [Nạp lại, đánh giá test một lần và dự đoán — best](#context-m022)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung predict_texts(texts). Giữ predict_texts(texts), reload features, ratio và classifier; đối chiếu validation trước khi chấm test. Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
def predict_texts(texts):
    normalized = [str(t) for t in texts]
    x = best["features"].transform(normalized).multiply(best["ratio"]).tocsr()
    return best["classifier"].predict(x)
```

### Tóm tắt context để dùng tiếp

<a id="context-m023"></a>

```text
Tóm tắt giao diện predict_texts(texts) từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]
```

```python
# Dán code hiện tại của predict_texts(texts) tại đây.
```

## Bước 17: Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…

<!-- module
ID: m024
Depends: m003, m009, m010, m013, m014, m020, m021, m022, m023
Kind: summary
Budget: input=1200; output=500; reserve=300
Summary: yes
Summary input: code
Target: baseline.ipynb#cell=11; lines=10-29
Config uses: LABEL_NAMES, PRIMARY_METRIC
Source: baseline_basic.ipynb#cell=11; lines=10-29
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [sha(text)](#context-m003)
2. [normalize_text(text)](#context-m009)
3. [canonical_rows(rows, split)](#context-m010)
4. [load_raw(split) — phần 3/3](#context-m013)
5. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
6. [classification_metrics(y, pred)](#context-m020)
7. [Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…](#context-m021)
8. [Nạp lại, đánh giá test một lần và dự đoán — best](#context-m022)
9. [predict_texts(texts)](#context-m023)

<a id="context-m024"></a>

### Prompt tóm tắt module

```text
Tóm tắt giao diện Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val… từ code hiện tại để dùng cho các module sau. Giữ chữ ký, I/O, shape/dtype/device và cấu hình cần dùng; chỉ trả chữ, báo rõ thông tin thiếu, không viết lại code.

Context m003 — sha(text):
[Dán tóm tắt m003]

Context m009 — normalize_text(text):
[Dán tóm tắt m009]

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m013 — load_raw(split) — phần 3/3:
[Dán tóm tắt m013]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m020 — classification_metrics(y, pred):
[Dán tóm tắt m020]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
reloaded_val = predict_texts([raw_train[i]["text"] for i in val_idx])
assert abs(classification_metrics(y_val, reloaded_val)[PRIMARY_METRIC]
           - best["validation"][PRIMARY_METRIC]) < 1e-12
raw_test = load_raw("test")
test_text = [r["text"] for r in raw_test]
y_test = np.array([r["label"] for r in raw_test])
test_fingerprint = fingerprint(raw_test)
test_groups = {sha(normalize_text(t).casefold()) for t in test_text if normalize_text(t)}
overlap = len(test_groups & {sha(t.casefold()) for t in texts_all if t})
print("Dấu vân tay test:", test_fingerprint)
print("Số văn bản không rỗng trùng giữa official train và test:", overlap)
if overlap:
    print("Có trùng dữ liệu nguồn: giữ split chuẩn để so sánh, diễn giải test thận trọng.")
test_pred = predict_texts(test_text)
test_scores = classification_metrics(y_test, test_pred)
print("Kết quả test của pipeline đã chốt:", test_scores)
print(classification_report(y_test, test_pred, labels=list(range(len(LABEL_NAMES))),
    target_names=LABEL_NAMES, zero_division=0, digits=4))
print("Ma trận nhầm lẫn (hàng=nhãn thật, cột=dự đoán):")
print(confusion_matrix(y_test, test_pred, labels=list(range(len(LABEL_NAMES)))))
```

## Bước 18: Nạp lại, đánh giá test một lần và dự đoán — f, pred, row

<!-- module
ID: m025
Depends: m010, m014, m021, m022, m023, m024
Kind: code
Budget: input=800; output=1000; reserve=200
Summary: yes
Target: baseline.ipynb#cell=11; lines=30-38
Summary budget: input=1200; output=500; reserve=300
Config owns: VERSIONS
Config uses: LABEL_NAMES, OUTPUT_DIR, VERSIONS
Source: baseline_basic.ipynb#cell=11; lines=30-38
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [canonical_rows(rows, split)](#context-m010)
2. [Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…](#context-m014)
3. [Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…](#context-m021)
4. [Nạp lại, đánh giá test một lần và dự đoán — best](#context-m022)
5. [predict_texts(texts)](#context-m023)
6. [Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…](#context-m024)

### Prompt tạo hoặc sửa module

```text
Sửa hoặc bổ sung Nạp lại, đánh giá test một lần và dự đoán — f, pred, row. Dùng pipeline đã reload để kiểm validation, chấm held-out theo đúng ID và metric, lưu đủ predictions/kết quả; không fit lại hoặc chọn tham số bằng test. Khai báo cùng phần này: VERSIONS được suy ra từ dữ liệu/context hoặc API tương ứng. Dùng lại LABEL_NAMES, OUTPUT_DIR từ context, không chọn hoặc khai báo lại giá trị. Trả code kèm imports cần dùng; dùng đúng I/O và các lựa chọn đã có trong context.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]

Context m024 — Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…:
[Dán tóm tắt m024]
```

### Code gửi kèm

Dùng code hiện tại nếu đã chỉnh sửa:

```python
(OUTPUT_DIR / "test_metrics.json").write_text(json.dumps({
    "validation": best["validation"], "test": test_scores,
    "test_fingerprint": test_fingerprint, "source_overlap": overlap,
    "split_info": SPLIT_INFO, "versions": VERSIONS}, ensure_ascii=False, indent=2), encoding="utf-8")
with (OUTPUT_DIR / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
    for row, pred in zip(raw_test, test_pred):
        f.write(json.dumps({"id": row["id"], "label": row["label"], "prediction": int(pred)},
            ensure_ascii=False) + "\n")
print("Ví dụ dự đoán:", LABEL_NAMES[int(predict_texts([test_text[0]])[0])])
```

### Tóm tắt context để dùng tiếp

<a id="context-m025"></a>

```text
Tóm tắt giao diện Nạp lại, đánh giá test một lần và dự đoán — f, pred, row từ code dưới đây. Giữ chữ ký, I/O, shape/dtype/device, các tham số đã chọn và trạng thái dùng chung; chỉ trả chữ, không viết lại code hoặc đoán thông tin thiếu.

Context m010 — canonical_rows(rows, split):
[Dán tóm tắt m010]

Context m014 — Đọc dữ liệu, chuẩn hóa và chia validation — SPLIT_INFO, groups, raw_train…:
[Dán tóm tắt m014]

Context m021 — Mô hình, huấn luyện và chọn checkpoint — CHECKPOINT_PATH, best_score, classifier…:
[Dán tóm tắt m021]

Context m022 — Nạp lại, đánh giá test một lần và dự đoán — best:
[Dán tóm tắt m022]

Context m023 — predict_texts(texts):
[Dán tóm tắt m023]

Context m024 — Nạp lại, đánh giá test một lần và dự đoán — overlap, raw_test, reloaded_val…:
[Dán tóm tắt m024]
```

```python
# Dán code hiện tại của Nạp lại, đánh giá test một lần và dự đoán — f, pred, row tại đây.
```

## Bước 19: Kiểm tra kết nối các module

<!-- module
ID: integration
Depends: m015, m025
Kind: integration
Budget: input=800; output=1000; reserve=200
Summary: no
-->

### Context cần chuẩn bị

Lần lượt dùng các prompt tóm tắt sau, hoặc dùng lại context khớp code hiện tại:

1. [Đọc dữ liệu, chuẩn hóa và chia validation](#context-m015)
2. [Nạp lại, đánh giá test một lần và dự đoán — f, pred, row](#context-m025)

### Prompt ghép và kiểm tra

```text
Viết phần kết nối và assertions còn thiếu, kèm imports cần dùng, theo các context dưới đây. Kiểm split/IDs, I/O, metric, cấu hình đã chọn và reload; dùng fit/predict cho pipeline sparse hoặc forward/loss/update cho neural. Không sinh lại toàn bộ notebook hoặc tự chọn lại tham số.

Context m015 — Đọc dữ liệu, chuẩn hóa và chia validation:
[Dán tóm tắt m015]

Context m025 — Nạp lại, đánh giá test một lần và dự đoán — f, pred, row:
[Dán tóm tắt m025]
```
