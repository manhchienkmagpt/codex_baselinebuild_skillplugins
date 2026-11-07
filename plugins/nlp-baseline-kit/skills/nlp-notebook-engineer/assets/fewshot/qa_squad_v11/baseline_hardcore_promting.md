# Prompt hardcore: SQuAD v1.1 — QA trích xuất span — bản tối ưu

<!-- prompt-guide
Format: 4
Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified
Session target: 1800
Session limit: 2000
-->

Mỗi prompt dùng một chat mới. Gửi prompt cùng code/context được chỉ định; dùng code hiện tại sau khi sửa. Mục tiêu mỗi phiên 1.800 token, tổng input + output phải dưới 2.000; chưa có đo token thực tế. Chỉ tóm tắt ở bước có liên kết dùng tiếp; web không đặt trần output riêng nên giới hạn độ dài tóm tắt là yêu cầu mềm.

## Bước 1: Load dữ liệu và chia train/validation/test

<!-- module
ID: s01
Group: data
Kind: code
Depends: none
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Config owns: splits
-->

### Prompt tạo hoặc sửa module

```text
Bổ sung loader train/validation/test, giữ nguồn/schema và split chính thức hoặc quy tắc nhóm/seed của bài. Giữ ID, nhãn/references và held-out gốc; kiểm nhóm trùng, chưa fit biểu diễn.
Nguồn/schema/split: {"dataset":"SQuAD v1.1","tfds_builder":"squad/v1.1:3.0.0","original_hint":"squad/v1.1.0 (đã sửa config thành v1.1, builder version3.0.0)"}. SQuAD v1.1 qua TFDS có id/title/context/question/answers, mỗi answer gồm text và answer_start. Khảo sát JSON gốc trước đây ghi nhận 87.599 train questions trên 442 titles và 10.570 dev questions trên 48 titles; kiểm offset gold không thấy mismatch. Có 262 annotation train và 76 dev có biên nằm giữa token regex. Nhiều câu hỏi chia sẻ cùng đoạn/bài Wikipedia, nên chia ngẫu nhiên câu hỏi dễ rò rỉ context. Baseline lấy 10% train titles làm validation nội bộ và giữ official dev làm held-out vì test không có nhãn công khai. Các answer có biên giữa token là giới hạn biểu diễn thật của head token-span; không tự sửa gold hoặc bỏ chúng để nâng EM.
Thiết lập split cần giữ: {"SEED":42,"VAL_FRACTION":0.1}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.
```

### Tóm tắt context để dùng tiếp

<a id="context-s01"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 2: Tiền xử lí dữ liệu — Biểu diễn và căn chỉnh

<!-- module
ID: s02
Group: preprocessing
Kind: code
Depends: s01
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=6; lines=5-6
Target: baseline.ipynb#cell=10; lines=5-25
Target: baseline.ipynb#cell=10; lines=32-43
Target: baseline.ipynb#cell=10; lines=69-77
Target: baseline.ipynb#cell=10; lines=94-100
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]
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
Depends: s01, s02
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=6; lines=43-46
Target: baseline.ipynb#cell=10; lines=45-67
Target: baseline.ipynb#cell=10; lines=79-95
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.
Ý tưởng tiền xử lí: 1. Giữ context nguyên văn, kể cả Unicode/khoảng trắng. Tokenize regex kèm start/end offsets; question tokenize cùng kiểu nhưng có trần riêng. Không strip/normalize context rồi dùng answer_start cũ. 2. Chia theo title trước khi fit word vocabulary và char vocabulary train-only; đếm context trùng trong train một lần để tránh một đoạn nhiều câu hỏi làm lệch thống kê. 3. Tạo mọi window context dài tối đa 384, bước 256 nên overlap 128; thêm CLS cho các window không chứa gold answer. Nhãn train là span token bao phủ tối thiểu annotation, không bỏ câu hỏi khi context dài. 4. Char feature tối đa 32 ký tự/từ; padding và attention có mask. Joint decoding chọn start/end hợp lệ trong cùng window, giới hạn độ dài answer và so margin so với CLS giữa các window. Gold text khi chấm vẫn nguyên vẹn.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"CLS_ID":2}.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s02 — Biểu diễn và căn chỉnh:
[Dán tóm tắt s02]
```

### Tóm tắt context để dùng tiếp

<a id="context-s03"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 4: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=12; lines=1-41
Config owns: architecture
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; attention giúp từng token context đọc phần question liên quan. Toàn bộ mô hình học từ đầu, không dùng BERT/tokenizer pretrained. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawSpanModel: __init__(); forward(rows).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]
```

### Tóm tắt context để dùng tiếp

<a id="context-s04"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 5: Siêu tham số

<!-- module
ID: s05
Group: hyperparameters
Kind: code
Depends: s03, s04
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s04)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Trả cấu hình cố định để bước training dùng; chưa viết vòng train.
Phương pháp: Chọn word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; Metric: exact_match. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]
```

### Tóm tắt context để dùng tiếp

<a id="context-s05"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 6: Loss, optimizer và training loop

<!-- module
ID: s06
Group: training
Kind: code
Depends: s01, s03, s04, s05
Budget: input=650; output=950; reserve=200
Summary: no
Config uses: splits, preprocess_2, architecture, training_config
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s04)
- [Siêu tham số](#context-s05)

### Prompt tạo hoặc sửa module

```text
Bổ sung huấn luyện thật vào notebook hardcore: loss đúng output/mask/target, optimizer chỉ cho tham số được học, train/eval và chọn checkpoint bằng metric validation. Tận dụng loader/model/config trong context; không tạo lại chúng. Với sparse dùng fit/predict và chọn tham số validation, không bịa optimizer/backward. Với neural kiểm forward/loss hữu hạn, backward và một bước update; reload config/weights và trạng thái biểu diễn rồi dùng lại inference hiện có. Giữ ID, metric và chiều tốt hơn của metric; test chỉ chấm sau khi chọn xong, không chọn tham số bằng test.
Phương pháp: Chọn word embedding + char-CNN → BiLSTM chung → attention context–question hai chiều → BiLSTM modeling/end → start/end heads. Word embedding 128, char embedding 16/Conv1d 32, encoder hidden 128 mỗi chiều. Tương đồng tri-linear kết hợp context, question và tích của chúng; Metric: exact_match.
Objective cần nối: cross-entropy theo head/target của bài; chỉ bỏ padding khi target có padding. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Model và giao diện sử dụng:
[Dán tóm tắt s04]

Context s05 — Siêu tham số:
[Dán tóm tắt s05]
```
