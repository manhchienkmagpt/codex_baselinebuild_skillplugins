# Prompt hardcore: Trích xuất aspect và phân loại polarity — bản tối ưu

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
Nguồn/schema/split: . SetFit SemEval Restaurants có các trường text, span, label, ordinal. Khảo sát đã ghi nhận 3.693 hàng aspect train và 1.134 hàng test, tương ứng 2.019 và 606 câu nguyên văn. Một câu có thể có nhiều hàng annotation; ordinal là lần xuất hiện của đúng chuỗi span, không phải thứ tự mọi aspect trong câu. Nhãn train gồm 2.164 positive, 805 negative, 633 neutral và 91 conflict trước gộp trùng; lớp conflict rất ít. Toàn bộ label test là chuỗi rỗng, nhưng còn span/ordinal để chấm extraction. Có ba annotation train kết thúc giữa một token regex. Vì vậy biểu diễn chỉ theo từ không thể khớp mọi ranh giới; dùng ký tự làm đơn vị dự đoán có căn cứ trực tiếp từ dữ liệu. Gộp annotation trùng của cùng câu, giữ nguyên nhãn nguồn;
Thiết lập split cần giữ: {"SEED":42,"VAL_FRACTION":0.2}.
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
Target: baseline.ipynb#cell=9; lines=5-31
Target: baseline.ipynb#cell=9; lines=73-74
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc. 2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold. 3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó. 4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1}.
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
Target: baseline.ipynb#cell=9; lines=34-42
Target: baseline.ipynb#cell=9; lines=45-62
Target: baseline.ipynb#cell=9; lines=65-72
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.
Ý tưởng tiền xử lí: 1. Giữ nguyên Unicode, dấu câu và khoảng trắng của câu đầu vào. Không viết lại chuỗi hoặc cắt câu vì nhãn được định nghĩa bằng vị trí ký tự gốc. 2. Tìm lần xuất hiện span theo ordinal, gộp hàng trùng và kiểm tra span chồng lấn/nhãn mâu thuẫn. Đây là điều kiện để chuyển annotation thành nhãn ký tự đúng, không phải lý do tự sửa gold. 3. Gán BIO: B là ký tự bắt đầu aspect, I là ký tự bên trong, O là ngoài aspect. Ký tự là đơn vị dự đoán; từ regex chỉ cung cấp embedding phụ cho các ký tự nằm trong từ đó. 4. Fit vocabulary ký tự và từ trên train sau khi chia tập; từ viết thường chỉ khi tra vocabulary. Padding có mask để loss/CRF không học các vị trí thêm vào. Giữ đầy đủ offset khi chuyển dự đoán về span. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1}.
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

## Bước 4: Xử lí mô hình — Thành phần LinearChainCRF

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=11; lines=1-53
Config owns: component_LinearChainCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết trọn thành phần LinearChainCRF phục vụ phương pháp bên dưới; định nghĩa API nhận tensor/mask và trả kết quả để model chính dùng. Với CRF: gồm NLL và Viterbi, transition hợp lệ, bỏ padding; không chỉ tạo khung lớp.
Ý tưởng mô hình: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawAspectModel: __init__(); forward(texts).
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

## Bước 5: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s05
Group: model
Kind: code
Depends: s03, s04
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=12; lines=1-43
Config owns: architecture
Config uses: preprocess_2, component_LinearChainCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Thành phần LinearChainCRF](#context-s04)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; inference chỉ dùng span do mô hình dự đoán. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawAspectModel: __init__(); forward(texts).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Thành phần LinearChainCRF:
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

## Bước 6: Siêu tham số

<!-- module
ID: s06
Group: hyperparameters
Kind: code
Depends: s03, s05
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Config owns: training_config
Config uses: preprocess_2, architecture
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s05)

### Prompt tạo hoặc sửa module

```text
Chọn các siêu tham số huấn luyện/solver phù hợp: batch, learning rate hoặc C, regularization, số epoch/lặp, patience và clip khi cần; nêu lý do bằng comment ngắn. Giữ kích thước, tokenizer và special IDs đã chốt ở context; không định nghĩa lại chúng. Đặt constants trước nơi khởi tạo tương ứng; tạo lại loader khi đổi batch. Trả cấu hình cố định để bước training dùng; chưa viết vòng train.
Phương pháp: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; Metric: metric đã khai báo trong notebook. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s05 — Model và giao diện sử dụng:
[Dán tóm tắt s05]
```

### Tóm tắt context để dùng tiếp

<a id="context-s06"></a>

```text
Tóm tắt code hiện tại trong tối đa 3 dòng, mục tiêu không quá 80 token và ngắn hơn input. Chỉ ghi chữ ký cần gọi, I/O (shape/mask nếu có), cấu hình và trạng thái mà bước sau cần. Không giải thích chức năng, không nhắc lại code; không đoán giá trị thiếu.
```

```python
# Dán code hiện tại vừa tạo/sửa của phần việc này.
```

## Bước 7: Loss, optimizer và training loop

<!-- module
ID: s07
Group: training
Kind: code
Depends: s01, s03, s05, s06
Budget: input=650; output=950; reserve=200
Summary: no
Config uses: splits, preprocess_2, architecture, training_config
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Dataset và batch](#context-s03)
- [Model và giao diện sử dụng](#context-s05)
- [Siêu tham số](#context-s06)

### Prompt tạo hoặc sửa module

```text
Bổ sung huấn luyện thật vào notebook hardcore: loss đúng output/mask/target, optimizer chỉ cho tham số được học, train/eval và chọn checkpoint bằng metric validation. Tận dụng loader/model/config trong context; không tạo lại chúng. Với sparse dùng fit/predict và chọn tham số validation, không bịa optimizer/backward. Với neural kiểm forward/loss hữu hạn, backward và một bước update; reload config/weights và trạng thái biểu diễn rồi dùng lại inference hiện có. Giữ ID, metric và chiều tốt hơn của metric; test chỉ chấm sau khi chọn xong, không chọn tham số bằng test.
Phương pháp: Chọn embedding ký tự + embedding từ → BiLSTM hai lớp → CRF, kèm head polarity cho mỗi span. Embedding ký tự 48 chiều và từ 128 chiều được nối; hidden mỗi chiều là 128. Head polarity ghép biểu diễn đầu/cuối/trung bình span với ngữ cảnh câu để phân biệt cảm xúc. Loss extraction kết hợp loss polarity trên gold span; Metric: metric đã khai báo trong notebook.
Objective cần nối: cross-entropy theo head/target của bài; chỉ bỏ padding khi target có padding; CRF NLL với transition/mask hợp lệ; decode Viterbi. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s01 — Load dữ liệu và chia train/validation/test:
[Dán tóm tắt s01]

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s05 — Model và giao diện sử dụng:
[Dán tóm tắt s05]

Context s06 — Siêu tham số:
[Dán tóm tắt s06]
```
