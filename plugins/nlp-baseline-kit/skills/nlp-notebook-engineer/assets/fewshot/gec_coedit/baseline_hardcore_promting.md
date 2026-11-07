# Prompt hardcore: Sửa ngữ pháp CoEdIT GEC — bản tối ưu

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
Nguồn/schema/split: {"dataset":"grammarly/coedit","revision":"e9a255c33ef910bc33a9d2b522653fa87521583e"}. grammarly/coedit chứa nhiều nhiệm vụ chỉnh sửa; chỉ chọn task=gec. Khảo sát đã ghi nhận 19.823 hàng GEC train trong 69.071 hàng train và 485 hàng GEC validation. Sau nhóm câu nguồn có 19.696 train sources, trong đó 60 nhóm có hai corrections khác nhau; official validation có 485 nguồn. Các src chứa instruction ở đầu câu. Nếu không tách đúng, model có thể học cách lặp prompt thay vì sửa ngữ pháp. Baseline nhận diện 27 prefix đã kiểm tra, nhóm theo câu nguồn sau bỏ prefix và giữ các target khác nhau. Tách 10% nhóm train thành 1.970 validation nội bộ, học từ 17.726 nhóm; 485 official validation làm held-out. Mọi correction của cùng nguồn ở cùng tập; không trộn thêm BEA/CoNLL để làm giàu train.
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
Target: baseline.ipynb#cell=11; lines=3-13
Target: baseline.ipynb#cell=11; lines=19-21
Target: baseline.ipynb#cell=11; lines=46-50
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu. 2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer. 3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm. 4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
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
Target: baseline.ipynb#cell=11; lines=23-45
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.
Ý tưởng tiền xử lí: 1. Lọc đúng task=gec; chỉ bỏ instruction khi khớp prefix đã biết ở đầu chuỗi. Không split mọi dấu hai chấm vì dấu đó có thể thuộc nội dung câu. 2. NFC và gộp khoảng trắng, giữ hoa/thường và dấu câu. Gộp hàng trùng theo source, giữ corrections khác nhau; chia nhóm trước khi fit tokenizer. 3. Fit BPE SentencePiece 8.000 trên train, byte fallback và identity normalizer sau chuẩn hóa ngoài. BPE chia từ thành đơn vị nhỏ giúp giảm UNK khi nguồn có lỗi chính tả hoặc từ hiếm. 4. Thêm BOS/EOS/PAD, cap source/target ở 160 token và mask PAD trong attention/loss. Huấn luyện mọi reference train, giữ gold không bị cap khi chấm. Inference chỉ đọc source, không dùng reference hoặc evaluator để chọn câu. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
Special IDs cố định: {"PAD_ID":0,"UNK_ID":1,"BOS_ID":2,"EOS_ID":3}.
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

## Bước 4: Xử lí mô hình — Encoder độc lập

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=13; lines=1-63
Config owns: encoder
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết một encoder hoàn chỉnh cho kiến trúc bên dưới, gồm embedding/backbone và mask/lengths. Xuất memory và trạng thái/shape cần cho attention/head; chưa viết decoder hay training loop.
Ý tưởng mô hình: Chọn BiGRU encoder + attention cộng + GRU decoder với pointer-copy, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold. Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawTextModel: __init__(); forward(texts).
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
Target: baseline.ipynb#cell=13; lines=1-63
Config owns: architecture
Config uses: preprocess_2, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Encoder độc lập](#context-s04)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn BiGRU encoder + attention cộng + GRU decoder với pointer-copy, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Beam search rộng 3 với length penalty 0.6 chọn chuỗi theo model, không dựa vào gold. Sửa ngữ pháp thường chỉ thay một phần câu, nên đường copy phù hợp nhu cầu bảo toàn tên/nội dung. Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawTextModel: __init__(); forward(texts).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Encoder độc lập:
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
Phương pháp: Chọn BiGRU encoder + attention cộng + GRU decoder với pointer-copy, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Metric: errant_f05. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
Phương pháp: Chọn BiGRU encoder + attention cộng + GRU decoder với pointer-copy, toàn bộ học từ đầu. Hidden encoder là 192 mỗi chiều, ghép thành 384; decoder 384 kết hợp context attention và embedding 128. Một cổng trộn xác suất sinh từ với xác suất copy token source. Metric: errant_f05.
Objective cần nối: token NLL từ log-probabilities, target dịch một bước; bỏ PAD, smoothing theo cấu hình. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
