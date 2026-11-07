# Prompt hardcore: FUNSD — nhận dạng thực thể tài liệu — bản tối ưu

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
Nguồn/schema/split: {"dataset":"nielsr/funsd","revision":"7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5"}. nielsr/funsd có 149 tài liệu train và 50 test với image, words, bboxes, ner_tags. Box dùng hệ tọa độ 0..1000; có bảy BIO tags: O và B/I cho HEADER, QUESTION, ANSWER. Baseline chia 149 tài liệu thành 119 train/30 validation theo seed cố định, giữ 50 test. Dữ liệu rất ít ở cấp trang, bố cục và hình thức chữ có thể mang tín hiệu quan trọng ngoài nội dung từ. Giữ mọi token của một trang cùng split; so trùng bằng nội dung words+boxes thay vì chỉ dùng ID vì ID có thể được đánh lại theo split. Đã khảo sát schema và mẫu ảnh/nhãn, chưa quét toàn bộ trang hoặc biết phân bố đủ mọi kiểu form.
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
Target: baseline.ipynb#cell=10; lines=1-4
Target: baseline.ipynb#cell=10; lines=9-11
Target: baseline.ipynb#cell=10; lines=47-47
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn. 2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu. 3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác. 4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
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
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=13-46
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.
Ý tưởng tiền xử lí: 1. Kiểm words, boxes và tags có cùng độ dài, box nằm trong miền hợp lệ. Giữ thứ tự từ nguồn; tự sắp lại từ theo box có thể làm sai chuỗi nhãn. 2. Tách theo tài liệu trước khi fit vocabulary từ/ký tự. Word lookup chữ thường; ký tự bổ sung hình thái/OOV, cap 32 ký tự trên mỗi từ nhưng không bỏ token/tài liệu. 3. Scale box về 0..1; model dùng x0,y0,x1,y1 và width/height. Ảnh RGB resize về 384, tensor 0..1; RoIAlign lấy vùng ảnh khớp box để bổ sung hình thức thị giác. 4. Chuẩn hóa I-tag không có B/I hợp lệ trước nó thành B cùng loại; dùng cùng quy tắc khi giải mã/chấm. Padding luôn có mask, không để O/PAD làm tăng F1. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint;
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

## Bước 4: Xử lí mô hình — Thành phần EntityCRF

<!-- module
ID: s04
Group: model
Kind: code
Depends: s03
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=12; lines=1-45
Config owns: component_EntityCRF
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết trọn thành phần EntityCRF phục vụ phương pháp bên dưới; định nghĩa API nhận tensor/mask và trả kết quả để model chính dùng. Với CRF: gồm NLL và Viterbi, transition hợp lệ, bỏ padding; không chỉ tạo khung lớp.
Ý tưởng mô hình: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans. Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawDocumentModel: __init__(); forward(rows).
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
Target: baseline.ipynb#cell=13; lines=2-42
Config owns: architecture
Config uses: preprocess_2, component_EntityCRF
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Thành phần EntityCRF](#context-s04)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. CRF chọn chuỗi bảy BIO tags hợp lệ và tạo các entity spans. Các modal bổ sung trực tiếp cho sự thiếu hụt dữ liệu từ đơn lẻ: bố cục phân biệt vị trí trường, char giúp từ lạ, vùng ảnh giúp kiểu trình bày; Không dùng pretrained cho tokenizer/embedding/model tác vụ.
API hiện có: RawDocumentModel: __init__(); forward(rows).
Chỉ trả code hoàn chỉnh kèm imports và comment chỉ chỗ thay/thêm. Tự chọn tham số điều chỉnh được; giữ nhãn/ID và ràng buộc bài, không ép giống các giá trị tham khảo.

Context s03 — Dataset và batch:
[Dán tóm tắt s03]

Context s04 — Thành phần EntityCRF:
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
Phương pháp: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. Metric: entity_f1. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
Phương pháp: Chọn word embedding + char-CNN + layout MLP + đặc trưng vùng ảnh → BiLSTM hai lớp → CRF. Word 128 chiều, char embedding 16/Conv1d 32; layout 6→32; CNN ảnh 3→32→64 và RoIAlign crop 2×4 cho từng từ, chiếu về 64 chiều. Ghép 256 chiều vào BiLSTM hidden 128 mỗi chiều. Metric: entity_f1.
Objective cần nối: CRF NLL với transition/mask hợp lệ; decode Viterbi. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
