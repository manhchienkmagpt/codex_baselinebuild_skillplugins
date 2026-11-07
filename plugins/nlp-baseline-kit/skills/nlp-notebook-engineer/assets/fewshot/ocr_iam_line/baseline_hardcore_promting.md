# Prompt hardcore: OCR IAM-line — bản tối ưu

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
Nguồn/schema/split: {"dataset":"Teklia/IAM-line","revision":"fbdad97500ce54635c0d1ba306bf535cb40656cf"}. Teklia/IAM-line có image và text, số dòng theo card đã ghi nhận là 6.482/976/2.915 train/validation/test. Mẫu nguồn có chiều cao 128; đã khảo sát mẫu ảnh/nhãn, chưa quét toàn bộ phân bố tỷ lệ ảnh/độ dài hoặc xác minh trùng người viết. HF schema không có writer_id nên giữ split nguồn và không khẳng định writer-disjoint. Ảnh có độ rộng thay đổi, chữ viết nối nét, nền và phong cách người viết khác nhau. Resize mọi dòng về một hình vuông dễ làm méo hoặc mất ký tự; việc biến ảnh thành một chuỗi đủ dài quan trọng hơn chỉ tăng số lớp CNN. Alphabet phải fit train để không dùng transcript test bổ sung nhãn; ký tự chưa gặp vẫn phải được tính lỗi khi đánh giá.
Thiết lập split cần giữ: {"SEED":42}.
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
Target: baseline.ipynb#cell=8; lines=12-13
Target: baseline.ipynb#cell=10; lines=2-5
Target: baseline.ipynb#cell=10; lines=11-23
Target: baseline.ipynb#cell=10; lines=53-56
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn. 2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học. 3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference. 4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"BLANK_ID":0,"UNK_ID":1}.
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
Target: baseline.ipynb#cell=10; lines=25-52
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.
Ý tưởng tiền xử lí: 1. Giữ official split; transcript NFC và gộp khoảng trắng, giữ hoa/thường/dấu câu theo cùng quy tắc khi chấm. Không xóa ký tự khó để làm CER đẹp hơn. 2. Fit alphabet chỉ trên train; blank=0 dành cho CTC, UNK=1 cho ký tự chưa biết. Không thêm ký tự từ val/test vào alphabet học. 3. Ảnh RGB resize theo tỷ lệ về height 64, chiều rộng được làm tròn theo stride; pad bên phải và chuẩn hóa theo encoder ImageNet. Không crop/truncate dòng, không lấy transcription gold để quyết định kích thước ảnh lúc inference. 4. Theo dõi chiều rộng thật và số bước sau CNN để mask/pad không thành văn bản. CTC cần ít nhất số ký tự target cộng số cặp ký tự lặp liền nhau; lỗi không đủ bước phải được phát hiện thay vì che bằng zero-infinity. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
Special IDs cố định: {"BLANK_ID":0,"UNK_ID":1}.
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
Target: baseline.ipynb#cell=12; lines=3-33
Config owns: architecture
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC. Encoder torchvision/resnet18 dùng ResNet18_Weights.IMAGENET1K_V1, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. CTC cộng xác suất nhiều alignment hợp lệ, không cần biết ký tự nằm ở pixel nào; greedy decoder gộp lặp và bỏ blank theo độ dài ảnh thật. Chỉ cho pretrained backbone ảnh resnet18/ResNet18_Weights.IMAGENET1K_V1; phần ngôn ngữ/head học mới.
API hiện có: RawImageTextModel: __init__(); forward(rows).
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
Phương pháp: Chọn ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC. Encoder torchvision/resnet18 dùng ResNet18_Weights.IMAGENET1K_V1, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. Metric: cer (càng thấp càng tốt). Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
Phương pháp: Chọn ResNet18 ImageNet sửa stride → BiLSTM hai lớp → linear CTC. Encoder torchvision/resnet18 dùng ResNet18_Weights.IMAGENET1K_V1, fine-tune toàn bộ; stride ngang tổng /4 để còn đủ bước cho dòng chữ, lấy trung bình chiều cao để tạo chuỗi feature 512. BiLSTM hidden 256 mỗi chiều bổ sung ngữ cảnh trước/sau. Metric: cer (càng thấp càng tốt).
Objective cần nối: CTC với blank của tokenizer; kiểm độ dài feature/target và nhãn lặp, không che mẫu lỗi bằng zero_infinity. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
