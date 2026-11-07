# Prompt hardcore: COCO Captions 2014 — bản tối ưu

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
Nguồn/schema/split: {"dataset":"MS COCO Captions 2014","reader":"torchvision.datasets.CocoCaptions","split_protocol":"train2014; val2014 chia image-disjoint validation5000/held-out còn lại"}. COCO Captions 2014 qua CocoCaptions có 82.783 ảnh train2014 và 40.504 ảnh val2014 theo thông tin nguồn đã ghi nhận. Mỗi ảnh có nhiều caption. Baseline học từ train2014; chia val2014 theo image ID và seed cố định thành 5.000 ảnh validation, phần còn lại 35.504 ảnh held-out có nhãn. Đây không phải COCO hidden test hoặc Karpathy split. Cùng một ảnh có thể tạo nhiều cặp train, nhưng mọi caption của ảnh phải cùng split. Khi đánh giá, mỗi ảnh chỉ sinh một dự đoán và dùng toàn bộ captions gốc làm references. Khó khăn là vật thể nhỏ, quan hệ giữa các vật và tên hiếm; một câu trôi chảy vẫn có thể mô tả chi tiết không tồn tại. Dự án đã khảo sát annotation và mẫu ảnh, chưa quét toàn bộ ảnh hay chứng minh không có ảnh gần trùng giữa các ID.
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
Target: baseline.ipynb#cell=10; lines=2-8
Target: baseline.ipynb#cell=10; lines=14-27
Target: baseline.ipynb#cell=10; lines=54-55
Config owns: preprocess_1
Config uses: splits
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)

### Prompt tạo hoặc sửa module

```text
Cập nhật biểu diễn/căn chỉnh; chỉ fit train, giữ gold/ID, special IDs và mask; giữ trạng thái và API khôi phục biểu diễn.
Ý tưởng tiền xử lí: 1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh. 2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng. 3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp. 4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
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
Target: baseline.ipynb#cell=6; lines=21-24
Target: baseline.ipynb#cell=10; lines=29-53
Config owns: preprocess_2
Config uses: splits, preprocess_1
-->

### Context cần chuẩn bị

- [Load dữ liệu và chia train/validation/test](#context-s01)
- [Biểu diễn và căn chỉnh](#context-s02)

### Prompt tạo hoặc sửa module

```text
Tạo Dataset/collate/DataLoader dùng biểu diễn đã chốt; thống nhất tensor, padding, mask và lengths với model; shuffle chỉ train.
Ý tưởng tiền xử lí: 1. Chia theo ảnh trước khi xử lý caption. Train sử dụng mọi caption; validation/held-out giữ một mẫu mỗi ảnh với toàn bộ references, tránh đánh giá lặp ảnh. 2. Chuyển ảnh RGB, resize giữ tỷ lệ rồi letterbox nền trắng về kích thước đã chọn. Giữ toàn ảnh giúp giảm nguy cơ crop mất đối tượng mà caption đang nhắc; chuẩn hóa theo encoder ImageNet được phép dùng. 3. Caption của mô hình dùng NFC, chữ thường và regex từ/dấu câu; fit vocabulary trên captions train. Thêm BOS/EOS/PAD, dịch target một bước và bỏ PAD khỏi loss để học dự đoán từ kế tiếp. 4. Giới hạn target train theo MAX_TOKENS và ghi nhận việc cắt; gold references dùng để chấm không bị rút ngắn hoặc thay từ lạ bằng UNK. Inference dừng ở EOS, không đọc caption gold. Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train.
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
Target: baseline.ipynb#cell=12; lines=2-57
Config owns: encoder
Config uses: preprocess_2
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)

### Prompt tạo hoặc sửa module

```text
Viết một encoder hoàn chỉnh cho kiến trúc bên dưới, gồm embedding/backbone và mask/lengths. Xuất memory và trạng thái/shape cần cho attention/head; chưa viết decoder hay training loop.
Ý tưởng mô hình: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản. Chỉ cho pretrained backbone ảnh resnet50/ResNet50_Weights.IMAGENET1K_V2; phần ngôn ngữ/head học mới.
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

## Bước 5: Xử lí mô hình — Model và giao diện sử dụng

<!-- module
ID: s05
Group: model
Kind: code
Depends: s03, s04
Budget: input=650; output=950; reserve=200
Summary: yes
Summary budget: input=1400; output=80; reserve=200
Target: baseline.ipynb#cell=12; lines=2-57
Config owns: architecture
Config uses: preprocess_2, encoder
-->

### Context cần chuẩn bị

- [Dataset và batch](#context-s03)
- [Encoder độc lập](#context-s04)

### Prompt tạo hoặc sửa module

```text
Viết model hoàn chỉnh dùng các thành phần trong context; giữ API hiện có hoặc thêm adapter tối thiểu. Gồm head/decoder, suy luận và method loss nếu vòng train cần; tự chọn kích thước/dropout, lưu cùng model config.
Ý tưởng mô hình: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Attention tính trọng số các vùng ảnh ở mỗi bước sinh từ; greedy decoding giữ inference đơn giản. Chỉ cho pretrained backbone ảnh resnet50/ResNet50_Weights.IMAGENET1K_V2; phần ngôn ngữ/head học mới.
API hiện có: RawImageTextModel: __init__(); forward(rows).
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
Phương pháp: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Metric: cider. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
Phương pháp: Chọn ResNet50 ImageNet → đặc trưng không gian → attention cộng → LSTM decoder scratch. Encoder dùng torchvision/resnet50 với ResNet50_Weights.IMAGENET1K_V2; chỉ layer4 được fine-tune, BatchNorm giữ eval. Feature 2.048 chiều được chiếu về memory 256 chiều; decoder có embedding 256 và hidden 512. Metric: cider.
Objective cần nối: cross-entropy theo head/target của bài; chỉ bỏ padding khi target có padding. Neural: precision thường, nhắm GPU 32 GB; không AMP, dò VRAM hoặc tự đổi batch.
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
