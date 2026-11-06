# Prompt hardcore: IAM-line: ResNet18, BiLSTM và CTC

Variant: hardcore
Model: gemini-3.8-flash
Thinking: low
Token verification: unverified

Đích là phương pháp/chức năng của baseline.ipynb, không phải cam kết cùng điểm số. Cell được đếm từ 1, gồm cả Markdown; dòng tính trong cell gốc. Target để người học đối chiếu, không phải code phải gửi cho chatbot. Các notebook và báo cáo mẫu được giữ nguyên; làm các nâng cấp trong bản làm việc riêng. Notebook hardcore gốc vẫn chỉ inference chưa huấn luyện.

Mỗi bước mở một phiên mới và chỉ gửi nội dung Input cùng đoạn Prompt, không gửi cả hướng dẫn. Basic dùng đoạn Source hiện tại sau các sửa đổi, không chép lại bản cũ làm mất thay đổi; nếu cell đã tách, tìm đúng hàm/đoạn tương ứng. Hardcore chỉ dùng mô tả và bản tóm tắt chữ, không gửi code. Những yêu cầu đặc tả trong Input là phần cần gửi cùng prompt; không dựa vào việc chatbot đọc được file này.

Đối chiếu code và bản tóm tắt API sau mỗi bước, lưu code cục bộ và chỉ chuyển hợp đồng cần thiết. Các nhánh có nhiều phụ thuộc chỉ gửi tóm tắt liên quan; nếu quá dài, dùng prompt gộp ở bước summary trong một phiên riêng trước module đó, với Depends là các bước đã hoàn tất. Gộp mất chữ ký/cấu hình thì bổ sung trong phiên nhỏ khác; không đoán. Bước gộp cuối chỉ giữ exports/cấu hình phục vụ glue, không tích lũy lịch sử hay toàn bộ thuật toán.

Giới hạn mỗi phiên là 2000 tokens gồm prompt, code, tóm tắt, system instructions, câu trả lời và thinking. Các Budget dưới đây là hạn mức dự kiến 600 input + 1100 thinking/output + 300 dự phòng, CHƯA đo bằng Gemini. Ưu tiên prompt dưới 100 tokens nhưng giữ yêu cầu bắt buộc. Dùng Gemini countTokens đo input thực tế; đặt max_output_tokens tối đa1100 nếu nền tảng hỗ trợ và kiểm usageMetadata gồm thinking. Nếu vượt600 input hoặc phần code/handoff không đủ output, chia module thành unit nhỏ và tính lại trước khi gửi. Không dùng số từ/ký tự hay tokenizer model khác làm số đo Gemini. Không có bộ đếm/cấu hình trần thì trạng thái vẫn chưa kiểm chứng; không hứa tự bảo đảm giới hạn bằng lời prompt.

Nếu output bị cắt, chỉ giữ unit hoàn chỉnh; phiên mới gửi tóm tắt cần thiết và yêu cầu unit nhỏ còn thiếu, không yêu cầu tiếp tục dựa vào lịch sử cũ. Giữ fixed constants, PyTorch precision thường nếu dùng PyTorch; không AMP, tự dò VRAM hay tự thay batch khi OOM. Fit biểu diễn chỉ train, chọn checkpoint trên validation, chấm held-out sau khi chốt. 32GB là mục tiêu thiết kế, không phải số đo đã chạy.

## Bước 1: Dữ liệu, split và schema
ID: data
Depends: none
Kind: code
Target: baseline.ipynb#cell=8; lines=1-6
Input: Prompt, đặc tả chữ: Teklia/IAM-line revision fbdad97500ce54635c0d1ba306bf535cb40656cf; train/validation/test có image,text; id=split:index, ảnh path/bytes. Giữ split gốc; chuẩn text theo normalize_text nguồn, không hứa writer-disjoint khi thiếu writer_id. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Không gửi mã nguồn.
Merge: Lưu riêng phần data để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Teklia/IAM-line revision fbdad97500ce54635c0d1ba306bf535cb40656cf; train/validation/test có image,text; id=split:index, ảnh path/bytes. Giữ split gốc; chuẩn text theo normalize_text nguồn, không hứa writer-disjoint khi thiếu writer_id. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Chuẩn hóa schema và chia nhóm
ID: splits
Depends: data
Kind: code
Target: baseline.ipynb#cell=8; lines=1-6
Input: Prompt, bản tóm tắt chữ của data, đặc tả chữ: Teklia/IAM-line revision fbdad97500ce54635c0d1ba306bf535cb40656cf; train/validation/test có image,text; id=split:index, ảnh path/bytes. Giữ split gốc; chuẩn text theo normalize_text nguồn, không hứa writer-disjoint khi thiếu writer_id. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Không gửi mã nguồn.
Merge: Lưu riêng phần splits để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Teklia/IAM-line revision fbdad97500ce54635c0d1ba306bf535cb40656cf; train/validation/test có image,text; id=split:index, ảnh path/bytes. Giữ split gốc; chuẩn text theo normalize_text nguồn, không hứa writer-disjoint khi thiếu writer_id. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần chuẩn hóa record và split đúng đặc tả, kiểm IDs và ngăn rò rỉ; không fit đặc trưng ở bước này. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Imports và cấu hình cố định
ID: config
Depends: splits
Kind: code
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt, bản tóm tắt chữ của splits, đặc tả chữ: AdamW hai nhóm lr head=0.001/backbone=0.0001,weight_decay=0.0001; epochs=50,batch=8,patience=8,clip=5; chọn cer validation giảm; không AMP hay tự dò VRAM. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Không gửi mã nguồn.
Merge: Lưu riêng phần config để ghép vào bản làm việc theo cell 5, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW hai nhóm lr head=0.001/backbone=0.0001,weight_decay=0.0001; epochs=50,batch=8,patience=8,clip=5; chọn cer validation giảm; không AMP hay tự dò VRAM. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Alphabet train-only
ID: alphabet
Depends: config
Kind: code
Target: baseline.ipynb#cell=10; lines=1-14
Input: Prompt, bản tóm tắt chữ của config, đặc tả chữ: char_vocab, id_to_char, TOKENIZER_STATE; blank=0,UNK=1; target_lengths thật, gold CER/WER giữ ký tự chưa thấy; IMAGE_HEIGHT=64. Không gửi mã nguồn.
Merge: Lưu riêng phần alphabet để ghép vào bản làm việc theo cell 10, dòng 1–14 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: char_vocab, id_to_char, TOKENIZER_STATE; blank=0,UNK=1; target_lengths thật, gold CER/WER giữ ký tự chưa thấy; IMAGE_HEIGHT=64. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ alphabet train-only, blank/PAD và UNK; lưu mapping để reload CTC nhất quán. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Ảnh giữ tỷ lệ và normalization
ID: images
Depends: alphabet
Kind: code
Target: baseline.ipynb#cell=10; lines=16-23
Input: Prompt, bản tóm tắt chữ của alphabet, đặc tả chữ: image_tensor(value)->float[3,64,W]; mean .485,.456,.406 std .229,.224,.225; open_rgb nhận path/bytes/PIL; resize bilinear. Không gửi mã nguồn.
Merge: Lưu riêng phần images để ghép vào bản làm việc theo cell 10, dòng 16–23 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: image_tensor(value)->float[3,64,W]; mean .485,.456,.406 std .229,.224,.225; open_rgb nhận path/bytes/PIL; resize bilinear. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Nâng image_tensor sang RGB cao64 giữ tỷ lệ, chuẩn ImageNet; không cắt chiều rộng làm mất chữ. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Padding và CTC lengths
ID: batch
Depends: images
Kind: code
Target: baseline.ipynb#cell=10; lines=35-48
Input: Prompt, bản tóm tắt chữ của images, đặc tả chữ: images[B,3,64,Wmax], widths nguyên bản; required=target length+số cặp ký tự lặp; widths//4>=required; không thay labels hay giảm đánh giá. Không gửi mã nguồn.
Merge: Lưu riêng phần batch để ghép vào bản làm việc theo cell 10, dòng 35–48 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images[B,3,64,Wmax], widths nguyên bản; required=target length+số cặp ký tự lặp; widths//4>=required; không thay labels hay giảm đánh giá. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Cập nhật padding nền trắng đã chuẩn hóa và kiểm tra số bước CTC kể cả ký tự lặp. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: ResNet18 và stride
ID: backbone
Depends: batch
Kind: code
Target: baseline.ipynb#cell=12; lines=4-21
Input: Prompt, bản tóm tắt chữ của batch, đặc tả chữ: OCRModel(cfg,initialize_pretrained=True); conv1(2,1), maxpool(2,2), layer2(2,1),layer3(2,2),layer4(2,1), cả conv1/downsample; tải sai weights phải lỗi, fc=Identity. Không gửi mã nguồn.
Merge: Lưu riêng phần backbone để ghép vào bản làm việc theo cell 12, dòng 4–21 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: OCRModel(cfg,initialize_pretrained=True); conv1(2,1), maxpool(2,2), layer2(2,1),layer3(2,2),layer4(2,1), cả conv1/downsample; tải sai weights phải lỗi, fc=Identity. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay CNN scratch bằng ResNet18 ImageNet V1, chỉnh stride để chiều ngang giảm4 và chiều cao giảm32. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: BiLSTM và CTC head
ID: head
Depends: backbone
Kind: code
Target: baseline.ipynb#cell=12; lines=22-22
Input: Prompt, bản tóm tắt chữ của backbone, đặc tả chữ: sequence LSTM input512,hidden_dim=256,num_layers=2,bidirectional=True,dropout=0.2,batch_first; head Linear512->num_classes; build_model(cfg,initialize_pretrained=True). Không gửi mã nguồn.
Merge: Lưu riêng phần head để ghép vào bản làm việc theo cell 12, dòng 22–22 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: sequence LSTM input512,hidden_dim=256,num_layers=2,bidirectional=True,dropout=0.2,batch_first; head Linear512->num_classes; build_model(cfg,initialize_pretrained=True). Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung BiLSTM hai lớp và CTC head vào constructor OCRModel, giữ cfg và build_model. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Feature sequence và packed LSTM
ID: forward
Depends: head
Kind: code
Target: baseline.ipynb#cell=12; lines=24-33
Input: Prompt, bản tóm tắt chữ của head, đặc tả chữ: forward(images,widths)->log_softmax[B,T,V], CPU lengths; head sau pad_packed_sequence total_length=T; không copy gold text làm prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần forward để ghép vào bản làm việc theo cell 12, dòng 24–33 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: forward(images,widths)->log_softmax[B,T,V], CPU lengths; head sau pad_packed_sequence total_length=T; không copy gold text làm prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay forward để lấy feature map ResNet, trung bình chiều cao và chạy packed BiLSTM theo widths//4. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: CTC loss và decoding
ID: ctc
Depends: forward
Kind: code
Target: baseline.ipynb#cell=15; lines=20-24
Input: Prompt, bản tóm tắt chữ của forward, đặc tả chữ: compute_loss(model,batch)->scalar; model output[B,T,V]; ctc_decode bỏ blank và gộp lặp, reset lặp khi blank; CER/WER theo gold đầy đủ. Khai báo ctc_loss=CTCLoss(blank=0,reduction mean,zero_infinity=False); lengths không vượt T. Không gửi mã nguồn.
Merge: Lưu riêng phần ctc để ghép vào bản làm việc theo cell 15, dòng 20–24 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: compute_loss(model,batch)->scalar; model output[B,T,V]; ctc_decode bỏ blank và gộp lặp, reset lặp khi blank; CER/WER theo gold đầy đủ. Khai báo ctc_loss=CTCLoss(blank=0,reduction mean,zero_infinity=False); lengths không vượt T. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ CTCLoss blank0 với logp đổi sang T,B,V, input lengths đúng và target_lengths thật. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Factory và cấu hình model
ID: factory
Depends: alphabet, images, batch, backbone, head, forward, ctc
Kind: code
Target: baseline.ipynb#cell=13; lines=1-1
Input: Prompt, bản tóm tắt chữ của alphabet, images, batch, backbone, head, forward, ctc, đặc tả chữ: Factory OCRModel theo chữ ký constructor đã tóm tắt; cấu hình  {"hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "num_classes": len(char_vocab)}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Không gửi mã nguồn.
Merge: Lưu riêng phần factory để ghép vào bản làm việc theo cell 13, dòng 1–1 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Factory OCRModel theo chữ ký constructor đã tóm tắt; cấu hình  {"hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "num_classes": len(char_vocab)}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Nối Dataset và DataLoader
ID: loaders
Depends: splits, config, alphabet, images, batch, backbone, head, forward, ctc, factory
Kind: code
Target: baseline.ipynb#cell=10; lines=35-48
Input: Prompt, bản tóm tắt chữ của splits, config, alphabet, images, batch, backbone, head, forward, ctc, factory, đặc tả chữ: images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. AdamW hai nhóm lr head=0.001/backbone=0.0001,weight_decay=0.0001; epochs=50,batch=8,patience=8,clip=5; chọn cer validation giảm; không AMP hay tự dò VRAM. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Không gửi mã nguồn.
Merge: Lưu riêng phần loaders để ghép vào bản làm việc theo cell 10, dòng 35–48 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. AdamW hai nhóm lr head=0.001/backbone=0.0001,weight_decay=0.0001; epochs=50,batch=8,patience=8,clip=5; chọn cer validation giảm; không AMP hay tự dò VRAM. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết Dataset/DataLoader còn thiếu theo hợp đồng batch, chỉ shuffle train và giữ IDs/order khi đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Helper edit_distance
ID: helper_edit_distance
Depends: loaders
Kind: code
Target: baseline.ipynb#cell=15; lines=1-8
Input: Prompt, bản tóm tắt chữ của loaders, đặc tả chữ: edit_distance nhận (a, b); Levenshtein DP hai hàng, áp dụng được chuỗi ký tự và danh sách từ. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_edit_distance để ghép vào bản làm việc theo cell 15, dòng 1–8 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: edit_distance nhận (a, b); Levenshtein DP hai hàng, áp dụng được chuỗi ký tự và danh sách từ. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng edit_distance theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Helper ctc_decode
ID: helper_ctc_decode
Depends: helper_edit_distance
Kind: code
Target: baseline.ipynb#cell=15; lines=10-17
Input: Prompt, bản tóm tắt chữ của helper_edit_distance, đặc tả chữ: ctc_decode nhận (path); Gộp nhãn lặp liền, blank reset previous; bỏ blank0, UNK thành replacement char, normalize_text như gold. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_ctc_decode để ghép vào bản làm việc theo cell 15, dòng 10–17 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: ctc_decode nhận (path); Gộp nhãn lặp liền, blank reset previous; bỏ blank0, UNK thành replacement char, normalize_text như gold. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng ctc_decode theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Metric chính thức và evaluation
ID: metrics
Depends: helper_ctc_decode
Kind: code
Target: baseline.ipynb#cell=15; lines=27-48
Input: Prompt, bản tóm tắt chữ của helper_ctc_decode, đặc tả chữ: ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Không gửi mã nguồn.
Merge: Lưu riêng phần metrics để ghép vào bản làm việc theo cell 15, dòng 27–48 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Target: baseline.ipynb#cell=18; lines=1-20
Input: Prompt, bản tóm tắt chữ của metrics, đặc tả chữ: AdamW hai nhóm lr head=0.001/backbone=0.0001,weight_decay=0.0001; epochs=50,batch=8,patience=8,clip=5; chọn cer validation giảm; không AMP hay tự dò VRAM. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Không gửi mã nguồn.
Merge: Lưu riêng phần training để ghép vào bản làm việc theo cell 18, dòng 1–20 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW hai nhóm lr head=0.001/backbone=0.0001,weight_decay=0.0001; epochs=50,batch=8,patience=8,clip=5; chọn cer validation giảm; không AMP hay tự dò VRAM. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Target: baseline.ipynb#cell=20; lines=1-19
Input: Prompt, bản tóm tắt chữ của training, đặc tả chữ: ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Không gửi mã nguồn.
Merge: Lưu riêng phần reload để ghép vào bản làm việc theo cell 20, dòng 1–19 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Target: baseline.ipynb#cell=22; lines=2-7
Input: Prompt, bản tóm tắt chữ của reload, đặc tả chữ: images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Hàm recognize_line nhận (image), giữ chữ ký và output schema đã chốt, không cần gold. Cần open_rgb nhận path/bytes/PIL và kiểm ảnh hỏng rõ. Không gửi mã nguồn.
Merge: Lưu riêng phần inference để ghép vào bản làm việc theo cell 22, dòng 2–7 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. ctc_decode greedy, CER/WER tổng edit distance chia tổng ký tự/từ gold; primary cer giảm; xuất đủ IDs theo split, không average sai theo batch. Hàm recognize_line nhận (image), giữ chữ ký và output schema đã chốt, không cần gold. Cần open_rgb nhận path/bytes/PIL và kiểm ảnh hỏng rõ. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 20: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Target: baseline.ipynb#cell=20; lines=1-19
Input: Prompt, bản tóm tắt chữ của summary, đặc tả chữ: images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Không gửi mã nguồn.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,64,W], widths CPU long[B], targets long tổng độ dài, target_lengths CPU, ids,texts; CTC blank=0; model trả logp[B,T,V], lengths=widths//4. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

