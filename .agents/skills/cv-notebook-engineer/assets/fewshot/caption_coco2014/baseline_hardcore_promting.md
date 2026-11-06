# Prompt hardcore: COCO2014: ResNet50 và attention LSTM captioning

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
Input: Prompt, đặc tả chữ: MS COCO Captions2014 local DATA_ROOT/train2014,val2014,annotations/captions_train2014.json và captions_val2014.json; CocoCaptions cần pycocotools. Record id,image_id,image,refs; train2014 là train; val2014 chia5000 validation và held-out còn lại theo image, seed42; không phải Karpathy/test server. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Không gửi mã nguồn.
Merge: Lưu riêng phần data để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: MS COCO Captions2014 local DATA_ROOT/train2014,val2014,annotations/captions_train2014.json và captions_val2014.json; CocoCaptions cần pycocotools. Record id,image_id,image,refs; train2014 là train; val2014 chia5000 validation và held-out còn lại theo image, seed42; không phải Karpathy/test server. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Chuẩn hóa schema và chia nhóm
ID: splits
Depends: data
Kind: code
Target: baseline.ipynb#cell=8; lines=1-6
Input: Prompt, bản tóm tắt chữ của data, đặc tả chữ: MS COCO Captions2014 local DATA_ROOT/train2014,val2014,annotations/captions_train2014.json và captions_val2014.json; CocoCaptions cần pycocotools. Record id,image_id,image,refs; train2014 là train; val2014 chia5000 validation và held-out còn lại theo image, seed42; không phải Karpathy/test server. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Không gửi mã nguồn.
Merge: Lưu riêng phần splits để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: MS COCO Captions2014 local DATA_ROOT/train2014,val2014,annotations/captions_train2014.json và captions_val2014.json; CocoCaptions cần pycocotools. Record id,image_id,image,refs; train2014 là train; val2014 chia5000 validation và held-out còn lại theo image, seed42; không phải Karpathy/test server. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần chuẩn hóa record và split đúng đặc tả, kiểm IDs và ngăn rò rỉ; không fit đặc trưng ở bước này. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Imports và cấu hình cố định
ID: config
Depends: splits
Kind: code
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt, bản tóm tắt chữ của splits, đặc tả chữ: AdamW head lr0.0003/backbone0.00001,weight_decay0.0001; epochs20,batch32,patience5,clip5; chọn cider validation tăng; BatchNorm ImageNet eval nhưng layer4 nhận gradient. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Không gửi mã nguồn.
Merge: Lưu riêng phần config để ghép vào bản làm việc theo cell 5, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW head lr0.0003/backbone0.00001,weight_decay0.0001; epochs20,batch32,patience5,clip5; chọn cider validation tăng; BatchNorm ImageNet eval nhưng layer4 nhận gradient. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Vocabulary và caption targets
ID: vocabulary
Depends: config
Kind: code
Target: baseline.ipynb#cell=10; lines=1-17
Input: Prompt, bản tóm tắt chữ của config, đặc tả chữ: tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Không gửi mã nguồn.
Merge: Lưu riêng phần vocabulary để ghép vào bản làm việc theo cell 10, dòng 1–17 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: tokenize,restore_tokenizer,CaptionDataset(records,training=False); train mọi caption, eval mỗi image một lần đủ refs; TOKENIZER_STATE vocab, id_to_word; MAX_TOKENS40. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ vocabulary chỉ từ captions train, regex NFC lowercase và BOS/EOS, không làm mất references đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Letterbox và normalization
ID: images
Depends: vocabulary
Kind: code
Target: baseline.ipynb#cell=10; lines=19-27
Input: Prompt, bản tóm tắt chữ của vocabulary, đặc tả chữ: image_tensor(value)->float[3,256,256]; mean .485,.456,.406/std .229,.224,.225; giữ center padding; PREPROCESS_CONFIG đồng bộ. Không gửi mã nguồn.
Merge: Lưu riêng phần images để ghép vào bản làm việc theo cell 10, dòng 19–27 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: image_tensor(value)->float[3,256,256]; mean .485,.456,.406/std .229,.224,.225; giữ center padding; PREPROCESS_CONFIG đồng bộ. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Nâng image_tensor thành letterbox RGB nền trắng 256 giữ toàn ảnh, rồi chuẩn ImageNet. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Caption batch
ID: batch
Depends: images
Kind: code
Target: baseline.ipynb#cell=10; lines=44-49
Input: Prompt, bản tóm tắt chữ của images, đặc tả chữ: collate_batch(items)->images,tokens,ids,records; CaptionDataset train index mọi refs, eval một/image; tokens BOS đầu EOS cuối, PAD0. Không gửi mã nguồn.
Merge: Lưu riêng phần batch để ghép vào bản làm việc theo cell 10, dòng 44–49 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: collate_batch(items)->images,tokens,ids,records; CaptionDataset train index mọi refs, eval một/image; tokens BOS đầu EOS cuối, PAD0. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ collate_batch với tokens PAD và image IDs khi đổi encoder; không lặp ảnh lúc đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: ResNet50 visual memory
ID: backbone
Depends: batch
Kind: code
Target: baseline.ipynb#cell=12; lines=3-10
Input: Prompt, bản tóm tắt chữ của batch, đặc tả chữ: CaptionModel(cfg,initialize_pretrained=True); fc Identity; projection2048->memory_dim256; train(mode) gọi super rồi backbone.eval; tải weights lỗi phải báo lỗi. Không gửi mã nguồn.
Merge: Lưu riêng phần backbone để ghép vào bản làm việc theo cell 12, dòng 3–10 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: CaptionModel(cfg,initialize_pretrained=True); fc Identity; projection2048->memory_dim256; train(mode) gọi super rồi backbone.eval; tải weights lỗi phải báo lỗi. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay CNN bằng ResNet50 ImageNet V2, chỉ fine-tune layer4 và giữ BatchNorm eval. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Khung attention LSTM
ID: decoder_init
Depends: backbone
Kind: code
Target: baseline.ipynb#cell=12; lines=11-20
Input: Prompt, bản tóm tắt chữ của backbone, đặc tả chữ: cfg vocab_size,embed_dim256,hidden_dim512,memory_dim256,dropout0.3; decoder input embed+memory; output Linear(hidden+memory,V); memory_attention,hidden_attention,attention_score. Không gửi mã nguồn.
Merge: Lưu riêng phần decoder_init để ghép vào bản làm việc theo cell 12, dòng 11–20 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: cfg vocab_size,embed_dim256,hidden_dim512,memory_dim256,dropout0.3; decoder input embed+memory; output Linear(hidden+memory,V); memory_attention,hidden_attention,attention_score. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung embedding, additive attention, initial_h/c, LSTMCell và output head vào CaptionModel. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Spatial memory và trạng thái
ID: encode
Depends: decoder_init
Kind: code
Target: baseline.ipynb#cell=12; lines=26-30
Input: Prompt, bản tóm tắt chữ của decoder_init, đặc tả chữ: encode(images)->[B,P,256]; start_state(memory)->h,c [B,512] qua tanh; giữ gradient layer4; thêm train(mode) giữ backbone BatchNorm eval. Không gửi mã nguồn.
Merge: Lưu riêng phần encode để ghép vào bản làm việc theo cell 12, dòng 26–30 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: encode(images)->[B,P,256]; start_state(memory)->h,c [B,512] qua tanh; giữ gradient layer4; thêm train(mode) giữ backbone BatchNorm eval. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết encode lấy feature map ResNet thành spatial memory và start_state từ trung bình memory. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Attention decoder step
ID: step
Depends: encode
Kind: code
Target: baseline.ipynb#cell=12; lines=34-40
Input: Prompt, bản tóm tắt chữ của encode, đặc tả chữ: step(token long[B],state=(h,c),memory[B,P,256])->logits[B,V],(h,c); forward(images,tokens)->logits[B,T,V] teacher forcing; trả riêng các method hoàn chỉnh. Không gửi mã nguồn.
Merge: Lưu riêng phần step để ghép vào bản làm việc theo cell 12, dòng 34–40 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: step(token long[B],state=(h,c),memory[B,P,256])->logits[B,V],(h,c); forward(images,tokens)->logits[B,T,V] teacher forcing; trả riêng các method hoàn chỉnh. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết step chọn spatial context bằng additive attention rồi cập nhật LSTMCell và logits. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Forward teacher forcing
ID: teacher_forward
Depends: step
Kind: code
Target: baseline.ipynb#cell=12; lines=41-45
Input: Prompt, bản tóm tắt chữ của step, đặc tả chữ: forward(images float[B,3,H,W],tokens long[B,T])->logits[B,T,V]; stack output theo chiều1, không softmax trước CE. Không gửi mã nguồn.
Merge: Lưu riêng phần teacher_forward để ghép vào bản làm việc theo cell 12, dòng 41–45 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: forward(images float[B,3,H,W],tokens long[B,T])->logits[B,T,V]; stack output theo chiều1, không softmax trước CE. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết forward(images,tokens) dùng encode, start_state và step theo từng target token để teacher forcing. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Greedy EOS decoding
ID: generate
Depends: step
Kind: code
Target: baseline.ipynb#cell=12; lines=47-57
Input: Prompt, bản tóm tắt chữ của step, đặc tả chữ: generate(images,max_tokens)->long[B,L]; BOS_ID2,EOS_ID3; đóng băng token EOS khi done; build_model(cfg,initialize_pretrained=True) giữ interface. Không gửi mã nguồn.
Merge: Lưu riêng phần generate để ghép vào bản làm việc theo cell 12, dòng 47–57 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: generate(images,max_tokens)->long[B,L]; BOS_ID2,EOS_ID3; đóng băng token EOS khi done; build_model(cfg,initialize_pretrained=True) giữ interface. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay generate bằng greedy decoding dùng step, chặn PAD/BOS và dừng EOS độc lập từng ảnh. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Masked teacher forcing
ID: loss
Depends: generate
Kind: code
Target: baseline.ipynb#cell=15; lines=7-10
Input: Prompt, bản tóm tắt chữ của generate, đặc tả chữ: compute_loss(model,batch)->scalar; forward(images,tokens[:,:-1]), target=tokens[:,1:]; decode_tokens dừng EOS; gold refs không truncate khi chấm. Không gửi mã nguồn.
Merge: Lưu riêng phần loss để ghép vào bản làm việc theo cell 15, dòng 7–10 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: compute_loss(model,batch)->scalar; forward(images,tokens[:,:-1]), target=tokens[:,1:]; decode_tokens dừng EOS; gold refs không truncate khi chấm. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ cross-entropy teacher forcing, bỏ PAD và dịch caption một token cho attention decoder. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Factory và cấu hình model
ID: factory
Depends: vocabulary, images, batch, backbone, decoder_init, encode, step, teacher_forward, generate, loss
Kind: code
Target: baseline.ipynb#cell=13; lines=1-1
Input: Prompt, bản tóm tắt chữ của vocabulary, images, batch, backbone, decoder_init, encode, step, teacher_forward, generate, loss, đặc tả chữ: Factory CaptionModel theo chữ ký constructor đã tóm tắt; cấu hình {"vocab_size":len(vocab),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"memory_dim":MEMORY_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Không gửi mã nguồn.
Merge: Lưu riêng phần factory để ghép vào bản làm việc theo cell 13, dòng 1–1 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Factory CaptionModel theo chữ ký constructor đã tóm tắt; cấu hình {"vocab_size":len(vocab),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"memory_dim":MEMORY_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Nối Dataset và DataLoader
ID: loaders
Depends: splits, config, vocabulary, images, batch, backbone, decoder_init, encode, step, teacher_forward, generate, loss, factory
Kind: code
Target: baseline.ipynb#cell=10; lines=44-49
Input: Prompt, bản tóm tắt chữ của splits, config, vocabulary, images, batch, backbone, decoder_init, encode, step, teacher_forward, generate, loss, factory, đặc tả chữ: images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. AdamW head lr0.0003/backbone0.00001,weight_decay0.0001; epochs20,batch32,patience5,clip5; chọn cider validation tăng; BatchNorm ImageNet eval nhưng layer4 nhận gradient. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Không gửi mã nguồn.
Merge: Lưu riêng phần loaders để ghép vào bản làm việc theo cell 10, dòng 44–49 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. AdamW head lr0.0003/backbone0.00001,weight_decay0.0001; epochs20,batch32,patience5,clip5; chọn cider validation tăng; BatchNorm ImageNet eval nhưng layer4 nhận gradient. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết Dataset/DataLoader còn thiếu theo hợp đồng batch, chỉ shuffle train và giữ IDs/order khi đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Helper decode_tokens
ID: helper_decode_tokens
Depends: loaders
Kind: code
Target: baseline.ipynb#cell=15; lines=1-6
Input: Prompt, bản tóm tắt chữ của loaders, đặc tả chữ: decode_tokens nhận (ids); Bỏ PAD/BOS, dừng EOS, map id_to_word và ghép từ theo baseline, không dùng references để sửa caption. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_decode_tokens để ghép vào bản làm việc theo cell 15, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: decode_tokens nhận (ids); Bỏ PAD/BOS, dừng EOS, map id_to_word và ghép từ theo baseline, không dùng references để sửa caption. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng decode_tokens theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Helper ptb_tokenize
ID: helper_ptb_tokenize
Depends: helper_decode_tokens
Kind: code
Target: baseline.ipynb#cell=16; lines=5-22
Input: Prompt, bản tóm tắt chữ của helper_decode_tokens, đặc tả chữ: ptb_tokenize nhận (mapping); Dùng Stanford PTBTokenizer đi kèm pycocoevalcap và Java, preserveLines/lowerCase, bỏ PTB punctuation; giữ key/count, lỗi khi thiếu Java. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_ptb_tokenize để ghép vào bản làm việc theo cell 16, dòng 5–22 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: ptb_tokenize nhận (mapping); Dùng Stanford PTBTokenizer đi kèm pycocoevalcap và Java, preserveLines/lowerCase, bỏ PTB punctuation; giữ key/count, lỗi khi thiếu Java. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng ptb_tokenize theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Helper score_corpus
ID: helper_score_corpus
Depends: helper_ptb_tokenize
Kind: code
Target: baseline.ipynb#cell=16; lines=24-29
Input: Prompt, bản tóm tắt chữ của helper_ptb_tokenize, đặc tả chữ: score_corpus nhận (records, predictions); PTB cả refs và predictions, Cider(n=4,sigma=6.0), trả cider raw không nhân100, image_count và tokenizer provenance. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_score_corpus để ghép vào bản làm việc theo cell 16, dòng 24–29 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: score_corpus nhận (records, predictions); PTB cả refs và predictions, Cider(n=4,sigma=6.0), trả cider raw không nhân100, image_count và tokenizer provenance. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng score_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Metric chính thức và evaluation
ID: metrics
Depends: helper_score_corpus
Kind: code
Target: baseline.ipynb#cell=17; lines=2-11
Input: Prompt, bản tóm tắt chữ của helper_score_corpus, đặc tả chữ: CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Không gửi mã nguồn.
Merge: Lưu riêng phần metrics để ghép vào bản làm việc theo cell 17, dòng 2–11 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Target: baseline.ipynb#cell=20; lines=1-20
Input: Prompt, bản tóm tắt chữ của metrics, đặc tả chữ: AdamW head lr0.0003/backbone0.00001,weight_decay0.0001; epochs20,batch32,patience5,clip5; chọn cider validation tăng; BatchNorm ImageNet eval nhưng layer4 nhận gradient. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Không gửi mã nguồn.
Merge: Lưu riêng phần training để ghép vào bản làm việc theo cell 20, dòng 1–20 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW head lr0.0003/backbone0.00001,weight_decay0.0001; epochs20,batch32,patience5,clip5; chọn cider validation tăng; BatchNorm ImageNet eval nhưng layer4 nhận gradient. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 21: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Target: baseline.ipynb#cell=22; lines=1-19
Input: Prompt, bản tóm tắt chữ của training, đặc tả chữ: CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Không gửi mã nguồn.
Merge: Lưu riêng phần reload để ghép vào bản làm việc theo cell 22, dòng 1–19 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 22: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Target: baseline.ipynb#cell=24; lines=2-5
Input: Prompt, bản tóm tắt chữ của reload, đặc tả chữ: images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Hàm describe_image nhận (image), giữ chữ ký và output schema đã chốt, không cần gold. Cần open_rgb nhận path/bytes/PIL và kiểm ảnh hỏng rõ. Không gửi mã nguồn.
Merge: Lưu riêng phần inference để ghép vào bản làm việc theo cell 24, dòng 2–5 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. CIDEr pycocoevalcap n=4,sigma=6.0 với Stanford PTB lowercase; kiểm Java, giữ mọi refs và mỗi image đúng một lần; cider raw không nhân100, primary tăng; không thêm metric khác. Hàm describe_image nhận (image), giữ chữ ký và output schema đã chốt, không cần gold. Cần open_rgb nhận path/bytes/PIL và kiểm ảnh hỏng rõ. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 23: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 24: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Target: baseline.ipynb#cell=22; lines=1-19
Input: Prompt, bản tóm tắt chữ của summary, đặc tả chữ: images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Không gửi mã nguồn.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,256,256], tokens long[B,T], ids,records; vocabulary train tối đa10000,minfreq3,PAD/UNK/BOS/EOS=0/1/2/3; token regex NFC lowercase; MAX_TOKENS=40. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

