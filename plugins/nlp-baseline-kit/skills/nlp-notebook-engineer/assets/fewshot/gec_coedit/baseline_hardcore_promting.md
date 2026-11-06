# Prompt hardcore: CoEdIT GEC: BPE, BiGRU attention và pointer-copy

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
Input: Prompt, đặc tả chữ: grammarly/coedit revision e9a255c33ef910bc33a9d2b522653fa87521583e; chỉ task=gec, schema _id,task,src,tgt. Bỏ đúng instruction prefix đã kiểm tra, không split mọi dấu hai chấm; nhóm src thành refs/source_ids. Official validation là held-out; train chia 10% theo source với seed=42, loại overlap held-out. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Không gửi mã nguồn.
Merge: Lưu riêng phần data để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: grammarly/coedit revision e9a255c33ef910bc33a9d2b522653fa87521583e; chỉ task=gec, schema _id,task,src,tgt. Bỏ đúng instruction prefix đã kiểm tra, không split mọi dấu hai chấm; nhóm src thành refs/source_ids. Official validation là held-out; train chia 10% theo source với seed=42, loại overlap held-out. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Chuẩn hóa schema và chia nhóm
ID: splits
Depends: data
Kind: code
Target: baseline.ipynb#cell=8; lines=1-6
Input: Prompt, bản tóm tắt chữ của data, đặc tả chữ: grammarly/coedit revision e9a255c33ef910bc33a9d2b522653fa87521583e; chỉ task=gec, schema _id,task,src,tgt. Bỏ đúng instruction prefix đã kiểm tra, không split mọi dấu hai chấm; nhóm src thành refs/source_ids. Official validation là held-out; train chia 10% theo source với seed=42, loại overlap held-out. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Không gửi mã nguồn.
Merge: Lưu riêng phần splits để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: grammarly/coedit revision e9a255c33ef910bc33a9d2b522653fa87521583e; chỉ task=gec, schema _id,task,src,tgt. Bỏ đúng instruction prefix đã kiểm tra, không split mọi dấu hai chấm; nhóm src thành refs/source_ids. Official validation là held-out; train chia 10% theo source với seed=42, loại overlap held-out. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần chuẩn hóa record và split đúng đặc tả, kiểm IDs và ngăn rò rỉ; không fit đặc trưng ở bước này. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Imports và cấu hình cố định
ID: config
Depends: splits
Kind: code
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt, bản tóm tắt chữ của splits, đặc tả chữ: AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Không gửi mã nguồn.
Merge: Lưu riêng phần config để ghép vào bản làm việc theo cell 5, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: BPE chung cho copy
ID: tokenizer
Depends: config
Kind: code
Target: baseline.ipynb#cell=11; lines=1-21
Input: Prompt, bản tóm tắt chữ của config, đặc tả chữ: sp encode/decode/get_piece_size, TOKENIZER_STATE spm_proto, PAD/UNK/BOS/EOS=0/1/2/3; MAX_SOURCE/TARGET/DECODE_TOKENS=160; restore_tokenizer(state). Không gửi mã nguồn.
Merge: Lưu riêng phần tokenizer để ghép vào bản làm việc theo cell 11, dòng 1–21 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: sp encode/decode/get_piece_size, TOKENIZER_STATE spm_proto, PAD/UNK/BOS/EOS=0/1/2/3; MAX_SOURCE/TARGET/DECODE_TOKENS=160; restore_tokenizer(state). Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay tokenizer bằng SentencePiece BPE 8000 train-only cho source và references, byte fallback và identity normalization. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Dataset và masks
ID: batch
Depends: tokenizer
Kind: code
Target: baseline.ipynb#cell=11; lines=35-41
Input: Prompt, bản tóm tắt chữ của tokenizer, đặc tả chữ: PairDataset(records,training=False); source/target long, lengths CPU, ids,records; dùng mọi refs train, giữ refs đầy đủ để chấm. Không gửi mã nguồn.
Merge: Lưu riêng phần batch để ghép vào bản làm việc theo cell 11, dòng 35–41 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: PairDataset(records,training=False); source/target long, lengths CPU, ids,records; dùng mọi refs train, giữ refs đầy đủ để chấm. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ collate_batch và PairDataset khi đổi BPE; padding 0, BOS/EOS và lengths tương thích pointer-copy. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Khung BiGRU attention-copy
ID: constructor
Depends: batch
Kind: code
Target: baseline.ipynb#cell=13; lines=2-14
Input: Prompt, bản tóm tắt chữ của batch, đặc tả chữ: cfg vocab_size,embed_dim=128,hidden_dim=192,dropout=0.3; memory/state dim=384; embedding chung, bridge, memory_projection,hidden_projection,attention_score,output,gate,dropout. Không gửi mã nguồn.
Merge: Lưu riêng phần constructor để ghép vào bản làm việc theo cell 13, dòng 2–14 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: cfg vocab_size,embed_dim=128,hidden_dim=192,dropout=0.3; memory/state dim=384; embedding chung, bridge, memory_projection,hidden_projection,attention_score,output,gate,dropout. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay constructor TextGenerator bằng BiGRU encoder, additive attention, GRUCell decoder và gate copy. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: Packed encoder
ID: encode
Depends: constructor
Kind: code
Target: baseline.ipynb#cell=13; lines=15-20
Input: Prompt, bản tóm tắt chữ của constructor, đặc tả chữ: encode -> memory[B,S,384], projected[B,S,384], hidden[B,384], mask bool[B,S] True ở PAD; lengths CPU; pad total_length bằng S. Không gửi mã nguồn.
Merge: Lưu riêng phần encode để ghép vào bản làm việc theo cell 13, dòng 15–20 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: encode -> memory[B,S,384], projected[B,S,384], hidden[B,384], mask bool[B,S] True ở PAD; lengths CPU; pad total_length bằng S. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết encode(source,lengths) dùng packed BiGRU và bridge trạng thái hai chiều. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Attention và copy distribution
ID: copy_step
Depends: encode
Kind: code
Target: baseline.ipynb#cell=13; lines=21-32
Input: Prompt, bản tóm tắt chữ của encode, đặc tả chữ: step(token[B],hidden[B,384],memory,projected,mask,source long[B,S])->logp[B,V],hidden; scatter_add cộng token lặp, clamp_min 1e-9 rồi normalize; không đưa gold vào inference. Không gửi mã nguồn.
Merge: Lưu riêng phần copy_step để ghép vào bản làm việc theo cell 13, dòng 21–32 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: step(token[B],hidden[B,384],memory,projected,mask,source long[B,S])->logp[B,V],hidden; scatter_add cộng token lặp, clamp_min 1e-9 rồi normalize; không đưa gold vào inference. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết step dùng masked additive attention, GRUCell và gate trộn vocabulary với copy bằng scatter_add. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Teacher forcing
ID: forward
Depends: copy_step
Kind: code
Target: baseline.ipynb#cell=13; lines=33-38
Input: Prompt, bản tóm tắt chữ của copy_step, đặc tả chữ: forward -> [B,T,V] log-prob; không log_softmax lại distribution copy; compute_loss lấy target dịch một token, bỏ PAD; LABEL_SMOOTHING=0. Không gửi mã nguồn.
Merge: Lưu riêng phần forward để ghép vào bản làm việc theo cell 13, dòng 33–38 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: forward -> [B,T,V] log-prob; không log_softmax lại distribution copy; compute_loss lấy target dịch một token, bỏ PAD; LABEL_SMOOTHING=0. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết forward(source,lengths,target) gọi step theo token target và trả chuỗi log-prob. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Khởi tạo copy beam
ID: beam_init
Depends: copy_step
Kind: code
Target: baseline.ipynb#cell=13; lines=40-49
Input: Prompt, bản tóm tắt chữ của copy_step, đặc tả chữ: generate(source,lengths,max_tokens,beam_size=3,length_penalty=0.6); seq BOS[B,K,1], scores chỉ beam đầu=0, done bool, lens; cần reindex hidden sau mỗi top-k. Không gửi mã nguồn.
Merge: Lưu riêng phần beam_init để ghép vào bản làm việc theo cell 13, dòng 40–49 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: generate(source,lengths,max_tokens,beam_size=3,length_penalty=0.6); seq BOS[B,K,1], scores chỉ beam đầu=0, done bool, lens; cần reindex hidden sau mỗi top-k. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần đầu generate, mở rộng memory, projected, mask, source và hidden cho beam search. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Chọn beam và trạng thái
ID: beam_loop
Depends: beam_init
Kind: code
Target: baseline.ipynb#cell=13; lines=50-63
Input: Prompt, bản tóm tắt chữ của beam_init, đặc tả chữ: Dùng step trả logp,next_hidden; gather seq/done/lens/next_hidden theo parent; EOS done có logp=0; dừng khi mọi done; trả phần vòng lặp để ghép. Không gửi mã nguồn.
Merge: Lưu riêng phần beam_loop để ghép vào bản làm việc theo cell 13, dòng 50–63 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Dùng step trả logp,next_hidden; gather seq/done/lens/next_hidden theo parent; EOS done có logp=0; dừng khi mọi done; trả phần vòng lặp để ghép. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng top-k beam, gather hidden theo parent, chặn PAD/BOS và giữ nguyên beam đã EOS. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Hoàn tất generate
ID: beam_finish
Depends: beam_loop
Kind: code
Target: baseline.ipynb#cell=13; lines=63-63
Input: Prompt, bản tóm tắt chữ của beam_loop, đặc tả chữ: scores/(((5+lens)/6)**0.6); output long[B,L]; MAX_DECODE_TOKENS=160; build_model(cfg)->TextGenerator; không đổi thứ tự batch. Không gửi mã nguồn.
Merge: Lưu riêng phần beam_finish để ghép vào bản làm việc theo cell 13, dòng 63–63 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: scores/(((5+lens)/6)**0.6); output long[B,L]; MAX_DECODE_TOKENS=160; build_model(cfg)->TextGenerator; không đổi thứ tự batch. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Hoàn tất generate bằng length penalty và chọn beam tốt nhất, bỏ BOS đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Factory và cấu hình model
ID: factory
Depends: tokenizer, batch, constructor, encode, copy_step, forward, beam_init, beam_loop, beam_finish
Kind: code
Target: baseline.ipynb#cell=14; lines=1-1
Input: Prompt, bản tóm tắt chữ của tokenizer, batch, constructor, encode, copy_step, forward, beam_init, beam_loop, beam_finish, đặc tả chữ: Factory TextGenerator theo chữ ký constructor đã tóm tắt; cấu hình {"vocab_size":sp.get_piece_size(),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Không gửi mã nguồn.
Merge: Lưu riêng phần factory để ghép vào bản làm việc theo cell 14, dòng 1–1 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Factory TextGenerator theo chữ ký constructor đã tóm tắt; cấu hình {"vocab_size":sp.get_piece_size(),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Nối Dataset và DataLoader
ID: loaders
Depends: splits, config, tokenizer, batch, constructor, encode, copy_step, forward, beam_init, beam_loop, beam_finish, factory
Kind: code
Target: baseline.ipynb#cell=11; lines=35-41
Input: Prompt, bản tóm tắt chữ của splits, config, tokenizer, batch, constructor, encode, copy_step, forward, beam_init, beam_loop, beam_finish, factory, đặc tả chữ: source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Không gửi mã nguồn.
Merge: Lưu riêng phần loaders để ghép vào bản làm việc theo cell 11, dòng 35–41 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết Dataset/DataLoader còn thiếu theo hợp đồng batch, chỉ shuffle train và giữ IDs/order khi đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Helper decode_tokens
ID: helper_decode_tokens
Depends: loaders
Kind: code
Target: baseline.ipynb#cell=17; lines=1-7
Input: Prompt, bản tóm tắt chữ của loaders, đặc tả chữ: decode_tokens nhận (ids); Giải mã BPE, bỏ PAD/BOS và dừng EOS; không sửa nội dung dự đoán theo gold. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_decode_tokens để ghép vào bản làm việc theo cell 17, dòng 1–7 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: decode_tokens nhận (ids); Giải mã BPE, bỏ PAD/BOS và dừng EOS; không sửa nội dung dự đoán theo gold. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng decode_tokens theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Helper errant_annotator
ID: helper_errant_annotator
Depends: helper_decode_tokens
Kind: code
Target: baseline.ipynb#cell=16; lines=8-32
Input: Prompt, bản tóm tắt chữ của helper_decode_tokens, đặc tả chữ: errant_annotator nhận (); Nạp errant3.0.2 và spaCy en_core_web_sm3.8.0 đã cài, chỉ evaluator; thiếu dependency báo lỗi rõ, không tự thay model. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_errant_annotator để ghép vào bản làm việc theo cell 16, dòng 8–32 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: errant_annotator nhận (); Nạp errant3.0.2 và spaCy en_core_web_sm3.8.0 đã cài, chỉ evaluator; thiếu dependency báo lỗi rõ, không tự thay model. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng errant_annotator theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Helper edits_as_rows
ID: helper_edits_as_rows
Depends: helper_errant_annotator
Kind: code
Target: baseline.ipynb#cell=16; lines=34-39
Input: Prompt, bản tóm tắt chữ của helper_errant_annotator, đặc tả chữ: edits_as_rows nhận (annotator, original, corrected, coder); Dùng annotator.parse/annotate lev=False,merging rules, trả edit offsets/type/correction/coder; noop khi không edits. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_edits_as_rows để ghép vào bản làm việc theo cell 16, dòng 34–39 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: edits_as_rows nhận (annotator, original, corrected, coder); Dùng annotator.parse/annotate lev=False,merging rules, trả edit offsets/type/correction/coder; noop khi không edits. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng edits_as_rows theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Helper errant_corpus
ID: helper_errant_corpus
Depends: helper_edits_as_rows
Kind: code
Target: baseline.ipynb#cell=16; lines=41-60
Input: Prompt, bản tóm tắt chữ của helper_edits_as_rows, đặc tả chữ: errant_corpus nhận (records, predictions); So edits theo API compare_m2 chính thức, beta0.5, nhiều references/coders; cộng TP/FP/FN corpus rồi F0.5, không average sentence F0.5. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_errant_corpus để ghép vào bản làm việc theo cell 16, dòng 41–60 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: errant_corpus nhận (records, predictions); So edits theo API compare_m2 chính thức, beta0.5, nhiều references/coders; cộng TP/FP/FN corpus rồi F0.5, không average sentence F0.5. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng errant_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Helper gleu_corpus
ID: helper_gleu_corpus
Depends: helper_errant_corpus
Kind: code
Target: baseline.ipynb#cell=16; lines=62-83
Input: Prompt, bản tóm tắt chữ của helper_errant_corpus, đặc tả chữ: gleu_corpus nhận (records, predictions); GLEU2016 source-aware ngram1..4, trừ giữ sai từ source, bootstrap reference theo seed42 và500iterations; không nltk corpus_gleu. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_gleu_corpus để ghép vào bản làm việc theo cell 16, dòng 62–83 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: gleu_corpus nhận (records, predictions); GLEU2016 source-aware ngram1..4, trừ giữ sai từ source, bootstrap reference theo seed42 và500iterations; không nltk corpus_gleu. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng gleu_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Helper score_corpus
ID: helper_score_corpus
Depends: helper_gleu_corpus
Kind: code
Target: baseline.ipynb#cell=16; lines=85-86
Input: Prompt, bản tóm tắt chữ của helper_gleu_corpus, đặc tả chữ: score_corpus nhận (records, predictions); Gộp errant_corpus và gleu_corpus; giữ keys errant_f05,gleu và provenance evaluator. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_score_corpus để ghép vào bản làm việc theo cell 16, dòng 85–86 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: score_corpus nhận (records, predictions); Gộp errant_corpus và gleu_corpus; giữ keys errant_f05,gleu và provenance evaluator. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng score_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 21: Metric chính thức và evaluation
ID: metrics
Depends: helper_score_corpus
Kind: code
Target: baseline.ipynb#cell=17; lines=17-26
Input: Prompt, bản tóm tắt chữ của helper_score_corpus, đặc tả chữ: ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Không gửi mã nguồn.
Merge: Lưu riêng phần metrics để ghép vào bản làm việc theo cell 17, dòng 17–26 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 22: Loss nối model và batch
ID: loss_wrapper
Depends: metrics
Kind: code
Target: baseline.ipynb#cell=17; lines=8-14
Input: Prompt, bản tóm tắt chữ của metrics, đặc tả chữ: AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Dùng các API forward/loss đã dựng, không dùng gold trong đường suy luận. Không gửi mã nguồn.
Merge: Lưu riêng phần loss_wrapper để ghép vào bản làm việc theo cell 17, dòng 8–14 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Dùng các API forward/loss đã dựng, không dùng gold trong đường suy luận. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết loss scalar nối model và batch theo hợp đồng, bỏ padding và giữ gradient; kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 23: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: loss_wrapper
Kind: code
Target: baseline.ipynb#cell=20; lines=1-19
Input: Prompt, bản tóm tắt chữ của loss_wrapper, đặc tả chữ: AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Không gửi mã nguồn.
Merge: Lưu riêng phần training để ghép vào bản làm việc theo cell 20, dòng 1–19 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.001,betas=(0.9,0.98),eps=1e-9,weight_decay=0.0001; warmup=0; epochs=40,batch=32,eval_batch=16,patience=8,clip=1; chọn errant_f05 validation tăng. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 24: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Target: baseline.ipynb#cell=22; lines=1-17
Input: Prompt, bản tóm tắt chữ của training, đặc tả chữ: ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Không gửi mã nguồn.
Merge: Lưu riêng phần reload để ghép vào bản làm việc theo cell 22, dòng 1–17 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 25: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Target: baseline.ipynb#cell=24; lines=2-9
Input: Prompt, bản tóm tắt chữ của reload, đặc tả chữ: source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Hàm rewrite_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Không gửi mã nguồn.
Merge: Lưu riêng phần inference để ghép vào bản làm việc theo cell 24, dòng 2–9 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. ERRANT 3.0.2 corpus F0.5 primary (thang100) và GLEU2016 source-aware whitespace,500iterations seed42, không phải nltk corpus_gleu; đủ mọi refs. spaCy en_core_web_sm3.8.0 chỉ evaluator, không dùng tạo prediction. Hàm rewrite_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 26: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 27: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Target: baseline.ipynb#cell=22; lines=1-17
Input: Prompt, bản tóm tắt chữ của summary, đặc tả chữ: source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Không gửi mã nguồn.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: source long[B,S], target long[B,T], lengths CPU long[B], ids,records; PAD/UNK/BOS/EOS=0/1/2/3; cùng vocabulary cho source và target. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

