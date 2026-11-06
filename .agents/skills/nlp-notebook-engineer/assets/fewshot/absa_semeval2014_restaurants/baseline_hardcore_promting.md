# Prompt hardcore: SemEval restaurants: character BIO, word features, BiLSTM-CRF và polarity

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
Target: baseline.ipynb#cell=7; lines=1-6
Input: Prompt, đặc tả chữ: tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Loại train trùng câu test; chia20% nhóm câu train seed42; polarity test có thể thiếu, không bịa điểm. span là chuỗi aspect, ordinal là lần xuất hiện zero-based khớp phân biệt hoa/thường; dùng end exclusive; reject gold span overlap. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Không gửi mã nguồn.
Merge: Lưu riêng phần data để ghép vào bản làm việc theo cell 7, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Loại train trùng câu test; chia20% nhóm câu train seed42; polarity test có thể thiếu, không bịa điểm. span là chuỗi aspect, ordinal là lần xuất hiện zero-based khớp phân biệt hoa/thường; dùng end exclusive; reject gold span overlap. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Chuẩn hóa schema và chia nhóm
ID: splits
Depends: data
Kind: code
Target: baseline.ipynb#cell=7; lines=1-6
Input: Prompt, bản tóm tắt chữ của data, đặc tả chữ: tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Loại train trùng câu test; chia20% nhóm câu train seed42; polarity test có thể thiếu, không bịa điểm. span là chuỗi aspect, ordinal là lần xuất hiện zero-based khớp phân biệt hoa/thường; dùng end exclusive; reject gold span overlap. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Không gửi mã nguồn.
Merge: Lưu riêng phần splits để ghép vào bản làm việc theo cell 7, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Loại train trùng câu test; chia20% nhóm câu train seed42; polarity test có thể thiếu, không bịa điểm. span là chuỗi aspect, ordinal là lần xuất hiện zero-based khớp phân biệt hoa/thường; dùng end exclusive; reject gold span overlap. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần chuẩn hóa record và split đúng đặc tả, kiểm IDs và ngăn rò rỉ; không fit đặc trưng ở bước này. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Imports và cấu hình cố định
ID: config
Depends: splits
Kind: code
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt, bản tóm tắt chữ của splits, đặc tả chữ: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Không gửi mã nguồn.
Merge: Lưu riêng phần config để ghép vào bản làm việc theo cell 5, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Word features theo offsets
ID: vocabulary
Depends: config
Kind: code
Target: baseline.ipynb#cell=9; lines=1-15
Input: Prompt, bản tóm tắt chữ của config, đặc tả chữ: PAD0 UNK1; word regex words+punct; WORD_MIN_FREQ1; char_vocab train-only; không normalize/cắt câu làm lệch span. Không gửi mã nguồn.
Merge: Lưu riêng phần vocabulary để ghép vào bản làm việc theo cell 9, dòng 1–15 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: PAD0 UNK1; word regex words+punct; WORD_MIN_FREQ1; char_vocab train-only; không normalize/cắt câu làm lệch span. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung words_with_offsets và word_vocab train-only, giữ trục thời gian ký tự nguyên văn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Căn word IDs lên ký tự
ID: encode
Depends: vocabulary
Kind: code
Target: baseline.ipynb#cell=9; lines=17-31
Input: Prompt, bản tóm tắt chữ của vocabulary, đặc tả chữ: encode_record(record)->char_ids,word_ids,tags,record cùng chiều dài text; space word_id UNK, B đầu span/I phần sau; không nhãn chồng lấn. Không gửi mã nguồn.
Merge: Lưu riêng phần encode để ghép vào bản làm việc theo cell 9, dòng 17–31 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: encode_record(record)->char_ids,word_ids,tags,record cùng chiều dài text; space word_id UNK, B đầu span/I phần sau; không nhãn chồng lấn. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Nâng encode_record để word ID phủ đúng khoảng ký tự, giữ BIO và spans gốc. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Padding và span targets
ID: batch
Depends: encode
Kind: code
Target: baseline.ipynb#cell=9; lines=45-62
Input: Prompt, bản tóm tắt chữ của encode, đặc tả chữ: lengths CPU, mask bool; gold_spans[M,4] chỉ polarity có nhãn; records giữ sid,offsets; AspectDataset/make_loader dùng encode_record mới. Không gửi mã nguồn.
Merge: Lưu riêng phần batch để ghép vào bản làm việc theo cell 9, dòng 45–62 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: lengths CPU, mask bool; gold_spans[M,4] chỉ polarity có nhãn; records giữ sid,offsets; AspectDataset/make_loader dùng encode_record mới. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Cập nhật collate_batch nhận word IDs, masks và gold polarity spans hợp lệ. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: CRF O/B/I
ID: crf_init
Depends: batch
Kind: code
Target: baseline.ipynb#cell=11; lines=3-11
Input: Prompt, bản tóm tắt chữ của batch, đặc tả chữ: LinearChainCRF(); transitions[3,3],start/end[3]; constrained()->transition,start illegal=-10000; buffer legal và legal_start. Không gửi mã nguồn.
Merge: Lưu riêng phần crf_init để ghép vào bản làm việc theo cell 11, dòng 3–11 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: LinearChainCRF(); transitions[3,3],start/end[3]; constrained()->transition,start illegal=-10000; buffer legal và legal_start. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Dựng LinearChainCRF O/B/I với start/end và cấm I đầu câu, cấm O sang I. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Masked CRF NLL
ID: crf_loss
Depends: crf_init
Kind: code
Target: baseline.ipynb#cell=11; lines=17-32
Input: Prompt, bản tóm tắt chữ của crf_init, đặc tả chữ: neg_log_likelihood(emissions[B,T,3],tags[B,T],mask[B,T])->scalar; cộng end theo last tag, chuẩn hóa từng chuỗi theo mask.sum. Không gửi mã nguồn.
Merge: Lưu riêng phần crf_loss để ghép vào bản làm việc theo cell 11, dòng 17–32 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: neg_log_likelihood(emissions[B,T,3],tags[B,T],mask[B,T])->scalar; cộng end theo last tag, chuẩn hóa từng chuỗi theo mask.sum. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết CRF neg_log_likelihood bằng gold path và logsumexp partition, chỉ tính ký tự thật. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: BIO Viterbi
ID: viterbi
Depends: crf_loss
Kind: code
Target: baseline.ipynb#cell=11; lines=35-53
Input: Prompt, bản tóm tắt chữ của crf_loss, đặc tả chữ: decode(emissions,mask)->list[list[int]]; mỗi path dài mask.sum; không dùng gold tags hoặc random labels. Không gửi mã nguồn.
Merge: Lưu riêng phần viterbi để ghép vào bản làm việc theo cell 11, dòng 35–53 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: decode(emissions,mask)->list[list[int]]; mỗi path dài mask.sum; không dùng gold tags hoặc random labels. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết decode Viterbi có constraints và mask, backtrack theo chiều dài thật. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: BiLSTM và polarity head
ID: constructor
Depends: viterbi
Kind: code
Target: baseline.ipynb#cell=12; lines=2-15
Input: Prompt, bản tóm tắt chữ của viterbi, đặc tả chữ: AspectModel(cfg,n_chars,n_words); cfg char_dim48,word_dim128,hidden_dim128,num_layers2,dropout0.3; encoded dim256; polarity head input4*256->128->4. Không gửi mã nguồn.
Merge: Lưu riêng phần constructor để ghép vào bản làm việc theo cell 12, dòng 2–15 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AspectModel(cfg,n_chars,n_words); cfg char_dim48,word_dim128,hidden_dim128,num_layers2,dropout0.3; encoded dim256; polarity head input4*256->128->4. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay constructor AspectModel bằng char/word embedding, BiLSTM hai lớp, tag head, CRF và polarity head. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Character sequence encoder
ID: forward
Depends: constructor
Kind: code
Target: baseline.ipynb#cell=12; lines=17-26
Input: Prompt, bản tóm tắt chữ của constructor, đặc tả chữ: forward(batch)->hidden[B,T,256],emissions[B,T,3]; lengths CPU,total_length=T; extraction_loss(emissions,batch) và decode(emissions,mask) gọi CRF. Không gửi mã nguồn.
Merge: Lưu riêng phần forward để ghép vào bản làm việc theo cell 12, dòng 17–26 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: forward(batch)->hidden[B,T,256],emissions[B,T,3]; lengths CPU,total_length=T; extraction_loss(emissions,batch) và decode(emissions,mask) gọi CRF. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết forward ghép char/word embedding theo ký tự, packed BiLSTM và BIO emissions. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Phân loại span
ID: polarity
Depends: forward
Kind: code
Target: baseline.ipynb#cell=12; lines=34-43
Input: Prompt, bản tóm tắt chữ của forward, đặc tả chữ: classify_spans(hidden,spans,mask)->logits[M,4]; mean span dùng prefix sum, end exclusive; M=0 trả tensor[0,4]; loss polarity chỉ dùng nhãn hiện hữu. Không gửi mã nguồn.
Merge: Lưu riêng phần polarity để ghép vào bản làm việc theo cell 12, dòng 34–43 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: classify_spans(hidden,spans,mask)->logits[M,4]; mean span dùng prefix sum, end exclusive; M=0 trả tensor[0,4]; loss polarity chỉ dùng nhãn hiện hữu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Nâng classify_spans dùng hai biên, mean span và mean câu có mask cho polarity. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Nối API CRF vào AspectModel
ID: crf_bridge
Depends: polarity
Kind: code
Target: baseline.ipynb#cell=12; lines=28-29
Input: Prompt, bản tóm tắt chữ của polarity, đặc tả chữ: extraction_loss(emissions,batch)->CRF.neg_log_likelihood với batch tags/mask; decode(emissions,mask)->CRF.decode; không cross-entropy BIO cũ, không gold khi decode. Không gửi mã nguồn.
Merge: Lưu riêng phần crf_bridge để ghép vào bản làm việc theo cell 12, dòng 28–29 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: extraction_loss(emissions,batch)->CRF.neg_log_likelihood với batch tags/mask; decode(emissions,mask)->CRF.decode; không cross-entropy BIO cũ, không gold khi decode. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết extraction_loss và decode của AspectModel để gọi CRF đã dựng, giữ chữ ký caller hiện tại. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Factory và cấu hình model
ID: factory
Depends: vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge
Kind: code
Target: baseline.ipynb#cell=13; lines=1-5
Input: Prompt, bản tóm tắt chữ của vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge, đặc tả chữ: Factory AspectModel theo chữ ký constructor đã tóm tắt; cấu hình  {"char_dim": CHAR_EMBED_DIM, "word_dim": WORD_EMBED_DIM,;                 "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}; model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE); ; print(f"Số tham số học từ đầu: {sum(p.numel() for p in model.parameters()):,}"); khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Không gửi mã nguồn.
Merge: Lưu riêng phần factory để ghép vào bản làm việc theo cell 13, dòng 1–5 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Factory AspectModel theo chữ ký constructor đã tóm tắt; cấu hình  {"char_dim": CHAR_EMBED_DIM, "word_dim": WORD_EMBED_DIM,;                 "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}; model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE); ; print(f"Số tham số học từ đầu: {sum(p.numel() for p in model.parameters()):,}"); khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Nối Dataset và DataLoader
ID: loaders
Depends: splits, config, vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge, factory
Kind: code
Target: baseline.ipynb#cell=9; lines=65-68
Input: Prompt, bản tóm tắt chữ của splits, config, vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge, factory, đặc tả chữ: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Không gửi mã nguồn.
Merge: Lưu riêng phần loaders để ghép vào bản làm việc theo cell 9, dòng 65–68 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết Dataset/DataLoader còn thiếu theo hợp đồng batch, chỉ shuffle train và giữ IDs/order khi đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Helper move_batch
ID: helper_move_batch
Depends: loaders
Kind: code
Target: baseline.ipynb#cell=15; lines=1-3
Input: Prompt, bản tóm tắt chữ của loaders, đặc tả chữ: move_batch nhận (batch); Chuyển tensor sang device trừ lengths giữ CPU, giữ metadata records nguyên dạng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_move_batch để ghép vào bản làm việc theo cell 15, dòng 1–3 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: move_batch nhận (batch); Chuyển tensor sang device trừ lengths giữ CPU, giữ metadata records nguyên dạng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng move_batch theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Helper bio_to_spans
ID: helper_bio_to_spans
Depends: helper_move_batch
Kind: code
Target: baseline.ipynb#cell=15; lines=6-15
Input: Prompt, bản tóm tắt chữ của helper_move_batch, đặc tả chữ: bio_to_spans nhận (tags); Đọc O/B/I thành character span end exclusive; I sau O xem B, không trim khoảng trắng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_bio_to_spans để ghép vào bản làm việc theo cell 15, dòng 6–15 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: bio_to_spans nhận (tags); Đọc O/B/I thành character span end exclusive; I sau O xem B, không trim khoảng trắng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng bio_to_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Helper set_prf
ID: helper_set_prf
Depends: helper_bio_to_spans
Kind: code
Target: baseline.ipynb#cell=15; lines=18-24
Input: Prompt, bản tóm tắt chữ của helper_bio_to_spans, đặc tả chữ: set_prf nhận (gold, predicted); TP từ giao tập gold/predicted, precision/recall/F1 từ tổng set sizes, empty chia0 trả0. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_set_prf để ghép vào bản làm việc theo cell 15, dòng 18–24 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: set_prf nhận (gold, predicted); TP từ giao tập gold/predicted, precision/recall/F1 từ tổng set sizes, empty chia0 trả0. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng set_prf theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Helper classification_metrics
ID: helper_classification_metrics
Depends: helper_set_prf
Kind: code
Target: baseline.ipynb#cell=15; lines=27-45
Input: Prompt, bản tóm tắt chữ của helper_set_prf, đặc tả chữ: classification_metrics nhận (gold, predicted); Tính confusion matrix bốn polarity, macro-F1 gồm mọi lớp; thiếu gold trả None và count0, không bịa nhãn. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_classification_metrics để ghép vào bản làm việc theo cell 15, dòng 27–45 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: classification_metrics nhận (gold, predicted); Tính confusion matrix bốn polarity, macro-F1 gồm mọi lớp; thiếu gold trả None và count0, không bịa nhãn. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng classification_metrics theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Metric chính thức và evaluation
ID: metrics
Depends: helper_classification_metrics
Kind: code
Target: baseline.ipynb#cell=15; lines=49-87
Input: Prompt, bản tóm tắt chữ của helper_classification_metrics, đặc tả chữ: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad. Không gửi mã nguồn.
Merge: Lưu riêng phần metrics để ghép vào bản làm việc theo cell 15, dòng 49–87 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 21: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Target: baseline.ipynb#cell=17; lines=1-20
Input: Prompt, bản tóm tắt chữ của metrics, đặc tả chữ: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Checkpoint ABSA lưu model_state,char_vocab,word_vocab,labels,tags,epoch; không resume state. Caller loss/evaluate theo tóm tắt. Không gửi mã nguồn.
Merge: Lưu riêng phần training để ghép vào bản làm việc theo cell 17, dòng 1–20 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Checkpoint ABSA lưu model_state,char_vocab,word_vocab,labels,tags,epoch; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 22: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Target: baseline.ipynb#cell=19; lines=1-25
Input: Prompt, bản tóm tắt chữ của training, đặc tả chữ: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Không gửi mã nguồn.
Merge: Lưu riêng phần reload để ghép vào bản làm việc theo cell 19, dòng 1–25 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 23: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Target: baseline.ipynb#cell=21; lines=2-14
Input: Prompt, bản tóm tắt chữ của reload, đặc tả chữ: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Hàm predict_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Không gửi mã nguồn.
Merge: Lưu riêng phần inference để ghép vào bản làm việc theo cell 21, dòng 2–14 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Hàm predict_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 24: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 25: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Target: baseline.ipynb#cell=19; lines=1-25
Input: Prompt, bản tóm tắt chữ của summary, đặc tả chữ: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Không gửi mã nguồn.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

