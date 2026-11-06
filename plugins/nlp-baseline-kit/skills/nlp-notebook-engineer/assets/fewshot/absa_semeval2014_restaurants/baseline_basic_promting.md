# Prompt basic: SemEval restaurants: character BIO, word features, BiLSTM-CRF và polarity

Variant: basic
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
Source: baseline_basic.ipynb#cell=7; lines=1-6
Target: baseline.ipynb#cell=7; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Hợp đồng cần giữ: tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Loại train trùng câu test; chia20% nhóm câu train seed42; polarity test có thể thiếu, không bịa điểm. span là chuỗi aspect, ordinal là lần xuất hiện zero-based khớp phân biệt hoa/thường; dùng end exclusive; reject gold span overlap. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 7, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 7, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: tomaarsen/setfit-absa-semeval-restaurants revision8885372fe73256f96bb60f65b550538ac5c26047; train/test có text,span,ordinal,label. Gom nguyên câu thành sid,text,spans[(start,end,polarity)], kiểm offsets/overlap. Loại train trùng câu test; chia20% nhóm câu train seed42; polarity test có thể thiếu, không bịa điểm. span là chuỗi aspect, ordinal là lần xuất hiện zero-based khớp phân biệt hoa/thường; dùng end exclusive; reject gold span overlap. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Kiểm tra đoạn load dữ liệu theo đặc tả, chỉ trả assertions còn thiếu và giữ split hiện tại. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Imports và cấu hình cố định
ID: config
Depends: data
Kind: code
Source: baseline_basic.ipynb#cell=5; lines=1-6
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của data. Hợp đồng cần giữ: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 5, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 5, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Word features theo offsets
ID: vocabulary
Depends: config
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=1-6
Target: baseline.ipynb#cell=9; lines=1-15
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của config. Hợp đồng cần giữ: PAD0 UNK1; word regex words+punct; WORD_MIN_FREQ1; char_vocab train-only; không normalize/cắt câu làm lệch span.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 1–8 của basic gốc. Đối chiếu phần đích cell 9, dòng 1–15; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: PAD0 UNK1; word regex words+punct; WORD_MIN_FREQ1; char_vocab train-only; không normalize/cắt câu làm lệch span. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung words_with_offsets và word_vocab train-only, giữ trục thời gian ký tự nguyên văn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Căn word IDs lên ký tự
ID: encode
Depends: vocabulary
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=13-18
Target: baseline.ipynb#cell=9; lines=17-31
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của vocabulary. Hợp đồng cần giữ: encode_record(record)->char_ids,word_ids,tags,record cùng chiều dài text; space word_id UNK, B đầu span/I phần sau; không nhãn chồng lấn.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 13–22 của basic gốc. Đối chiếu phần đích cell 9, dòng 17–31; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: encode_record(record)->char_ids,word_ids,tags,record cùng chiều dài text; space word_id UNK, B đầu span/I phần sau; không nhãn chồng lấn. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Nâng encode_record để word ID phủ đúng khoảng ký tự, giữ BIO và spans gốc. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Padding và span targets
ID: batch
Depends: encode
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=36-41
Target: baseline.ipynb#cell=9; lines=45-62
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của encode. Hợp đồng cần giữ: lengths CPU, mask bool; gold_spans[M,4] chỉ polarity có nhãn; records giữ sid,offsets; AspectDataset/make_loader dùng encode_record mới.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 36–53 của basic gốc. Đối chiếu phần đích cell 9, dòng 45–62; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: lengths CPU, mask bool; gold_spans[M,4] chỉ polarity có nhãn; records giữ sid,offsets; AspectDataset/make_loader dùng encode_record mới. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Cập nhật collate_batch nhận word IDs, masks và gold polarity spans hợp lệ. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: CRF O/B/I
ID: crf_init
Depends: batch
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=2-7
Target: baseline.ipynb#cell=11; lines=3-11
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của batch. Hợp đồng cần giữ: LinearChainCRF(); transitions[3,3],start/end[3]; constrained()->transition,start illegal=-10000; buffer legal và legal_start.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 2–11 của basic gốc. Đối chiếu phần đích cell 11, dòng 3–11; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: LinearChainCRF(); transitions[3,3],start/end[3]; constrained()->transition,start illegal=-10000; buffer legal và legal_start. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Dựng LinearChainCRF O/B/I với start/end và cấm I đầu câu, cấm O sang I. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: Masked CRF NLL
ID: crf_loss
Depends: crf_init
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=23-25
Target: baseline.ipynb#cell=11; lines=17-32
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của crf_init. Hợp đồng cần giữ: neg_log_likelihood(emissions[B,T,3],tags[B,T],mask[B,T])->scalar; cộng end theo last tag, chuẩn hóa từng chuỗi theo mask.sum.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 23–25 của basic gốc. Đối chiếu phần đích cell 11, dòng 17–32; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: neg_log_likelihood(emissions[B,T,3],tags[B,T],mask[B,T])->scalar; cộng end theo last tag, chuẩn hóa từng chuỗi theo mask.sum. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết CRF neg_log_likelihood bằng gold path và logsumexp partition, chỉ tính ký tự thật. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: BIO Viterbi
ID: viterbi
Depends: crf_loss
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=27-29
Target: baseline.ipynb#cell=11; lines=35-53
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của crf_loss. Hợp đồng cần giữ: decode(emissions,mask)->list[list[int]]; mỗi path dài mask.sum; không dùng gold tags hoặc random labels.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 27–29 của basic gốc. Đối chiếu phần đích cell 11, dòng 35–53; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: decode(emissions,mask)->list[list[int]]; mỗi path dài mask.sum; không dùng gold tags hoặc random labels. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết decode Viterbi có constraints và mask, backtrack theo chiều dài thật. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: BiLSTM và polarity head
ID: constructor
Depends: viterbi
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=2-7
Target: baseline.ipynb#cell=12; lines=2-15
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của viterbi. Hợp đồng cần giữ: AspectModel(cfg,n_chars,n_words); cfg char_dim48,word_dim128,hidden_dim128,num_layers2,dropout0.3; encoded dim256; polarity head input4*256->128->4.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 2–11 của basic gốc. Đối chiếu phần đích cell 12, dòng 2–15; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AspectModel(cfg,n_chars,n_words); cfg char_dim48,word_dim128,hidden_dim128,num_layers2,dropout0.3; encoded dim256; polarity head input4*256->128->4. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay constructor AspectModel bằng char/word embedding, BiLSTM hai lớp, tag head, CRF và polarity head. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Character sequence encoder
ID: forward
Depends: constructor
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=13-18
Target: baseline.ipynb#cell=12; lines=17-26
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của constructor. Hợp đồng cần giữ: forward(batch)->hidden[B,T,256],emissions[B,T,3]; lengths CPU,total_length=T; extraction_loss(emissions,batch) và decode(emissions,mask) gọi CRF.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 13–21 của basic gốc. Đối chiếu phần đích cell 12, dòng 17–26; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: forward(batch)->hidden[B,T,256],emissions[B,T,3]; lengths CPU,total_length=T; extraction_loss(emissions,batch) và decode(emissions,mask) gọi CRF. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết forward ghép char/word embedding theo ký tự, packed BiLSTM và BIO emissions. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Phân loại span
ID: polarity
Depends: forward
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=31-36
Target: baseline.ipynb#cell=12; lines=34-43
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của forward. Hợp đồng cần giữ: classify_spans(hidden,spans,mask)->logits[M,4]; mean span dùng prefix sum, end exclusive; M=0 trả tensor[0,4]; loss polarity chỉ dùng nhãn hiện hữu.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 31–38 của basic gốc. Đối chiếu phần đích cell 12, dòng 34–43; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: classify_spans(hidden,spans,mask)->logits[M,4]; mean span dùng prefix sum, end exclusive; M=0 trả tensor[0,4]; loss polarity chỉ dùng nhãn hiện hữu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Nâng classify_spans dùng hai biên, mean span và mean câu có mask cho polarity. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Nối API CRF vào AspectModel
ID: crf_bridge
Depends: polarity
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=23-25
Target: baseline.ipynb#cell=12; lines=28-29
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của polarity. Hợp đồng cần giữ: extraction_loss(emissions,batch)->CRF.neg_log_likelihood với batch tags/mask; decode(emissions,mask)->CRF.decode; không cross-entropy BIO cũ, không gold khi decode.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 23–25 của basic gốc. Đối chiếu phần đích cell 12, dòng 28–29; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: extraction_loss(emissions,batch)->CRF.neg_log_likelihood với batch tags/mask; decode(emissions,mask)->CRF.decode; không cross-entropy BIO cũ, không gold khi decode. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết extraction_loss và decode của AspectModel để gọi CRF đã dựng, giữ chữ ký caller hiện tại. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Factory và cấu hình model
ID: factory
Depends: vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge
Kind: code
Source: baseline_basic.ipynb#cell=12; lines=1-6
Target: baseline.ipynb#cell=13; lines=1-5
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge. Hợp đồng cần giữ: Factory AspectModel theo chữ ký constructor đã tóm tắt; cấu hình  {"char_dim": CHAR_EMBED_DIM, "word_dim": WORD_EMBED_DIM,;                 "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}; model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE); ; print(f"Số tham số học từ đầu: {sum(p.numel() for p in model.parameters()):,}"); khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 12, dòng 1–7 của basic gốc. Đối chiếu phần đích cell 13, dòng 1–5; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Factory AspectModel theo chữ ký constructor đã tóm tắt; cấu hình  {"char_dim": CHAR_EMBED_DIM, "word_dim": WORD_EMBED_DIM,;                 "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT}; model = AspectModel(MODEL_CONFIG, len(char_vocab), len(word_vocab)).to(DEVICE); ; print(f"Số tham số học từ đầu: {sum(p.numel() for p in model.parameters()):,}"); khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Helper move_batch
ID: helper_move_batch
Depends: vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge, factory
Kind: code
Source: baseline_basic.ipynb#cell=14; lines=1-3
Target: baseline.ipynb#cell=15; lines=1-3
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của vocabulary, encode, batch, crf_init, crf_loss, viterbi, constructor, forward, polarity, crf_bridge, factory. Hợp đồng cần giữ: move_batch nhận (batch); Chuyển tensor sang device trừ lengths giữ CPU, giữ metadata records nguyên dạng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 14, dòng 1–3 của basic gốc. Đối chiếu phần đích cell 15, dòng 1–3; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: move_batch nhận (batch); Chuyển tensor sang device trừ lengths giữ CPU, giữ metadata records nguyên dạng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng move_batch theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Helper bio_to_spans
ID: helper_bio_to_spans
Depends: helper_move_batch
Kind: code
Source: baseline_basic.ipynb#cell=14; lines=6-11
Target: baseline.ipynb#cell=15; lines=6-15
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_move_batch. Hợp đồng cần giữ: bio_to_spans nhận (tags); Đọc O/B/I thành character span end exclusive; I sau O xem B, không trim khoảng trắng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 14, dòng 6–15 của basic gốc. Đối chiếu phần đích cell 15, dòng 6–15; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: bio_to_spans nhận (tags); Đọc O/B/I thành character span end exclusive; I sau O xem B, không trim khoảng trắng. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng bio_to_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Helper set_prf
ID: helper_set_prf
Depends: helper_bio_to_spans
Kind: code
Source: baseline_basic.ipynb#cell=14; lines=18-23
Target: baseline.ipynb#cell=15; lines=18-24
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_bio_to_spans. Hợp đồng cần giữ: set_prf nhận (gold, predicted); TP từ giao tập gold/predicted, precision/recall/F1 từ tổng set sizes, empty chia0 trả0. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 14, dòng 18–24 của basic gốc. Đối chiếu phần đích cell 15, dòng 18–24; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: set_prf nhận (gold, predicted); TP từ giao tập gold/predicted, precision/recall/F1 từ tổng set sizes, empty chia0 trả0. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng set_prf theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Helper classification_metrics
ID: helper_classification_metrics
Depends: helper_set_prf
Kind: code
Source: baseline_basic.ipynb#cell=14; lines=27-32
Target: baseline.ipynb#cell=15; lines=27-45
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_set_prf. Hợp đồng cần giữ: classification_metrics nhận (gold, predicted); Tính confusion matrix bốn polarity, macro-F1 gồm mọi lớp; thiếu gold trả None và count0, không bịa nhãn. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 14, dòng 27–45 của basic gốc. Đối chiếu phần đích cell 15, dòng 27–45; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: classification_metrics nhận (gold, predicted); Tính confusion matrix bốn polarity, macro-F1 gồm mọi lớp; thiếu gold trả None và count0, không bịa nhãn. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng classification_metrics theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Metric chính thức và evaluation
ID: metrics
Depends: helper_classification_metrics
Kind: code
Source: baseline_basic.ipynb#cell=14; lines=49-54
Target: baseline.ipynb#cell=15; lines=49-87
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_classification_metrics. Hợp đồng cần giữ: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 14, dòng 49–87 của basic gốc. Đối chiếu phần đích cell 15, dòng 49–87; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. evaluate(model,loader,include_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Source: baseline_basic.ipynb#cell=16; lines=1-6
Target: baseline.ipynb#cell=17; lines=1-20
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của metrics. Hợp đồng cần giữ: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Checkpoint ABSA lưu model_state,char_vocab,word_vocab,labels,tags,epoch; không resume state. Caller loss/evaluate theo tóm tắt.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 16, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 17, dòng 1–20; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.002,weight_decay0.0001; epochs40,batch16,patience8,clip1; CRF NLL + 0.25*polarity CE khi có nhãn; chọn metrics aspect_span.f1 validation tăng, scratch toàn bộ. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Checkpoint ABSA lưu model_state,char_vocab,word_vocab,labels,tags,epoch; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Source: baseline_basic.ipynb#cell=20; lines=2-7
Target: baseline.ipynb#cell=19; lines=1-25
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của training. Hợp đồng cần giữ: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 20, dòng 2–14 của basic gốc. Đối chiếu phần đích cell 19, dòng 1–25; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 21: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Source: baseline_basic.ipynb#cell=20; lines=2-7
Target: baseline.ipynb#cell=21; lines=2-14
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của reload. Hợp đồng cần giữ: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Hàm predict_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 20, dòng 2–14 của basic gốc. Đối chiếu phần đích cell 21, dòng 2–14; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Exact span F1 primary theo (sid,start,end), polarity macro-F1 trên gold spans có nhãn và joint F1 khi có gold; test thiếu polarity ghi null, không đánh giá bằng gold giả. Hàm predict_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 22: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 23: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Source: baseline_basic.ipynb#cell=20; lines=2-7
Target: baseline.ipynb#cell=19; lines=1-25
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của summary. Hợp đồng cần giữ: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: char_ids,word_ids,tags long[B,T] theo ký tự gốc; lengths CPU,mask bool[B,T],gold_spans long[M,4]=(batch,start,end,polarity),records; O/B/I=0/1/2; polarity negative,neutral,positive,conflict. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

