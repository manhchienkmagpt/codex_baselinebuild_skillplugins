# Prompt basic: SQuAD v1.1: word/char, attention và span decoding

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
Source: baseline_basic.ipynb#cell=8; lines=1-6
Target: baseline.ipynb#cell=8; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Hợp đồng cần giữ: SQuAD v1.1 TFDS squad/v1.1:3.0.0; schema id,title,context,question,answers[{answer_start,text}]. Context giữ nguyên để kiểm offsets; official validation là held-out, 10% train titles làm validation với seed=42; assert context/ID không trùng split. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 8, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 8, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: SQuAD v1.1 TFDS squad/v1.1:3.0.0; schema id,title,context,question,answers[{answer_start,text}]. Context giữ nguyên để kiểm offsets; official validation là held-out, 10% train titles làm validation với seed=42; assert context/ID không trùng split. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Kiểm tra đoạn load dữ liệu theo đặc tả, chỉ trả assertions còn thiếu và giữ split hiện tại. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Imports và cấu hình cố định
ID: config
Depends: data
Kind: code
Source: baseline_basic.ipynb#cell=5; lines=1-6
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của data. Hợp đồng cần giữ: AdamW lr=0.001,weight_decay=0.0001,warmup=0; epochs=20,batch=24,patience=5,clip=1; chọn exact_match validation tăng; vocabulary chỉ train. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 5, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 5, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.001,weight_decay=0.0001,warmup=0; epochs=20,batch=24,patience=5,clip=1; chọn exact_match validation tăng; vocabulary chỉ train. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Vocabulary word và char
ID: vocabulary
Depends: config
Kind: code
Source: baseline_basic.ipynb#cell=10; lines=11-16
Target: baseline.ipynb#cell=10; lines=1-34
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của config. Hợp đồng cần giữ: word_vocab tối đa40000, char_vocab PAD=0 UNK=1; regex words+punct; TOKENIZER_STATE chứa word_vocab,char_vocab; context_tokens(text) giữ start/end; không normalize context. Word vocab có CLS_ID=2; context_tokens trả words và offsets, question_words rút ngắn64 hoặc UNK nếu rỗng.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 10, dòng 11–18 của basic gốc. Đối chiếu phần đích cell 10, dòng 1–34; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: word_vocab tối đa40000, char_vocab PAD=0 UNK=1; regex words+punct; TOKENIZER_STATE chứa word_vocab,char_vocab; context_tokens(text) giữ start/end; không normalize context. Word vocab có CLS_ID=2; context_tokens trả words và offsets, question_words rút ngắn64 hoặc UNK nếu rỗng. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung character vocabulary train-only cho QA, giữ word vocabulary và offsets context gốc. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Mã hóa char và batch
ID: characters
Depends: vocabulary
Kind: code
Source: baseline_basic.ipynb#cell=10; lines=68-72
Target: baseline.ipynb#cell=10; lines=69-77
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của vocabulary. Hợp đồng cần giữ: encode_words(rows,width)->word IDs long[B,width], char IDs long[B,width,32]; cập nhật collate_batch thêm context_chars/question_chars, giữ targets[B,2], items/ids/masks/lengths. CLS_ID=2, PAD0 UNK1.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 10, dòng 68–72 của basic gốc. Đối chiếu phần đích cell 10, dòng 69–77; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: encode_words(rows,width)->word IDs long[B,width], char IDs long[B,width,32]; cập nhật collate_batch thêm context_chars/question_chars, giữ targets[B,2], items/ids/masks/lengths. CLS_ID=2, PAD0 UNK1. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung char IDs trong encode_words và cập nhật batch context/question để model dùng char-CNN. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Windows và gold alignment
ID: windows
Depends: characters
Kind: code
Source: baseline_basic.ipynb#cell=10; lines=44-49
Target: baseline.ipynb#cell=10; lines=45-67
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của characters. Hợp đồng cần giữ: QADataset(records,training=False); MAX_CONTEXT_TOKENS=384,WINDOW_STEP=256,MAX_QUESTION_TOKENS=64,MAX_ANSWER_TOKENS=64,NEGATIVE_WINDOW_WEIGHT=0.5; giữ đủ question IDs khi hợp nhất.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 10, dòng 44–66 của basic gốc. Đối chiếu phần đích cell 10, dòng 45–67; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: QADataset(records,training=False); MAX_CONTEXT_TOKENS=384,WINDOW_STEP=256,MAX_QUESTION_TOKENS=64,MAX_ANSWER_TOKENS=64,NEGATIVE_WINDOW_WEIGHT=0.5; giữ đủ question IDs khi hợp nhất. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ windows 384 bước 256, căn gold theo offsets và câu hỏi tối đa64; không mất đáp án ở biên. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Khung QA attention
ID: constructor
Depends: windows
Kind: code
Source: baseline_basic.ipynb#cell=12; lines=2-7
Target: baseline.ipynb#cell=12; lines=2-16
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của windows. Hợp đồng cần giữ: cfg n_words,n_chars,hidden_dim=128,dropout=0.2; word128,char16,charCNN32 kernel3; encoder input160, dim256; modeling input4*dim, end_encoder dim; heads input5*dim.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 12, dòng 2–10 của basic gốc. Đối chiếu phần đích cell 12, dòng 2–16; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: cfg n_words,n_chars,hidden_dim=128,dropout=0.2; word128,char16,charCNN32 kernel3; encoder input160, dim256; modeling input4*dim, end_encoder dim; heads input5*dim. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay constructor QAModel bằng word/char-CNN, BiLSTM và attention hai chiều với start/end heads. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: Char-CNN pooling
ID: char_features
Depends: constructor
Kind: code
Source: baseline_basic.ipynb#cell=12; lines=11-16
Target: baseline.ipynb#cell=12; lines=17-22
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của constructor. Hợp đồng cần giữ: char_features(long[B,N,32])->float[B,N,32]; conv1d16->32 kernel3 padding1; mask char=0 trước max; giữ gradient cho từ hợp lệ.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 12, dòng 11–17 của basic gốc. Đối chiếu phần đích cell 12, dòng 17–22; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: char_features(long[B,N,32])->float[B,N,32]; conv1d16->32 kernel3 padding1; mask char=0 trước max; giữ gradient cho từ hợp lệ. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết char_features bỏ PAD khi max-pooling, từ toàn PAD trả vector không. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Packed sequence
ID: sequence
Depends: constructor
Kind: code
Source: baseline_basic.ipynb#cell=12; lines=11-16
Target: baseline.ipynb#cell=12; lines=23-26
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của constructor. Hợp đồng cần giữ: sequence(module,x[B,N,D],lengths CPU long[B])->[B,N,2H]; enforce_sorted=False, total_length=N; không đưa lengths lên GPU.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 12, dòng 11–17 của basic gốc. Đối chiếu phần đích cell 12, dòng 23–26; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: sequence(module,x[B,N,D],lengths CPU long[B])->[B,N,2H]; enforce_sorted=False, total_length=N; không đưa lengths lên GPU. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết sequence(module,x,lengths) đóng gói BiLSTM, dropout và khôi phục đúng chiều dài tensor. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Attention hai chiều
ID: attention
Depends: char_features, sequence
Kind: code
Source: baseline_basic.ipynb#cell=12; lines=11-16
Target: baseline.ipynb#cell=12; lines=27-40
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của char_features, sequence. Hợp đồng cần giữ: QAModel.forward(batch); similarity[B,C,Q]=sim_context(c)+sim_question(q)+bmm(c*sim_product,q.T); mask question trước softmax; attended_q và attended_c theo mask context.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 12, dòng 11–17 của basic gốc. Đối chiếu phần đích cell 12, dòng 27–40; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: QAModel.forward(batch); similarity[B,C,Q]=sim_context(c)+sim_question(q)+bmm(c*sim_product,q.T); mask question trước softmax; attended_q và attended_c theo mask context. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần đầu forward: word/char features, encoder chung và similarity trilinear có masks đúng. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Start/end logits
ID: span_heads
Depends: attention
Kind: code
Source: baseline_basic.ipynb#cell=12; lines=11-16
Target: baseline.ipynb#cell=12; lines=41-41
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của attention. Hợp đồng cần giữ: fused=[c,attended_q,c*attended_q,c*attended_c]; start/end float[B,C], PAD context logits=-10000; build_model(cfg); giữ interface compute_loss và best_spans.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 12, dòng 11–17 của basic gốc. Đối chiếu phần đích cell 12, dòng 41–41; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: fused=[c,attended_q,c*attended_q,c*attended_c]; start/end float[B,C], PAD context logits=-10000; build_model(cfg); giữ interface compute_loss và best_spans. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Hoàn tất forward bằng fusion bốn thành phần, modeling/end BiLSTM và hai span heads. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Weighted span loss
ID: loss
Depends: span_heads
Kind: code
Source: baseline_basic.ipynb#cell=15; lines=17-21
Target: baseline.ipynb#cell=15; lines=17-21
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của span_heads. Hợp đồng cần giữ: compute_loss(model,batch)->scalar; targets=batch targets[B,2]; CE(start,targets[:,0])+CE(end,targets[:,1]); weights=0.5 khi start_target=0, ngược lại1; weighted sum/weights.sum; gold không vào forward.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 15, dòng 17–21 của basic gốc. Đối chiếu phần đích cell 15, dòng 17–21; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: compute_loss(model,batch)->scalar; targets=batch targets[B,2]; CE(start,targets[:,0])+CE(end,targets[:,1]); weights=0.5 khi start_target=0, ngược lại1; weighted sum/weights.sum; gold không vào forward. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ cross-entropy start/end có window weights và gold alignment khi đổi kiến trúc. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Factory và cấu hình model
ID: factory
Depends: vocabulary, characters, windows, constructor, char_features, sequence, attention, span_heads, loss
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=1-1
Target: baseline.ipynb#cell=13; lines=1-1
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của vocabulary, characters, windows, constructor, char_features, sequence, attention, span_heads, loss. Hợp đồng cần giữ: Factory QAModel theo chữ ký constructor đã tóm tắt; cấu hình {"n_words":len(word_vocab),"n_chars":len(char_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 1–1 của basic gốc. Đối chiếu phần đích cell 13, dòng 1–1; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Factory QAModel theo chữ ký constructor đã tóm tắt; cấu hình {"n_words":len(word_vocab),"n_chars":len(char_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Helper answer_token_span
ID: helper_answer_token_span
Depends: vocabulary, characters, windows, constructor, char_features, sequence, attention, span_heads, loss, factory
Kind: code
Source: baseline_basic.ipynb#cell=10; lines=35-40
Target: baseline.ipynb#cell=10; lines=36-43
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của vocabulary, characters, windows, constructor, char_features, sequence, attention, span_heads, loss, factory. Hợp đồng cần giữ: answer_token_span nhận (context, answer); Căn token span bao phủ tối thiểu qua offsets context gốc bằng bisect, báo exact alignment; không sửa gold text. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 10, dòng 35–42 của basic gốc. Đối chiếu phần đích cell 10, dòng 36–43; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: answer_token_span nhận (context, answer); Căn token span bao phủ tối thiểu qua offsets context gốc bằng bisect, báo exact alignment; không sửa gold text. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng answer_token_span theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Helper normalize_answer
ID: helper_normalize_answer
Depends: helper_answer_token_span
Kind: code
Source: baseline_basic.ipynb#cell=15; lines=1-5
Target: baseline.ipynb#cell=15; lines=1-5
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_answer_token_span. Hợp đồng cần giữ: normalize_answer nhận (text); Chuẩn SQuAD: lowercase, bỏ ASCII punctuation và articles a/an/the, collapse whitespace. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 15, dòng 1–5 của basic gốc. Đối chiếu phần đích cell 15, dòng 1–5; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: normalize_answer nhận (text); Chuẩn SQuAD: lowercase, bỏ ASCII punctuation và articles a/an/the, collapse whitespace. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng normalize_answer theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Helper answer_metrics
ID: helper_answer_metrics
Depends: helper_normalize_answer
Kind: code
Source: baseline_basic.ipynb#cell=15; lines=6-11
Target: baseline.ipynb#cell=15; lines=6-15
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_normalize_answer. Hợp đồng cần giữ: answer_metrics nhận (prediction, references); Tính exact match và bag-token F1, max qua mọi refs; common0 trả F1=0 theo evaluator v1.1. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 15, dòng 6–15 của basic gốc. Đối chiếu phần đích cell 15, dòng 6–15; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: answer_metrics nhận (prediction, references); Tính exact match và bag-token F1, max qua mọi refs; common0 trả F1=0 theo evaluator v1.1. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng answer_metrics theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Helper best_spans
ID: helper_best_spans
Depends: helper_answer_metrics
Kind: code
Source: baseline_basic.ipynb#cell=15; lines=23-28
Target: baseline.ipynb#cell=15; lines=23-30
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_answer_metrics. Hợp đồng cần giữ: best_spans nhận (start, end, batch); Chọn joint logits với start>=1,end>=start,length<=64 và context mask; trả starts,ends,margin so CLS scores để gộp windows. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 15, dòng 23–30 của basic gốc. Đối chiếu phần đích cell 15, dòng 23–30; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: best_spans nhận (start, end, batch); Chọn joint logits với start>=1,end>=start,length<=64 và context mask; trả starts,ends,margin so CLS scores để gộp windows. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng best_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Metric chính thức và evaluation
ID: metrics
Depends: helper_best_spans
Kind: code
Source: baseline_basic.ipynb#cell=15; lines=33-38
Target: baseline.ipynb#cell=15; lines=33-50
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_best_spans. Hợp đồng cần giữ: Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 15, dòng 33–50 của basic gốc. Đối chiếu phần đích cell 15, dòng 33–50; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Source: baseline_basic.ipynb#cell=17; lines=1-6
Target: baseline.ipynb#cell=17; lines=1-19
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của metrics. Hợp đồng cần giữ: AdamW lr=0.001,weight_decay=0.0001,warmup=0; epochs=20,batch=24,patience=5,clip=1; chọn exact_match validation tăng; vocabulary chỉ train. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 17, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 17, dòng 1–19; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.001,weight_decay=0.0001,warmup=0; epochs=20,batch=24,patience=5,clip=1; chọn exact_match validation tăng; vocabulary chỉ train. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Source: baseline_basic.ipynb#cell=21; lines=2-7
Target: baseline.ipynb#cell=19; lines=1-17
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của training. Hợp đồng cần giữ: Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 21, dòng 2–13 của basic gốc. Đối chiếu phần đích cell 19, dòng 1–17; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Source: baseline_basic.ipynb#cell=21; lines=2-7
Target: baseline.ipynb#cell=21; lines=2-13
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của reload. Hợp đồng cần giữ: context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Hàm answer_question nhận (context, question), giữ chữ ký và output schema đã chốt, không cần gold.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 21, dòng 2–13 của basic gốc. Đối chiếu phần đích cell 21, dòng 2–13; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Official SQuAD exact_match primary và token_f1 thang0..100, max mọi gold; normalize lowercase,bỏ punctuation/articles,gộp khoảng trắng. best_spans chỉ 1<=start<=end, dài<=64; so margin(start+end trừ CLS scores) giữa mọi windows theo question ID và map offsets gốc. Hàm answer_question nhận (context, question), giữ chữ ký và output schema đã chốt, không cần gold. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 21: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 22: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Source: baseline_basic.ipynb#cell=21; lines=2-7
Target: baseline.ipynb#cell=19; lines=1-17
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của summary. Hợp đồng cần giữ: context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: context/question long[B,C/Q], context_chars/question_chars long[B,C/Q,32], lengths/question_lengths CPU; context_mask/question_mask bool, targets long[B,2], ids và items chứa record/offsets/windows. PAD0,UNK1,CLS2; CLS ở context index0 và targets(0,0) khi window không có answer. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

