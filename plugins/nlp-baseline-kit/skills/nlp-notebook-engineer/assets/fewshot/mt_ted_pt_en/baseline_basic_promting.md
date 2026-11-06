# Prompt basic: TED PT→EN: BPE và Transformer scratch

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
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Hợp đồng cần giữ: TED HRLR PT→EN TFDS ted_hrlr_translate/pt_to_en:1.0.0; local JSONL train/validation/test có pt,en. Nếu dùng cùng corpus HF neulab/ted_hrlr thì revision cf21722111c95aa73c30e13df430032db463736d. Record id,src,refs; NFC và gộp khoảng trắng; lọc source train trùng val/test, val trùng test; giữ IDs/split như tham chiếu. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 8, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 8, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: TED HRLR PT→EN TFDS ted_hrlr_translate/pt_to_en:1.0.0; local JSONL train/validation/test có pt,en. Nếu dùng cùng corpus HF neulab/ted_hrlr thì revision cf21722111c95aa73c30e13df430032db463736d. Record id,src,refs; NFC và gộp khoảng trắng; lọc source train trùng val/test, val trùng test; giữ IDs/split như tham chiếu. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Kiểm tra đoạn load dữ liệu theo đặc tả, chỉ trả assertions còn thiếu và giữ split hiện tại. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Imports và cấu hình cố định
ID: config
Depends: data
Kind: code
Source: baseline_basic.ipynb#cell=5; lines=1-6
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của data. Hợp đồng cần giữ: AdamW lr=0.0005, betas=(0.9,0.98), eps=1e-9, weight_decay=0.0001; warmup=1000 rồi inverse-sqrt; epochs=45,batch=64,eval_batch=16,patience=8,clip=1; chọn BLEU validation. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 5, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 5, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.0005, betas=(0.9,0.98), eps=1e-9, weight_decay=0.0001; warmup=1000 rồi inverse-sqrt; epochs=45,batch=64,eval_batch=16,patience=8,clip=1; chọn BLEU validation. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: BPE train-only
ID: tokenizer
Depends: config
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=4-9
Target: baseline.ipynb#cell=11; lines=1-21
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của config. Hợp đồng cần giữ: sp.encode(text,out_type=int), sp.decode(ids), sp.get_piece_size(); PAD/UNK/BOS/EOS=0/1/2/3; TOKENIZER_STATE chứa spm_proto và IDs; MAX_SOURCE/TARGET/DECODE_TOKENS=128.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 4–11 của basic gốc. Đối chiếu phần đích cell 11, dòng 1–21; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: sp.encode(text,out_type=int), sp.decode(ids), sp.get_piece_size(); PAD/UNK/BOS/EOS=0/1/2/3; TOKENIZER_STATE chứa spm_proto và IDs; MAX_SOURCE/TARGET/DECODE_TOKENS=128. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay tokenizer bằng SentencePiece BPE 8000 train-only, byte fallback và normalization identity; lưu proto để reload. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Dataset và padding
ID: batch
Depends: tokenizer
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=45-50
Target: baseline.ipynb#cell=11; lines=35-41
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của tokenizer. Hợp đồng cần giữ: PairDataset(records,training=False); collate_batch(items)->source,target,lengths,ids,records; train dùng mọi refs; padding 0, MAX_SOURCE/TARGET_TOKENS=128.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 45–51 của basic gốc. Đối chiếu phần đích cell 11, dòng 35–41; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: PairDataset(records,training=False); collate_batch(items)->source,target,lengths,ids,records; train dùng mọi refs; padding 0, MAX_SOURCE/TARGET_TOKENS=128. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ PairDataset và collate_batch khi chuyển BPE, target có BOS/EOS; chỉ rút ngắn input train, không rút gold metric. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Vị trí sinusoidal
ID: position
Depends: batch
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=2-7
Target: baseline.ipynb#cell=13; lines=1-8
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của batch. Hợp đồng cần giữ: PositionEncoding(dim,dropout,max_length).forward(x[B,T,E])->[B,T,E]; max_length=130, không pretrained.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 2–8 của basic gốc. Đối chiếu phần đích cell 13, dòng 1–8; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: PositionEncoding(dim,dropout,max_length).forward(x[B,T,E])->[B,T,E]; max_length=130, không pretrained. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết PositionEncoding sinusoidal với buffer và dropout cho tensor batch-first. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Khung Transformer
ID: constructor
Depends: tokenizer, position
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=2-7
Target: baseline.ipynb#cell=13; lines=11-20
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của tokenizer, position. Hợp đồng cần giữ: TextGenerator(cfg); cfg vocab_size,embed_dim=256,hidden_dim=256,dropout=0.2,heads=8,ff_dim=1024,layers=4,max_position=130; nn.Transformer layers norm_first=True; PAD_ID=0.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 2–8 của basic gốc. Đối chiếu phần đích cell 13, dòng 11–20; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: TextGenerator(cfg); cfg vocab_size,embed_dim=256,hidden_dim=256,dropout=0.2,heads=8,ff_dim=1024,layers=4,max_position=130; nn.Transformer layers norm_first=True; PAD_ID=0. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay constructor TextGenerator bằng Transformer encoder-decoder scratch, tied embedding/output và batch-first. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: Encoder có padding mask
ID: encode
Depends: constructor
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=9-12
Target: baseline.ipynb#cell=13; lines=21-23
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của constructor. Hợp đồng cần giữ: encode(long[B,S])->memory[B,S,E], mask bool[B,S] True tại PAD; lengths giữ trên CPU nhưng Transformer không cần pack.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 9–12 của basic gốc. Đối chiếu phần đích cell 13, dòng 21–23; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: encode(long[B,S])->memory[B,S,E], mask bool[B,S] True tại PAD; lengths giữ trên CPU nhưng Transformer không cần pack. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết encode(source) dùng embedding scale căn embed_dim, positional encoding và source padding mask. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Decoder causal
ID: decode
Depends: encode
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=13-18
Target: baseline.ipynb#cell=13; lines=24-28
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của encode. Hợp đồng cần giữ: decode(long[B,T],memory[B,S,E],bool[B,S])->logits[B,T,V]; forward(source,lengths,target)->log_softmax[B,T,V]; output tied embedding.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 13–18 của basic gốc. Đối chiếu phần đích cell 13, dòng 24–28; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: decode(long[B,T],memory[B,S,E],bool[B,S])->logits[B,T,V]; forward(source,lengths,target)->log_softmax[B,T,V]; output tied embedding. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết decode(target,memory,source_mask) có causal mask và padding masks nhất quán kiểu bool. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Forward teacher forcing
ID: teacher_forward
Depends: decode
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=13-18
Target: baseline.ipynb#cell=13; lines=29-31
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của decode. Hợp đồng cần giữ: forward(source long[B,S],lengths CPU,target long[B,T])->logp[B,T,V]; lengths giữ interface, không đưa gold khi generate.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 13–18 của basic gốc. Đối chiếu phần đích cell 13, dòng 29–31; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: forward(source long[B,S],lengths CPU,target long[B,T])->logp[B,T,V]; lengths giữ interface, không đưa gold khi generate. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết forward(source,lengths,target) nối encode và decode, trả log_softmax cho teacher forcing. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Khởi tạo beam search
ID: beam_init
Depends: decode
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=20-24
Target: baseline.ipynb#cell=13; lines=33-40
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của decode. Hợp đồng cần giữ: generate(source,lengths,max_tokens,beam_size=4,length_penalty=0.6); seq[B,K,1], scores[B,K] chỉ beam 0 hữu hạn, done bool, lens; trả đoạn đầu phương thức để ghép bước sau.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 20–24 của basic gốc. Đối chiếu phần đích cell 13, dòng 33–40; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: generate(source,lengths,max_tokens,beam_size=4,length_penalty=0.6); seq[B,K,1], scores[B,K] chỉ beam 0 hữu hạn, done bool, lens; trả đoạn đầu phương thức để ghép bước sau. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần khởi tạo beam cho generate, lặp memory/mask theo beam và bắt đầu BOS. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Cập nhật beam và EOS
ID: beam_loop
Depends: beam_init
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=25-30
Target: baseline.ipynb#cell=13; lines=41-52
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của beam_init. Hợp đồng cần giữ: Chặn PAD/BOS; EOS của beam done có logp=0; gather seq/done/lens theo parent; dừng khi mọi done; không tính length thêm cho beam đã đóng; ghép sau phần khởi tạo generate.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 25–30 của basic gốc. Đối chiếu phần đích cell 13, dòng 41–52; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Chặn PAD/BOS; EOS của beam done có logp=0; gather seq/done/lens theo parent; dừng khi mọi done; không tính length thêm cho beam đã đóng; ghép sau phần khởi tạo generate. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng beam chọn top-k theo log-prob, gom lại parent beam và đóng băng beam đã EOS. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Chọn beam cuối
ID: beam_finish
Depends: beam_loop
Kind: code
Source: baseline_basic.ipynb#cell=13; lines=25-30
Target: baseline.ipynb#cell=13; lines=51-52
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của beam_loop. Hợp đồng cần giữ: score_norm=scores/(((5+lens)/6)**length_penalty); generate -> long[B,L] gồm EOS nếu sinh; forward(source,lengths,target) và build_model(cfg) giữ interface.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 13, dòng 25–30 của basic gốc. Đối chiếu phần đích cell 13, dòng 51–52; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: score_norm=scores/(((5+lens)/6)**length_penalty); generate -> long[B,L] gồm EOS nếu sinh; forward(source,lengths,target) và build_model(cfg) giữ interface. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Hoàn tất generate bằng length penalty và chọn beam tốt nhất, bỏ BOS đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Teacher forcing và masked loss
ID: loss
Depends: batch, decode
Kind: code
Source: baseline_basic.ipynb#cell=17; lines=8-13
Target: baseline.ipynb#cell=17; lines=8-14
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của batch, decode. Hợp đồng cần giữ: compute_loss(model,batch)->scalar hữu hạn; target[:,1:] đối chiếu input target[:,:-1]; return model.log_softmax; LABEL_SMOOTHING=0.1.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 17, dòng 8–14 của basic gốc. Đối chiếu phần đích cell 17, dòng 8–14; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: compute_loss(model,batch)->scalar hữu hạn; target[:,1:] đối chiếu input target[:,:-1]; return model.log_softmax; LABEL_SMOOTHING=0.1. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ loss teacher forcing dịch target một token, bỏ PAD và label smoothing 0.1 cho log-prob Transformer. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Factory và cấu hình model
ID: factory
Depends: tokenizer, batch, position, constructor, encode, decode, teacher_forward, beam_init, beam_loop, beam_finish, loss
Kind: code
Source: baseline_basic.ipynb#cell=14; lines=1-1
Target: baseline.ipynb#cell=14; lines=1-2
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của tokenizer, batch, position, constructor, encode, decode, teacher_forward, beam_init, beam_loop, beam_finish, loss. Hợp đồng cần giữ: Factory TextGenerator theo chữ ký constructor đã tóm tắt; cấu hình {"vocab_size":sp.get_piece_size(),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; cấu hình bổ sung({"heads":8,"ff_dim":1024,"layers":4,"max_position":max(MAX_SOURCE_TOKENS,MAX_TARGET_TOKENS,MAX_DECODE_TOKENS)+2}); khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 14, dòng 1–1 của basic gốc. Đối chiếu phần đích cell 14, dòng 1–2; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Factory TextGenerator theo chữ ký constructor đã tóm tắt; cấu hình {"vocab_size":sp.get_piece_size(),"embed_dim":EMBED_DIM,"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; cấu hình bổ sung({"heads":8,"ff_dim":1024,"layers":4,"max_position":max(MAX_SOURCE_TOKENS,MAX_TARGET_TOKENS,MAX_DECODE_TOKENS)+2}); khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Helper decode_tokens
ID: helper_decode_tokens
Depends: tokenizer, batch, position, constructor, encode, decode, teacher_forward, beam_init, beam_loop, beam_finish, loss, factory
Kind: code
Source: baseline_basic.ipynb#cell=17; lines=1-6
Target: baseline.ipynb#cell=17; lines=1-7
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của tokenizer, batch, position, constructor, encode, decode, teacher_forward, beam_init, beam_loop, beam_finish, loss, factory. Hợp đồng cần giữ: decode_tokens nhận (ids); Giải mã IDs bằng SentencePiece, bỏ PAD/BOS và dừng EOS; giữ clean_text NFC và whitespace. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 17, dòng 1–7 của basic gốc. Đối chiếu phần đích cell 17, dòng 1–7; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: decode_tokens nhận (ids); Giải mã IDs bằng SentencePiece, bỏ PAD/BOS và dừng EOS; giữ clean_text NFC và whitespace. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng decode_tokens theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Helper score_corpus
ID: helper_score_corpus
Depends: helper_decode_tokens
Kind: code
Source: baseline_basic.ipynb#cell=16; lines=4-9
Target: baseline.ipynb#cell=16; lines=4-9
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_decode_tokens. Hợp đồng cần giữ: score_corpus nhận (records, predictions); Dùng sacreBLEU BLEU/chrF corpus cho toàn bộ refs; giữ signature và primary bleu. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 16, dòng 4–9 của basic gốc. Đối chiếu phần đích cell 16, dòng 4–9; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: score_corpus nhận (records, predictions); Dùng sacreBLEU BLEU/chrF corpus cho toàn bộ refs; giữ signature và primary bleu. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng score_corpus theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Metric chính thức và evaluation
ID: metrics
Depends: helper_score_corpus
Kind: code
Source: baseline_basic.ipynb#cell=17; lines=17-22
Target: baseline.ipynb#cell=17; lines=17-26
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của helper_score_corpus. Hợp đồng cần giữ: sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 17, dòng 17–26 của basic gốc. Đối chiếu phần đích cell 17, dòng 17–26; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Source: baseline_basic.ipynb#cell=20; lines=1-6
Target: baseline.ipynb#cell=20; lines=1-19
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của metrics. Hợp đồng cần giữ: AdamW lr=0.0005, betas=(0.9,0.98), eps=1e-9, weight_decay=0.0001; warmup=1000 rồi inverse-sqrt; epochs=45,batch=64,eval_batch=16,patience=8,clip=1; chọn BLEU validation. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 20, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 20, dòng 1–19; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr=0.0005, betas=(0.9,0.98), eps=1e-9, weight_decay=0.0001; warmup=1000 rồi inverse-sqrt; epochs=45,batch=64,eval_batch=16,patience=8,clip=1; chọn BLEU validation. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Source: baseline_basic.ipynb#cell=25; lines=2-7
Target: baseline.ipynb#cell=22; lines=1-17
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của training. Hợp đồng cần giữ: sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 25, dòng 2–9 của basic gốc. Đối chiếu phần đích cell 22, dòng 1–17; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Source: baseline_basic.ipynb#cell=25; lines=2-7
Target: baseline.ipynb#cell=25; lines=2-9
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của reload. Hợp đồng cần giữ: source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Hàm rewrite_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 25, dòng 2–9 của basic gốc. Đối chiếu phần đích cell 25, dòng 2–9; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. sacreBLEU BLEU và chrF corpus với đầy đủ refs; primary bleu tăng. COMET chỉ final scoring nếu được cho phép, không dùng sinh output hoặc chọn checkpoint. Hàm rewrite_text nhận (text), giữ chữ ký và output schema đã chốt, không cần gold. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
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
Source: baseline_basic.ipynb#cell=25; lines=2-7
Target: baseline.ipynb#cell=22; lines=1-17
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của summary. Hợp đồng cần giữ: source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: source long[B,S], target long[B,T], lengths long[B] trên CPU, ids và records; PAD/UNK/BOS/EOS=0/1/2/3; refs giữ đầy đủ khi chấm. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

