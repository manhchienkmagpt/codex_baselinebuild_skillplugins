# Prompt hardcore: FUNSD: word/char/layout/ROI, BiLSTM và CRF

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
Input: Prompt, đặc tả chữ: nielsr/funsd revision 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5; id,image,words,bboxes,ner_tags; boxes0..1000, words/boxes/tags cùng độ dài. Tags O,B/I-HEADER,B/I-QUESTION,B/I-ANSWER; sửa I sai thành B cùng loại. Train chia20% documents seed42; giữ test50; không có gold K-V links. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Không gửi mã nguồn.
Merge: Lưu riêng phần data để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: nielsr/funsd revision 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5; id,image,words,bboxes,ner_tags; boxes0..1000, words/boxes/tags cùng độ dài. Tags O,B/I-HEADER,B/I-QUESTION,B/I-ANSWER; sửa I sai thành B cùng loại. Train chia20% documents seed42; giữ test50; không có gold K-V links. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm đọc nguồn dữ liệu theo đặc tả và trả record thô, báo lỗi rõ khi thiếu nguồn; chưa chia hay fit dữ liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Chuẩn hóa schema và chia nhóm
ID: splits
Depends: data
Kind: code
Target: baseline.ipynb#cell=8; lines=1-6
Input: Prompt, bản tóm tắt chữ của data, đặc tả chữ: nielsr/funsd revision 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5; id,image,words,bboxes,ner_tags; boxes0..1000, words/boxes/tags cùng độ dài. Tags O,B/I-HEADER,B/I-QUESTION,B/I-ANSWER; sửa I sai thành B cùng loại. Train chia20% documents seed42; giữ test50; không có gold K-V links. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Không gửi mã nguồn.
Merge: Lưu riêng phần splits để ghép vào bản làm việc theo cell 8, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: nielsr/funsd revision 7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5; id,image,words,bboxes,ner_tags; boxes0..1000, words/boxes/tags cùng độ dài. Tags O,B/I-HEADER,B/I-QUESTION,B/I-ANSWER; sửa I sai thành B cùng loại. Train chia20% documents seed42; giữ test50; không có gold K-V links. Đầu ra train_records,val_records,test_records theo đúng thứ tự, cùng fingerprints như bản tham chiếu. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần chuẩn hóa record và split đúng đặc tả, kiểm IDs và ngăn rò rỉ; không fit đặc trưng ở bước này. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: Imports và cấu hình cố định
ID: config
Depends: splits
Kind: code
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt, bản tóm tắt chữ của splits, đặc tả chữ: AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Không gửi mã nguồn.
Merge: Lưu riêng phần config để ghép vào bản làm việc theo cell 5, dòng 1–6 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Word và char vocabulary
ID: vocabulary
Depends: config
Kind: code
Target: baseline.ipynb#cell=10; lines=1-11
Input: Prompt, bản tóm tắt chữ của config, đặc tả chữ: PAD0 UNK1; TOKENIZER_STATE word_vocab,char_vocab,tag_names; MAX_WORD_CHARS32,IMAGE_SIZE384; không học vocabulary từ validation/test. Không gửi mã nguồn.
Merge: Lưu riêng phần vocabulary để ghép vào bản làm việc theo cell 10, dòng 1–11 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: PAD0 UNK1; TOKENIZER_STATE word_vocab,char_vocab,tag_names; MAX_WORD_CHARS32,IMAGE_SIZE384; không học vocabulary từ validation/test. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung char vocabulary train-only và lưu cùng word_vocab, TAG_NAMES để reload. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: Ảnh tài liệu và batch
ID: dataset
Depends: vocabulary
Kind: code
Target: baseline.ipynb#cell=10; lines=13-21
Input: Prompt, bản tóm tắt chữ của vocabulary, đặc tả chữ: FormDataset.__getitem__ -> image[3,384,384],record; collate_batch bổ sung images,chars[B,N,32]; boxes/1000, lengths CPU, mask; không normalize ảnh ImageNet vì visual CNN scratch. Không gửi mã nguồn.
Merge: Lưu riêng phần dataset để ghép vào bản làm việc theo cell 10, dòng 13–21 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: FormDataset.__getitem__ -> image[3,384,384],record; collate_batch bổ sung images,chars[B,N,32]; boxes/1000, lengths CPU, mask; không normalize ảnh ImageNet vì visual CNN scratch. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Bổ sung RGB image tensor và chars vào FormDataset/collate_batch, giữ box/tag/word alignment. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: CRF transitions hợp lệ
ID: crf_init
Depends: dataset
Kind: code
Target: baseline.ipynb#cell=12; lines=2-15
Input: Prompt, bản tóm tắt chữ của dataset, đặc tả chữ: EntityCRF(names); I-X không đầu chuỗi, chỉ sau B-X/I-X; potentials()->transitions[Ntags,Ntags],start[Ntags] với illegal=-10000. Không gửi mã nguồn.
Merge: Lưu riêng phần crf_init để ghép vào bản làm việc theo cell 12, dòng 2–15 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: EntityCRF(names); I-X không đầu chuỗi, chỉ sau B-X/I-X; potentials()->transitions[Ntags,Ntags],start[Ntags] với illegal=-10000. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Dựng EntityCRF với start/end/transitions và ràng buộc BIO theo TAG_NAMES. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: CRF log-partition
ID: crf_loss
Depends: crf_init
Kind: code
Target: baseline.ipynb#cell=12; lines=18-29
Input: Prompt, bản tóm tắt chữ của crf_init, đặc tả chữ: loss(emissions[B,N,K],tags[B,N],mask[B,N])->scalar; start/end transitions, bỏ PAD và giữ gradient; không truyền gold vào decode. Không gửi mã nguồn.
Merge: Lưu riêng phần crf_loss để ghép vào bản làm việc theo cell 12, dòng 18–29 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: loss(emissions[B,N,K],tags[B,N],mask[B,N])->scalar; start/end transitions, bỏ PAD và giữ gradient; không truyền gold vào decode. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết loss CRF bằng logsumexp partition và gold path có mask, chuẩn hóa theo lengths. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Viterbi BIO
ID: viterbi
Depends: crf_loss
Kind: code
Target: baseline.ipynb#cell=12; lines=31-45
Input: Prompt, bản tóm tắt chữ của crf_loss, đặc tả chữ: decode(emissions,mask)->list[list[int]]; dùng potentials, end scores và backpointers; mỗi đường dài mask.sum, không phát I sai loại. Không gửi mã nguồn.
Merge: Lưu riêng phần viterbi để ghép vào bản làm việc theo cell 12, dòng 31–45 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: decode(emissions,mask)->list[list[int]]; dùng potentials, end scores và backpointers; mỗi đường dài mask.sum, không phát I sai loại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết Viterbi có mask và backtracking, trả chuỗi nhãn đúng độ dài mỗi tài liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Khung multimodal FormModel
ID: constructor
Depends: viterbi
Kind: code
Target: baseline.ipynb#cell=13; lines=3-16
Input: Prompt, bản tóm tắt chữ của viterbi, đặc tả chữ: cfg n_words,n_chars,hidden_dim128,dropout0.3; word128,char16->CNN32,layout6->32; visual Conv3->32->64 mỗi lớp maxpool2, ROI projection512->64; BiLSTM input256,2layers,bidir. Không gửi mã nguồn.
Merge: Lưu riêng phần constructor để ghép vào bản làm việc theo cell 13, dòng 3–16 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: cfg n_words,n_chars,hidden_dim128,dropout0.3; word128,char16->CNN32,layout6->32; visual Conv3->32->64 mỗi lớp maxpool2, ROI projection512->64; BiLSTM input256,2layers,bidir. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay constructor FormModel bằng word/char/layout/visual CNN, BiLSTM và EntityCRF. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Char-CNN và layout
ID: char_layout
Depends: constructor
Kind: code
Target: baseline.ipynb#cell=13; lines=17-28
Input: Prompt, bản tóm tắt chữ của constructor, đặc tả chữ: chars[B,N,32]->features[B,N,32], toàn PAD trả0; layout input[x0,y0,x1,y1,w,h] chuẩn0..1 ->[B,N,32]; không đổi word order. Không gửi mã nguồn.
Merge: Lưu riêng phần char_layout để ghép vào bản làm việc theo cell 13, dòng 17–28 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: chars[B,N,32]->features[B,N,32], toàn PAD trả0; layout input[x0,y0,x1,y1,w,h] chuẩn0..1 ->[B,N,32]; không đổi word order. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần đầu forward trích char-CNN masked pooling và layout từ tọa độ cùng kích thước box. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: ROI visual features
ID: roi
Depends: char_layout
Kind: code
Target: baseline.ipynb#cell=13; lines=29-38
Input: Prompt, bản tóm tắt chữ của char_layout, đặc tả chữ: visual CNN image/4; boxes normalized*image_width; torchvision.ops.roi_align output_size(2,4), spatial_scale0.25, aligned=True; projection512->64, scatter về[B,N,64]. Không gửi mã nguồn.
Merge: Lưu riêng phần roi để ghép vào bản làm việc theo cell 13, dòng 29–38 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: visual CNN image/4; boxes normalized*image_width; torchvision.ops.roi_align output_size(2,4), spatial_scale0.25, aligned=True; projection512->64, scatter về[B,N,64]. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần ROI Align theo boxes và ghép visual features về đúng word của từng tài liệu. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: BiLSTM emissions và CRF
ID: fusion
Depends: roi
Kind: code
Target: baseline.ipynb#cell=13; lines=38-38
Input: Prompt, bản tóm tắt chữ của roi, đặc tả chữ: forward(batch)->emissions[B,N,7]; concat dim256; loss(emissions,batch), decode(emissions,batch); build_model(cfg,initialize_pretrained=True) không tải pretrained; lengths CPU. Không gửi mã nguồn.
Merge: Lưu riêng phần fusion để ghép vào bản làm việc theo cell 13, dòng 38–38 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: forward(batch)->emissions[B,N,7]; concat dim256; loss(emissions,batch), decode(emissions,batch); build_model(cfg,initialize_pretrained=True) không tải pretrained; lengths CPU. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Hoàn tất forward: ghép word,char,layout,ROI, chạy packed BiLSTM và tag head, nối loss/decode CRF. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 13: Factory và cấu hình model
ID: factory
Depends: vocabulary, dataset, crf_init, crf_loss, viterbi, constructor, char_layout, roi, fusion
Kind: code
Target: baseline.ipynb#cell=14; lines=1-1
Input: Prompt, bản tóm tắt chữ của vocabulary, dataset, crf_init, crf_loss, viterbi, constructor, char_layout, roi, fusion, đặc tả chữ: Factory FormModel theo chữ ký constructor đã tóm tắt; cấu hình  {"n_words":len(word_vocab),"n_chars":len(char_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Không gửi mã nguồn.
Merge: Lưu riêng phần factory để ghép vào bản làm việc theo cell 14, dòng 1–1 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Factory FormModel theo chữ ký constructor đã tóm tắt; cấu hình  {"n_words":len(word_vocab),"n_chars":len(char_vocab),"hidden_dim":HIDDEN_DIM,"dropout":DROPOUT}; khai báo mọi constants được tham chiếu bằng giá trị cố định từ config và API trước đó; không tự tải checkpoint task. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết MODEL_CONFIG và phần khởi tạo model theo constructor đã chốt, không sinh lại toàn class hay thêm wrapper không cần thiết. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 14: Nối Dataset và DataLoader
ID: loaders
Depends: splits, config, vocabulary, dataset, crf_init, crf_loss, viterbi, constructor, char_layout, roi, fusion, factory
Kind: code
Target: baseline.ipynb#cell=10; lines=23-42
Input: Prompt, bản tóm tắt chữ của splits, config, vocabulary, dataset, crf_init, crf_loss, viterbi, constructor, char_layout, roi, fusion, factory, đặc tả chữ: images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Không gửi mã nguồn.
Merge: Lưu riêng phần loaders để ghép vào bản làm việc theo cell 10, dòng 23–42 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. Dùng seed42, num_workers0; định nghĩa to_device giữ lengths CPU nếu cần. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết Dataset/DataLoader còn thiếu theo hợp đồng batch, chỉ shuffle train và giữ IDs/order khi đánh giá. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 15: Helper entity_spans
ID: helper_entity_spans
Depends: loaders
Kind: code
Target: baseline.ipynb#cell=16; lines=1-10
Input: Prompt, bản tóm tắt chữ của loaders, đặc tả chữ: entity_spans nhận (tags); Sửa BIO bằng canonical_bio, đóng entity khi O/B/khác type; trả start_word,end_word exclusive,type, giữ thứ tự words. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Không gửi mã nguồn.
Merge: Lưu riêng phần helper_entity_spans để ghép vào bản làm việc theo cell 16, dòng 1–10 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: entity_spans nhận (tags); Sửa BIO bằng canonical_bio, đóng entity khi O/B/khác type; trả start_word,end_word exclusive,type, giữ thứ tự words. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng entity_spans theo thuật toán trong đặc tả, giữ chữ ký và kiểu đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 16: Metric chính thức và evaluation
ID: metrics
Depends: helper_entity_spans
Kind: code
Target: baseline.ipynb#cell=16; lines=16-37
Input: Prompt, bản tóm tắt chữ của helper_entity_spans, đặc tả chữ: Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Không gửi mã nguồn.
Merge: Lưu riêng phần metrics để ghép vào bản làm việc theo cell 16, dòng 16–37 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. evaluate(model,loader,return_predictions=False) dùng model.eval/no_grad. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết riêng evaluate gọi các helper đã dựng, giữ đầy đủ mẫu, IDs và references; không dùng gold tạo prediction. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 17: Loss nối model và batch
ID: loss_wrapper
Depends: metrics
Kind: code
Target: baseline.ipynb#cell=16; lines=12-13
Input: Prompt, bản tóm tắt chữ của metrics, đặc tả chữ: AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Dùng các API forward/loss đã dựng, không dùng gold trong đường suy luận. Không gửi mã nguồn.
Merge: Lưu riêng phần loss_wrapper để ghép vào bản làm việc theo cell 16, dòng 12–13 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Dùng các API forward/loss đã dựng, không dùng gold trong đường suy luận. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết loss scalar nối model và batch theo hợp đồng, bỏ padding và giữ gradient; kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 18: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: loss_wrapper
Kind: code
Target: baseline.ipynb#cell=19; lines=1-20
Input: Prompt, bản tóm tắt chữ của loss_wrapper, đặc tả chữ: AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Không gửi mã nguồn.
Merge: Lưu riêng phần training để ghép vào bản làm việc theo cell 19, dòng 1–20 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: AdamW lr0.001,weight_decay0.0001; epochs80,batch4,patience12,clip1; chọn entity_f1 validation tăng; toàn model scratch, không pretrained document parser. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng train trực tiếp và chọn checkpoint theo metric validation, giữ cấu hình cố định và kiểm loss hữu hạn. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 19: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Target: baseline.ipynb#cell=21; lines=1-19
Input: Prompt, bản tóm tắt chữ của training, đặc tả chữ: Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Không gửi mã nguồn.
Merge: Lưu riêng phần reload để ghép vào bản làm việc theo cell 21, dòng 1–19 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 20: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Target: baseline.ipynb#cell=23; lines=2-14
Input: Prompt, bản tóm tắt chữ của reload, đặc tả chữ: images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Hàm parse_form nhận (image, words, bboxes), giữ chữ ký và output schema đã chốt, không cần gold. Cần open_rgb nhận path/bytes/PIL và kiểm ảnh hỏng rõ. Không gửi mã nguồn.
Merge: Lưu riêng phần inference để ghép vào bản làm việc theo cell 23, dòng 2–14 của bản đích; giữ các phần đã sinh. Nếu là lát method, nối đúng thứ tự trong cùng class; không chạy khi class còn thiếu phần. Cập nhật caller theo API dưới đây.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Entity span precision/recall/F1 exact theo BIO và type, không token accuracy; entity_f1 primary tăng; K-V linking F1=null vì không gold links. Hàm parse_form nhận (image, words, bboxes), giữ chữ ký và output schema đã chốt, không cần gold. Cần open_rgb nhận path/bytes/PIL và kiểm ảnh hỏng rõ. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
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
Target: baseline.ipynb#cell=21; lines=1-19
Input: Prompt, bản tóm tắt chữ của summary, đặc tả chữ: images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Không gửi mã nguồn.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: images float[B,3,384,384], words long[B,N], chars long[B,N,32], boxes float[B,N,4] chuẩn0..1, tags long[B,N], lengths CPU,mask bool,ids,records; giữ thứ tự words nguồn. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

