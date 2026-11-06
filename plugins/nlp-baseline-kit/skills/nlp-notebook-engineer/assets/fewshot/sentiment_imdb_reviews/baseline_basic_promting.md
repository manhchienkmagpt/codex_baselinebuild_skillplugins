# Prompt basic: IMDb: TF-IDF từ/ký tự và NB-SVM

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
Source: baseline_basic.ipynb#cell=5; lines=1-6
Target: baseline.ipynb#cell=5; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Hợp đồng cần giữ: TFDS imdb_reviews; local JSONL train/test có id, text, label với neg=0, pos=1. Chuẩn hóa HTML/NFC, gộp khoảng trắng; giữ test chuẩn. Train chia StratifiedGroupKFold theo hash text chuẩn hóa, seed=42, giữ train_idx/val_idx (fold đầu của 5 folds) và SPLIT_INFO như bản tham chiếu. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 5, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 5, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: TFDS imdb_reviews; local JSONL train/test có id, text, label với neg=0, pos=1. Chuẩn hóa HTML/NFC, gộp khoảng trắng; giữ test chuẩn. Train chia StratifiedGroupKFold theo hash text chuẩn hóa, seed=42, giữ train_idx/val_idx (fold đầu của 5 folds) và SPLIT_INFO như bản tham chiếu. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Kiểm tra đoạn load dữ liệu theo đặc tả, chỉ trả assertions còn thiếu và giữ split hiện tại. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 2: Imports và cấu hình cố định
ID: config
Depends: data
Kind: code
Source: baseline_basic.ipynb#cell=3; lines=1-6
Target: baseline.ipynb#cell=3; lines=1-6
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của data. Hợp đồng cần giữ: Vòng C fit sklearn trên train, không epochs/backprop; chọn macro_f1 validation tăng; joblib đủ features,ratio,classifier,label_names,validation. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 3, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 3, dòng 1–6; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Vòng C fit sklearn trên train, không epochs/backprop; chọn macro_f1 validation tăng; joblib đủ features,ratio,classifier,label_names,validation. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Cấu hình mô hình sẽ được bổ sung đúng khi dựng từng module; không đặt pretrained text model. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết imports và constants cần cho các module theo đặc tả, giữ seed42 và device cuda/cpu đơn giản. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 3: TF-IDF từ và ký tự
ID: features
Depends: config
Kind: code
Source: baseline_basic.ipynb#cell=7; lines=1-3
Target: baseline.ipynb#cell=7; lines=1-8
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của config. Hợp đồng cần giữ: build_features() -> FeatureUnion; WORD_MAX_FEATURES=120000, CHAR_MAX_FEATURES=100000, MIN_DF=2, sublinear_tf=True, dtype=float32; thêm import TfidfVectorizer, FeatureUnion.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 7, dòng 1–3 của basic gốc. Đối chiếu phần đích cell 7, dòng 1–8; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: build_features() -> FeatureUnion; WORD_MAX_FEATURES=120000, CHAR_MAX_FEATURES=100000, MIN_DF=2, sublinear_tf=True, dtype=float32; thêm import TfidfVectorizer, FeatureUnion. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay build_features bằng FeatureUnion TF-IDF word 1–2 và char_wb 3–5, trọng số 1 và 0.5; chỉ fit train. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 4: Log-ratio NB train-only
ID: ratio
Depends: features
Kind: code
Source: baseline_basic.ipynb#cell=7; lines=5-9
Target: baseline.ipynb#cell=7; lines=10-23
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của features. Hợp đồng cần giữ: X_train/X_val: CSR[N,D]; ratio: float32[D], y_train có 0/1; USE_NB_RATIO=True, NB_RATIO_ALPHA=1; không fit/ước lượng từ validation/test.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 7, dòng 5–9 của basic gốc. Đối chiếu phần đích cell 7, dòng 10–23; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: X_train/X_val: CSR[N,D]; ratio: float32[D], y_train có 0/1; USE_NB_RATIO=True, NB_RATIO_ALPHA=1; không fit/ước lượng từ validation/test. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Tạo NB log-ratio nhị phân từ presence train với smoothing 1; nhân cùng ratio vào CSR train và validation. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 5: LinearSVC scratch
ID: classifier
Depends: ratio
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=2-3
Target: baseline.ipynb#cell=9; lines=2-4
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của ratio. Hợp đồng cần giữ: make_classifier(parameter) -> LinearSVC(C=parameter, dual=True, max_iter=10000, tol=1e-4, random_state=42); import LinearSVC, ConvergenceWarning, warnings; train fit trên CSR đã nhân ratio.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 2–3 của basic gốc. Đối chiếu phần đích cell 9, dòng 2–4; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: make_classifier(parameter) -> LinearSVC(C=parameter, dual=True, max_iter=10000, tol=1e-4, random_state=42); import LinearSVC, ConvergenceWarning, warnings; train fit trên CSR đã nhân ratio. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Thay make_classifier(parameter) bằng LinearSVC, giữ interface và xử lý lỗi không hội tụ. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 6: Chọn C bằng macro-F1
ID: selection
Depends: classifier
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=9-14
Target: baseline.ipynb#cell=9; lines=10-25
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của classifier. Hợp đồng cần giữ: PARAMETERS=C_VALUES=(0.25,1.0,4.0); classification_metrics(y,pred)->macro_f1,accuracy; joblib lưu features,ratio,classifier,label_names,validation, không fit lại train+validation.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 9–16 của basic gốc. Đối chiếu phần đích cell 9, dòng 10–25; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: PARAMETERS=C_VALUES=(0.25,1.0,4.0); classification_metrics(y,pred)->macro_f1,accuracy; joblib lưu features,ratio,classifier,label_names,validation, không fit lại train+validation. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Đổi vòng chọn mô hình sang C bằng 0.25, 1 và 4, tối đa hóa macro-F1 validation; lưu pipeline tốt nhất. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 7: Reload và dự đoán
ID: prediction
Depends: selection
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=4-7
Target: baseline.ipynb#cell=11; lines=4-7
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của selection. Hợp đồng cần giữ: predict_texts(list[str])->nhãn int[N]; normalize_text rồi features.transform().multiply(ratio).tocsr(); IDs và official test giữ nguyên; không học thêm vocabulary.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 4–7 của basic gốc. Đối chiếu phần đích cell 11, dòng 4–7; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: predict_texts(list[str])->nhãn int[N]; normalize_text rồi features.transform().multiply(ratio).tocsr(); IDs và official test giữ nguyên; không học thêm vocabulary. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Giữ predict_texts(texts), reload features, ratio và classifier; đối chiếu validation trước khi chấm test. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 8: Metric chính thức và evaluation
ID: metrics
Depends: features, ratio, classifier, selection, prediction
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=5-7
Target: baseline.ipynb#cell=9; lines=6-8
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của features, ratio, classifier, selection, prediction. Hợp đồng cần giữ: macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. classification_metrics(y,pred) giữ interface hiện tại.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 5–7 của basic gốc. Đối chiếu phần đích cell 9, dòng 6–8; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. classification_metrics(y,pred) giữ interface hiện tại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết classification_metrics(y,pred) trả macro_f1 và accuracy, giữ đủ nhãn và zero_division bằng0. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 9: Huấn luyện hoặc fit và chọn checkpoint
ID: training
Depends: metrics
Kind: code
Source: baseline_basic.ipynb#cell=9; lines=1-6
Target: baseline.ipynb#cell=9; lines=10-25
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của metrics. Hợp đồng cần giữ: Vòng C fit sklearn trên train, không epochs/backprop; chọn macro_f1 validation tăng; joblib đủ features,ratio,classifier,label_names,validation. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 9, dòng 1–6 của basic gốc. Đối chiếu phần đích cell 9, dòng 10–25; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: Vòng C fit sklearn trên train, không epochs/backprop; chọn macro_f1 validation tăng; joblib đủ features,ratio,classifier,label_names,validation. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. Checkpoint PyTorch chỉ lưu model_state,tokenizer và epoch cần thiết; sparse dùng joblib pipeline; không resume state. Caller loss/evaluate theo tóm tắt. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết vòng fit và chọn pipeline theo macro-F1 validation, lưu đủ thành phần suy luận bằng joblib. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 10: Reload pipeline và đánh giá cuối
ID: reload
Depends: training
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=4-7
Target: baseline.ipynb#cell=11; lines=1-12
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của training. Hợp đồng cần giữ: macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 4–7 của basic gốc. Đối chiếu phần đích cell 11, dòng 1–12; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Reload tokenizer/vocabulary rồi build model cùng cfg; visual pretrained không tải lại khi initialize_pretrained=False; không đổi official split hoặc fit lại. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết phần reload đầy đủ preprocessing và model đã chọn, kiểm lại validation trước khi chấm held-out đúng một lần. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 11: Suy luận đầu vào mới
ID: inference
Depends: reload
Kind: code
Source: baseline_basic.ipynb#cell=11; lines=4-7
Target: baseline.ipynb#cell=11; lines=4-7
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của reload. Hợp đồng cần giữ: train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. Hàm predict_texts nhận (texts), giữ chữ ký và output schema đã chốt, không cần gold.
Merge: Source là đoạn input rút gọn; vị trí basic để sửa hoặc dùng làm điểm chèn là cell 11, dòng 4–7 của basic gốc. Đối chiếu phần đích cell 11, dòng 4–7; thay đúng hàm/phương thức được nêu, thêm helper mới trước caller, không thay cả class chỉ vì gửi trích đoạn constructor. Các lát cùng một method được nối theo thứ tự trong bản làm việc; kiểm và cập nhật mọi caller theo API mới, giữ phần khác. Nếu cell đã thay đổi, dùng vị trí hàm hiện tại tương ứng.
Budget: input=600; output=1100; reserve=300
Handoff: train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. macro-F1 và accuracy sklearn, labels=[0,1], zero_division=0; kết quả test không dùng chọn C. Hàm predict_texts nhận (texts), giữ chữ ký và output schema đã chốt, không cần gold. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết hàm dự đoán đầu vào mới theo API đã chốt, dùng pipeline reload và giữ đúng schema đầu ra. Trả riêng code phần này và tóm tắt API ngắn cho phiên sau.

## Bước 12: Gộp hợp đồng cho phiên ghép
ID: summary
Depends: inference
Kind: summary
Input: Prompt và bản tóm tắt chữ đã đối chiếu của inference; không gửi code hay lịch sử phiên trước. Chỉ giữ API/cấu hình mà phần ghép cuối cần.
Merge: Dùng bản tóm tắt chữ cho phiên tích hợp kế tiếp; đối chiếu với code lưu cục bộ, không thay code đã sinh.
Budget: input=600; output=1100; reserve=300
Handoff: Chỉ exports và config được glue/integration dùng; interface,shape/dtype,IDs,mask,metric và thứ tự gọi; giới hạn250 tokens, không code. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Gộp các bản tóm tắt thành hợp đồng tích hợp ngắn, giữ chữ ký API, kiểu/shape, cấu hình và ràng buộc cần dùng. Chỉ trả bản tóm tắt chữ tối đa 250 tokens, bỏ chi tiết nội bộ và không dựng lại code.

## Bước 13: Ghép và kiểm tra tối thiểu
ID: integration
Depends: summary
Kind: integration
Source: baseline_basic.ipynb#cell=11; lines=4-7
Target: baseline.ipynb#cell=11; lines=1-12
Input: Prompt và đoạn code hiện tại tương ứng Source, chỉ các dòng đã chỉ; bổ sung chữ ký hiện tại nếu đoạn trích nằm giữa hàm. Gửi bản tóm tắt chữ của summary. Hợp đồng cần giữ: train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit.
Merge: Trong bản làm việc, nối các module đã lưu theo chữ ký tóm tắt; chỉ thêm glue/assertions trả về, không thay bằng toàn notebook mới. Run All từ kernel sạch sau khi kiểm imports/config và thứ tự.
Budget: input=600; output=1100; reserve=300
Handoff: train_text/val_text: list[str]; y_train/y_val: int64[N]; nhãn [neg,pos]; dữ liệu thưa CSR; không cần GPU. Kiểm fresh-kernel, split IDs và leakage; fit/forward-backward-update khi phù hợp; reload tương đương trước test; metric/ID/count đúng, không bịa score hoặc GPU fit. Giữ thêm chữ ký/import/cấu hình từ các phụ thuộc còn được caller dùng; chỉ tóm tắt chữ, đối chiếu với code vừa sinh.
### Prompt
Viết glue và assertions tối thiểu nối các module theo hợp đồng đã gửi, giữ đúng split, IDs và metric. Chỉ trả phần ghép còn thiếu cùng tóm tắt API, không sinh lại toàn bộ pipeline.

