# Fewshot NLP chọn từ outputs

Thư viện mới được chọn từ tám nhóm output hiện có và thay thế toàn bộ bộ mẫu đánh số trước đây. Bản gốc tuyển chọn ở `fewshot/<task_slug>/`; skill/plugin đóng gói bản sao riêng trong assets/fewshot để dùng độc lập với dự án. Các bản sao giữ nguyên nội dung bốn tệp của từng output, không đổi dataset, model hoặc metadata. Các notebook nguồn đã có outputs rỗng và execution_count=null.

## Chọn đúng tác vụ và biến thể

| Nhóm mẫu | Tệp tham khảo | Phần nên học |
|---|---|---|
| Phân loại văn bản sparse | [optimized](../assets/fewshot/sentiment_imdb_reviews/baseline.ipynb) · [basic](../assets/fewshot/sentiment_imdb_reviews/baseline_basic.ipynb) · [hardcore](../assets/fewshot/sentiment_imdb_reviews/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/sentiment_imdb_reviews/huong_giai_quyet.md) | TF-IDF/NB-SVM optimized; unigram counts/Naive Bayes basic; model byte scratch hardcore. Học cách chỉ dùng siêu tham số có nghĩa với sklearn. |
| Sequence labeling / span / polarity | [optimized](../assets/fewshot/absa_semeval2014_restaurants/baseline.ipynb) · [basic](../assets/fewshot/absa_semeval2014_restaurants/baseline_basic.ipynb) · [hardcore](../assets/fewshot/absa_semeval2014_restaurants/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/absa_semeval2014_restaurants/huong_giai_quyet.md) | Ký tự + từ, BiLSTM-CRF, span offsets và nhiều head; basic giữ đúng BIO/mask với RNN đơn giản. |
| Seq2seq Transformer scratch | [optimized](../assets/fewshot/mt_ted_pt_en/baseline.ipynb) · [basic](../assets/fewshot/mt_ted_pt_en/baseline_basic.ipynb) · [hardcore](../assets/fewshot/mt_ted_pt_en/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/mt_ted_pt_en/huong_giai_quyet.md) | BPE/causal masks/beam cho optimized; whitespace vocabulary + RNN cho basic; raw-text inference không training cho hardcore. |
| Extractive QA | [optimized](../assets/fewshot/qa_squad_v11/baseline.ipynb) · [basic](../assets/fewshot/qa_squad_v11/baseline_basic.ipynb) · [hardcore](../assets/fewshot/qa_squad_v11/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/qa_squad_v11/huong_giai_quyet.md) | Word/char features, context–question attention, window/offset alignment và start/end decoding; basic bỏ char features không dùng. |
| Seq2seq attention / pointer-copy | [optimized](../assets/fewshot/gec_coedit/baseline.ipynb) · [basic](../assets/fewshot/gec_coedit/baseline_basic.ipynb) · [hardcore](../assets/fewshot/gec_coedit/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/gec_coedit/huong_giai_quyet.md) | Giữ source groups/references, copy distribution và metric theo tác vụ; giúp tham khảo các bài sửa/viết lại/tóm tắt có nhu cầu giữ nội dung. |
| Ảnh → mô tả | [optimized](../assets/fewshot/caption_coco2014/baseline.ipynb) · [basic](../assets/fewshot/caption_coco2014/baseline_basic.ipynb) · [hardcore](../assets/fewshot/caption_coco2014/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/caption_coco2014/huong_giai_quyet.md) | Image encoder + scratch attention/LSTM; basic CNN/RNN; phân biệt thành phần ảnh pretrained và decoder mới. |
| OCR dòng / CTC | [optimized](../assets/fewshot/ocr_iam_line/baseline.ipynb) · [basic](../assets/fewshot/ocr_iam_line/baseline_basic.ipynb) · [hardcore](../assets/fewshot/ocr_iam_line/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/ocr_iam_line/huong_giai_quyet.md) | Resize theo tỷ lệ, variable-width padding, alphabet train-only, CTC lengths và decoder; basic giữ các điều kiện bắt buộc. |
| Tài liệu: word + layout + ảnh | [optimized](../assets/fewshot/document_funsd/baseline.ipynb) · [basic](../assets/fewshot/document_funsd/baseline_basic.ipynb) · [hardcore](../assets/fewshot/document_funsd/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/document_funsd/huong_giai_quyet.md) | Căn chỉnh words/boxes/tags, vùng ảnh, BiLSTM-CRF; basic bỏ ảnh/char không được model dùng. |

Ưu tiên đúng họ tác vụ, sau đó đúng biến thể cần tạo. Chỉ đọc các mẫu gần nhất và các phần source liên quan; không nạp toàn bộ thư viện mặc định. GEC/pointer-copy có thể gợi ý cách tổ chức cho rewriting/summarization, nhưng không tự chứng minh model đó tối ưu cho dataset khác. Các tác vụ chưa có mẫu như retrieval, dialogue hoặc table extraction dùng playbook và dữ liệu thực; không giả vờ thư viện có ví dụ tương ứng.

## Cách dùng bốn tệp

- `baseline.ipynb`: phong cách optimized, pipeline dữ liệu/model đúng nhiệm vụ, loop trực tiếp và checkpoint inference gọn. Text-only vẫn scratch; ảnh → text theo [pretraining policy](../../nlp-baseline-workflow/references/pretraining_policy.md).
- `baseline_basic.ipynb`: tham khảo riêng cho tiền xử lý tối thiểu và model đơn giản có full training/evaluation. Không lấy BPE/features/augmentation của optimized đưa vào basic.
- `baseline_hardcore.ipynb`: đúng hai phần load dữ liệu và một model scratch nhận raw input rồi inference. Không chép loss, optimizer, training, checkpoint hoặc preprocessing helpers từ hai bản còn lại.
- `huong_giai_quyet.md`: mẫu giải thích theo [năm trọng tâm](../../nlp-solution-architect/references/report_contract.md), gồm lý do tiền xử lý/model và bảng siêu tham số có vai trò, căn cứ, độ nhạy. Notebook-only không phải tạo thêm báo cáo.

## Quy tắc chuyển mẫu

Giữ cách viết tiếng Việt dễ đọc, constants cố định và natural Markdown/code; số cell linh hoạt. Dùng [training style](training_style.md): loop epoch/batch trực tiếp cho pha đơn giản, helper khi có ích; chỉ lưu trạng thái resume khi được yêu cầu. GPU32GB là mục tiêu thiết kế; sklearn có thể chạy CPU, không ép batch/epochs cho sklearn. Không AMP hoặc tự dò VRAM/batch.

Các ví dụ là tài liệu authoring, không phải dependency runtime hay lời bảo đảm chất lượng. Không copy scores, đường dẫn dữ liệu, split, label mapping, model/provenance hoặc assumptions sang bài mới. Ví dụ ABSA thiếu polarity test, FUNSD thiếu linking gold, CoEdIT/COCO/QA có held-out riêng; luôn kiểm lại nguồn thật. Trạng thái kiểm chứng trước đây không phải chứng nhận full training cho mọi mẫu.

Không notebook mới nào được import/execute các file trong thư viện, đọc output khác để lấy checkpoint/vocabulary, hoặc mở workbook/fewshot khi Run All. Áp dụng [notebook contract](notebook_contract.md), [runtime settings](runtime_config.md) và [three-variant contract](../../nlp-baseline-workflow/references/dual_notebook_contract.md). Giải thích lựa chọn tối ưu phải dựa dữ liệu/metric/ngân sách; chưa đo thì ghi đề xuất có cơ sở.
