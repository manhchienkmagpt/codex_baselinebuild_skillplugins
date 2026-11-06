# Fewshot thị giác chọn từ outputs cho CV

Trong outputs hiện có chưa có bài CV thuần classification, image-pair, detection, segmentation, keypoints, restoration, video/action recognition hoặc tracking. Ba nhóm dưới đây là **NLP/image-to-text có thành phần ảnh**, được copy từ outputs để tham khảo cách tổ chức và các thành phần thị giác. Chúng không phải ví dụ đã giải đúng một tác vụ CV thuần, không được đổi nhãn thành classification/detection chỉ để gọi là CV.

Skill/plugin có bản đóng gói riêng trong assets/fewshot; không cần truy cập folder outputs hoặc fewshot gốc khi authoring ở máy khác. Metadata nguồn `nlp_baseline` được giữ nguyên để không xuyên tạc loại tác vụ. Notebook mới cho CV phải khai báo `cv_baseline` và task_type thật theo contract CV.

## Mẫu có sẵn và giới hạn chuyển giao

| Nhóm mẫu | Tệp tham khảo | Phần nên học |
|---|---|---|
| Mẫu chính cho pipeline ảnh | [optimized](../assets/fewshot/caption_coco2014/baseline.ipynb) · [basic](../assets/fewshot/caption_coco2014/baseline_basic.ipynb) · [hardcore](../assets/fewshot/caption_coco2014/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/caption_coco2014/huong_giai_quyet.md) | Tham khảo RGB/resize/tensor, Dataset/collation, encoder ResNet50 có freeze/fine-tune, loop trực tiếp và checkpoint gọn. Bản basic minh họa CNN scratch và chuyển ảnh tối thiểu. |
| Mẫu bổ sung cho hình học/độ dài ảnh | [optimized](../assets/fewshot/ocr_iam_line/baseline.ipynb) · [basic](../assets/fewshot/ocr_iam_line/baseline_basic.ipynb) · [hardcore](../assets/fewshot/ocr_iam_line/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/ocr_iam_line/huong_giai_quyet.md) | Tham khảo giữ tỷ lệ, padding ảnh rộng khác nhau, đọc dữ liệu và truyền length. CTC/alphabet/decoder là phần OCR, không áp vào classification/detection/segmentation một cách máy móc. |
| Mẫu bổ sung cho ảnh và tọa độ | [optimized](../assets/fewshot/document_funsd/baseline.ipynb) · [basic](../assets/fewshot/document_funsd/baseline_basic.ipynb) · [hardcore](../assets/fewshot/document_funsd/baseline_hardcore.ipynb) · [báo cáo](../assets/fewshot/document_funsd/huong_giai_quyet.md) | Tham khảo đồng bộ ảnh/boxes/thực thể, mask và trích vùng ảnh. Đây là entity recognition có OCR words, không phải detector/segmenter hoặc tracker. |

Chọn captioning làm mẫu chung cho image Dataset/encoder/loop; chỉ mở thêm OCR khi cần học cách giữ tỷ lệ/padding, FUNSD khi cần đối chiếu tọa độ và ảnh. Không đọc mọi file mặc định. Các phần sinh văn bản, tokenization, CTC, entity labels và language metrics thuộc tác vụ nguồn, không phải loss/head/output/metric của bài CV mới.

## Chọn đúng biến thể

- `baseline.ipynb`: tham khảo freeze/fine-tune visual encoder, cấu trúc loop/checkpoint. CV vẫn theo [backbone-only policy](../../cv-baseline-workflow/references/pretraining_policy.md), không được lấy một pretrained detector/segmenter/tracker hoàn chỉnh.
- `baseline_basic.ipynb`: dùng phần chuyển ảnh tối thiểu và model scratch phù hợp; với FUNSD basic chỉ dùng word/layout nên không phải mẫu pipeline ảnh. Không áp letterbox/CTC như một quy tắc cho mọi bài CV.
- `baseline_hardcore.ipynb`: tham khảo hai phần raw loading + model scratch có chuyển input tối thiểu bên trong và inference trực tiếp; không có training/checkpoint/preprocessing helpers. CV phải tự tạo output schema và raw-input model phù hợp tác vụ thật.
- `huong_giai_quyet.md`: tham khảo phong cách [báo cáo năm phần](../../cv-solution-architect/references/report_contract.md) và cách giải thích siêu tham số, không chép nội dung nhiệm vụ/metric NLP vào báo cáo CV.

## Khi chưa có mẫu đúng họ CV

Theo [CV task guide](../../cv-baseline-workflow/references/cv_tasks.md) và dữ liệu/schema thực để chọn model, target encoding, loss, metric và split. Detection cần boxes đúng tọa độ; segmentation cần mask và ignore index; keypoints cần visibility/geometry; video/tracking cần grouping theo sequence và thứ tự frame. Các mẫu ảnh → text không kiểm chứng các hành vi này. Không giả lập một mẫu video hoặc tuyên bố đã thử dataset CV chỉ vì static validation qua.

Giữ [notebook contract](notebook_contract.md), [fixed runtime settings](runtime_config.md), [training style](training_style.md) và [three-variant contract](../../cv-baseline-workflow/references/dual_notebook_contract.md). Loop trực tiếp khi đơn giản; checkpoint inference gọn; basic/hardcore scratch; không AMP/hardware adaptation. Không copy đường dẫn, nhãn, số đo hoặc split nguồn như mặc định chung. Fewshot chỉ là authoring reference, không bao giờ là runtime dependency của notebook mới.

## Hai file prompt đi kèm

Mỗi nhóm mẫu đóng gói có thêm `baseline_basic_promting.md` và `baseline_hardcore_promting.md`, theo [hợp đồng prompt](../../cv-baseline-workflow/references/prompting_contract.md). Basic chỉ vị trí đoạn code cần sửa; hardcore dùng mô tả và tóm tắt interface để xây dựng tới phương pháp tối ưu trong các phiên độc lập. Các ngân sách là hạn mức chưa đo bằng Gemini, không phải chứng nhận token thực tế. Đọc mẫu đúng họ tác vụ, không sao chép prompt chung sang dữ liệu khác. Các mẫu OCR/captioning/FUNSD giữ nguyên nhiệm vụ NLP nguồn kể cả khi được đóng gói trong bộ CV; chúng không phải mẫu detector/segmenter/tracker.
