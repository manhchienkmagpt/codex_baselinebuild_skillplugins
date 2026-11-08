# Bộ skill NLP và Computer Vision cho Codex

Hai plugin giúp tạo notebook và báo cáo giải bài toán bằng tiếng Việt:

| Plugin | Dùng cho |
| --- | --- |
| `nlp-baseline-kit` | NLP, OCR, captioning, VQA và tác vụ đầu ra ngôn ngữ |
| `cv-baseline-kit` | Phân loại ảnh, detection, segmentation, keypoints, restoration và video/tracking |

## Cài vào Codex lần đầu

Chuẩn bị Codex CLI đã đăng nhập và tải đầy đủ dự án, gồm cả các thư mục ẩn `.agents`, thư mục `plugins` và tài nguyên đi kèm. Mở PowerShell tại thư mục gốc dự án rồi chạy:

```powershell
codex plugin marketplace add .
codex plugin add nlp-baseline-kit@olympic-ai-nlp
codex plugin add cv-baseline-kit@olympic-ai-nlp
codex plugin list --marketplace olympic-ai-nlp
```

Chỉ cần NLP hoặc CV thì cài plugin tương ứng. Marketplace được khai báo sẵn tại [`.agents/plugins/marketplace.json`](.agents/plugins/marketplace.json); không cần tự copy từng skill.

Sau khi cài, mở **phiên Codex mới** để sử dụng skill. Nếu CLI chưa có lệnh `plugin`, cập nhật Codex CLI trước.

Tài liệu tham khảo: [lệnh Codex](https://learn.chatgpt.com/docs/developer-commands#codex-plugin) · [hướng dẫn plugin chính thức](https://learn.chatgpt.com/docs/plugins).

## Sử dụng

Trong hội thoại Codex, gọi workflow và gửi mô tả bài toán, đường dẫn/schema dữ liệu, split và metric mong muốn.

### Ví dụ NLP

```text
Dùng $nlp-baseline-workflow tạo baseline cho phân loại cảm xúc.
Dữ liệu: data/reviews.csv; cột text và label; metric chính: macro-F1.
```

### Ví dụ Computer Vision

```text
Dùng $cv-baseline-workflow tạo baseline phân loại ảnh.
Dữ liệu: data/images/ chia sẵn train/val/test theo thư mục nhãn.
Metric chính: accuracy.
```

### Sử dụng với bảng bài toán

Chuẩn bị bảng mô tả bài toán như ví dụ dưới đây, sau đó gửi **một hoặc nhiều dòng** cùng lời gọi skill trong Codex. Có thể sao chép các ô từ Excel hoặc dán bảng Markdown.

| STT | Nhóm bài toán | Bài toán cụ thể | Dataset khuyến nghị | Mức ưu tiên nguồn | Thư viện / cách lấy | Link nguồn | Lệnh tải / khởi tạo | Split có sẵn | Metric chính | Lưu ý đánh giá & tiền xử lý | Mức độ triển khai |
| :---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Image-to-Text | OCR (Optical Character Recognition) | IAM-line (Teklia) | Hugging Face | HF datasets | [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | `load_dataset("Teklia/IAM-line")` | train / validation / test | CER ↓; WER ↓ | OCR chữ viết tay tiếng Anh ở mức dòng; không đại diện OCR tiếng Việt hoặc văn bản in. | Dễ |
| 2 | Image-to-Text | Image Captioning | MS COCO Captions 2014 | Thư viện | torchvision.datasets | [CocoCaptions](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html) | `CocoCaptions(root="val2014/", annFile="annotations/captions_val2014.json")` | train2014 / val2014; test COCO không mở nhãn | CIDEr ↑ | Torchvision chỉ đọc dữ liệu; phải tải ảnh và COCO captions annotations riêng; cài pycocotools. | Cần tải thủ công |
| 3 | Image-to-Text | VQA (Visual Question Answering) | VQAv2 (LMMs-Lab Encoder) | Hugging Face | HF datasets | [lmms-lab-encoder/VQAv2](https://huggingface.co/datasets/lmms-lab-encoder/VQAv2) | `load_dataset("lmms-lab-encoder/VQAv2", split="validation", streaming=True)` | train / validation / test (test không có đáp án công khai) | VQA Accuracy ↑ | Dung lượng ảnh rất lớn; dùng streaming hoặc chọn validation để tự đánh giá có nhãn. | Nặng / streaming |
| 4 | Image-to-Text | Document Understanding / Form Parsing | FUNSD (nielsr) | Hugging Face | HF datasets | [nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) | `load_dataset("nielsr/funsd")` | train (149) / test (50) | Entity F1 ↑; liên kết key-value F1 ↑ khi có liên kết | Bản HF có image, words, bboxes, ner_tags; chưa cung cấp đầy đủ liên kết câu hỏi–đáp án. Muốn K-V linking xem bản gốc. | Dễ |

Dùng `$nlp-baseline-workflow` cho bài toán NLP và Image-to-Text trong bảng trên; dùng `$cv-baseline-workflow` cho các bài toán CV như phân loại ảnh, detection hoặc segmentation.

Ví dụ tin nhắn gửi trong Codex để xử lý hai bài đầu:

> `$nlp-baseline-workflow`
>
> | STT | Nhóm bài toán | Bài toán cụ thể | Dataset khuyến nghị | Mức ưu tiên nguồn | Thư viện / cách lấy | Link nguồn | Lệnh tải / khởi tạo | Split có sẵn | Metric chính | Lưu ý đánh giá & tiền xử lý | Mức độ triển khai |
> | :---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | 1 | Image-to-Text | OCR (Optical Character Recognition) | IAM-line (Teklia) | Hugging Face | HF datasets | [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) | `load_dataset("Teklia/IAM-line")` | train / validation / test | CER ↓; WER ↓ | OCR chữ viết tay tiếng Anh ở mức dòng; không đại diện OCR tiếng Việt hoặc văn bản in. | Dễ |
> | 2 | Image-to-Text | Image Captioning | MS COCO Captions 2014 | Thư viện | torchvision.datasets | [CocoCaptions](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html) | `CocoCaptions(root="val2014/", annFile="annotations/captions_val2014.json")` | train2014 / val2014; test COCO không mở nhãn | CIDEr ↑ | Torchvision chỉ đọc dữ liệu; phải tải ảnh và COCO captions annotations riêng; cài pycocotools. | Cần tải thủ công |

### Chọn skill

Mỗi plugin có ba skill:

| Skill | Phạm vi |
| --- | --- |
| `*-baseline-workflow` | Toàn bộ quy trình: notebook, báo cáo và hai file prompt |
| `*-notebook-engineer` | Tạo hoặc chỉnh sửa notebook và file prompt của biến thể tương ứng |
| `*-solution-architect` | Phân tích bài toán và viết báo cáo |

Có thể yêu cầu riêng một biến thể hoặc chỉ báo cáo.

## Tệp đầu ra

Mặc định mỗi bài tạo sáu tệp:

| Tệp | Nội dung |
| --- | --- |
| `baseline.ipynb` | Bản tối ưu |
| `baseline_basic.ipynb` | Bản cơ bản, có huấn luyện |
| `baseline_hardcore.ipynb` | Mô hình scratch chưa huấn luyện, chỉ suy luận |
| `huong_giai_quyet.md` | Báo cáo hướng giải quyết bằng tiếng Việt |
| `baseline_basic_promting.md` | Prompt nâng cấp tiền xử lí, mô hình và siêu tham số từ code basic |
| `baseline_hardcore_promting.md` | Prompt bổ sung dữ liệu, tiền xử lí, mô hình, siêu tham số và huấn luyện từ hardcore |

### Sử dụng hai file prompt

Định dạng 4 nhóm theo công việc cần thay đổi để triển khai **ý tưởng của baseline tối ưu**. Basic tập trung vào tiền xử lí, mô hình và siêu tham số, tận dụng pipeline huấn luyện sẵn có. Hardcore thêm load/chia dữ liệu và loss, optimizer, training loop. Nhóm lớn có thể tách thành các phần việc hoàn chỉnh như encoder và model sử dụng encoder; không đi lần lượt từng hàm phụ hoặc cắt ngang thân hàm.

Mỗi prompt dùng một chat mới. Người dùng chỉ cần ý tưởng cơ bản và nguồn đính kèm, không phải nhớ hoặc tự ghép danh sách yêu cầu kỹ thuật. Guide phân biệt **Code cần sửa**, **Code liên quan** và **Nguồn tham khảo**; luôn dùng code hiện tại sau khi chỉnh. Nguồn ưu tiên đoạn báo cáo liên quan và bổ sung code tối ưu khi mô tả chưa đủ. Nội dung được đính kèm ngay trong guide để chatbot ở chat mới đọc được; đường dẫn chỉ phục vụ đối chiếu.

Với code có sẵn, lượt phân tích đọc code và nguồn theo ý tưởng, tự rút tên/chữ ký cần giữ và các ràng buộc, đồng thời đánh dấu điều chưa rõ hoặc lỗi nghi ngờ. Lượt sửa chỉ nhận ý tưởng ngắn, kết quả phân tích, code cần sửa và context; dùng câu “Giữ tương thích với code hiện tại; báo trước nếu cần thay đổi thêm.” Module tạo mới dùng ý tưởng, nguồn tham khảo và context trực tiếp. Tên/chữ ký cần giữ lấy từ code hiện tại; code mẫu cung cấp phương pháp. Nguồn đính kèm bảo toàn dữ liệu/version, split, ID, nhãn, metric và chính sách pretrained. Tham số tham khảo là gợi ý để chọn phù hợp và dùng nhất quán.

Giao diện ngắn, ổn định được đưa trực tiếp vào context. Chỉ tạo prompt tóm tắt khi có phần dùng kết quả: giữ chữ ký, I/O (shape/mask nếu có), cấu hình, trạng thái, ràng buộc và logic xử lý chính để sinh module tương tự hoặc viết caller. Không giới hạn cứng số dòng/token, không yêu cầu ngắn hơn input hoặc bắt người dùng nêu bước tiếp theo. Bỏ các lượt riêng cho hàm phụ không đổi, imports, fingerprint, metric, xuất kết quả và reload.

Ngân sách an toàn là 1.800 token mỗi phiên trong trần `input + output < 2.000`. Lượt trích ràng buộc và tóm tắt mặc định phân bổ `input=1100; output=500; reserve=200`; output có thể dài hơn input. Nếu không đủ chỗ, chia theo thành phần hoàn chỉnh thay vì bỏ thông tin cần thiết. Web chatbot dùng Antigravity CLI không có trần output riêng; các ngân sách là dự kiến phân bổ. Kiểm tra offline không xác nhận token thực tế hoặc code do chatbot sinh đã chạy thành công.

Dùng `migrate_prompt_guides.py <task_dir>` để xem trước chuyển định dạng 1–3 sang 4; thêm `--write` để lưu. `--refresh` tái tạo định dạng 4 từ notebook/báo cáo hiện tại. Notebook và báo cáo được giữ nguyên; hardcore gốc vẫn chỉ suy luận với trọng số chưa huấn luyện, còn guide hướng dẫn bổ sung huấn luyện.

Workflow đầy đủ tạo sáu file. Gọi notebook engineer hoặc chọn riêng basic/hardcore sẽ có file prompt tương ứng, trừ khi yêu cầu rõ chỉ xuất notebook. Chỉ báo cáo hoặc chỉ optimized không cần tạo prompt mới. Các bài đã sinh không được tự cập nhật hàng loạt.

Kiểm tra hai file bằng `python <workflow_skill>/scripts/validate_prompt_guides.py <task_dir>`; thêm `--variant basic` hoặc `--variant hardcore` khi chỉ chọn một biến thể. Kiểm tra offline không xác nhận token Gemini thực tế.

### Vị trí lưu

| Nhóm bài toán | Thư mục mặc định |
| --- | --- |
| NLP | `outputs/<STT>_<Tên bài toán>/` |
| Computer Vision | `outputs/cv/<STT>_<Tên bài toán>/` |

Ví dụ:

```text
outputs/
├── 14_Intent Classification/
└── cv/
    └── 1_Image Classification/
```

### Quy ước đánh số và đặt tên

Quy ước được áp dụng kể cả trong phiên chat mới không nhắc lại cách đặt tên:

- **Có STT trong yêu cầu hoặc bảng:** giữ nguyên STT.
- **Không có STT:** dùng lại thư mục cùng bài toán/dataset; với bài mới, lấy số tiếp theo trong thư mục đầu ra, bắt đầu từ 1 khi chưa có.
- **Có đường dẫn do người dùng chỉ định:** ưu tiên đường dẫn đó.
- **Thư mục đã tồn tại:** không tự đổi tên.

Chi tiết: [quy ước NLP](.agents/skills/nlp-baseline-workflow/references/output_naming.md) · [quy ước CV](.agents/skills/cv-baseline-workflow/references/output_naming.md).

## Mẫu tham khảo

Fewshot được đóng gói sẵn trong plugin:

- [Mẫu NLP](plugins/nlp-baseline-kit/skills/nlp-notebook-engineer/assets/fewshot/)
- [Mẫu CV](plugins/cv-baseline-kit/skills/cv-notebook-engineer/assets/fewshot/)

Mẫu thị giác hiện lấy từ captioning/OCR/FUNSD để tham khảo thành phần ảnh, chưa có mẫu CV thuần hoặc video.

## Quản lý tệp với Git

Git bỏ qua:

- Thư mục `outputs/` và `artifacts/`.
- Bản sao `fewshot/` ngoài cùng.
- Môi trường Python, cache và tệp tạm.

Tài nguyên cần để cài plugin và dùng skill trong `.agents/` và `plugins/` vẫn được giữ lại.
