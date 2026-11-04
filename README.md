# Bộ skill NLP và Computer Vision cho Codex

Hai plugin giúp tạo notebook và báo cáo giải bài toán bằng tiếng Việt:

| Plugin | Dùng cho |
|---|---|
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

Chỉ cần NLP hoặc CV thì cài plugin tương ứng. Marketplace được khai báo sẵn tại [`.agents/plugins/marketplace.json`](.agents/plugins/marketplace.json); không cần tự copy từng skill. Cú pháp theo [tài liệu lệnh Codex](https://learn.chatgpt.com/docs/developer-commands#codex-plugin).

Sau khi cài, mở **phiên Codex mới** để sử dụng skill. Nếu CLI chưa có lệnh `plugin`, cập nhật Codex CLI trước. Xem [hướng dẫn plugin chính thức](https://learn.chatgpt.com/docs/plugins).

## Sử dụng

Trong hội thoại Codex, gọi workflow và gửi mô tả bài toán, đường dẫn/schema dữ liệu, split và metric mong muốn. Ví dụ:

```text
Dùng $nlp-baseline-workflow tạo baseline cho phân loại cảm xúc.
Dữ liệu: data/reviews.csv; cột text và label; metric chính: macro-F1.
```

```text
Dùng $cv-baseline-workflow tạo baseline phân loại ảnh.
Dữ liệu: data/images/ chia sẵn train/val/test theo thư mục nhãn.
Metric chính: accuracy.
```

Mỗi plugin có ba skill: `*-baseline-workflow` cho toàn bộ quy trình, `*-notebook-engineer` cho notebook và `*-solution-architect` cho báo cáo. Có thể yêu cầu riêng một biến thể hoặc chỉ báo cáo.

Mặc định mỗi bài tạo `baseline.ipynb` (tối ưu), `baseline_basic.ipynb` (cơ bản, có training), `baseline_hardcore.ipynb` (model scratch chưa train, chỉ inference) và `huong_giai_quyet.md`. Kết quả NLP ở `outputs/<task_slug>/`, CV ở `outputs/cv/<task_slug>/`; thư mục `outputs/` được Git bỏ qua.

Fewshot được đóng gói sẵn trong plugin: [mẫu NLP](plugins/nlp-baseline-kit/skills/nlp-notebook-engineer/assets/fewshot/) và [mẫu CV](plugins/cv-baseline-kit/skills/cv-notebook-engineer/assets/fewshot/). Mẫu thị giác hiện lấy từ captioning/OCR/FUNSD để tham khảo thành phần ảnh, chưa có mẫu CV thuần hoặc video.

Git bỏ qua `outputs/`, `artifacts/`, bản sao `fewshot/` ngoài cùng, môi trường Python, cache và tệp tạm. Tài nguyên cần để cài plugin và dùng skill trong `.agents/` và `plugins/` vẫn được giữ lại.
