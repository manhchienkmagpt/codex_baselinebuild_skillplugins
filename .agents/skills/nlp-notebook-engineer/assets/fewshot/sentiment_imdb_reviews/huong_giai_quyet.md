# Phân tích bài toán và đề xuất hướng giải quyết

## 1. Giải thích bài toán

Bài toán phân loại cảm xúc nhận một review phim tiếng Anh và trả nhãn **negative** hoặc **positive**. Ví dụ minh họa: “The acting is excellent.” → positive. Không có lớp neutral; nhận xét trung tính hoặc review tiếng Việt nằm ngoài phạm vi dữ liệu đang dùng.

Metric chính **Macro F1** lấy trung bình F1 của hai lớp, nên cần nhận đúng cả khen lẫn chê; điểm 0–1 càng cao càng tốt. Accuracy được báo bổ sung. Nhãn 0 là neg, 1 là pos; không nhầm Macro F1 với F1 chỉ của positive.

## 2. Phân tích dữ liệu và các hướng tiếp cận

[IMDb Reviews trong TFDS](https://www.tensorflow.org/datasets/catalog/imdb_reviews) có text:str và label:int. Theo nguồn/khảo sát trước, có 25.000 train và 25.000 test, mỗi split cân bằng hai lớp; tập unsupervised 50.000 không được dùng trong phương pháp này. Dữ liệu là review phân cực mạnh nên không đại diện mọi sắc thái cảm xúc.

Nội dung review có HTML, từ phủ định, cách viết đa dạng và có thể dài. Baseline tách validation khoảng 20% bằng StratifiedGroupKFold theo nhóm văn bản chuẩn hóa để bản sao không vượt biên train/validation. Lần khảo sát trước ghi nhận 123 nhóm nội dung không rỗng giao nhau giữa official train/test; giữ test chuẩn nhưng không coi đây là đánh giá hoàn toàn sạch leakage. Không có mã phim ổn định để chia theo phim.

Các thống kê trên được giữ từ thông tin nguồn và khảo sát đã ghi nhận trước đây của dự án; không phải điểm mô hình và chưa được thu thập lại trong lần viết báo cáo này.

| Hướng tiếp cận | Điểm phù hợp dữ liệu | Giới hạn cần cân nhắc |
|---|---|---|
| Unigram counts + Naive Bayes | Rất nhẹ và dễ học với bag-of-words | Giả định đặc trưng độc lập, khó giữ cụm phủ định/ngữ cảnh. |
| TF-IDF word/char + NB-ratio + LinearSVC | Tận dụng cụm từ, hình thái và tín hiệu sentiment với dữ liệu cỡ vừa | Dùng RAM CPU, không hiểu sâu mỉa mai hoặc quan hệ dài. |
| Embedding trainable + CNN/BiGRU | Có thể học thứ tự/ngữ cảnh tốt hơn | Cần nhiều lượt học; không tự bảo đảm hơn sparse khi cấm pretrained text. |

## 3. Tiền xử lý dữ liệu tối ưu

TF-IDF biến văn bản thành vector thưa: mỗi chiều là một từ/cụm, trọng số giảm cho từ xuất hiện quá phổ biến trong nhiều văn bản. N-gram là nhóm các đơn vị liền nhau, chẳng hạn bigram là hai từ liên tiếp.

1. NFKC, HTML unescape, đổi thẻ br thành khoảng trắng rồi gộp whitespace. Không xóa stopword hoặc từ phủ định như “not”, vì chúng có thể đảo sentiment.

2. Chia nhóm train/validation trước khi fit; lowercase do vectorizer thực hiện. Không cắt review theo độ dài nên những từ sentiment ở cuối vẫn có thể đóng góp.

3. Fit TF-IDF word unigram/bigram và char_wb n-gram 3–5 trên train. Word bigram giữ cụm như “not good”; character features giúp biến thể từ/cách viết. Giới hạn vocabulary và min_df để kiểm soát nhiễu/chi phí.

4. Tính binary NB log-count ratio từ document presence của train với smoothing, rồi nhân đặc trưng TF-IDF trước LinearSVC. Ratio có thể âm nên không đưa ma trận này vào MultinomialNB. Chỉ transform val/test; mẫu rỗng vẫn được chấm.

Mọi biến đổi có học thống kê/vocabulary chỉ fit trên train. Validation dùng chọn cấu hình/checkpoint; held-out chỉ dùng đánh giá sau khi chốt, không dùng để sửa mô hình hoặc ép tăng điểm.

## 4. Lựa chọn mô hình tối ưu

Chọn **NB-SVM sparse**: TF-IDF word/char → log-count ratio giám sát nhị phân → LinearSVC. LinearSVC học biên phân tách giữa positive và negative trên đặc trưng thưa; không có embeddings, tokenizer hoặc encoder pretrained. Hai nhánh có weight word=1.0, char=0.5 để character features bổ sung thay vì lấn át word features.

Review sentiment thường có tín hiệu từ/cụm rõ, nên một bộ phân loại tuyến tính với biểu diễn tốt là lựa chọn có căn cứ cho khoảng 20 nghìn mẫu train học và chính sách scratch. Không cần ép dùng neural network vì có GPU; phương pháp này chạy CPU. Giới hạn là bag-of-features không biểu diễn đầy đủ mỉa mai và cấu trúc dài.

**Bằng chứng đã được ghi nhận:** lần chạy đầy đủ trước đây của optimized trên cùng phương pháp/giao thức (20.000 train học / 5.000 validation / 25.000 test) chọn C=4 trong ba ứng viên theo Macro F1 validation. Kết quả này được giữ từ báo cáo trước, không phải lần huấn luyện mới trong lần viết này.

| Tập đánh giá | Macro F1 | Accuracy |
|---|---:|---:|
| Validation | 0.914386 | 0.914400 |
| Held-out test | 0.905360 | 0.905360 |

Đây là kết quả một seed/giao thức đã ghi nhận, không phải chứng minh tốt nhất trên mọi mô hình, mọi C hoặc nguồn dữ liệu khác. Không dùng test để chọn lại C.

## 5. Siêu tham số và lý do lựa chọn

Siêu tham số là giá trị người thiết kế đặt trước, khác với trọng số được model học từ dữ liệu.

Các giá trị dưới đây khớp `baseline.ipynb`. C=4 là tốt nhất đã đo **trong lưới ba C ở lần chạy đã ghi nhận**; các trần features và tham số solver còn lại là cấu hình có cơ sở, chưa được chứng minh tối ưu riêng bằng ablation.

| Siêu tham số | Giá trị trong baseline | Vai trò | Căn cứ lựa chọn | Ảnh hưởng khi tăng/giảm |
|---|---|---|---|---|
| `WORD_MAX_FEATURES` | 120000 | Trần word features | 120.000 giữ các unigram/bigram phổ biến trong train mà chặn tăng chiều quá lớn. | Tăng giữ thêm cụm hiếm và tốn RAM; giảm có thể mất dấu hiệu phân loại. |
| `CHAR_MAX_FEATURES` | 100000 | Trần character features | 100.000 bổ sung hình thái/cách viết nhưng cần giới hạn nhánh có rất nhiều n-gram. | Tăng tăng RAM/nhiễu; giảm mất khả năng hỗ trợ OOV. |
| `MIN_DF` | 2 | Số document train tối thiểu | 2 bỏ feature chỉ xuất hiện ở một văn bản; là thỏa hiệp giữa giảm nhiễu và giữ đặc trưng hiếm. | Tăng lọc mạnh hơn nhưng có thể mất tên/chủ đề hiếm; giảm tăng chiều/nhiễu. |
| `NB_RATIO_ALPHA` | 1 | Smoothing log-ratio | 1.0 tránh chia/log của đếm bằng0 khi một feature chỉ hiện ở một sentiment. | Tăng làm ratio bớt cực đoan; giảm tăng nhạy với feature hiếm. |
| `C_VALUES` | (0.25, 1, 4) | Ứng viên điều chuẩn SVM | Đây là ba ứng viên được thử trên validation, không phải ba giá trị đồng thời áp vào một model. | C lớn giảm điều chuẩn và có thể quá khớp; C nhỏ tăng điều chuẩn nhưng có thể thiếu khả năng phân tách. |
| `MAX_ITER` | 10000 | Trần lặp solver | 10.000 cho solver cơ hội hội tụ; nếu chưa hội tụ, code báo lỗi thay vì coi nghiệm đó là tối ưu. | Tăng chỉ giúp khi solver cần thêm bước; giảm có thể dừng ở nghiệm chưa hội tụ. |
| `TOLERANCE` | 0.0001 | Ngưỡng hội tụ solver | 1e-4 là tiêu chí dừng số học của baseline, không phải bằng chứng về Macro F1 cao hơn. | Tăng dừng sớm hơn nhưng nghiệm kém chính xác hơn; giảm tốn nhiều bước. |
| Word/char n-gram | word=(1,2); char_wb=(3,5) | Phạm vi ngữ cảnh đặc trưng | Bigram giữ cụm, char3–5 giữ biến thể trong từ; đây là lựa chọn biểu diễn của code. | Phạm vi rộng hơn tăng chiều/nhiễu; hẹp hơn mất cụm/hình thái. |
| Trọng số nhánh | word=1.0; char=0.5 | Kết hợp hai biểu diễn | Giảm mức ảnh hưởng character features khi chúng chỉ là tín hiệu bổ sung. | Tăng char weight có thể giúp OOV nhưng cũng tăng shortcut; giảm làm gần word-only. |

Cần giữ nguyên split/metric khi đánh giá thêm một thay đổi như bỏ char hoặc thay min_df, để biết lợi ích đến từ đâu. Không gọi MAX_ITER hoặc việc chạy CPU là nguyên nhân bảo đảm F1 tối ưu.
