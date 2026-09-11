# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/9/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
`class_id` = 468, 
`class_name` = "cab", 
`rank` = 1, 
`score` = 0.510915, 
`taxonomy_name` = "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? 
Đây là prediction ở cấp toàn ảnh, không phải prediction cho một object riêng. Checkpoint `yolo11n-cls.pt` xếp hạng các lớp ImageNet-1K cho toàn bộ ảnh; `rank=1` là lớp có model score cao nhất, không phải ground truth đã được con người xác nhận. Score 0.51 nghĩa là model gán khoảng 51% xác suất cho toàn ảnh thuộc lớp "cab" (taxi), so với "minibus" (16.4%), "police_van" (8.6%)... ở các hạng thấp hơn.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Danh sách 1000 lớp do bộ dữ liệu ImageNet-1K (ILSVRC) định nghĩa từ trước — đây là taxonomy cố định mà checkpoint được huấn luyện để phân loại. Model không thể tự tạo lớp mới ngoài danh sách này, dù object thật trong ảnh không khớp lớp nào có sẵn.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
`class_id` là khóa ổn định để đối chiếu chương trình/so khớp dữ liệu chính xác; `class_name` giúp con người đọc hiểu và kiểm tra kết quả; `taxonomy_name` xác định hệ quy chiếu của ID/tên đó. Cùng một `class_id` có thể mang nghĩa khác nhau ở taxonomy khác — ví dụ COCO-80 (dùng cho detection/segmentation trong bài này) có ID và tên lớp hoàn toàn khác ImageNet-1K. Thiếu `taxonomy_name` thì `class_id`/`class_name` dễ bị hiểu nhầm khi trộn dữ liệu từ nhiều nguồn.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline phải nói rõ: ảnh được gán một nhãn duy nhất hay nhiều nhãn (single-label hay multi-label); quy tắc chọn nhãn đại diện nếu chỉ được gán một nhãn (ví dụ chủ thể chiếm diện tích lớn nhất, hoặc là tiêu điểm bố cục); và khi nào cần escalation thay vì để annotator tự suy đoán. Bài lab chỉ đang quan sát một ví dụ single-label classification — không nên mặc định object nổi bật nhất là ground truth nếu guideline chưa quy định điều đó.
- Vì sao model score không phải ground truth?
`score` chỉ là mức độ tự tin của model dựa trên trọng số đã học, dùng để xếp hạng prediction — không phải nhãn đã qua xác nhận của con người. Bằng chứng rõ nhất trong chính bộ evidence: ở sample `kitchen`, rank 1 của model là "gong" với score 0.42, dù ảnh thực chất là một căn bếp ("dining_table" chỉ ở rank 2 với score 0.07) — model tự tin nhưng sai hoàn toàn. Ground truth chỉ được tạo ra qua quy trình gán nhãn/QC của con người theo guideline, không phải bằng cách chép lại output của model. Guide cũng yêu cầu không coi score/confidence là điểm chất lượng của ground truth.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `class_name`: `person`
`score`: `0.912625`
`bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`
`bbox_width`: `113.58`
`bbox_height`: `279.68`
- Diễn giải vị trí box bằng lời:
Box bắt đầu ở khoảng `(385.33, 69.24)` — góc trên bên trái — và kết thúc ở khoảng `(498.92, 348.92)` — góc dưới bên phải. Cạnh trái box cách mép trái ảnh 385px (≈60% chiều rộng 640px), cạnh phải ở ≈78% chiều rộng; theo chiều dọc box kéo dài từ 69px gần như xuống hết ảnh cao 427px. Đây là một hình chữ nhật đứng (cao 279.68px, rộng 113.58px) ở nửa phải, gần góc trên của ảnh. Như vậy box bao quanh một người ở khu vực phía bên phải của ảnh. Tọa độ `xyxy` dùng đơn vị pixel, gốc `(0,0)` ở góc trên bên trái.
- So sánh số prediction ở hai threshold:
Toàn bộ 11 record của `kitchen` trong file đều được sinh ở `score_threshold = 0.35`, nên tại threshold 0.35 có 11 predictions (`person` ×2, `bowl` ×5, `oven` ×2, `cup` ×2). Lọc lại chính output này tại threshold 0.50, chỉ còn 6 predictions đạt ngưỡng: `person` 0.9126, `bowl` 0.7191, `bowl` 0.7013, `oven` 0.6868, `oven` 0.6343, `person` 0.6109 — 5 record bị loại (3 `bowl` score 0.50/0.46/0.38 và 2 `cup` score 0.45/0.38). Đây là phép lọc lại trên output có sẵn, không phải một lần inference riêng. Threshold cao hơn sẽ loại thêm các prediction có score thấp.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Threshold thấp (0.35) cho độ bao phủ (recall) cao hơn — giữ được cả các object nhỏ/khó như 2 cup — nhưng khối lượng review tăng vì nhiều box hơn cần kiểm tra, kể cả các box điểm thấp dễ sai. Threshold cao (0.50) giảm số box cần duyệt (11 → 6, nhàn hơn cho reviewer) nhưng có nguy cơ bỏ sót object thật mà model dự đoán với score thấp. Threshold chỉ là cơ chế lọc prediction để hiển thị, không phải quy tắc quyết định object nào được đưa vào ground truth.
- Đề xuất một quy tắc box chặt:
Box phải bao sát tối đa biên ngoài của phần object nhìn thấy ở cả 4 cạnh, không chừa margin thừa, không cắt vào phần object quan sát được, và không bao gồm vùng nền không cần thiết.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định trước: có annotate phần object bị che/cắt hay chỉ phần nhìn thấy; ngưỡng % diện tích tối thiểu còn nhìn thấy để vẫn được gán nhãn; có gắn nhãn phụ occluded/truncated hay không. Khi không thể xác định chắc chắn biên hoặc độ đầy đủ của object, annotator cần escalation thay vì tự suy đoán — ví dụ câu hỏi escalation: "Object bị cắt khỏi mép ảnh/che khuất một phần thì box chỉ bao phần nhìn thấy hay cần quy tắc đặc biệt?"

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
`instance_id=kitchen-001`, `class_name=person`, `score=0.899318`, `bbox_xyxy=[385.45, 66.44, 498.02, 348.58]`, `polygon_point_count=348`. Một phần `polygon_xy` (3 điểm đầu trong tổng 348 điểm): `[446.0, 70.0], [445.0, 71.0], [444.0, 71.0]` — các điểm pixel nối tiếp nhau bám theo biên ngoài của người trong ảnh.
- Polygon bổ sung chi tiết gì so với box?
Box chỉ mô tả vùng hình chữ nhật bao quanh object bằng 4 tọa độ. Polygon mô tả đường biên thực tế của instance bằng hàng trăm điểm `[x, y]` bám sát đường viền (348 điểm cho `kitchen-001`), nên loại trừ được phần nền/khoảng trống lọt vào trong box mà không thuộc vật thể, và thể hiện được các phần lồi/lõm mà box không thể hiện.
- `instance_id` dùng để làm gì và không phải loại ID nào?
`instance_id` (dạng `kitchen-001`, `kitchen-002`...) dùng để phân biệt từng object riêng lẻ trong cùng một ảnh, kể cả khi nhiều object cùng chung một lớp — ví dụ ảnh kitchen có 5 instance `bowl` khác nhau (`kitchen-002`, `003`, `006`, `010`...), mỗi cái là một instance riêng dù cùng `class_name = "bowl"`. Đây không phải `class_id` (không cho biết vật thể thuộc lớp gì) và không phải tracking ID (không dùng để theo dõi cùng một object xuyên suốt nhiều ảnh/frame khác nhau) — nó chỉ có ý nghĩa nội bộ trong phạm vi một ảnh của output lab này.
- Đề xuất một quy tắc biên mask:
Polygon phải bám theo biên ngoài thực tế của phần object nhìn thấy, không "cắt góc" bỏ qua các chi tiết lồi lõm rõ ràng của biên, và không mở rộng ra ngoài để bao luôn vùng nền hoặc vật thể liền kề. Với các vùng biên không rõ, không nên tự suy đoán.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định: khi hai instance cùng lớp tiếp xúc/chồng lấn nhau, ranh giới giữa hai polygon được xác định thế nào để không bị dính liền thành một mask; khi biên bị mờ/nhòe không thể xác định chính xác từng pixel, annotator có được "làm tròn" theo ước lượng hay phải escalation cho reviewer xác nhận. Hai ví dụ biên thực tế đáng chú ý trong evidence: `kitchen-004 `(`potted plant`, score 0.63) có `bbox_xyxy = [0.05, 1.4, 63.45, 148.89]` — nằm sát cả mép trên lẫn mép trái ảnh; `kitchen-009` (`person`, score 0.43) có `bbox_xyxy = [0.3, 263.27, 59.5, 309.92]` — sát mép trái. Cả hai đều là trường hợp object bị cắt mép cần áp dụng đúng quy tắc trên, và `kitchen-010` (`bowl`, score 0.37, chỉ 12 điểm polygon so với 348 điểm của `kitchen-001`) là dấu hiệu object nhỏ/khó xác định biên, nên được reviewer xem kỹ trước khi chấp nhận thay vì annotator tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh theo taxonomy đã quy định; với multi-label thì một ảnh có thể có nhiều nhãn nếu guideline yêu cầu. | Rank 1 của model trên kitchen là "gong" (score 0.42) — hoàn toàn sai so với nội dung ảnh, minh chứng rõ score không phải chất lượng nhãn | Đối chiếu ảnh với guideline, chọn 1 lớp đại diện đúng; không copy class_name/score của model thành ground truth | Kiểm tra nhãn có đúng taxonomy, đúng phạm vi cấp ảnh, và trường hợp nhiều chủ thể có xử lý nhất quán theo guideline không |
| Phát hiện vật thể | Mỗi object là một đơn vị nhãn gồm class_name và bbox_xyxy/kích thước box. | Nhiều object sát mép ảnh (person x≈0.12, oven x≈0.11); một số box score thấp (0.38–0.46, ví dụ 2 cup nhỏ) dễ bị bỏ sót nếu tăng threshold | Vẽ box sát phần object nhìn thấy; áp dụng đúng rule cho object bị che/cắt; escalation khi không đủ thông tin. | Kiểm tra class, độ chặt của box, có bỏ sót object nào không, và cách xử lý các trường hợp crop/che khuất |
| Instance segmentation | Mỗi instance gồm instance_id, class_name, polygon_xy bám biên riêng | Độ phức tạp polygon chênh lệch lớn giữa các instance (348 điểm ở kitchen-001 so với 12 điểm ở kitchen-010); một số instance nằm sát mép ảnh (kitchen-004, kitchen-009) | Trace polygon theo biên phần nhìn thấy; tách rõ các instance cùng lớp tiếp xúc nhau; escalation khi biên mờ/che khuất | Kiểm tra polygon có bám biên, có ăn nền/cắt object không, và các instance cùng lớp có bị gộp nhầm không |

Trong vòng đời này, prediction của model chỉ được dùng làm evidence để quan sát và kiểm tra, không thay thế ground truth — ground truth luôn đến từ quy trình gán nhãn/QC theo guideline của con người. Rubric cũng yêu cầu nối evidence qua guideline, ground truth, model, QC/rework và tách hành động của annotator với reviewer.

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Chỉ dùng ảnh mẫu COCO được notebook tự tải kèm giấy phép CC BY 2.0 đã ghi rõ trong IMAGE_ATTRIBUTION.md/THIRD_PARTY_NOTICES.md; không đưa ảnh riêng, dữ liệu khách hàng, khuôn mặt/biển số thật, hay bất kỳ thông tin cá nhân (họ tên, MSSV, email, số điện thoại) vào output/báo cáo công khai — các thông tin định danh đó chỉ được phép nằm ở tên repository theo đúng hướng dẫn nộp bài.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: 
Lab Coach/mentor phụ trách bài lab, thay vì tự tiếp tục xử lý hoặc thay thế bằng dữ liệu cá nhân. Hướng dẫn cũng yêu cầu báo mentor nếu gặp dữ liệu/ảnh không đúng phạm vi hoặc lỗi liên quan đến nguồn ảnh.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
