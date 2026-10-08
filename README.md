# Chọn vị trí đeo cảm biến cho nhận dạng hoạt động, kiểm chứng lời giải thích bằng huấn luyện lại (MHEALTH)

*Choosing Body Positions for Wearable Activity Recognition by Retraining-validated Explanations on MHEALTH with Ten Classifiers*

Assignment 2, môn ADY201m.

---

## 1. Bài toán

Ứng dụng sức khỏe di động thường yêu cầu người dùng **đeo nhiều cảm biến** (ngực, cổ tay, cổ chân…) để nhận dạng họ đang đi bộ, chạy, ngồi hay nằm. Nhưng mỗi thiết bị đeo thêm làm người dùng dễ bỏ cuộc hơn. Vì vậy người thiết kế muốn biết: **có thể bỏ bớt vị trí đeo nào mà không mất nhiều độ chính xác?**

Cách hay được dùng là nhìn biểu đồ giải thích mô hình (SHAP, LIME…) rồi bỏ cảm biến bị chấm "ít quan trọng". Cách này có hai vấn đề:

1. **Lời giải thích hiếm khi được kiểm chứng.** Phép thử phổ biến là che đặc trưng (thay bằng giá trị trung bình) rồi xem điểm giảm bao nhiêu. Khi đó mô hình nhận dữ liệu khác hẳn lúc học, nên điểm sụt rất mạnh và tầm quan trọng bị **phóng đại**.
2. **Nhiều bài trên MHEALTH chia dữ liệu ngẫu nhiên theo cửa sổ**, nên cùng một người có mặt ở cả tập huấn luyện lẫn tập kiểm thử, và điểm gần như tuyệt đối. Ngoài đời, mô hình luôn phải đoán cho **người mới chưa từng gặp**.

Đề tài này kiểm chứng lời giải thích bằng **huấn luyện lại** (ROAR, *Remove And Retrain*), và đánh giá bằng **chia theo người**.

## 2. Dữ liệu

[MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) (Banos et al., 2014), UCI Machine Learning Repository, giấy phép CC BY 4.0.

- 10 người, 12 hoạt động: đứng, ngồi, nằm, đi bộ, leo cầu thang, cúi người, giơ tay, gập gối, đạp xe, chạy chậm, chạy nhanh, nhảy.
- Cảm biến 50 Hz ở 3 vị trí: **ngực** (gia tốc + 2 đạo trình điện tim ECG), **cổ chân trái** và **cổ tay phải** (gia tốc, con quay, từ kế).
- 1,215,745 dòng thô, khoảng 28% có nhãn hoạt động.
- Dữ liệu được chia thành cửa sổ 2 giây, bước 0.2 giây, mỗi cửa sổ tính trung bình và độ lệch chuẩn của từng kênh. Kết quả là **33,148 cửa sổ** và **42 đặc trưng**. Từ kế bị bỏ vì phụ thuộc phòng đo.

Dataset **không có trong repo**; Notebook A tự tải về từ UCI.

## 3. Ba câu hỏi nghiên cứu

| | Câu hỏi | Cách trả lời |
|---|---|---|
| **RQ1** | Vị trí nào chuyển động trong hoạt động nào, và các người khác nhau bao nhiêu? | SQL (DuckDB), hệ số biến thiên giữa người, kiểm định Friedman |
| **RQ2** | Trong 10 mô hình thuộc 4 họ, mô hình nào nhận dạng tốt nhất khi gặp **người mới**? | Leave-one-subject-out: lần lượt để riêng 1 trong 10 người làm tập kiểm thử. Đo macro-F1, có mốc naive và kiểm định Wilcoxon. |
| **RQ3** | Đặc trưng mà 5 phương pháp giải thích cho là quan trọng có thật sự quan trọng khi **huấn luyện lại**? Vị trí đeo được lời giải thích chọn có phải vị trí tốt nhất không? | ROAR với TreeSHAP, KernelSHAP, LIME, Permutation, Gain; huấn luyện lại với một vị trí hoặc bỏ một vị trí; độ ổn định xếp hạng giữa hai nhóm người |

**10 mô hình:** Logistic, Linear SVM, k-NN, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, LightGBM, XGBoost, MLP nhỏ.

## 4. Kết quả chính

### RQ1: Mỗi vị trí "thấy" một nhóm hoạt động
| Hoạt động | Ngực | Cổ chân | Cổ tay |
|---|---|---|---|
| đứng / ngồi / nằm | 0.14–0.16 | 0.10 | 0.10–0.12 |
| giơ tay | 0.31 | **0.11** | **2.75** |
| đi bộ | 2.20 | **5.20** | 2.21 |
| chạy nhanh | 8.58 | 7.01 | 6.70 |

*(Độ lệch chuẩn của độ lớn gia tốc, m/s²; càng lớn càng chuyển động mạnh.)*

- Cổ chân tách nhóm di chuyển, cổ tay tách các động tác tay.
- Khác biệt giữa người lớn nhất ở các động tác tự do: cúi người có CV 33%, giơ tay 27%.

### RQ2: Năm mô hình đứng đầu ngang nhau
| Mô hình | macro-F1 (TB ± sd trên 10 người) |
|---|---|
| Extra Trees | 0.962 ± 0.048 |
| XGBoost | 0.958 ± 0.050 |
| Logistic | 0.955 ± 0.043 |
| HistGB | 0.951 ± 0.049 |
| Random Forest | 0.950 ± 0.056 |
| LightGBM | 0.933 ± 0.053 |
| Mốc naive (đoán lớp nhiều nhất) | 0.014 |

Chênh lệch giữa các mô hình đứng đầu (dưới 0.012) nhỏ hơn nhiều so với chênh lệch giữa các người (khoảng 0.05). Kiểm định Wilcoxon giữa Extra Trees và XGBoost cho p = 0.945. **Chọn mô hình nào không quan trọng bằng việc gặp người khác.** Lỗi lớn nhất đến từ hai người có thói quen khác số đông (xem `report/table_diagnosis.csv`):
- Người 3 ngồi với tay buông thõng, nên bị đoán là đứng.
- Người 6 chạy chậm mạnh như người khác chạy nhanh.

### RQ3: Che đặc trưng phóng đại tầm quan trọng
Bỏ 10 đặc trưng mà phương pháp giải thích xếp đầu (điểm gốc là 0.947):

| Phương pháp | Che đặc trưng, không học lại | Huấn luyện lại (ROAR) |
|---|---|---|
| TreeSHAP | 0.370 | 0.932 |
| KernelSHAP | 0.445 | 0.912 |
| LIME | 0.312 | 0.933 |
| Permutation | 0.536 | 0.907 |
| Gain | 0.373 | 0.948 |
| Bỏ ngẫu nhiên | 0.942 | 0.950 |

- Che đặc trưng làm điểm sập xuống 0.3–0.5, trong khi huấn luyện lại chỉ mất tối đa 0.04. Lý do: các cảm biến chứa thông tin trùng nhau, nên mô hình học lại được từ phần còn lại.
- ROAR **không phân biệt được** phương pháp giải thích nào trung thực hơn.

### RQ3: Bỏ được thiết bị ở ngực
| Cấu hình (LightGBM, leave-one-subject-out) | macro-F1 | Người kém nhất |
|---|---|---|
| Đủ 3 vị trí | 0.934 | 0.867 |
| **Bỏ thiết bị ngực (gia tốc + ECG)** | **0.894** | 0.809 |
| Bỏ cổ tay | 0.865 | 0.748 |
| Bỏ cổ chân | 0.851 | 0.742 |
| Chỉ cổ chân / chỉ ngực / chỉ cổ tay | 0.792 / 0.786 / 0.749 | 0.53–0.64 |
| Chỉ ECG | 0.118 | 0.053 |

- Lời giải thích xếp cổ chân khoảng 40%, cổ tay khoảng 36%, ngực khoảng 23%, ECG khoảng 1%. Thứ tự này **khớp với thí nghiệm bỏ một vị trí**.
- **Kết luận thực tế:** đeo cổ chân và cổ tay, bỏ thiết bị ngực, chỉ mất khoảng 0.04 macro-F1. Không nên dùng một cảm biến duy nhất, vì có người chỉ còn được nhận đúng khoảng một nửa số hoạt động.

## 5. Giới hạn

- Chỉ có 10 người, nên chênh lệch dưới khoảng 0.03 không đủ tin cậy để kết luận.
- ROAR làm trên một cách chia (người 8–10) và một mô hình (LightGBM). LightGBM được dùng vì TreeSHAP và Gain cần mô hình cây tăng cường.
- Đặc trưng thống kê đơn giản, không có cảm biến ở đùi, nên ngồi và đứng dễ nhầm.
- Đặc trưng ECG là độ lệch chuẩn điện áp, **không phải nhịp tim**.
- Tham số mô hình được giữ cố định, chưa tinh chỉnh.

## 6. Chạy lại

Cần Python 3.11. Trên Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mở bằng VS Code (extension Jupyter) hoặc Jupyter, rồi chạy theo thứ tự:

1. **`Notebook_A_MHEALTH_SensorPositionRetrain.ipynb`** (khoảng 15 giây, chưa tính thời gian tải 72 MB dữ liệu): tải dữ liệu, cắt cửa sổ, làm sạch, chạy SQL, vẽ hình RQ1, ghi `data/processed/mhealth.parquet` và `manifest.json`.
2. **`Notebook_B_MHEALTH_SensorPositionRetrain_executed.ipynb`** (khoảng 22 phút trên CPU 8 luồng): kiểm tra md5 với manifest, chạy 10 mô hình, các phép thử RQ3, vẽ hình SHAP.

Nếu chạy đúng, md5 của `data/processed/mhealth.parquet` là `3d81b90d136d90450e7650e80612e8e9`.

## 7. Cấu trúc repo

```
├── Notebook_A_MHEALTH_SensorPositionRetrain.ipynb        dữ liệu, làm sạch, SQL, RQ1
├── Notebook_B_MHEALTH_SensorPositionRetrain_executed.ipynb   mô hình (RQ2), giải thích (RQ3)
├── sql/queries.sql                  truy vấn Q1–Q5 (DuckDB)
├── data/processed/manifest.json     số dòng, số cột, md5 của file bàn giao
├── report/
│   ├── table_q*.csv, table_rq1_tests.csv          RQ1
│   ├── table_rq2.csv, table_rq2_folds.csv         RQ2 (trung bình và từng người)
│   ├── table_rq3_roar.csv                         che đặc trưng và huấn luyện lại
│   ├── table_rq3_positions.csv                    một vị trí và bỏ một vị trí
│   ├── table_rq3_group_share.csv, table_rq3_methods.csv, table_top10_by_method.csv
│   ├── table_diagnosis.csv                        nguyên nhân các lỗi lớn, có bằng chứng
│   └── fig_*.png                                  hình
└── requirements.txt                 phiên bản thư viện đã chạy thành công
```

## 8. Tài liệu tham khảo chính

- Banos, O., et al. (2014). mHealthDroid: A novel framework for agile development of mobile health applications. *IWAAL 2014*, LNCS 8868, 91–98. https://doi.org/10.1007/978-3-319-13105-4_14
- Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS 32*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*. https://doi.org/10.1145/2939672.2939778
