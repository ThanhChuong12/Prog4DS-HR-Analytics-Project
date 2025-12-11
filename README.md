# Homework 2: HR Analytics Project using NumPy
**Môn học:** CSC17104 - Lập trình cho Khoa học Dữ liệu  
**Sinh viên:** Lê Hà Thanh Chương - 23120195

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
---

## Mục lục

- [Homework 2: HR Analytics Project using NumPy](#homework-2-hr-analytics-project-using-numpy)
  - [](#)
  - [Mục lục](#mục-lục)
  - [Giới thiệu](#giới-thiệu)
    - [Bài toán](#bài-toán)
    - [Động lực](#động-lực)
    - [Mục tiêu cụ thể](#mục-tiêu-cụ-thể)
    - [Ứng dụng thực tiễn](#ứng-dụng-thực-tiễn)
  - [Dataset](#dataset)
    - [Mô tả các features](#mô-tả-các-features)
  - [Method](#method)
    - [1. Xử lý Dữ liệu](#1-xử-lý-dữ-liệu)
    - [2. Mô hình hóa (Logistic Regression from Scratch)](#2-mô-hình-hóa-logistic-regression-from-scratch)
  - [Installation \& Setup](#installation--setup)
    - [1. Cài đặt môi trường](#1-cài-đặt-môi-trường)
    - [2. Chạy dự án](#2-chạy-dự-án)
  - [Results](#results)
    - [1. Hiệu năng Mô hình (Model Performance)](#1-hiệu-năng-mô-hình-model-performance)
    - [2. Độ ổn định (Stability)](#2-độ-ổn-định-stability)
    - [3. Insight Quan trọng (Business Insight)](#3-insight-quan-trọng-business-insight)
  - [Project Structure](#project-structure)
  - [Future Improvements](#future-improvements)
    - [Thử nghiệm các Mô hình Phi tuyến (Non-linear Models)](#thử-nghiệm-các-mô-hình-phi-tuyến-non-linear-models)
    - [Tối ưu hóa dựa trên Bài toán Kinh tế (Cost-Benefit Analysis)](#tối-ưu-hóa-dựa-trên-bài-toán-kinh-tế-cost-benefit-analysis)
    - [Phân tích Nguyên nhân Cốt lõi (Explainable AI - XAI)](#phân-tích-nguyên-nhân-cốt-lõi-explainable-ai---xai)
  - [Contributors](#contributors)
  - [Licenses](#licenses)

---

## Giới thiệu

### Bài toán
Bài toán này tập trung vào việc xây dựng và đánh giá một mô hình phân loại nhị phân (Binary Classification) nhằm dự đoán xác suất một Nhà khoa học dữ liệu (Data Scientist) có xu hướng tìm kiếm cơ hội việc làm mới hay tiếp tục gắn bó với tổ chức hiện tại. Bài toán này thuộc nhóm mô hình dự báo hành vi nhân sự, trong đó đầu ra là biến nhị phân biểu diễn ý định nghỉ việc.

### Động lực
Trong bối cảnh thị trường lao động công nghệ cạnh tranh mạnh mẽ, việc duy trì đội ngũ nhân sự chất lượng cao đóng vai trò then chốt đối với năng lực đổi mới của doanh nghiệp. Tình trạng "chảy máu chất xám" không chỉ làm gián đoạn hoạt động mà còn kéo theo chi phí đáng kể liên quan tới tuyển dụng, đào tạo và chuyển giao kiến thức. Việc phát triển một mô hình dự báo rời việc hỗ trợ bộ phận nhân sự (HR) trong việc:
* Nhận diện sớm các cá nhân có nguy cơ rời tổ chức, từ đó triển khai chiến lược giữ chân phù hợp.
* Phân tích mức độ ảnh hưởng của các yếu tố như môi trường làm việc, cơ hội đào tạo và kinh nghiệm tích lũy đến quyết định nghỉ việc của nhân viên.

### Mục tiêu cụ thể

Mục tiêu trọng tâm của bài toán là xây dựng một mô hình dự báo ý định nghỉ việc có độ tin cậy cao, khả năng khái quát hóa tốt, và có thể tích hợp trực tiếp vào quy trình ra quyết định nhân sự. 

* Phát triển mô hình Logistic Regression từ đầu bằng NumPy nhằm đảm bảo tính minh bạch của thuật toán và khả năng kiểm toán mô hình.
* Đánh giá hiệu quả mô hình dựa trên các thước đo như F1-Score, Recall và Accuracy, vốn là các chỉ số quan trọng trong các bài toán có mất cân bằng dữ liệu.
* Kiểm định độ ổn định của mô hình trên nhiều tập dữ liệu khác nhau thông qua kỹ thuật Stratified K-Fold Cross Validation, từ đó xác định mức độ tin cậy trong điều kiện triển khai thực tế.

### Ứng dụng thực tiễn

Mô hình dự báo có thể được tích hợp vào hệ thống quản trị nhân sự để:

* Hỗ trợ ra quyết định: cảnh báo sớm nhân sự có nguy cơ rời đi, giúp HR ưu tiên tiếp cận, hỗ trợ hoặc điều chỉnh chính sách phù hợp.
* Tối ưu hóa chi phí: giảm thiểu chi phí tuyển dụng và đào tạo nhờ vào khả năng giữ chân nhân sự chiến lược.
* Nâng cao chất lượng môi trường làm việc: cung cấp thông tin định lượng giúp ban lãnh đạo hiểu rõ yếu tố nào ảnh hưởng mạnh đến quyết định nghỉ việc, từ đó điều chỉnh chính sách phúc lợi và phát triển nghề nghiệp.

---

## Dataset

**Nguồn:** *HR Analytics: Job Change of Data Scientists (Kaggle)*

**Kích thước:** 19,158 mẫu (Training set)

**Đặc điểm:** Dữ liệu hỗn hợp (Số, Phân loại, Thứ tự) và mất cân bằng

### Mô tả các features

- `enrollee_id`: id định danh duy nhất của mỗi ứng viên.
- `city`: Mã định danh của thành phố nơi ứng viên sinh sống hoặc làm việc.
- `city_development_index` là biến liên tục (continuous) trong khoảng [0, 1], thể hiện mức độ phát triển của thành phố – yếu tố có thể ảnh hưởng đáng kể đến cơ hội việc làm và khả năng dịch chuyển nghề nghiệp.
- `gender`: Giới tính của ứng viên.
- `relevent_experience`: Ứng viên có kinh nghiệm liên quan đến lĩnh vực Data Science hay không.
- `enrolled_university`: Tình trạng học tập hiện tại.
- `education_level`: Trình độ học vấn cao nhất.
- `major_discipline`: Ngành học chính.
- `experience`: Số năm kinh nghiệm tổng cộng.
- `company_size`: Quy mô công ty hiện tại.
- `company_type`: Loại hình tổ chức công ty nơi ứng viên đang làm việc, nhưng chứa tỷ lệ missing tương đối lớn.
- `last_new_job`: Số năm kể từ công việc mới gần nhất.
- `training_hours`: Số giờ đào tạo mà ứng viên đã hoàn thành.
- `target`: Cho biết nhân viên có nghỉ việc hay không

---

## Method

### 1. Xử lý Dữ liệu

- **Đọc dữ liệu:** Sử dụng `np.genfromtxt` với `dtype=str` để xử lý định dạng hỗn hợp.
- **Mã hóa (Encoding):**
  - **Ordinal Encoding:** Ánh xạ thủ công cho `experience` (`<1 → 0`, `>20 → 21`) và `company_size`.
  - **Frequency Encoding:** Áp dụng cho cột `city` (do số lượng category quá lớn).
  - **One-Hot Encoding:** Tự cài đặt thuật toán one-hot cho các biến định danh (`gender`, `major`).
- **Xử lý Missing Values:** Điền giá trị thiếu bằng Median (cho biến số) hoặc Mode (cho biến phân loại).

### 2. Mô hình hóa (Logistic Regression from Scratch)

Chúng tôi tự xây dựng class `LogisticRegressionFromScratch` với các cải tiến kỹ thuật nhằm tối ưu hóa cho dữ liệu mất cân bằng:

- **Hàm kích hoạt:** Sigmoid Function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Hàm mất mát (Weighted Log-Loss with L2 Regularization):**
  Sử dụng hàm Binary Cross Entropy có gắn trọng số cho từng lớp để phạt nặng các dự đoán sai ở lớp thiểu số, kết hợp với L2 Regularization để kiểm soát Overfitting:

  $$
  J(w, b) = - \frac{1}{N} \sum_{i=1}^{N} \left[ w_{1} y^{(i)} \log(\hat{y}^{(i)}) + w_{0} (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right] + \frac{\lambda}{2} \sum_{j=1}^{M} w_j^2
  $$

  *Trong đó:*
  - $N$: Kích thước batch.
  - $w_1, w_0$: Trọng số lớp (Class Weights).
  - $\lambda$: Tham số điều chuẩn (Regularization parameter `reg_lambda`).

- **Thuật toán Tối ưu hóa:** **Mini-batch Gradient Descent**. Phương pháp này cân bằng giữa tốc độ tính toán của SGD và sự ổn định của Batch Gradient Descent.
- **Kỹ thuật xử lý mất cân bằng dữ liệu:**
  - **Class Weighting:** Thiết lập tỷ lệ trọng số **1:4** (`{'0': 1.0, '1': 4.0}`). Nghĩa là mô hình sẽ chịu mức phạt gấp 4 lần nếu dự đoán sai một nhân viên "Sẽ nghỉ việc".
  - **Threshold Tuning:** Tối ưu hóa ngưỡng quyết định (Decision Threshold) dựa trên việc cực đại hóa **F1-Score** trên tập kiểm thử thay vì sử dụng ngưỡng mặc định 0.5.

---

## Installation & Setup

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/ThanhChuong12/Prog4DS-HR-Analytics-Project.git
cd HR-Analytics-Project

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chạy dự án

Thực thi các notebook theo thứ tự:

1. **01_data_exploration.ipynb**: Khám phá dữ liệu và kiểm định giả thuyết "Chảy máu chất xám" cũng như câu hỏi và phân bố các đặc trưng.
2. **02_preprocessing.ipynb**: Chạy pipeline tiền xử lý và lưu file `.npy`.
3. **03_modeling.ipynb**: Huấn luyện mô hình và đánh giá độ ổn định.

---

## Results

### 1. Hiệu năng Mô hình (Model Performance)

Kết quả đánh giá trên tập Test (20%) với ngưỡng tối ưu **Threshold = 0.6**:

| Metric      | Score   | Ý nghĩa                                                                 |
|------------|---------|------------------------------------------------------------------------|
| Accuracy   | 75.8%   | Độ chính xác tổng thể                                                   |
| Precision  | 50.8%   | Tỷ lệ báo động đúng (giảm thiểu Spam cảnh báo)                         |
| Recall     | 71.3%   | Quan trọng nhất: Phát hiện được >71% nhân sự muốn nghỉ việc           |
| F1-Score   | 0.593   | Sự cân bằng tốt giữa Precision và Recall                               |


### 2. Độ ổn định (Stability)

- Kiểm định **Stratified 5-Fold Cross-Validation** cho thấy mô hình hoạt động rất ổn định:  
  **F1-Score:** 0.5963 ± 0.0142 (Độ lệch chuẩn cực thấp)

**Kết luận:** Mô hình đủ tin cậy để triển khai thực tế.


### 3. Insight Quan trọng (Business Insight)

- **Bác bỏ giả thuyết "Brain Drain":** Phân tích cho thấy việc đào tạo nhiều (High Training) không làm tăng tỷ lệ nghỉ việc tại các vùng kém phát triển.  
- **Yếu tố then chốt:** `City Development Index` là yếu tố dự báo mạnh nhất (**Trọng số** $W = -0.62$).  
  Nhân viên ở vùng kém phát triển có nguy cơ rời đi cao nhất, bất kể được đào tạo hay không.

---

## Project Structure

```text
HR-Analytics-Project/
├── data/                      # Thư mục chứa dữ liệu
│   ├── raw/                   # Dữ liệu gốc (aug_train.csv)
│   └── processed/             # Dữ liệu đã xử lý (.npy)
├── notebooks/                     # Thư mục chứa các notebook 
│   ├── 01_data_exploration.ipynb  # Khám phá và trực quan hoá dữ liệu
│   ├── 02_preprocessing.ipynb     # Xử lý dữ liệu
│   └── 03_modeling.ipynb          # Training, Tuning Threshold, Cross-Validation
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Các hàm xử lý dữ liệu NumPy
│   ├── models.py              # Class LogisticRegression, Metrics, KFold tự viết
│   └── visualization.py       # Các hàm vẽ biểu đồ chuẩn hóa (Matplotlib/Seaborn)
├── requirements.txt           # Danh sách thư viện
└── README.md                  # Tài liệu dự án
```

---
## Future Improvements
Mặc dù mô hình Logistic Regression hiện tại đã đạt được độ ổn định cao và hiệu suất khá tốt (F1 ~ 0.59), dự án vẫn còn dư địa để phát triển nhằm đạt được độ chính xác cao hơn và giá trị thực tiễn sâu sắc hơn:

### Thử nghiệm các Mô hình Phi tuyến (Non-linear Models)
* **Giới hạn của Logistic Regression:** Mô hình hiện tại giả định mối quan hệ tuyến tính giữa các biến đầu vào và log-odds của biến mục tiêu. Tuy nhiên, hành vi con người thường phức tạp hơn.
* **Đề xuất:** Triển khai các thuật toán dựa trên cây quyết định như **Random Forest**, **XGBoost** hoặc **LightGBM**. Các mô hình này có khả năng tự động bắt được các mối quan hệ phi tuyến và tương tác phức tạp giữa các biến mà không cần feature engineering thủ công quá nhiều.

### Tối ưu hóa dựa trên Bài toán Kinh tế (Cost-Benefit Analysis)
* Hiện tại, ngưỡng quyết định (Threshold = 0.6) được chọn để tối ưu hóa F1-Score (cân bằng kỹ thuật).
* **Hướng cải tiến:** Xây dựng **Ma trận Lợi nhuận (Profit Matrix)** dựa trên chi phí thực tế:
    * *Chi phí giữ chân (Cost of Retention):* Chi phí bỏ ra để giữ một nhân viên (ví dụ: thưởng, training).
    * *Chi phí thay thế (Cost of Replacement):* Chi phí tuyển dụng và đào tạo người mới nếu nhân viên cũ nghỉ việc.
* Từ đó, tìm ra ngưỡng tối ưu để **giảm thiểu tổng chi phí tài chính** cho công ty thay vì chỉ tối ưu hóa chỉ số thống kê.

### Phân tích Nguyên nhân Cốt lõi (Explainable AI - XAI)
* Sử dụng **SHAP (SHapley Additive exPlanations)** hoặc **LIME** để giải thích dự đoán cho từng cá nhân cụ thể. Điều này giúp bộ phận HR không chỉ biết *ai* sắp nghỉ việc, mà còn biết chính xác *tại sao* (ví dụ: "Nhân viên A có nguy cơ cao chủ yếu vì họ có 15 năm kinh nghiệm nhưng vẫn làm ở công ty quá nhỏ"), từ đó đưa ra gói giải pháp giữ chân cá nhân hóa.

## Contributors

* Thông tin tác giả: Lê Hà Thanh Chương
* Phương thức liên lạc (Contact):
    * Email: chuongle241205@gmail.com
    * Github: https://github.com/ThanhChuong12

## Licenses
Đồ án được phân phối dưới giấy phép MIT License.