# Coffee Shop BI Project

## 1. Giới thiệu dự án

**Coffee Shop BI Project** là bài tập lớn môn **Data Warehouse & Decision Support System (DW & DSS)**, được xây dựng dựa trên bộ dữ liệu **Coffee Shop Sales** từ Kaggle.

Dự án mô phỏng quy trình triển khai một hệ thống hỗ trợ ra quyết định cho chuỗi cửa hàng cà phê, bao gồm:

* Tiếp nhận dữ liệu giao dịch gốc từ Kaggle
* Tiền xử lý và làm sạch dữ liệu bằng **Pentaho Data Integration (PDI)**
* Xây dựng mô hình **Data Warehouse** theo hướng **Star Schema**
* Thực hiện **Data Mining** để tìm ra các mẫu và tri thức hữu ích
* Xây dựng **dashboard/ứng dụng Streamlit** phục vụ trực quan hóa và hỗ trợ ra quyết định

### Bộ dữ liệu sử dụng

Nguồn dữ liệu: **Coffee Shop Sales**
Dataset gồm các bản ghi giao dịch của một chuỗi cửa hàng cà phê giả lập hoạt động tại 3 địa điểm ở New York.

### Mục tiêu chính của dự án

* Xây dựng quy trình **ETL** để làm sạch và chuẩn hóa dữ liệu
* Thiết kế kho dữ liệu phục vụ phân tích kinh doanh
* Phân tích xu hướng doanh thu theo thời gian, cửa hàng và sản phẩm
* Áp dụng **Association Rules** để tìm các sản phẩm thường đi cùng nhau
* Áp dụng **K-Means Clustering** để phân nhóm hành vi bán hàng theo cửa hàng/thời gian
* Trực quan hóa kết quả bằng dashboard để hỗ trợ ra quyết định

### Các kỹ thuật chính được sử dụng

* **ETL / Data Cleaning**: Pentaho Data Integration
* **Data Warehouse**: Star Schema
* **Association Rule Mining**: Apriori / FP-Growth
* **Clustering**: K-Means
* **Dimensionality Reduction / Visualization**: PCA
* **Dashboard / DSS**: Streamlit, Plotly

---

## 2. Cấu trúc thư mục dự án

```text
Coffee_Shop_BI_Project/
├── data/
│   ├── raw/                    # Dữ liệu gốc tải từ Kaggle
│   ├── processed/              # Dữ liệu sạch, dim/fact CSV xuất từ Pentaho
│   └── sql/                    # File SQL tạo schema và bảng trong Data Warehouse
│
├── etl/
│   ├── transformations/        # Các file .ktr xử lý ETL
│   ├── jobs/                   # Các file .kjb điều phối toàn bộ pipeline ETL
│   └── config/                 # Cấu hình kết nối DB hoặc tham số ETL (nếu có)
│
├── mining_app/
│   ├── notebooks/              # Notebook thử nghiệm EDA, Apriori, K-Means, PCA
│   ├── src/                    # Mã nguồn xử lý logic data mining và visualization
│   │   ├── __init__.py         # Đánh dấu thư mục src là Python package
│   │   ├── data_loader.py      # Đọc dữ liệu fact/dim và join thành bảng phân tích
│   │   ├── feature_engineering.py # Tạo feature phục vụ mining và dashboard
│   │   ├── association.py      # Logic Association Rules (Apriori / FP-Growth)
│   │   ├── clustering.py       # Logic K-Means, Elbow, Silhouette, PCA
│   │   └── visualization.py    # Hàm dựng biểu đồ Plotly
│   ├── app.py                  # Ứng dụng Streamlit chính
│   └── requirements.txt        # Danh sách thư viện Python cần cài
│
├── docs/
│   ├── report/                 # Báo cáo cuối kỳ (Word/PDF)
│   ├── presentation/           # Slide thuyết trình
│   └── images/                 # Ảnh ETL, chart, dashboard để chèn vào báo cáo
│
├── README.md                   # Hướng dẫn tổng quan dự án
└── .gitignore                  # Loại bỏ file rác, file log, file không cần đẩy lên Git
```

---

## 3. Vai trò của từng thành phần

### 3.1. Thư mục `data/`

Đây là trung tâm dữ liệu của toàn bộ dự án.

#### `data/raw/`

* Lưu trữ file dữ liệu gốc tải từ Kaggle
* Không chỉnh sửa trực tiếp để đảm bảo tính nguyên bản của dữ liệu

#### `data/processed/`

* Lưu dữ liệu sau khi làm sạch và chuẩn hóa từ Pentaho
* Lưu các bảng đầu ra phục vụ Data Warehouse và Data Mining như:

  * `Coffee_Shop_Cleaned.csv`
  * `dim_product.csv`
  * `dim_store.csv`
  * `dim_date.csv`
  * `dim_time.csv`
  * `fact_sales.csv`
* Đây là nơi kết nối giữa nhóm ETL và nhóm Data Mining

#### `data/sql/`

* Chứa các file SQL để tạo bảng trong Data Warehouse
* Có thể gồm:

  * tạo schema
  * tạo dimension tables
  * tạo fact table
  * truy vấn kiểm tra dữ liệu

---

### 3.2. Thư mục `etl/`

Đây là khu vực làm việc chính của nhóm phụ trách ETL bằng Pentaho.

#### `etl/transformations/`

Chứa các file `.ktr`, mỗi file thực hiện một tác vụ ETL riêng.

Ví dụ:

* `clean_transaction.ktr`: đọc dữ liệu gốc, xử lý lỗi định dạng, lọc dữ liệu rác, xuất file sạch
* `load_dim_product.ktr`: tạo bảng chiều sản phẩm
* `load_dim_store.ktr`: tạo bảng chiều cửa hàng
* `load_dim_date.ktr`: tạo bảng chiều ngày
* `load_dim_time.ktr`: tạo bảng chiều thời gian
* `load_fact_sales.ktr`: tạo bảng fact bán hàng

#### `etl/jobs/`

Chứa file `.kjb` để điều phối việc chạy các transformation theo đúng thứ tự.

Ví dụ:

* `etl.kjb`: job tổng để chạy toàn bộ pipeline ETL từ dữ liệu gốc đến các bảng đầu ra

#### `etl/config/`

* Lưu các cấu hình liên quan đến ETL nếu cần
* Ví dụ: cấu hình đường dẫn file, cấu hình database connection

---

### 3.3. Thư mục `mining_app/`

Đây là phần chính của nhóm Data Mining và Dashboard.

#### `mining_app/notebooks/`

* Dùng để thử nghiệm nhanh bằng Jupyter Notebook
* Phù hợp cho:

  * EDA
  * thử các ngưỡng support/confidence/lift
  * chạy thử Elbow Method, Silhouette Score
  * kiểm tra PCA trước khi đưa vào app chính

#### `mining_app/src/`

Chứa toàn bộ logic Python cho phần Data Mining.

* `data_loader.py`
  Đọc các file `fact_sales.csv`, `dim_product.csv`, `dim_date.csv`, `dim_time.csv`, `dim_store.csv` và join thành bảng phân tích.

* `feature_engineering.py`
  Tạo các biến mới như doanh thu, weekday/weekend, peak hour, tỷ lệ doanh thu theo nhóm sản phẩm, tập feature cho clustering.

* `association.py`
  Cài đặt Apriori/FP-Growth, tạo basket data, sinh luật kết hợp và lọc các luật tốt theo support/confidence/lift.

* `clustering.py`
  Chuẩn hóa dữ liệu, tính Elbow Method, Silhouette Score, chạy K-Means và PCA.

* `visualization.py`
  Chứa các hàm dựng biểu đồ cho dashboard như trend chart, top products, association chart, cluster charts.

#### `mining_app/app.py`

* File chính để chạy ứng dụng Streamlit
* Kết nối tất cả các phần trên vào dashboard hoàn chỉnh

#### `mining_app/requirements.txt`

* Danh sách thư viện cần cài để chạy phần Python/Streamlit

---

### 3.4. Thư mục `docs/`

Đây là nơi lưu các tài liệu phục vụ nộp bài và thuyết trình.

#### `docs/report/`

* Báo cáo cuối kỳ dạng Word/PDF
* Trình bày toàn bộ mục tiêu, ETL, Data Warehouse, Data Mining, kết quả và kết luận

#### `docs/presentation/`

* Slide thuyết trình dùng để bảo vệ dự án

#### `docs/images/`

* Lưu ảnh chụp màn hình workflow Pentaho, kết quả dashboard, biểu đồ mining để chèn vào báo cáo và slide

---

## 4. Kiến trúc dữ liệu tổng quát

Luồng xử lý dữ liệu của dự án:

```text
Kaggle raw data
    -> Pentaho cleaning
    -> Cleaned dataset
    -> Build dimensions + fact table
    -> Data mining (association, clustering)
    -> Streamlit dashboard
    -> Decision support insights
```

### Các bảng chính trong Data Warehouse

#### Fact table

* `fact_sales.csv`

  * `transaction_id`
  * `date_skey`
  * `time_skey`
  * `product_id`
  * `store_id`
  * `transaction_qty`
  * `unit_price`
  * `line_item_amount`

#### Dimension tables

* `dim_product.csv`

  * `product_id`
  * `product_category`
  * `product_type`
  * `product_detail`

* `dim_store.csv`

  * `store_id`
  * `store_location`

* `dim_date.csv`

  * `date_skey`
  * `transaction_date`
  * `year`
  * `month_number`
  * `day_in_month`
  * `day_name`

* `dim_time.csv`

  * `time_skey`
  * `hour_number`
  * `time_period`

---

## 5. Hướng dẫn chạy dự án

### 5.1. Yêu cầu môi trường

Cần cài đặt:

* **Java** và **Pentaho Data Integration (PDI / Spoon)**
* **Python 3.10+** hoặc phiên bản tương đương
* pip để cài thư viện Python

---

### 5.2. Bước 1: Chuẩn bị dữ liệu gốc

1. Tải dataset Coffee Shop Sales từ Kaggle
2. Đặt file dữ liệu gốc vào thư mục:

```text
data/raw/
```

Ví dụ:

```text
data/raw/Coffee Shop Sales.csv
```

---

### 5.3. Bước 2: Chạy ETL bằng Pentaho

1. Mở Pentaho Spoon
2. Mở file job tổng:

```text
etl/jobs/etl.kjb
```

3. Run job để thực hiện toàn bộ pipeline ETL

Kết quả mong đợi trong `data/processed/`:

* `Coffee_Shop_Cleaned.csv`
* `dim_product.csv`
* `dim_store.csv`
* `dim_date.csv`
* `dim_time.csv`
* `fact_sales.csv`

> Lưu ý: Nếu nhóm đang làm theo từng bước, có thể chạy riêng các file `.ktr` trong `etl/transformations/` trước khi chạy `etl.kjb`.

---

### 5.4. Bước 3: Cài thư viện Python cho dashboard

Di chuyển vào thư mục `mining_app/`:

```bash
cd mining_app
```

Cài các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

### 5.5. Bước 4: Chạy ứng dụng Streamlit

Từ thư mục `mining_app/`, chạy lệnh:

```bash
streamlit run app.py
```

Sau khi chạy, Streamlit sẽ mở dashboard trên trình duyệt.

---

## 6. Các chức năng chính của dashboard

Ứng dụng Streamlit hỗ trợ các nhóm chức năng sau:

### Overview

* Tổng doanh thu
* Tổng số lượng bán
* Tổng số giao dịch
* Giá trị đơn hàng trung bình
* Trend doanh thu theo thời gian
* Doanh thu theo thứ trong tuần
* Doanh thu theo giờ
* Doanh thu theo cửa hàng và danh mục sản phẩm

### Association Rules

* Tạo basket data từ `transaction_id`
* Chạy Apriori để tìm luật kết hợp
* Đánh giá theo:

  * support
  * confidence
  * lift
* Hiển thị top luật kết hợp bằng bảng và biểu đồ

### Clustering

* Tạo feature ở mức `store-day`
* Chuẩn hóa dữ liệu
* Chọn số cụm bằng Elbow Method và Silhouette Score
* Chạy K-Means
* Trực quan hóa bằng PCA scatter plot
* So sánh cluster theo doanh thu, peak hour, weekend ratio

### Raw Data

* Xem nhanh dữ liệu sau khi join fact và dimension

---

## 7. Phân công công việc nhóm gợi ý

### Nhóm ETL / Pentaho

* Làm sạch dữ liệu
* Tạo các bảng dimension
* Tạo fact table
* Điều phối toàn bộ pipeline ETL

### Nhóm Data Warehouse / SQL

* Thiết kế star schema
* Viết SQL tạo bảng
* Kiểm tra ràng buộc khóa

### Nhóm Data Mining / Python

* Xây dựng Association Rules
* Xây dựng K-Means Clustering + PCA
* Tạo feature engineering

### Nhóm Dashboard / Report

* Xây dựng Streamlit app
* Chụp ảnh kết quả
* Viết báo cáo và làm slide

---

## 8. Kết quả mong đợi của dự án

Sau khi hoàn thành, dự án có thể hỗ trợ trả lời các câu hỏi như:

* Doanh thu của chuỗi cửa hàng thay đổi như thế nào theo thời gian?
* Ngày nào trong tuần bận nhất?
* Khung giờ nào có doanh thu cao nhất?
* Những sản phẩm nào bán chạy nhất và mang lại doanh thu cao nhất?
* Những sản phẩm nào thường được mua cùng nhau?
* Các cửa hàng hoặc giai đoạn bán hàng có thể được phân nhóm thành những kiểu hành vi nào?

---

## 9. Ghi chú

* Đường dẫn file nên dùng dạng tương đối trong project để nhóm dễ chạy trên nhiều máy khác nhau
* Nên thống nhất tên cột giữa Pentaho, SQL và Python để tránh lỗi khi tích hợp
* Có thể mở rộng dự án bằng cách bổ sung dashboard nâng cao, dự báo doanh thu hoặc tối ưu gợi ý bán kèm trong tương lai

---

## 10. Tác giả / Nhóm thực hiện

Điền thông tin thành viên nhóm tại đây:

* Thành viên 1: ...
* Thành viên 2: ...
* Thành viên 3: ...
* Thành viên 4: ...
* Thành viên 5: ...

---

## 11. Tài liệu tham khảo

* Kaggle Coffee Shop Sales Dataset
* Tài liệu Pentaho Data Integration
* Tài liệu scikit-learn
* Tài liệu mlxtend
* Tài liệu Streamlit
