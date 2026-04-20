import os

def create_project_structure():
    # Tên thư mục gốc
    base_dir = ""
    # Danh sách các thư mục cần tạo
    dirs = [
        "data/raw",
        "data/processed",
        "data/sql",
        "etl/transformations",
        "etl/jobs",
        "etl/config",
        "mining_app/notebooks",
        "mining_app/src",
        "docs/report",
        "docs/presentation",
        "docs/images",
    ]

    # Danh sách các file và nội dung mẫu (nếu có)
    files = {
        "README.md": "# Coffee Shop BI Project\n\nProject kết hợp Pentaho (ETL) và Streamlit (Data Mining).",
        ".gitignore": "__pycache__/\n*.py[cod]\n.venv/\n.env\n/data/raw/*\n!/data/raw/.gitkeep",
        "mining_app/requirements.txt": "streamlit\npandas\nplotly\nscikit-learn\nmlxtend\nmatplotlib\nseaborn",
        "mining_app/app.py": "import streamlit as st\n\nst.set_page_config(page_title='Coffee Shop DSS', layout='wide')\nst.title('☕ Coffee Shop Decision Support System')\n\nst.sidebar.info('Chọn các Tab để xem phân tích')",
        "mining_app/src/association.py": "# Code cho thuật toán FP-Growth / Apriori",
        "mining_app/src/clustering.py": "# Code cho thuật toán K-Means & PCA",
        "data/sql/schema.sql": "-- Script tạo bảng Star Schema\n-- Dim_Product, Dim_Store, Dim_Date, Fact_Sales",
        "data/raw/.gitkeep": "", # Giữ thư mục trống trên git
    }

    print(f"--- Đang tạo cấu trúc dự án trong: {base_dir} ---")

    # Tạo thư mục
    for d in dirs:
        dir_path = os.path.join(base_dir, d)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Tạo file
    for file_path, content in files.items():
        full_path = os.path.join(base_dir, file_path)
        # Tạo thư mục cha cho file nếu chưa có (trong trường hợp file nằm sâu)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {full_path}")

    print("\n--- Hoàn tất! Nhóm có thể bắt đầu làm việc ngay ---")

if __name__ == "__main__":
    create_project_structure()