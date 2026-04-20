import os

# Đường dẫn gốc (Sửa lại cho đúng với máy của bạn)
BASE_DIR = "/Users/duoc.phamdinh/Documents/HK252/Kho dữ liệu quyết định/Coffee_Shop_DW_Project"

# Biến này PHẢI là DATA_PATH để khớp với file app.py
DATA_PATH = os.path.join(BASE_DIR, "data/raw/Coffee Shop Sales.xlsx")

# Các cấu hình khác
MIN_SUPPORT_DEFAULT = 0.01
MIN_CONFIDENCE_DEFAULT = 0.2
CLUSTERS_DEFAULT = 3