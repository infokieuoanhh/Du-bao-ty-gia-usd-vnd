import os
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler 

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


import matplotlib.pyplot as plt
import seaborn as sns


import shap


import joblib

# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
DATA_FOLDER = "Dữ liệu khai phá/" 

try:
    df_vnd = pd.read_csv(DATA_FOLDER + "Dữ liệu USD_VND.csv")
    df_cny = pd.read_csv(DATA_FOLDER + "Dữ liệu Lịch sử CNY_USD.csv")
    df_dxy = pd.read_csv(DATA_FOLDER + "Dữ liệu Lịch sử Chỉ số Đô la Mỹ.csv")
    df_gold = pd.read_csv(DATA_FOLDER + "Dữ liệu Lịch sử Hợp đồng Tương lai Vàng.csv")
    df_bond = pd.read_csv(DATA_FOLDER + "Dữ liệu Lịch sử Suất Thu lợi Trái phiếu 10 Năm Việt Nam.csv")
    df_cpi = pd.read_csv(DATA_FOLDER + "Dữ liệu CPI.csv")
    df_dff = pd.read_csv(DATA_FOLDER + "Fed Funds Rate.csv")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file trong thư mục '{DATA_FOLDER}'. Vui lòng kiểm tra lại đường dẫn.")
    os._exit(1)


# Dữ lại dữ liệu cần thiết và đổi tên cột 'Lần cuối' cho từng DataFrame
df_vnd = df_vnd[["Ngày", "Lần cuối"]].rename(columns={'Lần cuối': 'Lần cuối_vnd'})
df_cny = df_cny[["Ngày", "Lần cuối"]].rename(columns={'Lần cuối': 'Lần cuối_cny'})
df_dxy = df_dxy[["Ngày", "Lần cuối"]].rename(columns={'Lần cuối': 'Lần cuối_dxy'})
df_gold = df_gold[["Ngày", "Lần cuối"]].rename(columns={'Lần cuối': 'Lần cuối_gold'})
df_bond = df_bond[["Ngày", "Lần cuối"]].rename(columns={'Lần cuối': 'Lần cuối_bond'})

def normalize_datetime_index(df, date_col='Ngày'):
    df = df.copy()

    # Nếu chưa phải DatetimeIndex thì chuẩn hóa
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col in df.columns:
            # Sửa lỗi 'coerce' cho phép ngày không hợp lệ trở thành NaT
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            df = df.set_index(date_col)

    # Làm sạch index
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    return df
    
df_vnd = normalize_datetime_index(df_vnd)
df_cny = normalize_datetime_index(df_cny)
df_dxy = normalize_datetime_index(df_dxy)
df_gold = normalize_datetime_index(df_gold)
df_bond = normalize_datetime_index(df_bond)
df_dff = normalize_datetime_index(df_dff)
df_cpi = normalize_datetime_index(df_cpi, date_col='Ngày')

for name, df_ in {
    'vnd': df_vnd, 'cny': df_cny, 'dxy': df_dxy,
    'gold': df_gold, 'bond': df_bond,
    'dff': df_dff
}.items():
    print(f"Index type for {name}: {type(df_.index)}")
    
# xử lý dữ liệu CPI 
# Đặt cột 'Ngày' làm chỉ mục trước khi thực hiện resample
if 'Ngày' in df_cpi.columns:
    df_cpi.set_index('Ngày', inplace=True)
if not isinstance(df_cpi.index, pd.DatetimeIndex):
    df_cpi.index = pd.to_datetime(df_cpi.index, dayfirst=True)

# Loại bỏ dữ liệu không hợp lệ (nếu có NaT do to_datetime)
df_cpi = df_cpi[df_cpi.index.notna()]
# Loại bỏ các ngày trùng lặp
df_cpi = df_cpi[~df_cpi.index.duplicated(keep='last')]
df_cpi = df_cpi.resample('D').ffill()
# df_cpi # Không nên hiển thị trực tiếp trong script

print(f"Index type df_vnd: {type(df_vnd.index)}")
print(f"Index type df_cny: {type(df_cny.index)}")
print(f"Index type df_dxy: {type(df_dxy.index)}")
print(f"Index type df_gold: {type(df_gold.index)}")
print(f"Index type df_bond: {type(df_bond.index)}")
print(f"Index type df_dff: {type(df_dff.index)}")
print(f"Index type df_cpi: {type(df_cpi.index)}")

#Gộp dữ liệu vào một dataframe
# Gộp tất cả các DataFrame đã được xử lý (với DatetimeIndex)
df = df_vnd.copy()
df = (
    df.merge(df_cny, left_index=True, right_index=True, how='left')
      .merge(df_dxy, left_index=True, right_index=True, how='left')
      .merge(df_gold, left_index=True, right_index=True, how='left')
      .merge(df_bond, left_index=True, right_index=True, how='left')
      .merge(df_dff, left_index=True, right_index=True, how='left')
      .merge(df_cpi, left_index=True, right_index=True, how='left')
)

df = df.sort_index()

# Chuyển dữ liệu sang dạng số
for col in ['Lần cuối_vnd', 'Lần cuối_cny', 'Lần cuối_dxy', 'Lần cuối_gold', 'Lần cuối_bond', 'DFF', 'CPI']:
    if col in df.columns:
        # Sửa lỗi: Cần làm sạch dữ liệu trước khi chuyển đổi
        df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True) # Loại bỏ ký tự không phải số, dấu chấm, dấu trừ
        df[col] = pd.to_numeric(df[col], errors='coerce')

# df.head() # Không nên hiển thị trực tiếp trong script

#Xây dựng các đặc trưng
base_cols = ['Lần cuối_vnd', 'Lần cuối_cny', 'Lần cuối_dxy', 'Lần cuối_gold', 'Lần cuối_bond', 'DFF', 'CPI']

for col in base_cols:
    df[f'{col}_ret'] = df[col].pct_change()

lag_cols = ['Lần cuối_vnd_ret', 'Lần cuối_cny_ret', 'Lần cuối_dxy_ret', 'Lần cuối_gold_ret', 'Lần cuối_bond_ret','CPI_ret', 'DFF_ret']

for col in lag_cols:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag3'] = df[col].shift(3)

##Tạo biến mục tiêu
# Biến mục tiêu: xu hướng USD/VND 3 ngày tới
df['target'] = (
    df['Lần cuối_vnd_ret']
    .rolling(window=3)
    .mean()
    .shift(-1)
)

# Loại bỏ NaN sinh ra từ rolling & shift
# Sử dụng reset_index() để biến DatetimeIndex thành một cột mới, sau đó gán lại làm index
df = df.dropna().reset_index() # 'index' is now a column with original DatetimeIndex
df = df.rename(columns={'index': 'Ngày'}) # Rename the new column to 'Ngày'

# Đảm bảo cột 'Ngày' là datetime và đặt lại làm index
df['Ngày'] = pd.to_datetime(df['Ngày'], dayfirst=True) # Ensure it's datetime
df = df.set_index('Ngày') # Set 'Ngày' back as the DatetimeIndex

df = df.dropna()
#Kiểm tra thông tin dữ liệu
print("\n--- Thông tin DataFrame cuối cùng ---")
df.info()
print(f"Kích thước DataFrame: {df.shape}")

##Phân tích tương quan
important_cols = [
    'target',
    'Lần cuối_vnd_ret',
    'Lần cuối_vnd_ret_lag1',
    'Lần cuối_vnd_ret_lag3',
    'Lần cuối_cny_ret',
    'Lần cuối_cny_ret_lag1',
    'Lần cuối_dxy_ret',
    'Lần cuối_dxy_ret_lag1',
    'Lần cuối_gold_ret',
    'Lần cuối_gold_ret_lag1',
    'Lần cuối_bond_ret',
    'Lần cuối_bond_ret_lag1',
    'DFF_ret',
    'DFF_ret_lag1',
    'CPI_ret',
    'CPI_ret_lag1'
]

# Chỉ giữ lại các cột số (đề phòng)
corr = df[important_cols].select_dtypes(include=np.number).corr()
plt.figure(figsize=(30, 25))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Biểu đồ tương quan')
plt.show() 


# Chọn các cột số
df_numeric = df[important_cols].select_dtypes(include = 'number')
print(f"\nCác cột số được sử dụng cho thống kê: {df_numeric.columns}")

mean = df_numeric.mean()
median = df_numeric.median()
mode = df_numeric.mode().iloc[0]
min = df_numeric.min()
max = df_numeric.max()
Q1 = df_numeric.quantile(0.25)
Q2 = df_numeric.quantile(0.5)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
std = df_numeric.std()
var = df_numeric.var()

# Tạo bảng thống kê
def thong_ke(mean,median,mode,min,max,Q1,Q2,Q3,IQR,std,var):
    data = {'Mean': list(mean),
            'Median': list(median),
            'Mode': list(mode),
            'Min': list(min),
            'Max': list(max),
            'Q1': list(Q1),
            'Q2': list(Q2),
            'Q3': list(Q3),
            'IQR': list(IQR),
            'Std': list(std),
            'Var': list(var)
            }
    df_stats = pd.DataFrame(data)
    df_stats.index = df_numeric.columns
    # df_complete = df_stats.transpose() # Giữ nguyên format ban đầu của hàm
    return df_stats.transpose()

df_complete = thong_ke(mean,median,mode,min,max,Q1,Q2,Q3,IQR,std,var)
print("\n--- Bảng thống kê mô tả ---")
print(df_complete)

# Biểu đồ Histogram phân phối giá đóng cửa
plt.figure(figsize=(10, 5))
sns.histplot(df['Lần cuối_vnd'], bins=50, kde=True)
plt.title('Phân phối giá đóng cửa USD/VND (Lần cuối)')
plt.xlabel('Giá đóng cửa')
plt.ylabel('Tần suất')
plt.grid(True)
plt.show() # 

# Biểu đồ Đường giá đóng cửa qua các năm
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Lần cuối_vnd'], label='Giá đóng cửa')
plt.title('Xu hướng giá đóng cửa USD/VND theo thời gian')
plt.xlabel('Ngày')
plt.ylabel('Giá đóng cửa')
plt.legend()
plt.grid(True)
plt.show() # 

# Tạo cột 'Năm' từ index nếu chưa có
if 'Năm' not in df.columns:
    df['Năm'] = df.index.year

# Biểu đồ Boxplot phân phối giá đóng cửa theo năm
plt.figure(figsize=(10, 4))
sns.boxplot(x='Năm', y='Lần cuối_vnd', data=df)
plt.title('Phân phối giá đóng cửa theo năm')
plt.xlabel('Năm')
plt.ylabel('Giá đóng cửa')
plt.xticks(rotation=45)
plt.grid(True)
plt.show() # 

# Xóa cột 'Năm' đã thêm vào DataFrame nếu không cần nữa
df.drop(columns=['Năm'], inplace=True, errors='ignore')

feature_cols = [
    'Lần cuối_vnd_ret_lag1',
    'Lần cuối_vnd_ret_lag3',
    'Lần cuối_cny_ret_lag1',
    'Lần cuối_dxy_ret_lag1',
    'Lần cuối_gold_ret_lag1',
    'Lần cuối_bond_ret_lag1',
    'DFF_ret_lag1'
]


X = df[feature_cols]
y = df['target']


split_ratio = 0.8
split_index = int(len(df) * split_ratio)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ** BỔ SUNG: Chuẩn hóa dữ liệu **
# Khởi tạo và fit StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Chuyển lại thành DataFrame với tên cột giữ nguyên
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)


print("\n--- Huấn luyện Mô hình 1 (Model Optimized) ---")
# **HUẤN LUYỆN MÔ HÌNH 1 (Dùng cho đánh giá hiệu suất) - Dùng dữ liệu CHƯA chuẩn hóa**
# Sử dụng dữ liệu CHƯA chuẩn hóa để giữ đúng logic ban đầu của bạn (vì RFR ít nhạy cảm với scale)
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Dự đoán trên tập Test (Dữ liệu CHƯA chuẩn hóa)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Đánh giá Hiệu suất trên Tập Kiểm tra (Mô hình 1 - Không Scale) ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2 ): {r2:.4f}")

# Tính r2 trên tập huấn luyện
y_pred_train = model.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)
# Tính r2 trên tập kiểm tra
r2_test = r2_score(y_test, y_pred) # y_pred đã được tính ở trên
print(f"R^2 trên tập Huấn luyện (Train): {r2_train:.4f}")
print(f"R^2 trên tập Kiểm tra (Test): {r2_test:.4f}")

k_folds = 10
# 1. R2
r2_cv = cross_val_score(
    model,
    X_train,
    y_train,
    scoring='r2',
    cv=k_folds,
    n_jobs=-1
)
# 2. RMSE (sklearn trả về neg_root_mean_squared_error → phải đổi dấu)
rmse_cv = -cross_val_score(
    model,
    X_train,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=k_folds,
    n_jobs=-1
)
# 3. MAE (sklearn trả về neg_mean_absolute_error → phải đổi dấu)
mae_cv = -cross_val_score(
    model,
    X_train,
    y_train,
    scoring='neg_mean_absolute_error',
    cv=k_folds,
    n_jobs=-1
)
# Tạo DataFrame tổng hợp
df_result = pd.DataFrame({
    'R2': r2_cv,
    'RMSE': rmse_cv,
    'MAE': mae_cv
})
print(f"\n--- Kết quả Đánh giá Chéo (Cross-Validation) ---")
print(f"R² CV Trung bình : {r2_cv.mean():.4f}")
print(f"R² Std Dev       : {r2_cv.std():.4f}")

print(f"RMSE CV Trung bình  : {rmse_cv.mean():.4f}")
print(f"RMSE Std Dev     : {rmse_cv.std():.4f}")

print(f"MAE CV Trung bình   : {mae_cv.mean():.4f}")
print(f"MAE Std Dev      : {mae_cv.std():.4f}")

print("\nBảng chi tiết CV:")
print(df_result)

# Tạo DataFrame so sánh cho tập huấn luyện
df_train_comparison = pd.DataFrame({
    'Thực tế (Train)': y_train,
    'Dự đoán (Train)': y_pred_train,
    'Chênh lệch': y_train - y_pred_train
})

print("\nBảng so sánh Giá trị thực tế và Dự đoán trên Tập Huấn luyện (5 mẫu đầu):")
print(df_train_comparison.head())

# Tạo DataFrame so sánh cho tập kiểm tra
df_test_comparison = pd.DataFrame({
    'Thực tế (Test)': y_test,
    'Dự đoán (Test)': y_pred,
    'Chênh lệch ': y_test - y_pred
})

print("\nBảng so sánh Giá trị thực tế và Dự đoán trên Tập Kiểm tra (5 mẫu đầu):")
print(df_test_comparison.head(5))

# Biểu đồ Giá trị thực tế so với Giá trị dự đoán
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực tế (y_test)')
plt.ylabel('Giá trị dự đoán (y_pred)')
plt.title('Biểu đồ Giá trị thực tế so với Giá trị dự đoán')
plt.grid(True)
plt.show() # 

# **BỔ SUNG: Tính toán phần dư (residuals)**
residuals = y_test - y_pred

#Biểu đồ Phần dư so với Giá trị dự đoán
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Giá trị dự đoán (y_pred)')
plt.ylabel('Phần dư (residuals)')
plt.title('Biểu đồ Phần dư so với Giá trị dự đoán')
plt.grid(True)
plt.show() # 


feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,12))
plt.barh(feat_imp['feature'].head(20),
          feat_imp['importance'].head(20))
plt.gca().invert_yaxis()
plt.title("Top Features quan trọng nhất")
plt.show() # 


# LƯU SCALER ĐÃ FIT 
joblib.dump(scaler, 'scaler_data.joblib')
print("\n[THÀNH CÔNG] Bộ chuẩn hóa (scaler_data.joblib) đã được lưu.")


# 5. HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG VÀ LƯU FILE
# Huấn luyện mô hình lần 2, sử dụng dữ liệu ĐÃ CHUẨN HÓA (thực tế thì RFR không cần scale) 
# và tham số đơn giản hơn để tránh huấn luyện lại mô hình 1
model_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_final.fit(X_train_scaled, y_train)

# LƯU MÔ HÌNH (Tạo ra file random_forest_model.joblib)
joblib.dump(model_final, 'random_forest_model.joblib')
print("[THÀNH CÔNG] Mô hình (random_forest_model.joblib) đã được lưu.")

# 6. ĐÁNH GIÁ (Chỉ để kiểm tra hiệu suất của mô hình đã lưu)
y_pred_test_scaled = model_final.predict(X_test_scaled)
r2_test_scaled = r2_score(y_test, y_pred_test_scaled)
print(f"\nKiểm tra R^2 trên tập Test (Mô hình Cuối cùng, Dữ liệu ĐÃ CHUẨN HÓA): {r2_test_scaled:.4f}")