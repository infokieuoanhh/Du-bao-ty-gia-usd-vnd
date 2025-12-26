import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

# Tải mô hình và bộ chuẩn hóa đã lưu
try:
    # Đảm bảo tên file model và scaler đã được đồng bộ với script train_and_save_model.py
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler_data.joblib')
    print("Mô hình và Scaler đã được tải thành công.")
except Exception as e:
    print(f"LỖI KHÔNG TẢI ĐƯỢC FILE JOBLIB: {e}")
    print("Vui lòng đảm bảo đã chạy script huấn luyện và 2 file joblib nằm cùng thư mục.")
    exit() 

FEATURE_COLS = [
    'Lần cuối_vnd_ret_lag1',
    'Lần cuối_vnd_ret_lag3',
    'Lần cuối_cny_ret_lag1',
    'Lần cuối_dxy_ret_lag1',
    'Lần cuối_gold_ret_lag1',
    'Lần cuối_bond_ret_lag1',
    'DFF_ret_lag1'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    
    data_list = list(request.form.values())
    
    # KIỂM TRA SỐ LƯỢNG ĐẶC TRƯNG (phải là 7)
    if len(data_list) != len(FEATURE_COLS):
          return render_template('index.html', 
                                 prediction_text=f'LỖI: Vui lòng nhập đủ {len(FEATURE_COLS)} đặc trưng.')
    
    try:
        # Chuyển đổi dữ liệu form thành DataFrame
        features_list = [float(x) for x in data_list] 
        # TẠO DATAFRAME ĐẦU VÀO VỚI TÊN CỘT CHÍNH XÁC
        input_df = pd.DataFrame([features_list], columns=FEATURE_COLS)
        
        # 1. CHUẨN HÓA DỮ LIỆU ĐẦU VÀO (Quan trọng: phải dùng tên cột FEATURE_COLS)
        scaled_features = scaler.transform(input_df)
        
        # 2. Dự đoán (kết quả là Tỷ suất sinh lời - return)
        predicted_return = model.predict(scaled_features)[0]
        
        # Hiển thị Tỷ suất sinh lời dự đoán (nhân 100 để ra %)
        output_pct = round(predicted_return * 100, 4)
        
        return render_template('index.html', 
                               prediction_text=f'Tỷ suất sinh lời USD/VND dự đoán 3 ngày tới là: {output_pct:.4f} %')

    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'LỖI XỬ LÝ DỮ LIỆU: Vui lòng kiểm tra lại định dạng số. Chi tiết lỗi: {e}')

if __name__ == "__main__":
    app.run(debug=True)