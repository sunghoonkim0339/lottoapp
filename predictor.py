import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown
import os

# 1. 구글드라이브에서 모델/데이터 자동 다운로드
def download_from_gdrive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

model_files = {
    'model_번호1.joblib': '1d0KGuuRvDeUMfkosnUBPvxr4PQ5_VFVQ',
    'model_번호2.joblib': '1X9X4pR1eParVIOaCYOk-gJdLASHODzRJ',
    'model_번호3.joblib': '1phx6vgYfTjF61yZ6x23HUeA_P6cM62L2',
    'model_번호4.joblib': '1ZOnrdEQ_QeGiaLfP1T-v8QMljcR3skx5',
    'model_번호5.joblib': '1ZOnrdEQ_QeGiaLfP1T-v8QMljcR3skx5',
    'model_번호6.joblib': '1_tsKo82xmrUY7KHRrYqqrKARFFc9UR0X',
    'scaler.joblib': '1umjYN8G2wDFCRLR6c1rvKdOX0quGC2lN',
    'lottol.xls.xlsx': '13ktLRl-NWLEwdjhi5EEf04bjMNHm9_NT'
}

for filename, file_id in model_files.items():
    download_from_gdrive(file_id, filename)

# 2. 예측기 클래스 정의
class HybridLottoPredictor:
    def __init__(self, model_dir: str, data_file: str):
        self.model_dir = model_dir
        self.data_file = data_file
        self.models = {}
        self.scaler = None
        self.number_range = range(1, 46)
        self.load_model()
        self.recent_draws = self.load_recent_draws(n=30)

    def load_model(self):
        for i in range(1, 7):
            model_path = os.path.join(self.model_dir, f'model_번호{i}.joblib') if self.model_dir else f'model_번호{i}.joblib'
            self.models[f'번호{i}'] = joblib.load(model_path)
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib') if self.model_dir else 'scaler.joblib'
        self.scaler = joblib.load(scaler_path)

    def load_recent_draws(self, n=30):
        df = pd.read_excel(self.data_file, skiprows=2)
        df.columns = ['년도', '회차', '추첨일', '1등당첨자수', '1등당첨금액', '2등당자수', '2등당첨금액',
                      '3등당첨자수', '3등당금액', '4등당첨자수', '4등당첨금액', '5등당자수', '5등당첨금액',
                      '번호1', '번호2', '번호3', '번호4', '번호5', '번호6', '보너스']
        df = df.sort_values(by=['회차']).reset_index(drop=True)
        return df.tail(n)

    def create_enhanced_features(self, df):
        features = []
        number_cols = [f'번호{i}' for i in range(1, 7)]
        for i in range(1, 7):
            col = f'번호{i}'
            for window in [3, 5, 10]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                features.extend([f'{col}_rolling_mean_{window}', f'{col}_rolling_std_{window}'])
            df[f'{col}_diff1'] = df[col].diff()
            df[f'{col}_diff2'] = df[col].diff().diff()
            features.extend([f'{col}_diff1', f'{col}_diff2'])
        df['nums_sum'] = df[number_cols].sum(axis=1)
        df['nums_mean'] = df[number_cols].mean(axis=1)
        df['nums_std'] = df[number_cols].std(axis=1)
        df['nums_max'] = df[number_cols].max(axis=1)
        df['nums_min'] = df[number_cols].min(axis=1)
        df['nums_range'] = df['nums_max'] - df['nums_min']
        features.extend(['nums_sum', 'nums_mean', 'nums_std', 'nums_max', 'nums_min', 'nums_range'])
        df['even_ratio'] = df[number_cols].apply(lambda x: sum(val % 2 == 0 for val in x) / 6, axis=1)
        df['consecutive_count'] = df[number_cols].apply(
            lambda x: sum(x.iloc[i+1] - x.iloc[i] == 1 for i in range(5)), axis=1
        )
        features.extend(['even_ratio', 'consecutive_count'])
        for start in [1, 16, 31]:
            end = start + 14
            col_name = f'range_{start}_{end}_ratio'
            df[col_name] = df[number_cols].apply(
                lambda x: sum((val >= start) & (val <= end) for val in x) / 6, axis=1
            )
            features.append(col_name)
        if '회차' in df.columns:
            df['회차_norm'] = (df['회차'] - df['회차'].min()) / (df['회차'].max() - df['회차'].min())
            df['회차_sin'] = np.sin(2 * np.pi * df['회차_norm'])
            df['회차_cos'] = np.cos(2 * np.pi * df['회차_norm'])
            features.extend(['회차_sin', '회차_cos'])
        return df[features].fillna(0)

    def model_based_prediction(self, features_scaled, n_top=10):
        predictions = {}
        for i in range(1, 7):
            pos = f'번호{i}'
            model = self.models.get(pos)
            if model:
                proba = model.predict_proba(features_scaled)[0]
                top_indices = np.argsort(proba)[-n_top:]
                top_numbers = model.classes_[top_indices]
                top_probas = proba[top_indices]
                predictions[pos] = list(zip(top_numbers, top_probas))
        return predictions

    def predict_next_numbers(self, n_predictions=3, random_ratio=0.5):
        features = self.create_enhanced_features(self.recent_draws)
        features_scaled = self.scaler.transform(features.iloc[-1:])
        model_predictions = self.model_based_prediction(features_scaled)
        hybrid_predictions = []
        for _ in range(n_predictions):
            prediction_set = []
            for i in range(1, 7):
                pos = f'번호{i}'
                use_random = np.random.rand() < random_ratio
                if use_random:
                    available_nums = [n for n in self.number_range if n not in prediction_set]
                    selected_num = np.random.choice(available_nums)
                elif pos in model_predictions and model_predictions[pos]:
                    nums, probs = zip(*model_predictions[pos])
                    probs = np.array(probs)
                    nums = np.array(nums)
                    probs = probs / probs.sum()
                    selected_num = np.random.choice(nums, p=probs)
                    while selected_num in prediction_set:
                        selected_num = np.random.choice(nums, p=probs)
                else:
                    available_nums = [n for n in self.number_range if n not in prediction_set]
                    selected_num = np.random.choice(available_nums)
                prediction_set.append(selected_num)
            hybrid_predictions.append(sorted(prediction_set))
        return hybrid_predictions

# 3. Streamlit UI
st.title("🎲 로또 번호 예측기 (Streamlit)")

if st.button("행운의 번호 생성"):
    predictor = HybridLottoPredictor(model_dir="", data_file="lottol.xls.xlsx")
    predictions = predictor.predict_next_numbers(n_predictions=3, random_ratio=0.5)
    for idx, nums in enumerate(predictions, 1):
        st.success(f"세트 {idx}: {', '.join(str(n) for n in nums)}")
