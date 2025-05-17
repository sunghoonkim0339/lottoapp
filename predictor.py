from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import os
import numpy as np
import pandas as pd
import joblib
from fastapi.responses import JSONResponse
import gdown

app = FastAPI()

# 모델 및 데이터 로딩
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
            model_path = os.path.join(self.model_dir, f'model_번호{i}.joblib')
            self.models[f'번호{i}'] = joblib.load(model_path)
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))

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

    def predict_next_numbers(self, n_predictions=5, random_ratio=0.5):
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

# predictor 인스턴스 생성 (실행 경로 기준)
predictor = HybridLottoPredictor(model_dir="./hybrid_lotto_model", data_file="./lottol.xls.xlsx")

@app.get("/predict")
@app.post("/predict")
async def predict(request: Request = None):
    predictions = predictor.predict_next_numbers(n_predictions=3, random_ratio=0.5)
    return JSONResponse(content={"result": [[int(num) for num in pred] for pred in predictions]})

# 로컬 테스트용
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

def download_from_gdrive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

model_files = {
    'model_번호1.joblib': '1aBcDeFgHiJkLmNoPqRstUvWxYz',
    'model_번호2.joblib': '파일ID2',
    # ... 추가 ...
    'scaler.joblib': '파일ID3',
    'lottol.xls.xlsx': '파일ID4'
}

for filename, file_id in model_files.items():
    download_from_gdrive(file_id, filename)
