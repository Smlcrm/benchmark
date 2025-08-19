import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler
from foundation_model import FoundationModel
from momentfm import MOMENTPipeline
from tqdm import tqdm


class MomentDataset(Dataset):
    """Dataset class for MOMENT model training and inference."""
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int, scaler: Optional[StandardScaler] = None):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_series, self.n_timesteps = data.shape
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(data.T)
        else:
            self.scaler = scaler
        self.scaled_data = self.scaler.transform(data.T).T
    def __len__(self):
        samples_per_series = max(0, self.n_timesteps - self.context_length - self.prediction_length + 1)
        return self.n_series * samples_per_series
    def __getitem__(self, idx):
        samples_per_series = max(0, self.n_timesteps - self.context_length - self.prediction_length + 1)
        series_idx = idx // samples_per_series
        time_idx = idx % samples_per_series
        start_idx = time_idx
        context_end = start_idx + self.context_length
        target_end = context_end + self.prediction_length
        context = self.scaled_data[series_idx, start_idx:context_end]
        target = self.scaled_data[series_idx, context_end:target_end]
        context = context.reshape(1, -1)
        target = target.reshape(1, -1)
        input_mask = np.ones(self.context_length)
        return torch.FloatTensor(context), torch.FloatTensor(target), torch.FloatTensor(input_mask)


class MomentModel(FoundationModel):
    """MOMENT model wrapper for time series forecasting, extending FoundationModel."""
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        super().__init__(config, config_file)
        self.model_path = self.config.get('model_path', 'AutonLab/MOMENT-1-large')
        self.context_length = int(self.config.get('context_length', 512))
        self.fine_tune_epochs = int(self.config.get('fine_tune_epochs', 3))
        self.batch_size = int(self.config.get('batch_size', 8))
        self.learning_rate = float(self.config.get('learning_rate', 1e-4))
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"MOMENT Model initialized - Device: {self.device}")
        print(f"Context length: {self.context_length}, Fine-tune epochs: {self.fine_tune_epochs}")

    def set_params(self, **params: Dict[str, Any]) -> 'MomentModel':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _load_model(self, prediction_length: int):
        print(f"Loading MOMENT model for prediction length: {prediction_length}")
        self.model = MOMENTPipeline.from_pretrained(
            self.model_path,
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': prediction_length,
                'head_dropout': 0.1,
                'freeze_encoder': True,
                'freeze_embedder': True,
                'freeze_head': False,
            },
        )
        self.model.init()
        self.model = self.model.to(self.device)
        print("MOMENT model loaded successfully!")

    def fit(self, df: pd.DataFrame, prediction_length: int, verbose: bool = True):
        self._load_model(prediction_length)
        data = df.values.T  # Shape: [n_series, time_steps]
        min_required = self.context_length + prediction_length
        if data.shape[1] < min_required:
            raise ValueError(
                f"Time series too short! Need at least {min_required} timesteps, "
                f"got {data.shape[1]}. Consider reducing context_length or prediction_length."
            )
        dataset = MomentDataset(data, self.context_length, prediction_length)
        self.scaler = dataset.scaler  # Store scaler for inference
        if len(dataset) == 0:
            raise ValueError("No valid training samples created. Check your data length and parameters.")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        if verbose:
            print(f"Fine-tuning MOMENT on {len(dataset)} samples...")
        for epoch in range(self.fine_tune_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.fine_tune_epochs}") if verbose else dataloader
            for context, target, input_mask in pbar:
                context = context.to(self.device)
                target = target.to(self.device)
                input_mask = input_mask.to(self.device)
                optimizer.zero_grad()
                output = self.model(x_enc=context, input_mask=input_mask)
                loss = criterion(output.forecast, target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                if verbose and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            avg_loss = np.mean(epoch_losses)
            if verbose:
                print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
        self.is_fitted = True
        print("MOMENT fine-tuning completed!")

    def predict(self,
            y_context: Optional[Union[pd.Series, np.ndarray]] = None,
            y_target: Union[pd.Series, np.ndarray] = None,
            y_context_timestamps = None,
            y_target_timestamps = None,
            **kwargs) -> Union[np.ndarray, Dict[str, List[float]]]:
        if y_context is None and y_target is not None:
            y_context = y_target
        if y_context is None:
            raise ValueError("y_context or y_target must be provided.")
        if y_context_timestamps is not None:
            if len(y_context.shape) == 1:
                columns = ['1']
            else:
                columns = list(range(y_context.shape[0]))
            df = pd.DataFrame(y_context, index=y_context_timestamps, columns=columns)
        else:
            if len(y_context.shape) == 1:
                columns = ['1']
                df = pd.DataFrame({columns[0]: y_context})
            else:
                columns = list(range(y_context.shape[0]))
                df = pd.DataFrame(y_context.T, columns=columns)
        prediction_length = self.config.get('pdt', self.config.get('prediction_length', 8))
        prediction_length = int(prediction_length)
        results = self._sub_predict(df, prediction_length)
        if len(list(results.keys())) == 1:
            return np.array(results[columns[0]])
        else:
            multivariate_values = []
            for key in results.keys():
                multivariate_values.append(results[key])
            return np.array(multivariate_values)

    def _sub_predict(self, df: pd.DataFrame, prediction_length: int) -> Dict[str, List[float]]:
        if not self.is_fitted:
            self.fit(df, prediction_length, verbose=True)
        if self.model.model.forecast_horizon != prediction_length:
            warnings.warn(
                f"Model was fitted for prediction_length={self.model.model.forecast_horizon}, "
                f"but predict() called with prediction_length={prediction_length}. "
                "Re-fitting model..."
            )
            self.fit(df, prediction_length, verbose=False)
        self.model.eval()
        results = {}
        with torch.no_grad():
            for series_name in df.columns:
                series_data = df[series_name].values
                if len(series_data) >= self.context_length:
                    context = series_data[-self.context_length:]
                else:
                    padding = np.zeros(self.context_length - len(series_data))
                    context = np.concatenate([padding, series_data])
                    warnings.warn(
                        f"Series '{series_name}' is shorter than context length {self.context_length}. "
                        "Padded with zeros.",
                        UserWarning
                    )
                context_scaled = self.scaler.transform(context.reshape(-1, 1)).flatten()
                context_tensor = torch.FloatTensor(context_scaled).unsqueeze(0).unsqueeze(0).to(self.device)
                input_mask = torch.ones(1, self.context_length).to(self.device)
                output = self.model(x_enc=context_tensor, input_mask=input_mask)
                forecast_scaled = output.forecast.cpu().numpy().flatten()
                forecast = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                results[series_name] = forecast.tolist()
        return results

    def predict_zero_shot(self, df: pd.DataFrame, prediction_length: int) -> Dict[str, List[float]]:
        self._load_model(prediction_length)
        data = df.values.T
        self.scaler = StandardScaler()
        self.scaler.fit(data.T)
        print("Generating zero-shot forecasts (no fine-tuning)...")
        return self._sub_predict(df, prediction_length)

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'MOMENT-1-large',
            'context_length': self.context_length,
            'is_fitted': self.is_fitted,
            'device': self.device,
            'fine_tune_epochs': self.fine_tune_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'forecast_horizon': self.model.model.forecast_horizon if self.model else None,
        }


# testing

if __name__ == "__main__":
    # Create sample data similar to your TimesFM example
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    # Generate synthetic time series
    series1 = 100 + np.arange(1000) * 0.1 + 10 * np.sin(2 * np.pi * np.arange(1000) / 30) + np.random.normal(0, 2, 1000)
    series2 = 200 + np.arange(1000) * 0.05 + 15 * np.cos(2 * np.pi * np.arange(1000) / 45) + np.random.normal(0, 3, 1000)
    df = pd.DataFrame({
        'sales': series1,
        'revenue': series2
    }, index=dates)
    print("Testing MOMENT Model...")
    # Initialize model
    config = {
        'context_length': 256,
        'fine_tune_epochs': 2,
        'batch_size': 4
    }
    model = MomentModel(config=config)
    try:
        forecasts = model._sub_predict(df, prediction_length=24)
        print("\nForecasts generated successfully!")
        for series_name, forecast in forecasts.items():
            print(f"{series_name}: {forecast[:5]}... (showing first 5 values)")
        print(f"\nModel info: {model.get_model_info()}")
    except Exception as e:
        print(f"Error: {e}")