import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import warnings
from sklearn.preprocessing import StandardScaler
from momentfm import MOMENTPipeline
from tqdm import tqdm


class MomentDataset(Dataset):
    """Dataset class for MOMENT model training and inference."""
    
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int, scaler: Optional[StandardScaler] = None):
        """
        Args:
            data: Time series data of shape [n_series, time_steps]
            context_length: Length of historical context
            prediction_length: Number of steps to forecast
            scaler: Optional pre-fitted scaler for normalization
        """
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_series, self.n_timesteps = data.shape
        
        # Apply scaling
        if scaler is None:
            self.scaler = StandardScaler()
            # Fit on all data (transpose for sklearn format)
            self.scaler.fit(data.T)
        else:
            self.scaler = scaler
            
        # Transform data
        self.scaled_data = self.scaler.transform(data.T).T  # Shape: [n_series, time_steps]
        
    def __len__(self):
        # Number of valid samples per series
        samples_per_series = max(0, self.n_timesteps - self.context_length - self.prediction_length + 1)
        return self.n_series * samples_per_series
    
    def __getitem__(self, idx):
        # Determine which series and which time window
        samples_per_series = max(0, self.n_timesteps - self.context_length - self.prediction_length + 1)
        series_idx = idx // samples_per_series
        time_idx = idx % samples_per_series
        
        # Extract context and target
        start_idx = time_idx
        context_end = start_idx + self.context_length
        target_end = context_end + self.prediction_length
        
        context = self.scaled_data[series_idx, start_idx:context_end]
        target = self.scaled_data[series_idx, context_end:target_end]
        
        # MOMENT expects shape [n_channels, seq_len]
        context = context.reshape(1, -1)  # [1, context_length]
        target = target.reshape(1, -1)    # [1, prediction_length]
        
        input_mask = np.ones(self.context_length)
        
        return torch.FloatTensor(context), torch.FloatTensor(target), torch.FloatTensor(input_mask)


class MomentForecaster:
    """MOMENT model wrapper for time series forecasting."""
    
    def __init__(
        self,
        model_path: str = "AutonLab/MOMENT-1-large",
        context_length: int = 512,
        fine_tune_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
    ):
        """
        Initializes the MOMENT model wrapper.
        
        Args:
            model_path: The Hugging Face path to the MOMENT model
            context_length: Length of historical context (default: 512)
            fine_tune_epochs: Number of epochs for fine-tuning (default: 3)
            batch_size: Batch size for training (default: 8)
            learning_rate: Learning rate for fine-tuning (default: 1e-4)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_length = context_length
        self.fine_tune_epochs = fine_tune_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
        print(f"MOMENT Forecaster initialized - Device: {self.device}")
        print(f"Context length: {context_length}, Fine-tune epochs: {fine_tune_epochs}")
    
    def _load_model(self, prediction_length: int):
        """Load and configure MOMENT model for specific prediction length."""
        print(f"Loading MOMENT model for prediction length: {prediction_length}")
        
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
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
    
    def fit(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        verbose: bool = True
    ):
        """
        Fine-tune MOMENT model on the provided time series data.
        
        Args:
            df: DataFrame with historical time series data
            prediction_length: Number of future time steps to predict
            verbose: Whether to show training progress
        """
        # Load model with correct prediction length
        self._load_model(prediction_length)
        
        # Prepare data
        data = df.values.T  # Shape: [n_series, time_steps]
        
        # Check if we have enough data
        min_required = self.context_length + prediction_length
        if data.shape[1] < min_required:
            raise ValueError(
                f"Time series too short! Need at least {min_required} timesteps, "
                f"got {data.shape[1]}. Consider reducing context_length or prediction_length."
            )
        
        # Create dataset
        dataset = MomentDataset(data, self.context_length, prediction_length)
        self.scaler = dataset.scaler  # Store scaler for inference
        
        if len(dataset) == 0:
            raise ValueError("No valid training samples created. Check your data length and parameters.")
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
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
    
    def predict(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        fit_if_needed: bool = True
    ) -> Dict[str, List[float]]:
        """
        Generate forecasts for future time steps.
        
        Args:
            df: DataFrame with historical time series data
            prediction_length: Number of future time steps to predict
            fit_if_needed: Whether to automatically fit the model if not already fitted
            
        Returns:
            Dictionary where keys are series names and values are forecasted values
        """
        # Auto-fit if needed
        if not self.is_fitted and fit_if_needed:
            self.fit(df, prediction_length, verbose=True)
        elif not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first or set fit_if_needed=True")
        
        # Check if model prediction length matches request
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
                
                # Get context (last context_length points)
                if len(series_data) >= self.context_length:
                    context = series_data[-self.context_length:]
                else:
                    # Pad with zeros if series is too short
                    padding = np.zeros(self.context_length - len(series_data))
                    context = np.concatenate([padding, series_data])
                    warnings.warn(
                        f"Series '{series_name}' is shorter than context length {self.context_length}. "
                        "Padded with zeros.",
                        UserWarning
                    )
                
                # Scale using fitted scaler
                context_scaled = self.scaler.transform(context.reshape(-1, 1)).flatten()
                
                # Prepare input for MOMENT [batch_size=1, n_channels=1, seq_len]
                context_tensor = torch.FloatTensor(context_scaled).unsqueeze(0).unsqueeze(0).to(self.device)
                input_mask = torch.ones(1, self.context_length).to(self.device)
                
                # Generate forecast
                output = self.model(x_enc=context_tensor, input_mask=input_mask)
                forecast_scaled = output.forecast.cpu().numpy().flatten()
                
                # Inverse transform to original scale
                forecast = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
                
                results[series_name] = forecast.tolist()
        
        return results
    
    def predict_zero_shot(
        self,
        df: pd.DataFrame,
        prediction_length: int,
    ) -> Dict[str, List[float]]:
        """
        Generate zero-shot forecasts without fine-tuning (will be poor quality).
        
        Args:
            df: DataFrame with historical time series data
            prediction_length: Number of future time steps to predict
            
        Returns:
            Dictionary where keys are series names and values are forecasted values
        """
        # Load model without fine-tuning
        self._load_model(prediction_length)
        
        # Create a simple scaler on the input data
        data = df.values.T
        self.scaler = StandardScaler()
        self.scaler.fit(data.T)
        
        print("Generating zero-shot forecasts (no fine-tuning)...")
        return self.predict(df, prediction_length, fit_if_needed=False)
    
    def get_model_info(self) -> Dict:
        """Get information about the current model state."""
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


# Example usage and testing
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
    
    print("Testing MOMENT Forecaster...")
    
    # Initialize forecaster
    forecaster = MomentForecaster(
        context_length=256,  # Smaller for demo
        fine_tune_epochs=2,
        batch_size=4
    )
    
    # Generate forecasts
    try:
        forecasts = forecaster.predict(df, prediction_length=24)
        
        print("\nForecasts generated successfully!")
        for series_name, forecast in forecasts.items():
            print(f"{series_name}: {forecast[:5]}... (showing first 5 values)")
        
        print(f"\nModel info: {forecaster.get_model_info()}")
        
    except Exception as e:
        print(f"Error: {e}")