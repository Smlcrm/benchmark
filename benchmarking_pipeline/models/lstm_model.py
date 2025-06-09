"""
LSTM model implementation.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .my_model.model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, config):
        """
        Initialize LSTM model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - input_shape: tuple, shape of input sequences (sequence_length, n_features)
                - lstm_units: list of int, number of LSTM units in each layer
                - dense_units: list of int, number of dense units in each layer
                - dropout_rate: float, dropout rate between layers
                - learning_rate: float, learning rate for optimization
                - loss: str, loss function name
                - metrics: list of str, metrics to track
                - early_stopping: bool, whether to use early stopping
                - patience: int, patience for early stopping
        """
        super().__init__(config)
        self.model = None
        self._build_model()
        
    def _build_model(self):
        """Build the LSTM model architecture."""
        input_shape = self.config.get('input_shape', (10, 1))
        lstm_units = self.config.get('lstm_units', [50, 30])
        dense_units = self.config.get('dense_units', [20, 1])
        dropout_rate = self.config.get('dropout_rate', 0.2)
        
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            if i == 0:
                # First layer needs input shape
                model.add(LSTM(
                    units=units,
                    input_shape=input_shape,
                    return_sequences=(i < len(lstm_units) - 1)
                ))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=(i < len(lstm_units) - 1)
                ))
            model.add(Dropout(dropout_rate))
            
        # Add Dense layers
        for units in dense_units[:-1]:
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(dropout_rate))
            
        # Output layer
        model.add(Dense(units=dense_units[-1]))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss=self.config.get('loss', 'mse'),
            metrics=self.config.get('metrics', ['mae'])
        )
        
        self.model = model
        
    def train(self, data):
        """
        Train the LSTM model on given data.
        
        Args:
            data: Dataset object containing train and validation data
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call _build_model first.")
            
        # Convert data to numpy arrays
        X_train = data.train.features.to_pandas().values
        y_train = data.train.labels.to_numpy()
        X_val = data.validation.features.to_pandas().values
        y_val = data.validation.labels.to_numpy()
        
        # Prepare callbacks
        callbacks = []
        if self.config.get('early_stopping', True):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('patience', 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
        
    def predict(self, data):
        """
        Make predictions using the trained LSTM model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call _build_model first.")
            
        # Convert data to numpy array
        X = data.to_pandas().values
        
        # Generate predictions
        return self.model.predict(X)
        
    def save(self, path):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        
    def load(self, path):
        """Load the model from disk."""
        self.model = tf.keras.models.load_model(path) 