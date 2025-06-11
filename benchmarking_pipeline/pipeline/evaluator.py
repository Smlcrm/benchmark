import pandas as pd
import numpy as np

"""
Model evaluation utilities.
"""

class Evaluator:
    def __init__(self, config=None):
        """
        Initialize evaluator with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with evaluation parameters.
                                     Can include 'metrics_to_calculate', 'target_col', 
                                     'pred_col', 'y_train_col_name', 'probabilistic_forecast_cols',
                                     'quantiles_q_values', 'interval_alpha', 'seasonal_period_mase'.
        """
        self.config = config if config is not None else {}
        self.metrics_to_calculate = self.config.get('metrics_to_calculate', ['mae', 'rmse'])
        self.target_col_name = self.config.get('target_col', 'y_true')
        self.pred_col_name = self.config.get('pred_col', 'y_pred')
        self.y_train_col_name = self.config.get('y_train_col_name', 'y_train') # Name of column in data if y_train is passed with it

    # Evaluation Metric Calculations
    def _rmse(self, y_true, y_pred):
        if y_true.ndim == 1:
            return np.sqrt(np.mean((y_true - y_pred)**2))
        elif y_true.ndim == 2:
            return np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
        return np.nan

    def _mae(self, y_true, y_pred):
        if y_true.ndim == 1:
            return np.mean(np.abs(y_true - y_pred))
        elif y_true.ndim == 2:
            return np.mean(np.abs(y_true - y_pred), axis=0)
        return np.nan

    def _mean_of_maes(self, y_true, y_pred):
        if y_true.ndim != 2: return np.nan
        maes_per_series = np.mean(np.abs(y_true - y_pred), axis=0)
        return np.mean(maes_per_series)

    def _mean_of_rmses(self, y_true, y_pred):
        if y_true.ndim != 2: return np.nan
        rmses_per_series = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
        return np.mean(rmses_per_series)

    def _mase(self, y_true, y_pred, y_train, seasonal_period=1):
        if y_train is None: return np.nan
        
        y_t_eval = np.asarray(y_true)
        y_p_eval = np.asarray(y_pred)
        y_tr_eval = np.asarray(y_train)

        if y_t_eval.ndim == 2: # Multivariate
            mase_scores = []
            for i in range(y_t_eval.shape[1]):
                y_t_series = y_t_eval[:, i]
                y_p_series = y_p_eval[:, i]
                y_tr_series = y_tr_eval[:, i] if y_tr_eval.ndim == 2 and y_tr_eval.shape[1] > i else y_tr_eval

                if len(y_tr_series) <= seasonal_period: 
                    mase_scores.append(np.nan) # Not enough train data for this series
                    continue
                
                forecast_errors = np.abs(y_t_series - y_p_series)
                naive_in_sample_errors = np.abs(y_tr_series[seasonal_period:] - y_tr_series[:-seasonal_period])
                mean_abs_naive_error = np.mean(naive_in_sample_errors)

                if mean_abs_naive_error == 0:
                    mase_scores.append(0.0 if np.mean(forecast_errors) == 0 else np.inf)
                else:
                    mase_scores.append(np.mean(forecast_errors) / mean_abs_naive_error)
            return np.array(mase_scores) if mase_scores else np.nan
        
        # Univariate case
        forecast_errors = np.abs(y_t_eval - y_p_eval)
        if len(y_tr_eval) <= seasonal_period: return np.nan
        
        naive_in_sample_errors = np.abs(y_tr_eval[seasonal_period:] - y_tr_eval[:-seasonal_period])
        mean_abs_naive_error = np.mean(naive_in_sample_errors)
        if mean_abs_naive_error == 0:
            return 0.0 if np.mean(forecast_errors) == 0 else np.inf
        return np.mean(forecast_errors) / mean_abs_naive_error

    def _crps(self, y_true, y_pred_dist_samples):
        if y_pred_dist_samples is None: return np.nan

        y_t = np.asarray(y_true)
        y_s = np.asarray(y_pred_dist_samples)

        if y_t.ndim == 1: # Univariate
            if y_t.shape[0] != y_s.shape[0]:
                raise ValueError("CRPS: Mismatch in number of forecast points for y_true and y_pred_dist_samples.")
            crps_values = []
            for i in range(len(y_t)):
                y_obs = y_t[i]
                y_ens = np.sort(y_s[i, :]) 
                m = len(y_ens)
                if m == 0: crps_values.append(np.nan); continue
                term1 = np.sum(np.abs(y_ens - y_obs)) / m
                term2 = np.sum([np.sum(np.abs(y_ens[j] - y_ens[k])) for j in range(m) for k in range(m)]) / (2 * m**2)
                crps_values.append(term1 - term2)
            return np.mean(crps_values)

        elif y_t.ndim == 2: # Multivariate
            if y_t.shape[0] != y_s.shape[0] or y_t.shape[1] != y_s.shape[1]:
                raise ValueError("CRPS: Mismatch in shape for y_true and y_pred_dist_samples.")
            mean_crps_per_series = []
            for s_idx in range(y_t.shape[1]):
                crps_values_series = []
                for i in range(y_t.shape[0]):
                    y_obs = y_t[i, s_idx]
                    y_ens = np.sort(y_s[i, s_idx, :])
                    m = len(y_ens)
                    if m == 0: crps_values_series.append(np.nan); continue
                    term1 = np.sum(np.abs(y_ens - y_obs)) / m
                    term2 = np.sum([np.sum(np.abs(y_ens[j] - y_ens[k])) for j in range(m) for k in range(m)]) / (2 * m**2)
                    crps_values_series.append(term1 - term2)
                mean_crps_per_series.append(np.mean(crps_values_series))
            return np.array(mean_crps_per_series)
        return np.nan

    def _quantile_loss(self, y_true, y_pred_quantiles, quantiles_q_values):
        if y_pred_quantiles is None or quantiles_q_values is None: return {}
        
        y_t = np.asarray(y_true)
        y_q_pred = np.asarray(y_pred_quantiles)
        q_vals = np.asarray(quantiles_q_values)

        if y_t.ndim == 1: # Univariate
            if y_t.shape[0] != y_q_pred.shape[0] or y_q_pred.shape[1] != len(q_vals):
                raise ValueError("Quantile Loss: Shape mismatch for univariate y_true/y_pred_quantiles/quantiles_q_values.")
            losses_all_quantiles = {}
            for i, q_val in enumerate(q_vals):
                errors = y_t - y_q_pred[:, i]
                loss = np.mean(np.maximum(q_val * errors, (q_val - 1) * errors))
                losses_all_quantiles[f"q_{q_val:.2f}"] = loss
            return losses_all_quantiles
        elif y_t.ndim == 2: # Multivariate
            if y_t.shape[0] != y_q_pred.shape[0] or \
               y_t.shape[1] != y_q_pred.shape[1] or \
               y_q_pred.shape[2] != len(q_vals):
                raise ValueError("Quantile Loss: Shape mismatch for multivariate y_true/y_pred_quantiles/quantiles_q_values.")
            
            avg_losses_per_quantile_across_series = {}
            for q_idx, q_val in enumerate(q_vals):
                series_losses = []
                for s_idx in range(y_t.shape[1]):
                    errors = y_t[:, s_idx] - y_q_pred[:, s_idx, q_idx]
                    loss = np.mean(np.maximum(q_val * errors, (q_val - 1) * errors))
                    series_losses.append(loss)
                avg_losses_per_quantile_across_series[f"q_{q_val:.2f}"] = np.mean(series_losses)
            return avg_losses_per_quantile_across_series
        return {}

    def _interval_score(self, y_true, y_pred_lower_bound, y_pred_upper_bound, interval_alpha):
        if y_pred_lower_bound is None or y_pred_upper_bound is None: return np.nan
        
        y_t = np.asarray(y_true)
        lower = np.asarray(y_pred_lower_bound)
        upper = np.asarray(y_pred_upper_bound)

        if y_t.shape != lower.shape or y_t.shape != upper.shape:
             raise ValueError("Interval Score: Shapes of y_true, lower, and upper bounds must match.")
        if interval_alpha <= 0 or interval_alpha >= 1:
            raise ValueError("Interval Score: interval_alpha must be between 0 and 1 (exclusive).")

        interval_width = upper - lower
        penalty_lower = (2 / interval_alpha) * np.maximum(0, lower - y_t)
        penalty_upper = (2 / interval_alpha) * np.maximum(0, y_t - upper)
        score = interval_width + penalty_lower + penalty_upper

        if y_t.ndim == 1:
            return np.mean(score)
        elif y_t.ndim == 2:
            return np.mean(score, axis=0) # Average score per series
        return np.nan

    def evaluate(self, model, data, y_train_series=None):
        """
        Evaluate model performance on given data.
        
        Args:
            model: Trained model instance (must have a .predict() method).
                   The .predict() method can return:
                   - A single NumPy array (deterministic point forecasts).
                   - A dictionary containing keys like 'point', 'samples', 'quantiles', 
                     'lower_bound', 'upper_bound' for probabilistic forecasts.
            data (pd.DataFrame): Evaluation data. Must contain self.target_col_name.
                                 Features (X_eval) will be data.drop(columns=[self.target_col_name]).
            y_train_series (pd.Series or np.array, optional): Original training target series,
                                                            required for MASE.
                                                            
        Returns:
            Dictionary of evaluation metrics
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Evaluation data must be a Pandas DataFrame.")
        if self.target_col_name not in data.columns:
            raise ValueError(f"Target column '{self.target_col_name}' not found in evaluation data.")

        y_true_eval = data[self.target_col_name].values
        
        # Prepare X_eval - assumes all other columns besides target are features
        # This might need to be more sophisticated based on how features were generated
        X_eval_cols = [col for col in data.columns if col != self.target_col_name]
        X_eval = data[X_eval_cols]

        # Get predictions from the model
        predictions_output = model.predict(X_eval)

        # Initialize prediction components
        y_pred_eval = None
        y_pred_dist_samples_eval = None
        y_pred_quantiles_eval = None
        y_pred_lower_bound_eval = None
        y_pred_upper_bound_eval = None

        if isinstance(predictions_output, dict): # Probabilistic model returning a dict
            y_pred_eval = predictions_output.get('point')
            y_pred_dist_samples_eval = predictions_output.get('samples')
            y_pred_quantiles_eval = predictions_output.get('quantiles')
            y_pred_lower_bound_eval = predictions_output.get('lower_bound')
            y_pred_upper_bound_eval = predictions_output.get('upper_bound')
        elif isinstance(predictions_output, (np.ndarray, pd.Series)): # Deterministic model
            y_pred_eval = predictions_output
        else:
            raise TypeError(f"Model prediction output type {type(predictions_output)} not recognized. "
                            "Expected np.ndarray, pd.Series, or dict.")

        # If y_pred_eval is still None but we have probabilistic components, derive it
        if y_pred_eval is None:
            if y_pred_quantiles_eval is not None and self.config.get('quantiles_q_values') is not None:
                q_vals = np.asarray(self.config.get('quantiles_q_values'))
                try:
                    median_idx = np.where(np.isclose(q_vals, 0.5))[0]
                    if len(median_idx) > 0:
                        median_idx = median_idx[0]
                        if y_true_eval.ndim == 1: # (points, quantiles)
                             y_pred_eval = np.asarray(y_pred_quantiles_eval)[:, median_idx]
                        elif y_true_eval.ndim == 2: # (points, series, quantiles)
                             y_pred_eval = np.asarray(y_pred_quantiles_eval)[:, :, median_idx]
                    else: # 0.5 not in quantiles, try to take middle quantile
                        median_idx = y_pred_quantiles_eval.shape[-1] // 2
                        if y_true_eval.ndim == 1: y_pred_eval = np.asarray(y_pred_quantiles_eval)[:, median_idx]
                        elif y_true_eval.ndim == 2: y_pred_eval = np.asarray(y_pred_quantiles_eval)[:, :, median_idx]
                        print("Warning: Median (q=0.5) not found. Using middle quantile for point forecast.")
                except Exception as e:
                    print(f"Warning: Could not derive point forecast from quantiles: {e}")
                    y_pred_eval = np.full_like(y_true_eval, np.nan)
            elif y_pred_dist_samples_eval is not None:
                if y_true_eval.ndim == 1:
                     y_pred_eval = np.mean(np.asarray(y_pred_dist_samples_eval), axis=1)
                elif y_true_eval.ndim == 2:
                     y_pred_eval = np.mean(np.asarray(y_pred_dist_samples_eval), axis=2) # samples assumed to be last axis
        
        if y_pred_eval is None:
            # If we still don't have a point forecast, create a NaN array for MAE/RMSE to not fail
            y_pred_eval = np.full_like(y_true_eval, np.nan)
            print("Warning: No point forecast available for MAE/RMSE calculation.")


        results = {}
        # Deterministic Metrics
        if 'rmse' in self.metrics_to_calculate:
            results['rmse'] = self._rmse(y_true_eval, y_pred_eval)
        if 'mae' in self.metrics_to_calculate:
            results['mae'] = self._mae(y_true_eval, y_pred_eval)
        if 'mean_of_maes' in self.metrics_to_calculate and y_true_eval.ndim == 2:
            results['mean_of_maes'] = self._mean_of_maes(y_true_eval, y_pred_eval)
        if 'mean_of_rmses' in self.metrics_to_calculate and y_true_eval.ndim == 2:
            results['mean_of_rmses'] = self._mean_of_rmses(y_true_eval, y_pred_eval)
        if 'mase' in self.metrics_to_calculate:
            results['mase'] = self._mase(y_true_eval, y_pred_eval, y_train_series, 
                                         seasonal_period=self.config.get('seasonal_period_mase', 1))
        # Probabilistic Metrics
        if 'crps' in self.metrics_to_calculate:
            results['crps'] = self._crps(y_true_eval, y_pred_dist_samples_eval)
        if 'quantile_loss' in self.metrics_to_calculate:
            results['quantile_loss'] = self._quantile_loss(y_true_eval, y_pred_quantiles_eval, 
                                                          self.config.get('quantiles_q_values'))
        if 'interval_score' in self.metrics_to_calculate:
            results['interval_score'] = self._interval_score(y_true_eval, y_pred_lower_bound_eval, 
                                                             y_pred_upper_bound_eval, 
                                                             self.config.get('interval_alpha', 0.1))
        return results