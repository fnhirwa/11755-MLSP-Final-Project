import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    dates: pd.DatetimeIndex
    indices: np.ndarray
    residuals: np.ndarray
    scores: np.ndarray
    directions: np.ndarray
    actual_values: np.ndarray
    predicted_values: np.ndarray
    threshold: float
    method: str
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': self.dates,
            'actual': self.actual_values,
            'predicted': self.predicted_values,
            'residual': self.residuals,
            'score': self.scores,
            'direction': np.where(self.directions > 0, 'positive', 'negative')
        })
    
    def summary(self) -> dict:
        return {
            'total': len(self),
            'positive': int(np.sum(self.directions > 0)),
            'negative': int(np.sum(self.directions < 0)),
            'mean_abs_residual': float(np.mean(np.abs(self.residuals))),
            'max_abs_residual': float(np.max(np.abs(self.residuals))),
        }


class ResidualAnomalyDetector:
    """
    Residual-based anomaly detector using z-score or IQR method.
    
    Parameters
    ----------
    method : str, default='zscore'
        Detection method: 'zscore' or 'iqr'
    threshold : float, default=2.5
        For zscore: number of standard deviations
        For iqr: multiplier for IQR (typically 1.5 or 3.0)
    """
    
    def __init__(self, method: str = 'zscore', threshold: float = 2.5):
        self.method = method.lower()
        self.threshold = threshold
        
    def detect(
        self,
        y_actual: Union[pd.Series, np.ndarray],
        y_predicted: Union[pd.Series, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None
    ) -> AnomalyResult:
        """
        Detect anomalies based on forecast residuals.
        
        Parameters
        ----------
        y_actual : array-like
            Actual observed values
        y_predicted : array-like
            Predicted values
        dates : DatetimeIndex, optional
            Dates for the observations
            
        Returns
        -------
        AnomalyResult
        """
        y_actual = np.asarray(y_actual)
        y_predicted = np.asarray(y_predicted)
        residuals = y_actual - y_predicted
        
        # Compute scores and identify anomalies
        if self.method == 'zscore':
            mean, std = np.mean(residuals), np.std(residuals, ddof=1)
            scores = (residuals - mean) / std if std > 0 else np.zeros_like(residuals)
            anomaly_mask = np.abs(scores) > self.threshold
        else:  # iqr
            q1, q3 = np.percentile(residuals, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - self.threshold * iqr, q3 + self.threshold * iqr
            scores = np.where(residuals > q3, (residuals - q3) / iqr,
                            np.where(residuals < q1, (q1 - residuals) / iqr, 0))
            anomaly_mask = (residuals < lower) | (residuals > upper)
        
        # Handle dates
        if dates is None:
            dates = pd.date_range('2020-01-01', periods=len(y_actual))
        
        idx = np.where(anomaly_mask)[0]
        
        return AnomalyResult(
            dates=dates[idx],
            indices=idx,
            residuals=residuals[anomaly_mask],
            scores=scores[anomaly_mask],
            directions=np.sign(residuals[anomaly_mask]),
            actual_values=y_actual[anomaly_mask],
            predicted_values=y_predicted[anomaly_mask],
            threshold=self.threshold,
            method=self.method
        )
    
    def detect_from_model(self, model, y: pd.Series) -> AnomalyResult:
        """Detect anomalies using fitted values from an ARIMA/ARIMAX model."""
        if model.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        fitted = model.fitted_model.fittedvalues
        y_aligned = y.loc[fitted.index]
        return self.detect(y_aligned.values, fitted.values, fitted.index)


def detect_anomalies(
    y_actual: Union[pd.Series, np.ndarray],
    y_predicted: Union[pd.Series, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    method: str = 'zscore',
    threshold: float = 2.5
) -> AnomalyResult:
    """
    Convenience function for anomaly detection.
    
    Examples
    --------
    >>> results = detect_anomalies(actual, predicted, threshold=2.5)
    >>> print(f"Found {len(results)} anomalies")
    >>> df = results.to_dataframe()
    """
    return ResidualAnomalyDetector(method, threshold).detect(y_actual, y_predicted, dates)


def evaluate_detection(
    detected: AnomalyResult,
    ground_truth_dates: pd.DatetimeIndex,
    tolerance_days: int = 0
) -> dict:
    """
    Evaluate detection against known event dates.
    
    Returns dict with precision, recall, f1_score.
    """
    detected_set = set(detected.dates.date)
    gt_set = set(ground_truth_dates.date)
    
    if tolerance_days > 0:
        expanded = set()
        for d in gt_set:
            for delta in range(-tolerance_days, tolerance_days + 1):
                expanded.add(d + pd.Timedelta(days=delta))
        gt_set = expanded
    
    tp = len(detected_set & gt_set)
    fp = len(detected_set - gt_set)
    fn = len(gt_set - detected_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


__all__ = ['AnomalyResult', 'ResidualAnomalyDetector', 'detect_anomalies', 'evaluate_detection']