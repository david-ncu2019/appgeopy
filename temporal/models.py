import numpy as np
import pandas as pd
from scipy.optimize import least_squares

class Models:
    """Sinusoidal fitting and synthetic data generation"""
    
    @staticmethod
    def synthetic_daily_signal(start_date="2020-01-01", end_date="2024-12-31", **kwargs):
        # synthetic_daily_signal() function content from modeling.py
        
    @staticmethod
    def prepare_sinusoidal_model_inputs(time_series_data, seasonality_info, select_col=None):
        # prepare_sinusoidal_model_inputs() function content from modeling.py
        
    @staticmethod
    def fit_sinusoidal_model(time_values, observed_values, amplitudes, periods, phase_shifts, baseline, predict_time=None):
        # fit_sinusoidal_model() function content from modeling.py
        
    @staticmethod
    def sinusoidal_model(time_values, amplitude_terms, baseline):
        # sinusoidal_model() function content from modeling.py
        
    @staticmethod
    def least_squares_loss(parameters, time_values, observed_values, amplitudes, periods):
        # least_squares_loss() function content from modeling.py