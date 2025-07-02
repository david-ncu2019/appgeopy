import numpy as np

class Displacement:
    """InSAR displacement calculations"""
    
    @staticmethod
    def degree_to_radian(degree):
        # degree_to_radian() function content from insartools/ts_disp.py
        
    @staticmethod
    def get_LOS_disp(dN, dE, dU, incidence_angle=37, heading_angle=347.6):
        # get_LOS_disp() function content from insartools/ts_disp.py
        
    @staticmethod
    def compare_LOS_disp(psc_df, gps_df, mutual_index):
        # compare_LOS_disp() function content from insartools/ts_disp.py
        
    @staticmethod
    def convert_cumdisp_to_disp(cumulative_series):
        # convert_cumdisp_to_disp() function content from insartools/ts_disp.py