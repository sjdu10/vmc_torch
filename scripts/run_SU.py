import sys
import os

L_list = [20, 40, 60, 80, 100, 120, 140, 160]

for L in L_list:
    os.system(f'python SU_run_1d_input_L.py {L}')