# Copyright (c) 2025 Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Authors: Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Email: giuseppe.stecca@gmail.com

from bilevel_params import *
dd = Data('I_0V', True, 'instances_bilevel.csv')
CO2 = dd.maxCO2()
print(f"max CO2 is : {CO2}")
print ("*******************")
C = {t: dd.maxCO2() *(1-(dd.delta/(dd.LT-1))*(t-1)) for t in dd.T}
print("CAP OVER T IS:")
print(C)