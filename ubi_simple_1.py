# -*- coding: utf-8 -*-
"""UBI Simple 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uk4jmEGu4NefRBOoqUIyPcPwKASrwP3z
"""

# @title Variables poblacionales iniciales { run: "auto", display-mode: "both" }
from utils import ModeloUBI, graficar_modelo_iva_ubi, graficar_demo_real_ubi

a0 = 11033056  # @param {type:"number"}
b0 = 28571872  # @param {type:"number"}
c0 = 4990720  # @param {type:"number"}
d0 = 0  # @param {type:"number"}

b0_1 = b0 * 0.5
b0_2 = b0 * 0.5

print("Población total (N) = {}".format(a0 + b0 + c0 + d0))

# @title Híper-parámetros { display-mode: "both" }
theta = 0.009  # @param {type:"slider", min:0.001, max:0.1, step:0.01}
kappa = 0.014  # @param {type:"slider", min:0.001, max:0.1, step:0.01}
phi = 0.055  # @param {type:"slider", min:0.001, max:0.5, step:0.001}

periodos_pase_B2 = 8  # @param {type:"number"}

omega = 0.1 / periodos_pase_B2

modelo = ModeloUBI(a0, b0, b0_1, b0_2, c0, d0, phi, theta, kappa, omega)

# @title Correr simulación poblacional (modelo 1) { run: "auto", vertical-output: true, display-mode: "both" }

t_deseado = 18  # @param {type:"integer"}

graficar_modelo_iva_ubi()
graficar_modelo_iva_ubi(comparado_con_iva=False)

modelo.graficar_modelo_progresivo(t_deseado)
modelo.graficar_modelo_shock(t_deseado)
graficar_demo_real_ubi()




