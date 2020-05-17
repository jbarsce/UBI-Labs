import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def graficar_modelo_iva_ubi(comparado_con_iva: bool = True, en_usd: bool = False) -> None:
    datos = pd.read_csv('data/UBI.csv')

    if comparado_con_iva:
        impuesto = 'PBI_IVA'
        title = 'Comparativa IVA-UBI'
        y_label = 'Diferencia IVA - UBI (en %)'
    else:
        impuesto = 'PBI_INDIRECTOS'
        title = 'Comparativa INDIRECTOS-UBI'
        y_label = 'Diferencia INDIRECTOS - UBI (en %)'

    if en_usd:
        d_per_capita = 'D_per_capita_USD'
        pbi = 'PBI_USD'
        title += ' en (USD)'
    else:
        d_per_capita = 'D_per_capita'
        pbi = 'PBI'

    datos['masa_total_d'] = datos['D'] * datos[d_per_capita]
    datos['masa_total_pbi'] = datos['masa_total_d'] / datos[pbi] * 100
    datos['diferencia_impuesto_ubi'] = datos[impuesto] - datos['masa_total_pbi']

    plt.plot([2004, 2018], [0, 0], '--', c='grey')
    plt.plot(datos['AÑO'], datos['diferencia_impuesto_ubi'])
    plt.title(title)
    plt.xlabel('Año')
    plt.ylabel(y_label)
    plt.show()


def graficar_demo_real_ubi() -> None:
    datos = pd.read_csv('data/UBI_demo.csv')

    plt.plot(datos['AÑO'], datos['N'], label='N')
    plt.plot(datos['AÑO'], datos['B'], label='B')
    plt.plot(datos['AÑO'], datos['C2004'], label='C')
    plt.plot(datos['AÑO'], datos['D'], label='D')
    plt.legend(loc='best', numpoints=1)
    plt.title('Modelo UBI a partir de datos reales')
    plt.xlabel('Período de tiempo')
    plt.ylabel('Población')
    plt.show()


class ModeloUBI:

    def __init__(self, a0, b0, b0_1, b0_2, c0, d0, phi, theta, kappa, omega):
        self.a0 = a0
        self.b0 = b0
        self.b0_1 = b0_1
        self.b0_2 = b0_2
        self.c0 = c0
        self.d0 = d0
        self.phi = phi
        self.theta = theta
        self.kappa = kappa
        self.omega = omega

        self.n0 = np.sum([a0, b0, c0, d0])

    def n_primer_modelo_t(self, t: int):
        if t == 0:
            return self.a0, self.b0, self.c0, self.d0
        else:
            at_prev, bt_prev, ct_prev, dt_prev = self.n_primer_modelo_t(t - 1)

            at = at_prev - self.phi * self.a0
            bt = bt_prev - self.phi * self.b0
            ct = ct_prev - self.theta * ct_prev
            dt = self.phi * self.a0 + self.phi * self.b0 + self.kappa * self.n0 + dt_prev
            return at, bt, ct, dt

    def n_segundo_modelo_t(self, t: int):
        if t == 0:
            return self.a0, self.b0, self.c0, self.d0
        elif t == 1:
            at = self.a0 - 0.2 * self.a0
            bt = self.b0_2 - self.omega * self.b0_2
            ct = self.c0 - self.theta * self.c0
            dt = 0.2 * self.a0 + self.omega * self.b0_2 + self.kappa * self.n0 + self.b0_1 + self.d0
            return at, bt, ct, dt
        else:
            at_prev, bt_prev, ct_prev, dt_prev = self.n_segundo_modelo_t(t - 1)

            at = at_prev - 0.066 * self.a0
            bt = bt_prev - self.omega * self.b0_2
            ct = ct_prev - self.theta * ct_prev
            dt = 0.066 * self.a0 + self.omega * self.b0_2 + self.kappa * self.n0 + dt_prev
            return at, bt, ct, dt

    def graficar_modelo_progresivo(self, t_deseado: int) -> None:
        t_list = np.arange(t_deseado + 1)
        a_list = [self.n_primer_modelo_t(t)[0] for t in t_list]
        b_list = [self.n_primer_modelo_t(t)[1] for t in t_list]
        c_list = [self.n_primer_modelo_t(t)[2] for t in t_list]
        d_list = [self.n_primer_modelo_t(t)[3] for t in t_list]
        res_list = np.sum((a_list, b_list, c_list, d_list), axis=0)

        plt.plot(t_list, res_list, label='N')
        plt.plot(t_list, a_list, label='A')
        plt.plot(t_list, b_list, label='B')
        plt.plot(t_list, c_list, label='C')
        plt.plot(t_list, d_list, label='D')
        plt.legend(loc='best', numpoints=1)
        plt.title('Modelo UBI Progresivo')
        plt.xlabel('Período de tiempo')
        plt.ylabel('Población')

        plt.show()

    def graficar_modelo_shock(self, t_deseado: int) -> None:
        t_list = np.arange(t_deseado + 1)
        a_list = [self.n_segundo_modelo_t(t)[0] for t in t_list]
        b_list = [self.n_segundo_modelo_t(t)[1] for t in t_list]
        c_list = [self.n_segundo_modelo_t(t)[2] for t in t_list]
        d_list = [self.n_segundo_modelo_t(t)[3] for t in t_list]
        res_list = np.sum((a_list, b_list, c_list, d_list), axis=0)

        plt.plot(t_list, res_list, label='N')
        plt.plot(t_list, a_list, label='A')
        plt.plot(t_list, b_list, label='B1')
        plt.plot(t_list, c_list, label='C')
        plt.plot(t_list, d_list, label='D')
        plt.legend(loc='best', numpoints=1)
        plt.title('Modelo UBI de Shock')
        plt.xlabel('Período de tiempo')
        plt.ylabel('Población')

        plt.show()


