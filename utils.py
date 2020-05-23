import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def graficar_modelo_iva_ubi(comparado_con_iva: bool = True, en_usd: bool = False) -> None:
    datos = pd.read_csv('data/UBI.csv')
    datos_demo = pd.read_csv('data/UBI_demo.csv')

    if comparado_con_iva:
        impuesto = 'PBI_IVA'
        titulo = 'Comparativa IVA-UBI'
        y_label = 'Diferencia IVA - UBI (en %)'
    else:
        impuesto = 'PBI_INDIRECTOS'
        titulo = 'Comparativa INDIRECTOS-UBI'
        y_label = 'Diferencia INDIRECTOS - UBI (en %)'

    if en_usd:
        d_per_capita = 'D_per_capita_USD'
        pbi = 'PBI_USD'
        titulo += ' en (USD)'
    else:
        d_per_capita = 'D_per_capita'
        pbi = 'PBI'

    datos['masa_total_d'] = datos['D'] * datos[d_per_capita]
    datos['masa_total_pbi'] = datos['masa_total_d'] / datos[pbi] * 100
    datos['diferencia_impuesto_ubi'] = datos[impuesto] - datos['masa_total_pbi']

    sns.set()
    ax = plt.gca()
    ax2 = ax.twinx()
    plt.grid()
    plt.plot([2004, 2018], [0, 0], '--', c='grey')
    ax2.plot(datos['AÑO'], datos['diferencia_impuesto_ubi'], color='black')
    ax.bar(datos['AÑO'],100 * datos['D']/(datos_demo['N'] - datos_demo['C2004']),
           label='Cobertura de UBI', width=0.65, color='lightblue')
    plt.title(titulo)
    ax.legend(loc='lower center', numpoints=1)
    plt.xlabel('Año')
    ax2.set_ylabel(y_label)
    ax.set_ylabel('grado de cobertura (%)')
    plt.show()


def graficar_demo_real_ubi() -> None:
    datos = pd.read_csv('data/UBI_demo.csv')

    sns.set()

    plt.plot(datos['AÑO'], datos['N']/1e6, label='N (población total)')
    plt.plot(datos['AÑO'], datos['B']/1e6, label='B (entre 0-64 años sin IBU)')
    plt.plot(datos['AÑO'], datos['C2004']/1e6, label='C (jubilados)')
    plt.plot(datos['AÑO'], datos['D']/1e6, label='D (población con IBU)')

    plt.legend(loc='best', numpoints=1)
    plt.title('Modelo UBI a partir de datos reales')
    plt.xlabel('Período de tiempo')
    plt.ylabel('Población (millones de habitantes)')
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

    def graficar_modelo(self, t_deseado: int, tipo_modelo:str = 'progresivo') -> None:

        if tipo_modelo == 'progresivo':
            modelo = self.n_primer_modelo_t
            titulo = 'Modelo UBI Progresivo'
            b_label = 'B'
        elif tipo_modelo == 'shock':
            modelo = self.n_segundo_modelo_t
            titulo = 'Modelo UBI de Shock'
            b_label = 'B1'
        else:
            raise NotImplementedError

        t_list = np.array([modelo(t) for t in np.arange(t_deseado + 1)])/1e6
        t_list_filas = np.arange(t_deseado + 1)

        a_list = t_list[:, 0]
        b_list = t_list[:, 1]
        c_list = t_list[:, 2]
        d_list = t_list[:, 3]
        res_list = np.sum((a_list, b_list, c_list, d_list), axis=0)

        sns.set()
        plt.plot(t_list_filas, res_list, label='N (población total)')
        plt.plot(t_list_filas, a_list, label='A (menores de 18 sin IBU)')
        plt.plot(t_list_filas, b_list, label='{} (entre 18 y 64 años sin IBU)'.format(b_label))
        plt.plot(t_list_filas, c_list, label='C (jubilados)')
        plt.plot(t_list_filas, d_list, label='D (población con IBU)')
        plt.legend(loc='best', numpoints=1)
        plt.title(titulo)
        plt.xlabel('Período de tiempo')
        plt.ylabel('Población (millones de habitantes)')

        plt.show()


