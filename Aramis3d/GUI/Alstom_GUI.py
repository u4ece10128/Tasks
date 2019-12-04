import os
# import Simulate
import tkinter
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from tkinter import *
from tkinter import ttk
import tkinter.messagebox as mbox
from PIL import Image, ImageTk


class ComputeRCM:
    """
    Computes, alpha beta parameters
    Computes total maintenance cost
    LCC of a component with a test all data must be provided in hours
    """
    def __init__(self):
        self.Weibull_parameter = np.array([], dtype=np.float64)

        self.costsche = None
        self.fallimenti = None
        self.maintenancesche = None
        self.testsche = None
        self.detected = None
        self.undetected = None
        self.tm_und = None

    def alpha_beta_comb(self, alpha, beta, n):
        """
        :param alpha: Scale Parameter <type: np.float64>
        :param beta: Shape Parameter <type: np.float64>
        :param n: Number of Monte carlo Simulations <type: int>
        :return: Weibull parameter
        """
        l = 1
        for i in range(0, len(n)):
            l = n[i] * l

        A = np.ones((len(n), l), dtype=np.float64)

        # # first column as ones
        A[0:len(n), 0] = 1

        for i in range(1, l):
            A[0, i] = A[0, i-1] + 1
            if A[0, i] == n[0] + 1:
                A[0, i] = 1
        for j in range(1, len(n)):
            m = 1
            for k in range(0, j):
                m = m * n[k]
            # (low_thr, high_thr, step_size)
            for i in range(m, l, m):
                A[j, i:i+m] = A[j, i-1] + 1
                if A[j, i] == n[j] + 1:
                    A[j, i:i+m] = 1

        self.Weibull_parameter = np.zeros((A.shape[0], A.shape[1], 2), dtype=np.float64)

        for i in range(0, len(n)):
            for j in range(1, n[i]+1):
                index_alpha = np.where(1*(A[i, :] == j))
                for index in index_alpha:
                    self.Weibull_parameter[i, index, 0] = alpha[i, j-1]
                    self.Weibull_parameter[i, index, 1] = beta[i, j-1]

    def cost_test(self, n, horizon, alpha_w, beta_w, hidden, ccor, cprev, tsched, ttest, warranty, compatibility,
                  compatibility_cm, co_crew_h, MTTR, time_preventive, time_test, chi_w, depreciation):
        """
        Cost test function computes he total cost of maintenance
        :param n:
        :param horizon: time at which simulation ends
        :param alpha_w: vector shape parameter of weibull distribution for n failure modes
        :param beta_w: vector scale parameter of weibull distribution for n failure modes
        :param hidden: hidden fault matrix
        :param ccor: cost of corrective maintenance
        :param cprev: m dimensional vector cost of preventive maintenance
        :param tsched: time between consecutive inspections
        :param ttest:  time between different tests
        :param warranty: time at which the warranty ends; if(warranty==0), we are not following the warranty policy
        :param compatibility: n x m matrix with binary entries, where n is the number of failure modes and m
        the number of possible preventive actions: comptibilty(i,j)=1
        means that preventive action j restores failure mode i
        :param compatibility_cm:
        :param co_crew_h: cost of hour of repair
        :param MTTR: mean time to repair
        :param time_preventive: m dimensional vector; m(j) is time to perform the preventive action
        :param time_test: mean time to repair
        :param chi_w: parameter indicating the increased risk of failure
        :param depreciation: Set to 1, depreciation of the component
        :return: costsche : cost of preventive maintenance
        :return: failures: Number of failures
        :return: maintenancesche:
        :return: testsche:
        :return: detected:
        :return: undetected:
        :return: tm_und
        """
        # set depreciation to 1
        nfailuremode = len(alpha_w)

        # initialize return variables
        self.costsche = np.zeros((n, 1), dtype=np.float64)
        self.fallimenti = np.zeros((n, 1), dtype=np.float64)
        self.maintenancesche = np.zeros((n, 1), dtype=np.float64)
        self.testsche = np.zeros((n, 1), dtype=np.float64)
        self.detected = np.zeros((n, 1), dtype=np.float64)
        self.undetected = np.zeros((n, 1), dtype=np.float64)
        self.tm_und = np.zeros((n, 1), dtype=np.float64)

        if np.sum(hidden) == 0:
            ttest = 2*horizon

        for k in range(0, n):
            print("Num of Monte carlo:", k)
            t_und = [0, 0]
            tbreakP = np.zeros((nfailuremode, 1), dtype=np.float64)
            tstopP = 0
            tmonitor = tsched
            tnexttest = ttest
            thidden = 2 * horizon

            # np.random.weibull(a) ---> Draw samples from a Weibull distribution.
            # Draw samples from a 1-parameter Weibull distribution with the given shape parameter a.
            for nf in range(0, nfailuremode):
                # scale parameter * np.random.weibull(shape_parameter)
                tbreakP[nf, 0] = alpha_w[nf] * np.random.weibull(beta_w[nf])

            while tstopP < horizon:
                next_failure = tbreakP.min(0)
                which_failuremode = np.argmin(tbreakP)
                # next_maintenance = tmonitor.min(0)
                next_maintenance = tmonitor
                which_maintenance = np.argmin(tmonitor)
                next_test = tnexttest
                next_event = min([next_failure, next_test, next_maintenance])
                which_event = np.argmin([next_failure, next_test, next_maintenance])
                tstopP = next_event

                if tstopP < horizon:
                    if which_event == 0:
                        print("FM happens")
                        if hidden[which_failuremode] == 0:
                            self.fallimenti[k] = self.fallimenti[k] + 1
                            if thidden < (2 * horizon):
                                t_und[0] = t_und[0] + (tstopP - thidden)
                                t_und[1] = t_und[1] + 1
                                self.undetected[k] = self.undetected[k] + 1
                            if np.random.rand() <= compatibility_cm[which_failuremode]:
                                for nf in range(0, nfailuremode):
                                    tbreakP[nf, 0] = tstopP + (alpha_w[nf] * np.random.weibull(beta_w[nf]))
                                if tstopP > warranty:
                                    self.costsche[k] = self.costsche[k] + (((MTTR[0] * co_crew_h) + ccor[0])
                                                                           * depreciation **
                                                                           np.floor(tbreakP[which_maintenance])[0])
                            else:
                                gamma = tstopP
                                tbreakP[which_failuremode, 0] = tstopP + alpha_w[which_failuremode] * \
                                                                ((gamma/alpha_w[which_failuremode]) **
                                                                 beta_w[which_failuremode] - np.log(1 - np.random.rand()
                                                                                                    )) ** \
                                                                (1/beta_w[which_failuremode]) - gamma
                                if tstopP > warranty:
                                    self.costsche[k] = self.costsche[k] + ((MTTR[1] * co_crew_h) + ccor[1]) * \
                                                   depreciation ** np.floor(tbreakP[which_maintenance])
                            thidden = 2 * horizon
                        else:
                            if thidden == 2 * horizon:
                                thidden = tstopP
                                for nf in range(0, nfailuremode):
                                    if nf != which_failuremode and chi_w[nf, which_failuremode] != 0:
                                        weibull_sample = 1 / chi_w ** (1 / beta_w[nf]) * alpha_w[nf] * \
                                                         np.random.weibull(beta_w[nf])
                                        tbreakP[nf, 0] = min(tbreakP[nf, 0], tstopP + weibull_sample, beta_w[nf])
                                    else:
                                        tbreakP[nf, 0] = 2 * horizon
                            tbreakP[which_failuremode, 0] = 2 * horizon
                    if which_event == 1:
                        print("Perform Test")
                        tnexttest = tnexttest + ttest
                        if next_test != next_maintenance:
                            self.costsche[k] = self.costsche[k] + \
                                               ((time_test * co_crew_h) * depreciation ** np.floor(tstopP))
                            self.testsche[k] = self.testsche[k] + 1
                        if thidden < 2 * horizon:
                            for fn in range(0, nfailuremode):
                                tbreakP[fn, 0] = tstopP + (alpha_w[fn] * np.random.weibull(beta_w[fn]))
                            tmonitor = tstopP + tsched
                            self.fallimenti[k] = self.fallimenti[k] + 1
                            self.detected[k] = self.detected[k] + 1
                            t_und[0] = t_und[0] + (tstopP - thidden)
                            t_und[1] = t_und[1] + 1
                            if tstopP > warranty:
                                self.costsche[k] = self.costsche[k] + \
                                                   ((MTTR[0] * co_crew_h) + ccor[0]) * depreciation ** np.floor(tstopP)
                            thidden = 2 * horizon
                            if next_test == next_maintenance:
                                # tmonitor[which_maintenance, 0] = tstopP + tsched[which_maintenance]
                                tmonitor = tstopP + tsched
                    if which_event == 2:
                        print("Perform PM")
                        self.costsche[k] = self.costsche[k] + ((time_test * co_crew_h) * depreciation **
                                                               np.floor(tstopP))
                        self.testsche[k] = self.testsche[k] + 1
                        if thidden < 2 * horizon:
                            for fn_2 in range(0, nfailuremode):
                                tbreakP[fn_2, 0] = tstopP + (alpha_w[fn_2] * np.random.weibull(beta_w[fn_2]))
                            tmonitor = tstopP + tsched
                            self.fallimenti[k] = self.fallimenti[k] + 1
                            self.detected[k] = self.detected[k] + 1
                            t_und[0] = t_und[0] + (tstopP - thidden)
                            t_und[1] = t_und[1] + 1

                            if tstopP > warranty:
                                self.costsche[k] = self.costsche[k] + (MTTR[0] * co_crew_h + ccor[0]) \
                                              * depreciation ** np.floor(tstopP)
                            thidden = 2 * horizon
                            if next_test == next_maintenance:
                                if tmonitor == next_maintenance:
                                    tmonitor = tstopP + tsched
                                else:
                                    tmonitor = tstopP
                        else:
                            qualiaggiorno = np.where(compatibility[:, which_maintenance] == 1)
                            for id_q in qualiaggiorno:
                                tbreakP[id_q, 0] = tstopP + (alpha_w[id_q] * np.random.weibull(beta_w[id_q]))
                            tmonitor = tstopP + tsched
                            self.maintenancesche[k] = self.maintenancesche[k] + 1
                            # time_preventive is a int
                            self.costsche[k] = self.costsche[k] + ((time_preventive * co_crew_h
                                                                   + cprev)
                                                                   * depreciation ** np.floor(tstopP))
            if thidden < horizon:
                self.fallimenti[k] = self.fallimenti[k] + 1
                self.undetected[k] = self.undetected[k] + 1
                t_und[0] = t_und[0] + (horizon-thidden)
                t_und[1] = t_und[1] + 1
            if t_und[1] == 0:
                self.tm_und[k] = 0
            else:
                self.tm_und[k] = t_und[0] / t_und[1]


class Simulate:
    def __init__(self, params):
        self.params = params
        self.obj_instance = ComputeRCM()
        self.w_parameter = self.get_w_parameter()

        self.fall = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                             dtype=np.float64)
        self.man = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                            dtype=np.float64)
        self.test = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                             dtype=np.float64)
        self.det = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                            dtype=np.float64)
        self.undet = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                            dtype=np.float64)
        self.t_fail_h = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                            dtype=np.float64)
        self.costsc = np.zeros((self.w_parameter.shape[1],
                             ((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                             len(self.params.fr_sens)),
                            dtype=np.float64)

        self.Costsc_max = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                    len(self.params.fr_sens)), dtype=np.float64)
        # self.Costsc_min = np.zeros((len(self.params.PmT), len(self.params.fr_sens)), dtype=np.float64)
        self.Costsc_min = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                    len(self.params.fr_sens)), dtype=np.float64)
        self.fall_max = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                  len(self.params.fr_sens)), dtype=np.float64)
        self.fall_min = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                  len(self.params.fr_sens)), dtype=np.float64)
        self.l_size = 2

        self.det_max = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                 len(self.params.fr_sens)), dtype=np.float64)
        self.det_min = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                 len(self.params.fr_sens)), dtype=np.float64)

        self.undet_max = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                  len(self.params.fr_sens)), dtype=np.float64)
        self.undet_min = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                  len(self.params.fr_sens)), dtype=np.float64)
        self.t_fail_h_max = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                      len(self.params.fr_sens)), dtype=np.float64)
        self.t_fail_h_min = np.zeros((((len(self.params.Tst)-1) * len(self.params.PmT)) + len(self.params.PmT),
                                     len(self.params.fr_sens)), dtype=np.float64)

    def print_params(self):
        print(self.params.__dict__)

    def get_w_parameter(self):
        w_parameter = np.zeros((len(self.params.n), np.prod(self.params.n), 2, len(self.params.fr_sens)))
        # self.obj_instance = utils.Utils()
        self.obj_instance = ComputeRCM()
        for k in range(0, len(self.params.fr_sens)):
            Fm = self.params.FT * self.params.U * self.params.P_failure_mode * self.params.fr_sens[k]
            alpha = np.zeros((len(self.params.n), max(self.params.n)))
            for i in range(0, len(self.params.n)):
                for j in range(0, self.params.n[i]):
                    alpha[i, j] = self.params.Tau[i] / ((Fm[i] * self.params.Tau[i]) ** (1 / self.params.beta[i, j]))
            self.obj_instance.alpha_beta_comb(alpha, self.params.beta, self.params.n)
            w_parameter[:, :, :, k] = self.obj_instance.Weibull_parameter
        return w_parameter

    def run_cost_test(self):
        #  Cycle that turns the motor once for each different combination of Weibull parameters
        for k in range(0, len(self.params.fr_sens)):
            for c in range(0, len(self.params.Tst)):
                t_test = self.params.Tst[c]
                t_test = t_test * 365 * self.params.h_day
                # here PmT is an int - 31
                for p in range(0, self.params.PmT.shape[0]):
                    tsc = self.params.PmT[p]
                    tsc = tsc * 365 * self.params.h_day
                    for l in range(0, self.w_parameter.shape[1]):
                        self.obj_instance.cost_test(self.params.n_sim,
                                                    self.params.Life,
                                                    self.w_parameter[:, l, 0, k],
                                                    self.w_parameter[:, l, 1, k],
                                                    self.params.hidden,
                                                    self.params.Ccor,
                                                    self.params.Csc,
                                                    tsc,
                                                    t_test,
                                                    self.params.Tgaranzia,
                                                    self.params.Comp,
                                                    self.params.Comp_cm,
                                                    self.params.co_crew_h,
                                                    self.params.wkl_cm,
                                                    self.params.wkl_pm,
                                                    self.params.wkl_ts,
                                                    self.params.chi_w,
                                                    1)

                    # Calculation of average values ​​on the lives and on the different combinations of parameters
                    self.fall[:, c * (self.params.PmT.shape[0]-1) + p, k] = np.mean(self.get_fallimenti())
                    self.man[:, c * (self.params.PmT.shape[0]-1) + p, k] = np.mean(self.get_maintenancesche())
                    self.test[:, c * self.params.PmT.shape[0] + p, k] = np.mean(self.get_testsche())
                    self.det[:, c * self.params.PmT.shape[0] + p, k] = np.mean(self.get_detected())
                    self.undet[:, c * self.params.PmT.shape[0] + p, k] = np.mean(self.get_undetected())
                    self.t_fail_h[:, c * self.params.PmT.shape[0] + p, k] = np.mean(self.get_tm_und())
                    self.costsc[:, c * self.params.PmT.shape[0] + p,  k] = np.mean(self.get_costsche())

    def get_costsche(self):
        return self.obj_instance.costsche

    def get_fallimenti(self):
        return self.obj_instance.fallimenti

    def get_maintenancesche(self):
        return self.obj_instance.maintenancesche

    def get_testsche(self):
        return self.obj_instance.testsche

    def get_detected(self):
        return self.obj_instance.detected

    def get_undetected(self):
        return self.obj_instance.undetected

    def get_tm_und(self):
        return self.obj_instance.tm_und

    def get_fall(self):
        return self.fall

    def get_man(self):
        return self.man

    def get_test(self):
        return self.test

    def get_det(self):
        return self.det

    def get_undet(self):
        return self.get_undet

    def get_t_fail_h(self):
        return self.t_fail_h

    def get_costsc(self):
        return self.costsc

    def plot_results(self):
        self.plot_mandatory_results()
        self.plot_is_hidden_results()

    def plot_mandatory_results(self):
        """
        Graph 1 - costs with varying PM intervals - the maximum value (continuous line)
        and minimum (dotted line) on the various beta / alpha are plotted for each FR
        :return:
        """
        cost_tst_max = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
        cost_tst_min = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
        cost_pmt_max = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
        cost_pmt_min = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)

        self.Costsc_max[:, :] = self.costsc.max(0)
        self.Costsc_min[:, :] = self.costsc.min(0)

        if not os.path.isdir(self.params.component):
            os.system('mkdir ' + self.params.component)

        x_time = np.zeros((self.params.Tst.shape[0]), dtype=np.int64)

        if len(self.params.Tst) > 1:
            x_time[:] = self.params.Tst

        for tst_sample in range(0, self.params.Tst.shape[0]):
            temp_max = self.Costsc_max[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                       self.params.PmT.shape[0] - 1, :]
            temp_min = self.Costsc_min[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                       self.params.PmT.shape[0] - 1, :]
            cost_tst_max[tst_sample, :] = temp_max.max(0)
            cost_tst_min[tst_sample, :] = temp_min.min(0)

        for pmt_sample in range(0, self.params.PmT.shape[0]):
            a = np.arange(pmt_sample, self.Costsc_max.shape[0], len(self.params.PmT))
            temp_max = self.Costsc_max[a, :]
            temp_min = self.Costsc_min[a, :]
            cost_pmt_max[pmt_sample, :] = temp_max.max(0)
            cost_pmt_min[pmt_sample, :] = temp_min.min(0)

        # Turn ON grid
        print("IN")
        plt.figure(0)
        plt.grid(True)
        print("Out")
        for i in range(0, cost_tst_max.shape[1]):
            plt.plot(x_time, cost_tst_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)
        for i in range(0, cost_tst_min.shape[1]):
            plt.plot(x_time, cost_tst_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)

        fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
        plt.title("Cost vs. FR")
        plt.legend(fr_legend, loc="upper right")
        plt.xlabel("Test interval [years]")
        plt.ylabel("Total maintenance cost")
        print("In")
        plt.savefig(self.params.component + '/' + self.params.component + "_Cost_TsT.png")

        plt.figure(1)
        plt.grid(True)

        x_time = np.zeros((self.params.PmT.shape[0]), dtype=np.int64)
        if len(self.params.PmT) > 1:
            x_time[:] = self.params.PmT

        for i in range(0, cost_pmt_max.shape[1]):
            plt.plot(x_time, cost_pmt_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)

        for i in range(0, cost_pmt_min.shape[1]):
            plt.plot(x_time, cost_pmt_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)

        fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
        plt.title("Cost vs. FR")
        plt.legend(fr_legend, loc="upper right")
        plt.xlabel("PM interval [years]")
        plt.ylabel("Total maintenance cost")
        plt.savefig(self.params.component + '/' + self.params.component + "_Cost_PmT.png")

        """
        costs according to lambda values ​​-
        The base case cost, corrective maintenance and best preventive are plotted for each lambda
        :return:
        """
        plt.figure(2)
        plt.grid(True)

        for i in range(0, self.Costsc_max.shape[1]):
            plt.plot([self.params.fr_sens[i], self.params.fr_sens[i]], [self.Costsc_min[0, i], self.Costsc_max[-1, i]],
                     linestyle='-', linewidth=self.l_size, color='black')

        plt.plot(self.params.fr_sens, self.Costsc_min[0, :], marker='o', color='g', linestyle='None',
                 ms=8, markerfacecolor='g', label="Base Case")
        plt.plot(self.params.fr_sens, self.Costsc_max[-1, :], marker='o', color='r', linestyle='None', ms=8,
                 markerfacecolor='r', label="Corrective")
        plt.plot(self.params.fr_sens, self.Costsc_max[1:self.Costsc_max.shape[0] - 1, :].min(0), marker='o',
                 linestyle='None', color='blue', ms=8, markerfacecolor='blue', label="Best Preventive")

        plt.title("Cost vs. FR")
        plt.legend(loc="best")
        plt.xlabel("Relative Lambda")
        plt.ylabel("Total maintenance cost")
        plt.savefig(self.params.component + '/' + self.params.component + "_Cost_Scheme.png")

        """
        ENF when PM intervals vary -
        The maximum value of ENF on the various beta / alpha is plotted for each FR
        :return:
        """
        fall_tst_max = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
        fall_tst_min = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
        fall_pmt_max = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
        fall_pmt_min = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)

        self.fall_max[:, :] = self.fall.max(0)
        self.fall_min[:, :] = self.fall.min(0)

        x_time = np.zeros((self.params.Tst.shape[0]), dtype=np.int64)

        if len(self.params.Tst) > 1:
            x_time[:] = self.params.Tst

        for tst_sample in range(0, self.params.Tst.shape[0]):
            temp_max = self.fall_max[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                     self.params.PmT.shape[0] - 1, :]
            temp_min = self.fall_min[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                     self.params.PmT.shape[0] - 1, :]
            fall_tst_max[tst_sample, :] = temp_max.max(0)
            fall_tst_min[tst_sample, :] = temp_min.min(0)

        for pmt_sample in range(0, self.params.PmT.shape[0]):
            a = np.arange(pmt_sample, self.Costsc_max.shape[0], len(self.params.PmT))
            temp_max = self.fall_max[a, :]
            temp_min = self.fall_min[a, :]
            fall_pmt_max[pmt_sample, :] = temp_max.max(0)
            fall_pmt_min[pmt_sample, :] = temp_min.min(0)

        plt.figure(3)
        plt.grid(True)
        # Turn ON grid
        for i in range(0, fall_tst_max.shape[1]):
            plt.plot(x_time, fall_tst_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)
        for i in range(0, fall_tst_min.shape[1]):
            plt.plot(x_time, fall_tst_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)

        fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
        plt.title("ENF vs. FR")
        plt.legend(fr_legend, loc="upper right")
        plt.xlabel("Test interval [years]")
        plt.ylabel("N. expected failures")
        plt.savefig(self.params.component + '/' + self.params.component + "_TsT_Failures.png")

        plt.figure(4)
        plt.grid(True)
        x_time = np.zeros((self.params.PmT.shape[0]), dtype=np.int64)

        if len(self.params.PmT) > 1:
            x_time[:] = self.params.PmT

        for i in range(0, fall_pmt_max.shape[1]):
            plt.plot(x_time, fall_pmt_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)
        for i in range(0, fall_pmt_min.shape[1]):
            plt.plot(x_time, fall_pmt_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=2)

        fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
        plt.title("ENF vs. FR")
        plt.legend(fr_legend, loc="upper right")
        plt.xlabel("PM interval [years]")
        plt.ylabel("N. expected failures")
        plt.savefig(self.params.component + '/' + self.params.component + "_PmT_Failures.png")

    def plot_is_hidden_results(self):
        if np.sum(self.params.hidden) > 0:
            det_tst_max = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            det_tst_min = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            det_pmt_max = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            det_pmt_min = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)

            self.det_max[:, :] = self.det.max(0)
            self.det_min[:, :] = self.det.min(0)

            undet_tst_max = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            undet_tst_min = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            undet_pmt_max = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            undet_pmt_min = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)

            self.undet_max[:, :] = self.undet.max(0)
            self.undet_min[:, :] = self.undet.min(0)

            t_fail_h_tst_max = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            t_fail_h_tst_min = np.zeros((self.params.Tst.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            t_fail_h_pmt_max = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)
            t_fail_h_pmt_min = np.zeros((self.params.PmT.shape[0], self.params.fr_sens.shape[0]), dtype=np.float64)

            self.t_fail_h_max[:, :] = (self.t_fail_h / 24.0).max(0)
            self.t_fail_h_min[:, :] = (self.t_fail_h / 24.0).min(0)

            """
            Failures detected by the test when the PM or Test intervals vary -
            The maximum value of DF on the various beta / alpha is plotted for each FR
            :return:
            """
            plt.figure(5)
            plt.grid(True)

            for tst_sample in range(0, self.params.Tst.shape[0]):
                temp_max = self.det_max[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                                                     self.params.PmT.shape[0] - 1, :]
                temp_min = self.det_min[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                                                     self.params.PmT.shape[0] - 1, :]
                det_tst_max[tst_sample, :] = temp_max.max(0)
                det_tst_min[tst_sample, :] = temp_min.min(0)

            for pmt_sample in range(0, self.params.PmT.shape[0]):
                a_max = np.arange(pmt_sample, self.det_max.shape[0], len(self.params.PmT))
                a_min = np.arange(pmt_sample, self.det_min.shape[0], len(self.params.PmT))
                temp_max = self.det_max[a_max, :]
                temp_min = self.det_min[a_min, :]
                det_pmt_max[pmt_sample, :] = temp_max.max(0)
                det_pmt_min[pmt_sample, :] = temp_min.min(0)

            x_time = np.zeros((self.params.Tst.shape[0]), dtype=np.int64)

            if len(self.params.Tst) > 1:
                x_time[:] = self.params.Tst

            for i in range(0, det_tst_max.shape[1]):
                plt.plot(x_time, det_tst_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                             color=self.params.plot_color[i], linewidth=2)
            for i in range(0, det_tst_min.shape[1]):
                plt.plot(x_time, det_tst_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                             color=self.params.plot_color[i], linewidth=2)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("Detected Failures vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("N. expected failures")
            plt.savefig(self.params.component + '/' + self.params.component + "_Det_TsT_Failures_Ext.png")

            x_time = np.zeros((self.params.PmT.shape[0]), dtype=np.int64)

            if len(self.params.PmT) > 1:
                x_time[:] = self.params.PmT

            for i in range(0, det_pmt_max.shape[1]):
                plt.plot(x_time, det_pmt_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)
            for i in range(0, det_pmt_min.shape[1]):
                plt.plot(x_time, det_pmt_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("Detected Failures vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("N. expected failures")
            plt.savefig(self.params.component + '/' + self.params.component + "_Det_PmT_Failures_Ext.png")

            """
            Failures not detected by the test when the PM or Test intervals vary
            The maximum NDF value on the various beta / alpha is plotted for each FR
            :return:
            """
            plt.figure(6)
            plt.grid(True)

            for tst_sample in range(0, self.params.Tst.shape[0]):
                temp_max = self.undet_max[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                          self.params.PmT.shape[0] - 1, :]
                temp_min = self.undet_min[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                          self.params.PmT.shape[0] - 1, :]
                undet_tst_max[tst_sample, :] = temp_max.max(0)
                undet_tst_min[tst_sample, :] = temp_min.min(0)

            for pmt_sample in range(0, self.params.PmT.shape[0]):
                a_max = np.arange(pmt_sample, self.undet_max.shape[0], len(self.params.PmT))
                a_min = np.arange(pmt_sample, self.undet_min.shape[0], len(self.params.PmT))
                temp_max = self.undet_max[a_max, :]
                temp_min = self.undet_min[a_min, :]
                undet_pmt_max[pmt_sample, :] = temp_max.max(0)
                undet_pmt_min[pmt_sample, :] = temp_min.min(0)

            x_time = np.zeros((self.params.Tst.shape[0]), dtype=np.int64)

            if len(self.params.Tst) > 1:
                x_time[:] = self.params.Tst

            for i in range(0, undet_tst_max.shape[1]):
                plt.plot(x_time, undet_tst_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)
            for i in range(0, undet_tst_min.shape[1]):
                plt.plot(x_time, undet_tst_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("UnDetected Failures vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("Test interval [years]")
            plt.ylabel("N. expected failures")
            plt.savefig(self.params.component + '/' + self.params.component + "_Undet_TsT_Failures_Ext.png")

            plt.figure(7)
            plt.grid(True)

            x_time = np.zeros((self.params.PmT.shape[0]), dtype=np.int64)

            if len(self.params.PmT) > 1:
                x_time[:] = self.params.PmT

            for i in range(0, undet_pmt_max.shape[1]):
                plt.plot(x_time, undet_pmt_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)
            for i in range(0, undet_pmt_min.shape[1]):
                plt.plot(x_time, undet_pmt_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("UnDetected Failures vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("N. expected failures")
            plt.savefig(self.params.component + '/' + self.params.component + "_Undet_PmT_Failures_Ext.png")

            """
            Average time for which a faulty component with an FM hidden remains faulty before being adjusted / detected
            The maximum value Time for the various beta / alpha is plotted for each FR
            :return:
            """
            plt.figure(8)
            plt.grid(True)

            for tst_sample in range(0, self.params.Tst.shape[0]):
                temp_max = self.t_fail_h_max[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                             self.params.PmT.shape[0] - 1, :]
                temp_min = self.t_fail_h_min[self.params.Tst.shape[0] * tst_sample:self.params.PmT.shape[0] * tst_sample +
                                             self.params.PmT.shape[0] - 1, :]
                t_fail_h_tst_max[tst_sample, :] = temp_max.max(0)
                t_fail_h_tst_min[tst_sample, :] = temp_min.min(0)

            for pmt_sample in range(0, self.params.PmT.shape[0]):
                a_max = np.arange(pmt_sample, self.t_fail_h_max.shape[0], len(self.params.PmT))
                a_min = np.arange(pmt_sample, self.t_fail_h_max.shape[0], len(self.params.PmT))
                temp_max = self.t_fail_h_max[a_max, :]
                temp_min = self.t_fail_h_max[a_min, :]
                t_fail_h_pmt_max[pmt_sample, :] = temp_max.max(0)
                t_fail_h_pmt_min[pmt_sample, :] = temp_min.min(0)

            x_time = np.zeros((self.params.Tst.shape[0]), dtype=np.int64)

            if len(self.params.Tst) > 1:
                x_time[:] = self.params.Tst

            for i in range(0, t_fail_h_tst_max.shape[1]):
                plt.plot(x_time, t_fail_h_tst_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)
            for i in range(0, t_fail_h_tst_min.shape[1]):
                plt.plot(x_time, t_fail_h_tst_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("Undetected Time vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("Time [days]")
            plt.savefig(self.params.component + '/' + self.params.component + "_Time_Failures_TsT_Ext.png")

            plt.figure(9)
            plt.grid(True)

            x_time = np.zeros((self.params.PmT.shape[0]), dtype=np.int64)

            if len(self.params.PmT) > 1:
                x_time[:] = self.params.PmT

            for i in range(0, t_fail_h_pmt_max.shape[1]):
                plt.plot(x_time, t_fail_h_pmt_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)
            for i in range(0, t_fail_h_pmt_min.shape[1]):
                plt.plot(x_time, t_fail_h_pmt_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                         color=self.params.plot_color[i], linewidth=2)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("Undetected Time vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("Time [days]")
            plt.savefig(self.params.component + '/' + self.params.component + "_Time_Failures_PmT_Ext.png")


class InputParams:
    def __init__(self, component):
        self.component = component


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        # Root Window Settings
        self.title("Alstom Control Application")
        self.geometry('960x670')
        self.geometry('1120x760')# MAC OS
        self.maxsize(width=1120, height=760)
        # self.maxsize(width=960, height=670)
        self.iconbitmap('aramis_icon.ico')
        self.resizable(0, 0)
        self.configure(background='DimGray')

        # variables
        self.life = IntVar()
        self.h_day = IntVar()
        self.cmp_fr = DoubleVar()
        self.cmp_uf = DoubleVar()
        self.cmp_type = StringVar()
        self.fail_mode = IntVar()
        self.warranty_bool = BooleanVar()
        self.plt_percent_bool = BooleanVar()
        self.duration = DoubleVar()
        self.set_cmp_name = StringVar()

        self.fm_1 = BooleanVar()
        self.fm_2 = BooleanVar()
        self.fm_3 = BooleanVar()
        self.fm_4 = BooleanVar()
        self.fm_5 = BooleanVar()

        self.is_access_roof_repair = BooleanVar()
        self.is_access_side_repair = BooleanVar()
        self.is_access_pit_repair = BooleanVar()
        self.is_access_pit_repair = BooleanVar()
        self.is_access_breaking_repair = BooleanVar()
        self.is_access_decoupling_repair = BooleanVar()

        self.is_access_roof_replace = BooleanVar()
        self.is_access_side_replace = BooleanVar()
        self.is_access_pit_replace = BooleanVar()
        self.is_access_breaking_replace = BooleanVar()
        self.is_access_decoupling_replace = BooleanVar()

        self.time_roof_replace = DoubleVar()
        self.time_side_replace = DoubleVar()
        self.time_pit_replace = DoubleVar()
        self.time_breaking_replace = DoubleVar()
        self.time_decoupling_replace = DoubleVar()

        self.time_roof_repair = DoubleVar()
        self.time_side_repair = DoubleVar()
        self.time_pit_repair = DoubleVar()
        self.time_breaking_repair = DoubleVar()
        self.time_decoupling_repair = DoubleVar()

        self.fm_percentage_1 = IntVar()
        self.fm_percentage_2 = IntVar()
        self.fm_percentage_3 = IntVar()
        self.fm_percentage_4 = IntVar()
        self.fm_percentage_5 = IntVar()

        self.prediction_interval_1 = IntVar()
        self.prediction_interval_2 = IntVar()
        self.prediction_interval_3 = IntVar()
        self.prediction_interval_4 = IntVar()
        self.prediction_interval_5 = IntVar()

        self.options = ["Select Type",
                        "Mechanical",
                        "Electrical",
                        "Worn Out",
                        "Unknown",
                        "Other"]

        self.range_beta = {"Mechanical": [1.0, 3.0],
                           "Electrical": [0.3, 1],
                           "Worn Out": [2.0, 5.0],
                           "Unknown": [0.5, 1.2]}

        self.options_menu_1 = StringVar()
        self.options_menu_2 = StringVar()
        self.options_menu_3 = StringVar()
        self.options_menu_4 = StringVar()
        self.options_menu_5 = StringVar()

        self.beta_min_1 = DoubleVar()
        self.beta_min_2 = DoubleVar()
        self.beta_min_3 = DoubleVar()
        self.beta_min_4 = DoubleVar()
        self.beta_min_5 = DoubleVar()

        self.beta_max_1 = DoubleVar()
        self.beta_max_2 = DoubleVar()
        self.beta_max_3 = DoubleVar()
        self.beta_max_4 = DoubleVar()
        self.beta_max_5 = DoubleVar()

        self.cost_crew = DoubleVar()
        self.mttr_corrective_replacement = DoubleVar()
        self.mttr_corrective_repair = DoubleVar()
        self.mttr_test = DoubleVar()
        self.mttr_preventive = DoubleVar()
        self.num_corrective = IntVar()
        self.num_preventive = IntVar()
        self.n_man_ts = IntVar()
        self.co_spare_cm_1 = DoubleVar()
        self.co_spare_cm_2 = DoubleVar()
        self.co_spare_pm = DoubleVar()

        self.co_pen_min = DoubleVar()
        self.co_pen_max = DoubleVar()
        self.night_time = DoubleVar()
        self.ld = DoubleVar()

        self.is_hidden = []
        self.percentage_fm = []
        self.prediction_interval_params = []
        self.options_menu = []
        self.popup_menu_list = []
        self.beta_min = []
        self.beta_max = []
        self.failure_entry_widget = []
        self.hidden_widget = []
        self.prediction_widget = []
        self.beta_min_widget = []
        self.beta_max_widget = []

        # FM Matrix
        self.fm_1_1 = IntVar()
        self.fm_1_2 = IntVar()
        self.fm_1_3 = IntVar()
        self.fm_1_4 = IntVar()
        self.fm_1_5 = IntVar()

        self.fm_2_1 = IntVar()
        self.fm_2_2 = IntVar()
        self.fm_2_3 = IntVar()
        self.fm_2_4 = IntVar()
        self.fm_2_5 = IntVar()

        self.fm_3_1 = IntVar()
        self.fm_3_2 = IntVar()
        self.fm_3_3 = IntVar()
        self.fm_3_4 = IntVar()
        self.fm_3_5 = IntVar()

        self.fm_4_1 = IntVar()
        self.fm_4_2 = IntVar()
        self.fm_4_3 = IntVar()
        self.fm_4_4 = IntVar()
        self.fm_4_5 = IntVar()

        self.fm_5_1 = IntVar()
        self.fm_5_2 = IntVar()
        self.fm_5_3 = IntVar()
        self.fm_5_4 = IntVar()
        self.fm_5_5 = IntVar()

        self.fm_matrix = [[self.fm_1_1, self.fm_1_2, self.fm_1_3, self.fm_1_4, self.fm_1_5],
                          [self.fm_2_1, self.fm_2_2, self.fm_2_3, self.fm_2_4, self.fm_2_5],
                          [self.fm_3_1, self.fm_3_2, self.fm_3_3, self.fm_3_4, self.fm_3_5],
                          [self.fm_4_1, self.fm_4_2, self.fm_4_3, self.fm_4_4, self.fm_4_5],
                          [self.fm_5_1, self.fm_5_2, self.fm_5_3, self.fm_5_4, self.fm_5_5]]

        self.fm_matrix_widget = []
        self.fm_matrix_5d = []

        self.fm_corrective_1 = DoubleVar()
        self.fm_corrective_2 = DoubleVar()
        self.fm_corrective_3 = DoubleVar()
        self.fm_corrective_4 = DoubleVar()
        self.fm_corrective_5 = DoubleVar()

        self.fm_preventive_1 = BooleanVar()
        self.fm_preventive_2 = BooleanVar()
        self.fm_preventive_3 = BooleanVar()
        self.fm_preventive_4 = BooleanVar()
        self.fm_preventive_5 = BooleanVar()

        self.fm_corrective = [self.fm_corrective_1, self.fm_corrective_2, self.fm_corrective_3, self.fm_corrective_4,
                              self.fm_corrective_5]

        self.fm_preventive = [self.fm_preventive_1, self.fm_preventive_2, self.fm_preventive_3, self.fm_preventive_4,
                              self.fm_preventive_5]

        self.fm_corrective_entry = []
        self.fm_preventive_entry = []

        self.N_sim = IntVar()
        self.step_size = DoubleVar()
        self.fr_sens_min = DoubleVar()
        self.fr_sens_max = DoubleVar()
        self.Tst_min = DoubleVar()
        self.Tst_max = DoubleVar()

        self.back_button = None
        self.progressbar = None
        self.stop_button = None
        self.start_button = None

        self.tab_control = ttk.Notebook(self)

        # self.tab_control.enable_traversal()

        self.tab1 = ttk.Frame(self.tab_control, width=500, height=500)

        self.tab_control.add(self.tab1, text="Mandatory")

        self.tab2 = ttk.Frame(self.tab_control, width=60, height=80)

        self.tab_control.add(self.tab2, text="Optional")

        self.tab_control.pack(expand=1, fill='both')

        # disable tab2
        self.tab_control.tab(1, state="disabled")
        # Add logo
        self.add_logo(self.tab1)
        self.add_logo(self.tab2)
        # Add Widgets
        self.add_widgets_sheet1()
        self.add_widgets_sheet2()

    def simulate_test(self):
        params = self.load_input_params()
        simulate = Simulate(params)
        simulate.print_params()
        # simulate.run_cost_test()
        # simulate.plot_results()

    def start_process(self):
        t1 = Process(target=self.simulate_test)
        return t1

    def start_rcm(self):
        global p
        if self.is_mandate() is not False:
            mbox.showinfo("Info", "RCM Analysis Start")
            self.disable_button()
            self.progressbar.start()
            p = self.start_process()
            p.daemon = True
            p.start()
            self.after(100, self.process)

    def process(self):
        if p.is_alive():
            self.after(100, self.process)
        else:
            p.terminate()
            self.progressbar.stop()
            self.enable_button()

    def stop_rcm(self):
        mbox.showinfo("Info", "RCM Analysis Stopped")
        p.terminate()
        self.progressbar.stop()
        self.enable_button()

    @staticmethod
    def is_alpha(val):
        """
        Check if the entry content is strictly Alphabetic
        :return: Boolean
        """
        if val.isalpha() is False:
            return False
        else:
            return True

    @staticmethod
    def is_digit(val):
        """
        Check if the entry content is strictly Alphabetic
        :return: Boolean
        """
        val = str(val)
        if val.replace(".", "", 1).isdigit() is False:
            return False
        else:
            return True

    def is_mandate(self):
        """
        Check if there exists a value in the mandatory fields, If not raise error
        :return: Boolean
        """
        # Train Life Parameters
        try:
            self.is_digit(self.life.get())
            if self.life.get() <= 0:
                mbox.showerror("Error", "Invalid Train Life Field \nProvide Positive Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Train Life Field \nProvide Positive Value")
            return False

        try:
            self.is_digit(self.h_day.get())
            if self.h_day.get() <= 0:
                mbox.showerror("Error", "Invalid Operative Hours \n Field")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Operative Hours \n Field")
            return False

        if self.fail_mode.get() == 0:
                mbox.showerror("Error", "Select Atleast 1 Failure Mode")
                return False

        if self.warranty_bool.get() == 1:
            try:
                self.is_digit(self.duration.get())
                if self.duration.get() <= 0:
                    mbox.showerror("Error", "Invalid Warranty Period")
                    return False
            except tkinter.TclError:
                mbox.showerror("Error", "Invalid Warranty Period")
                return False

        # Component Parameters

        if not self.is_alpha(self.set_cmp_name.get()):
            if len(self.set_cmp_name.get()) == 0:
                mbox.showerror("Error", "Invalid Component Field")
                return False
            mbox.showerror("Error", "Invalid Component Field")
            return False

        try:
            self.is_digit(self.cmp_fr.get())
            if self.cmp_fr.get() <= 0:
                mbox.showerror("Error", "Invalid Failure Rate")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Failure Rate")
            return False

        try:
            self.is_digit(self.cmp_uf.get())
            if self.cmp_uf.get() <= 0:
                mbox.showerror("Error", "Invalid Utilization Factor")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Utilization Factor")
            return False

        for num_mode in range(self.fail_mode.get()):
            try:
                self.is_digit(self.prediction_interval_params[num_mode].get())
                if self.options_menu[num_mode].get() == "Other":
                    self.is_digit(self.beta_min[num_mode].get())
                    self.is_digit(self.beta_max[num_mode].get())
                    if self.beta_min[num_mode].get() > self.beta_max[num_mode].get():
                        mbox.showerror("Error", "Invalid Failure Mode Params - Beta Min must be <= Beta Max")
                        return False
                if self.prediction_interval_params[num_mode].get() > 360:
                    mbox.showerror("Error", "Invalid - Maximum Allowable Prediction Interval Length is 360")
                    return False
                if self.prediction_interval_params[num_mode].get() <= 0 and self.beta_min[num_mode].get() < 0 \
                        and self.beta_max[num_mode].get() < 0:
                    mbox.showerror("Error", "Invalid Failure Mode Parameters")
                    return False
                if self.options_menu[num_mode].get() == "Select Type":
                    mbox.showerror("Error", "Invalid Type of Failure Mode")
                    return False
            except tkinter.TclError:
                mbox.showerror("Error", "Invalid Failure Mode Parameters")
                return False

        # Failure Modes
        if self.check_percentage():
            pass
        else:
            mbox.showerror("Error", "Invalid Percentage Split - Failure Modes \nSum must be equal to 100")
            return False
        # LCC - Parameters I
        try:
            self.is_digit(self.time_roof_replace.get())
            if self.time_roof_replace.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Roof")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Roof")
            return False

        try:
            self.is_digit(self.time_side_replace.get())
            if self.time_side_replace.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Side")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Side")
            return False

        try:
            self.is_digit(self.time_pit_replace.get())
            if self.time_pit_replace.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Pit")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Pit")
            return False

        try:
            self.is_digit(self.time_decoupling_replace.get())
            if self.time_decoupling_replace.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Decoupling")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Decoupling")
            return False

        try:
            self.is_digit(self.time_breaking_replace.get())
            if self.time_breaking_replace.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Breaking")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Breaking")
            return False

        try:
            self.is_digit(self.time_roof_repair.get())
            if self.time_roof_repair.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Roof")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Roof")
            return False

        try:
            self.is_digit(self.time_side_repair.get())
            if self.time_side_repair.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Side")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Side")
            return False

        try:
            self.is_digit(self.time_pit_repair.get())
            if self.time_pit_repair.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Pit")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Pit")
            return False

        try:
            self.is_digit(self.time_decoupling_repair.get())
            if self.time_decoupling_repair.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Decoupling")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Decoupling")
            return False

        try:
            self.is_digit(self.time_breaking_repair.get())
            if self.time_breaking_repair.get() < 0:
                mbox.showerror("Error", "Invalid LCC I Parameters - Breaking")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid LCC I Parameters - Breaking")
            return False

        # LCC - Parameters II
        try:
            self.is_digit(self.mttr_corrective_replacement.get())
            if self.mttr_corrective_replacement.get() < 0:
                mbox.showerror("Error", "Invalid MTTR Corrective\nReplacement Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid MTTR Corrective\nReplacement Value")
            return False

        try:
            self.is_digit(self.mttr_corrective_repair.get())
            if self.mttr_corrective_repair.get() < 0:
                mbox.showerror("Error", "Invalid MTTR Corrective\nRepair Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid MTTR Corrective\nRepair Value")
            return False

        try:
            self.is_digit(self.mttr_test.get())
            if self.mttr_test.get() < 0:
                mbox.showerror("Error", "Invalid MTTR Test Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid MTTR Test Value")
            return False

        try:
            self.is_digit(self.num_corrective.get())
            if self.num_corrective.get() < 0:
                mbox.showerror("Error", "Invalid N. of Maintainer \nCorrective Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid N. of Maintainer \nCorrective Value")
            return False

        try:
            self.is_digit(self.num_preventive.get())
            if self.num_preventive.get() < 0:
                mbox.showerror("Error", "Invalid N. of Maintainer\npreventive Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid N. of Maintainer\npreventive Value")
            return False

        try:
            self.is_digit(self.n_man_ts.get())
            if self.n_man_ts.get() < 0:
                mbox.showerror("Error", "Invalid N. Maintainer\nTest Value")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid N. Maintainer\nTest Value")
            return False
        try:
            self.is_digit(self.cost_crew.get())
            if self.cost_crew.get() < 0:
                mbox.showerror("Error", "Invalid Crew Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Crew Cost")
            return False
        try:
            self.is_digit(self.co_spare_cm_1.get())
            if self.co_spare_cm_1.get() < 0:
                mbox.showerror("Error", "Invalid Replacement Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Replacement Cost")
            return False

        try:
            self.is_digit(self.co_spare_cm_2.get())
            if self.co_spare_cm_2.get() < 0:
                mbox.showerror("Error", "Invalid Repair Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Repair Cost")
            return False

        try:
            self.is_digit(self.co_spare_pm.get())
            if self.co_spare_pm.get() < 0:
                mbox.showerror("Error", "Invalid Replacement\n PM Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Replacement\n PM Cost")
            return False

        try:
            self.is_digit(self.co_pen_min.get())
            if self.co_pen_min.get() < 0:
                mbox.showerror("Error", "Invalid Min Penalty Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Min Penalty Cost")
            return False

        try:
            self.is_digit(self.co_pen_max.get())
            if self.co_pen_max.get() < 0:
                mbox.showerror("Error", "Invalid Max Penalty Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Max Penalty Cost")
            return False

        try:
            self.is_digit(self.co_spare_cm_2.get())
            if self.co_spare_cm_2.get() < 0:
                mbox.showerror("Error", "Invalid Repair Cost")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Repair Cost")
            return False

        try:
            self.is_digit(self.night_time.get())
            if self.night_time.get() < 0:
                mbox.showerror("Error", "Invalid Intervention Period")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Intervention Period")
            return False

        try:
            self.is_digit(self.ld.get())
            if self.ld.get() < 0:
                mbox.showerror("Error", "Invalid Time for Reaching Deposit")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Time for Reaching Deposit")
            return False

        # optional Page:
        # Effect Matrix
        try:
            for row in range(self.fail_mode.get()):
                for col in range(self.fail_mode.get()):
                    if self.fm_matrix[row][col].get() != 0 and self.fm_matrix[row][col].get() != 1:
                        mbox.showerror("Error", "Invalid Entry in the Effect Matrix \ninput either 0 or 1")
                        return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Entry in the Effect Matrix")
            return False

        # Compatibility Matrix
        # Corrective Maintenance
        try:
            for mode in range(self.fail_mode.get()):
                self.is_digit(self.fm_corrective[mode].get())
                self.is_digit(self.fm_preventive[mode].get())
                # not between 0 - 1
                if 1 < self.fm_corrective[mode].get() < 0:
                    mbox.showerror("Error", "Invalid -  Input Positive value between 0 - 1")
                    return False
                if self.fm_preventive[mode].get() !=0 and self.fm_preventive[mode].get() != 1:
                    mbox.showerror("Error", "Invalid -  Input Either 0 or 1")
                    return
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Entry in Compatibility Matrix")
            return False

        # Simulation Parameters
        # Number of Monte Carlo simulation
        try:
            self.is_digit(self.N_sim.get())
            self.is_digit(self.step_size.get())
            if self.N_sim.get() <= 0:
                mbox.showerror("Error", "Invalid Simulation Params - Atleast 1 Simulation is Required")
                return False
            if self.step_size.get() <= 0:
                mbox.showerror("Error", "Invalid Simulation Params - Step Size Must Be > 0")
                return False
            if self.fr_sens_min.get() < 0 and self.fr_sens_max.get() < 0:
                mbox.showerror("Error", "Invalid Simulation Params - FR Sensitivity Range")
                return False
            if self.fr_sens_min.get() > self.fr_sens_max.get():
                mbox.showerror("Error", "Invalid Simulation Params - FR Sensitivity Min must be < FR Sensitivity Max")
                return False
            if self.Tst_min.get() < 0 and self.Tst_max.get() < 0:
                mbox.showerror("Error", "Invalid Simulation Params - Proposed Test Interval Range")
                return False
            if self.Tst_min.get() > self.Tst_min.get():
                mbox.showerror("Error", "Invalid Simulation Params - \nProposed Test Interval Min must be < "
                                        "Proposed Test Interval Max")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Entry -  N. MonteCarlo Simulations")
            return False

    def disable_button(self):
        """
        Disable Entry and Button Widgets - Sheet 2
        :return:
        """
        self.back_button.config(state=DISABLED)
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=ACTIVE)

    def enable_button(self):
        """
        Enable Entry and Button Widgets
        :return:
        """
        self.stop_button.config(state=DISABLED)
        self.back_button.config(state=ACTIVE)
        self.start_button.config(state=ACTIVE)

    def to_tab1(self):
        self.tab_control.tab(0, state="normal")
        self.tab_control.select(self.tab1)
        self.geometry('1120x760') # mac OS
        # self.geometry('960x670')
        self.tab_control.tab(1, state="disabled")

    def to_tab2(self):
        self.tab_control.tab(1, state="normal")
        self.tab_control.select(self.tab2)
        self.geometry('680x700') # MAC OS
        # self.geometry('530x620') # Windows
        self.tab_control.tab(0, state="disabled")

    def load_input_params(self):
        """
        Build Input Parameters object to run the simulation
        :return: Object Instance
        """
        # input_params = Input_component.Parameters(self.set_cmp_name.get())
        input_params = InputParams(self.set_cmp_name.get())
        # Color vector for plots according to FR (length equal to fr_sens)
        input_params.plot_color = ['g', 'b', 'r']

        # Marker vector for plots according to FR (length equal to fr_sens)
        input_params.plot_marker = ['o', 'x', '*']

        # Train Life
        input_params.Life = self.life.get()
        input_params.h_day = self.h_day.get()
        input_params.n_fm = self.fail_mode.get()
        if self.warranty_bool.get() == 1:
            input_params.Warranty = self.duration.get()
        else:
            input_params.Warranty = 0

        input_params.plot_percentage = bool(self.plt_percent_bool.get())

        # Component Parameters
        # Component is already passed as an initialization parameter
        input_params.FT = self.cmp_fr.get()
        input_params.U = self.cmp_uf.get()

        # Failure Modes
        input_params.n = np.r_[3, 3]

        # Repartition percentage of the FR on the FMs
        # Number of rows =n_fm, Vector sum =1
        input_params.P_failure_mode = np.zeros((input_params.n_fm, 1), dtype=np.float64)

        # Predictive interval "as good as new" implemented for the component
        # rows =n_fm, value in months, value =360 if going to failure
        input_params.Tau = np.zeros((input_params.n_fm, 1), dtype=np.int64)

        # Hidden matrix, it defines which FMs are hidden
        # rows =n_fm, value =1 if the FM is hidden, value =0 if not
        input_params.hidden = np.zeros((input_params.n_fm, 1), dtype=np.int64)

        # hard coded to 3 peta values per failure mode
        input_params.beta = np.zeros((input_params.n_fm, 3), dtype=np.float64)

        # Compatibility matrix� FM/PM
        # rows =n_fm, value =1 if maintenance interval effective on the FM, value =0 if not
        # PM
        input_params.Comp = np.zeros((input_params.n_fm, 1), dtype=np.float64)

        # rows =n_fm, Values between 0 and 1 and equal to the replacement probability
        # EXAMPLE: 0.1 shows that in the 10percent of cases it is expected the replacement("as good as new"),
        # in the 90percent of cases replacement NOT "as good as new")
        # from optional - Compatibility matrix
        # Compatibility Matrix FM/CM
        input_params.Comp_cm = np.zeros((input_params.n_fm, 1), dtype=np.float64)

        for fail_mode in range(0, input_params.n_fm):
            input_params.P_failure_mode[fail_mode, 0] = self.percentage_fm[fail_mode].get() / 100.0
            input_params.Tau[fail_mode, 0] = self.prediction_interval_params[fail_mode].get()
            input_params.hidden[fail_mode, 0] = self.is_hidden[fail_mode].get()
            # PM
            input_params.Comp[fail_mode, 0] = self.fm_preventive[fail_mode].get()
            # CM
            input_params.Comp_cm[fail_mode, 0] = self.fm_corrective[fail_mode].get()

            if self.options_menu[fail_mode].get() == "Mechanical":
                input_params.beta[fail_mode, 0:3] = np.linspace(self.range_beta["Mechanical"][0],
                                                                self.range_beta["Mechanical"][1],
                                                                3, dtype=np.float64)

            if self.options_menu[fail_mode].get() == "Electrical":
                input_params.beta[fail_mode, 0:3] = np.linspace(self.range_beta["Electrical"][0],
                                                                self.range_beta["Electrical"][1],
                                                                3, dtype=np.float64)

            if self.options_menu[fail_mode].get() == "Worn Out":
                input_params.beta[fail_mode, 0:3] = np.linspace(self.range_beta["Worn Out"][0],
                                                                self.range_beta["Worn Out"][1],
                                                                3, dtype=np.float64)
            if self.options_menu[fail_mode].get() == "Unknown":
                input_params.beta[fail_mode, 0:3] = np.linspace(self.range_beta["Unknown"][0],
                                                                self.range_beta["Unknown"][1],
                                                                3, dtype=np.float64)
            if self.options_menu[fail_mode].get() == "Other":
                input_params.beta[fail_mode, 0:3] = np.linspace(self.beta_min[fail_mode].get(),
                                                                self.beta_max[fail_mode].get(),
                                                                3, dtype=np.float64)
        # LCC parameters I
        # Accessibility procedure
        input_params.acc = np.zeros((5, 3, 2), dtype=np.float64)

        input_params.acc[0, 1, 0] = self.is_access_roof_replace.get()
        input_params.acc[1, 1, 0] = self.is_access_side_replace.get()
        input_params.acc[2, 1, 0] = self.is_access_pit_replace.get()
        input_params.acc[3, 1, 0] = self.is_access_breaking_replace.get()
        input_params.acc[4, 1, 0] = self.is_access_decoupling_replace.get()

        input_params.acc[0, 1, 1] = self.is_access_roof_repair.get()
        input_params.acc[1, 1, 1] = self.is_access_side_repair.get()
        input_params.acc[2, 1, 1] = self.is_access_pit_repair.get()
        input_params.acc[3, 1, 1] = self.is_access_breaking_repair.get()
        input_params.acc[4, 1, 1] = self.is_access_decoupling_repair.get()

        input_params.acc[0, 0, :] = self.time_roof_replace.get()
        input_params.acc[1, 0, :] = self.time_side_replace.get()
        input_params.acc[2, 0, :] = self.time_pit_replace.get()
        input_params.acc[3, 0, :] = self.time_breaking_replace.get()
        input_params.acc[4, 0, :] = self.time_decoupling_replace.get()

        input_params.acc[0, 2, :] = self.time_roof_repair.get()
        input_params.acc[1, 2, :] = self.time_side_repair.get()
        input_params.acc[2, 2, :] = self.time_pit_repair.get()
        input_params.acc[3, 2, :] = self.time_breaking_repair.get()
        input_params.acc[4, 2, :] = self.time_decoupling_repair.get()

        # LCC parameters II
        input_params.MTTRc = np.r_[self.mttr_corrective_replacement.get(), self.mttr_corrective_repair.get()]
        input_params.MTTRsc = self.mttr_preventive.get()
        input_params.MTTRts = self.mttr_test.get()

        input_params.n_man_cm = self.num_corrective.get()
        input_params.n_man_pm = self.num_preventive.get()
        input_params.n_man_ts = self.n_man_ts.get()

        input_params.co_crew_h = self.cost_crew.get()

        # corrective maintenance - replacement and repair cost
        input_params.co_spare_cm = np.zeros((2, 1), dtype=np.float64)
        input_params.co_spare_cm[0] = self.co_spare_cm_1.get()
        input_params.co_spare_cm[1] = self.co_spare_cm_2.get()

        # preventive maintenance
        input_params.co_spare_pm = self.co_spare_pm.get()
        input_params.co_pen_min = self.co_pen_min.get()
        input_params.co_pen_max = self.co_pen_max.get()
        input_params.night_time = self.night_time.get()
        input_params.ld = self.ld.get()

        # Tab - optional
        # Effect Matrix
        input_params.chi_w = np.zeros((input_params.n_fm, input_params.n_fm), dtype=np.int64)

        for row in range(input_params.n_fm):
            for col in range(input_params.n_fm):
                input_params.chi_w[row, col] = self.fm_matrix[row][col].get()

        # Number of MC simulations
        input_params.n_sim = self.N_sim.get()
        # Multiplication Factor FR (sensitivity, in general between 0.5 and 2)
        input_params.fr_sens = np.r_[self.fr_sens_min.get(), 1, self.fr_sens_max.get()]

        # Proposed preventive interval (y)
        hidden_list = []
        non_hidden_list = []
        # Proposed test interval (y)
        for mode in range(input_params.n_fm):
            if input_params.hidden[mode, 0] == 1:
                hidden_list.append(input_params.Tau[mode, 0])
            else:
                non_hidden_list.append(input_params.Tau[mode, 0])

        # minimum of non hidden failure mode prediction interval
        if len(non_hidden_list) != 0:
            input_params.PmT = np.arange(min(non_hidden_list)/12, input_params.Life, self.step_size.get())
        else:
            input_params.PmT = np.r_[input_params.Life]

        # minimum of minimum prediction interval and Min Proposed Tst interval
        if len(hidden_list) != 0:
            if min(hidden_list)/12 <= self.Tst_min.get():
                input_params.Tst = np.arange(min(hidden_list)/12, self.Tst_max.get(), self.step_size.get())
            else:
                input_params.Tst = np.arange(self.Tst_min.get(), self.Tst_max.get()+1, self.step_size.get())
        else:
            input_params.Tst = np.r_[input_params.Life]
        # Conversions
        input_params.conversion_factor = np.round((365 / 12) * input_params.h_day, decimals=5)
        input_params.Tau = input_params.Tau * input_params.conversion_factor

        input_params.Life = input_params.Life * 365 * input_params.h_day

        # Calculation of CMT replacement
        input_params.cmt = np.zeros((2, 1), dtype=np.float64)
        # Calculation of man-hours and Cost spare/penalty
        input_params.wkl_cm = np.zeros((2, 1), dtype=np.float64)

        for cmt_id in range(0, 2):
            dot_prod_1 = np.dot(np.dot(input_params.acc[:, 0, cmt_id], input_params.acc[:, 1, cmt_id]),
                                input_params.acc[:, 2, cmt_id])
            dot_prod_2 = np.dot(np.dot(input_params.acc[:, 0, cmt_id], input_params.acc[:, 1, cmt_id]),
                                1*(input_params.acc[:, 2, cmt_id] == 0))
            input_params.cmt[cmt_id, 0] = input_params.MTTRc[cmt_id] + dot_prod_1[0] +\
                                           (dot_prod_2*input_params.n_man_cm)[0]
            input_params.wkl_cm[cmt_id, 0] = input_params.MTTRc[cmt_id] + (dot_prod_1[0]*input_params.n_man_cm) + \
                                             (dot_prod_2*input_params.n_man_cm)[0]

        input_params.co_pen = input_params.co_pen_min * 1*((input_params.cmt+input_params.ld) > input_params.night_time)
        # Total fixed costs CM
        input_params.Ccor = input_params.co_pen + input_params.co_spare_cm
        input_params.pmt = input_params.MTTRsc
        input_params.wkl_pm = input_params.MTTRsc * input_params.n_man_pm
        input_params.Csc = input_params.co_spare_pm

        input_params.tst = input_params.MTTRts
        input_params.wkl_ts = input_params.MTTRts * input_params.n_man_ts

        input_params.Tgaranzia = input_params.Warranty * 365 * input_params.h_day

        return input_params

    def check_percentage(self):
        sum_per = self.fm_percentage_1.get() + self.fm_percentage_2.get() + self.fm_percentage_3.get() + \
                  self.fm_percentage_4.get() + self.fm_percentage_5.get()
        if sum_per != 100.0:
            return False
        else:
            return True

    def enable_range_beta(self, *args):
        val = 0
        for val in range(0, self.fail_mode.get()+1):
            if val != 0:
                if self.options_menu[val-1].get() == "Other":
                    self.beta_min_widget[val-1].config(state=ACTIVE)
                    self.beta_max_widget[val - 1].config(state=ACTIVE)
                else:
                    self.beta_min_widget[val - 1].config(state=DISABLED)
                    self.beta_max_widget[val - 1].config(state=DISABLED)
        for i in range(val + 1, 6):
            self.beta_min_widget[i - 1].config(state=DISABLED)
            self.beta_max_widget[i - 1].config(state=DISABLED)
            self.beta_min[i - 1].set(float(0))
            self.beta_max[i - 1].set(float(0))

    def change_drop_down_fail_mode(self, *args):
        # Activate Buttons
        val = 0
        for val in range(0, self.fail_mode.get()+1):
            if val != 0:
                widget_percentage = self.failure_entry_widget[val-1]
                widget_percentage.config(state=ACTIVE)
                widget_hidden = self.hidden_widget[val - 1]
                widget_hidden.config(state=ACTIVE)
                widget_predictive = self.prediction_widget[val-1]
                widget_predictive.config(state=ACTIVE)
                self.prediction_widget[val-1].config(state=ACTIVE)
                self.fm_corrective_entry[val-1].config(state=ACTIVE)
                self.fm_preventive_entry[val-1].config(state=ACTIVE)
                self.popup_menu_list[val-1].config(state=ACTIVE)

        for i in range(val+1, 6):
            widget_percentage = self.failure_entry_widget[i - 1]
            widget_percentage.config(state=DISABLED)
            widget_hidden = self.hidden_widget[i - 1]
            widget_hidden.config(state=DISABLED)
            self.prediction_widget[i - 1].config(state=DISABLED)
            self.fm_corrective_entry[i - 1].config(state=DISABLED)
            self.fm_preventive_entry[i - 1].config(state=DISABLED)
            self.popup_menu_list[i - 1].config(state=DISABLED)
            self.beta_min_widget[i - 1].config(state=DISABLED)
            self.beta_max_widget[i - 1].config(state=DISABLED)
            # set the default Value
            self.is_hidden[i-1].set(0)
            self.percentage_fm[i-1].set(0)
            self.prediction_interval_params[i-1].set(0)
            self.fm_corrective[i-1].set(1.0)
            self.fm_preventive[i-1].set(1)
            self.options_menu[i-1].set(self.options[0])

        size = self.fail_mode.get()
        row = 0
        col = 0
        if size != 0:
            for row in range(0, size):
                for col in range(0, size):
                    if row != col:
                        self.fm_matrix_5d[row][col].config(state=ACTIVE)

        for j in range(row+1, 5):
            if col == row:
                for k in range(0, 5):
                    self.fm_matrix_5d[j][k].config(state=DISABLED)
                    self.fm_matrix[j][k].set(0)

        for j in range(col+1, 5):
            if col == row:
                for k in range(0, 5):
                    self.fm_matrix_5d[k][j].config(state=DISABLED)
                    self.fm_matrix[j][k].set(0)

    @staticmethod
    def add_logo(tab):
        # Logo - Start
        image = Image.open("aramis_logo.jpg")
        image = image.resize((350, 150), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        label = ttk.Label(tab, image=photo)
        label.image = photo
        label.grid(row=0, column=0, columnspan=10, rowspan=10)

    def add_widgets_sheet1(self):
        # label Frame - Train Mission Parameters
        label_frame_0 = ttk.LabelFrame(self.tab1, text="Train Mission Parameters", height=20, width=40)

        label_frame_0.grid(column=0, row=12, padx=(5, 0))

        # add Labels abd Entry widgets
        set_train_life = ttk.Label(label_frame_0, text="Train Life: ")
        set_train_life.grid(column=0, row=12, sticky='W')

        set_train_life_box = ttk.Entry(label_frame_0, textvariable=self.life, width=8)
        set_train_life_box.grid(row=12, column=1, sticky='W', padx=(0, 10))

        set_h_day = ttk.Label(label_frame_0, text="Operative Hours (per day): ")
        set_h_day.grid(row=14, column=0, sticky='W')

        set_h_day_box = ttk.Entry(label_frame_0, textvariable=self.h_day, width=8)
        set_h_day_box.grid(row=14, column=1, sticky='W')

        number_fail = [0, 1, 2, 3, 4, 5]

        self.fail_mode.set(number_fail[0])  # set the default option

        popup_menu_fail = ttk.OptionMenu(label_frame_0, self.fail_mode, *number_fail)
        popup_menu_fail.config(width=8)
        ttk.Label(label_frame_0, text="Number of Failure Modes: ").grid(column=0, row=16, sticky="W")
        popup_menu_fail.grid(column=1, row=16, sticky="W", padx=(0, 40))

        # link function to change drop down
        self.fail_mode.trace('w', self.change_drop_down_fail_mode)

        ttk.Label(label_frame_0, text="Other Parameters:").grid(column=0, row=18, sticky="W", pady=(8, 0))

        # Warranty Block - Start
        warranty = ttk.Label(label_frame_0, text="Period(yrs):")
        warranty.grid(row=22, column=0, sticky="W", pady=(2, 0))
        #
        warranty = ttk.Label(label_frame_0, text="Warranty: ")
        warranty.grid(row=19, column=0, sticky="W", pady=(2, 0))
        #
        warranty_bool = ttk.Checkbutton(label_frame_0, variable=self.warranty_bool, onvalue=1, offvalue=0, width=2)
        warranty_bool.grid(row=19, column=1, sticky="W")
        #
        warranty_period = ttk.Entry(label_frame_0, textvariable=self.duration, width=8)
        warranty_period.grid(row=22, column=1, columnspan=5, sticky='W')
        # Warranty Block End

        # plot Percentage Block - Start
        plt_percent = ttk.Label(label_frame_0, text="Plot Percentage: ")
        plt_percent.grid(row=24, column=0, sticky="W")

        plt_percent_bool = ttk.Checkbutton(label_frame_0, variable=self.plt_percent_bool, onvalue=1, offvalue=0,
                                           width=2)
        plt_percent_bool.grid(row=24, column=1, sticky="W")
        # plot Percentage Block - End

        # Label Frame
        label_frame_1 = ttk.LabelFrame(self.tab1, text="Component Parameters", height=20, width=20)
        label_frame_1.config(width=100, height=20)
        label_frame_1.grid(column=0, row=16, padx=(5, 0))

        # add Labels abd Entry widgets
        ttk.Label(label_frame_1, text="Component: ").grid(column=0, row=0, sticky='W')
        set_component_name = ttk.Entry(label_frame_1, textvariable=self.set_cmp_name, width=12)
        set_component_name.grid(row=0, column=1, sticky='W', padx=(0, 100))

        set_failure_rate = ttk.Label(label_frame_1, text="Failure Rate: ")
        set_failure_rate.grid(column=0, row=1, sticky='W')

        set_failure_rate_box = ttk.Entry(label_frame_1, textvariable=self.cmp_fr, width=8)
        set_failure_rate_box.grid(row=1, column=1, sticky='W', padx=(0, 100))

        set_uf = ttk.Label(label_frame_1, text="Utilization factor: ")
        set_uf.grid(row=2, column=0, sticky='W', pady=(0, 120))

        set_uf_box = ttk.Entry(label_frame_1, textvariable=self.cmp_uf, width=8)
        set_uf_box.grid(row=2, column=1, sticky='W', padx=(0, 100), pady=(0, 120))

        # Check Buttons for warranty and Plot Percentage
        # label Frame - Failure Modes
        label_frame_3 = ttk.LabelFrame(self.tab1, text="Failure Modes", height=20, width=10)
        label_frame_3.config(width=200, height=20)
        label_frame_3.grid(column=10, row=0, columnspan=10)

        mode_percent = ttk.Label(label_frame_3, text="Percentage(%)")
        mode_percent.grid(row=0, column=10, columnspan=5, sticky="W", pady=(5, 0))

        hidden = ttk.Label(label_frame_3, text="Hidden")
        hidden.grid(row=0, column=20, columnspan=4, sticky="W", pady=(5, 0))

        cmp_type = ttk.Label(label_frame_3, text="Type")
        cmp_type.grid(row=0, column=25, sticky=N+E+W+S, pady=(5, 0), padx=(55, 0), columnspan=4)

        ttk.Label(label_frame_3, text="Beta_min").grid(row=0, column=30,
                                                       sticky=N + E + W + S, pady=(5, 0), padx=(10, 0), columnspan=4)

        ttk.Label(label_frame_3, text="Beta_max").grid(row=0, column=35,
                                                       sticky=N + E + W + S, pady=(5, 0), padx=(4, 0))

        prediction_interval = ttk.Label(label_frame_3, text="Prediction \nInterval(Months)")
        prediction_interval.grid(row=0, column=16, columnspan=4, sticky="W", pady=(5, 0))

        self.is_hidden = [self.fm_1, self.fm_2, self.fm_3, self.fm_4, self.fm_5]
        self.percentage_fm = [self.fm_percentage_1, self.fm_percentage_2, self.fm_percentage_3, self.fm_percentage_4,
                              self.fm_percentage_5]
        self.prediction_interval_params = [self.prediction_interval_1, self.prediction_interval_2,
                                           self.prediction_interval_3, self.prediction_interval_4,
                                           self.prediction_interval_5]

        self.options_menu = [self.options_menu_1, self.options_menu_2, self.options_menu_3, self.options_menu_4,
                             self.options_menu_5]

        self.cmp_type.trace('w', self.enable_range_beta)

        self.beta_min = [self.beta_min_1, self.beta_min_2, self.beta_min_3, self.beta_min_4, self.beta_min_5]
        self.beta_max = [self.beta_max_1, self.beta_max_2, self.beta_max_3, self.beta_max_4, self.beta_max_5]

        options = self.options

        for mode in range(1, 6):
            failure_mode_label = ttk.Label(label_frame_3, text="FM" + str(mode) + ": ")
            failure_mode_label.grid(row=mode+1, column=0, columnspan=10, sticky="W", padx=(20, 0))

            mode_percent_entry = ttk.Entry(label_frame_3, textvariable=self.percentage_fm[mode-1], width=5)
            mode_percent_entry.config(state=DISABLED)
            mode_percent_entry.grid(row=mode+1, column=11, columnspan=5, sticky='W')

            hidden_check = ttk.Checkbutton(label_frame_3, variable=self.is_hidden[mode-1], onvalue=1, offvalue=0,
                                           width=2)
            hidden_check.config(state=DISABLED)
            hidden_check.grid(row=mode+1, column=20, columnspan=4, sticky='W', padx=(10, 0))

            prediction_interval_entry = ttk.Entry(label_frame_3,
                                                  textvariable=self.prediction_interval_params[mode-1], width=5)
            prediction_interval_entry.config(state=DISABLED)
            prediction_interval_entry.grid(row=mode+1, column=16, columnspan=4, sticky='W')

            beta_min_entry = ttk.Entry(label_frame_3, textvariable=self.beta_min[mode-1], width=5)
            beta_min_entry.config(state=DISABLED)
            beta_min_entry.grid(column=30, row=mode+1, sticky="W", padx=(15, 0), columnspan=4)

            beta_max_entry = ttk.Entry(label_frame_3, textvariable=self.beta_max[mode - 1], width=5)
            beta_max_entry.config(state=DISABLED)
            beta_max_entry.grid(column=35, row=mode + 1, sticky="W", columnspan=4, padx=(2, 0))

            self.failure_entry_widget.append(mode_percent_entry)
            self.hidden_widget.append(hidden_check)
            self.prediction_widget.append(prediction_interval_entry)
            self.beta_min_widget.append(beta_min_entry)
            self.beta_max_widget.append(beta_max_entry)

        popup_menu_1 = ttk.OptionMenu(label_frame_3, self.options_menu[0], *options)
        popup_menu_1.config(width=9, state=DISABLED)
        popup_menu_1.grid(column=25, row=1 + 1, sticky="W", columnspan=4)

        self.options_menu[0].trace('w', self.enable_range_beta)

        popup_menu_2 = ttk.OptionMenu(label_frame_3, self.options_menu[1], *options)
        popup_menu_2.config(width=9, state=DISABLED)
        popup_menu_2.grid(column=25, row=2 + 1, sticky="W", columnspan=4)

        self.options_menu[1].trace('w', self.enable_range_beta)

        popup_menu_3 = ttk.OptionMenu(label_frame_3, self.options_menu[2], *options)
        popup_menu_3.config(width=9, state=DISABLED)
        popup_menu_3.grid(column=25, row=3 + 1, sticky="W", columnspan=4)

        self.options_menu[2].trace('w', self.enable_range_beta)

        popup_menu_4 = ttk.OptionMenu(label_frame_3, self.options_menu[3], *options)
        popup_menu_4.config(width=9, state=DISABLED)
        popup_menu_4.grid(column=25, row=4 + 1, sticky="W", columnspan=4)

        self.options_menu[3].trace('w', self.enable_range_beta)

        popup_menu_5 = ttk.OptionMenu(label_frame_3, self.options_menu[4], *options)
        popup_menu_5.config(width=9, state=DISABLED)
        popup_menu_5.grid(column=25, row=5 + 1, sticky="W", columnspan=4)

        self.options_menu[4].trace('w', self.enable_range_beta)

        self.popup_menu_list = [popup_menu_1, popup_menu_2, popup_menu_3, popup_menu_4, popup_menu_5]

        # label Frame - LCC parameters - I

        access_params = {"Roof:": [self.is_access_roof_replace, self.is_access_roof_repair],
                         "Side/Intern:": [self.is_access_side_replace, self.is_access_side_repair],
                         "Pit:": [self.is_access_pit_replace, self.is_access_pit_repair],
                         "Breaking:": [self.is_access_breaking_replace, self.is_access_breaking_repair],
                         "Decoupling:": [self.is_access_decoupling_replace, self.is_access_decoupling_repair]
                         }

        time_access_params_replace = [self.time_roof_replace, self.time_side_replace, self.time_pit_replace,
                                      self.time_breaking_replace, self.time_decoupling_replace]

        time_access_params_repair = [self.time_roof_repair, self.time_side_repair, self.time_pit_repair,
                                     self.time_breaking_repair, self.time_decoupling_repair]

        label_frame_4 = ttk.LabelFrame(self.tab1, text="LCC Parameters - I ", height=20, width=12)
        label_frame_4.config(width=200, height=20)
        label_frame_4.grid(column=10, row=12, columnspan=8)

        maintenance_replacement = ttk.Label(label_frame_4, text="Replacement")
        maintenance_replacement.grid(row=0, column=2, columnspan=20, sticky=E)

        maintenance_repair = ttk.Label(label_frame_4, text="Repair")
        maintenance_repair.grid(row=0, column=27, columnspan=12, sticky=E)

        idx = 0
        idx_list = 0

        for key, val in access_params.items():
            label_replace = ttk.Label(label_frame_4, text=key)
            label_replace.grid(row=1+idx, column=0, columnspan=5, sticky="W", pady=(5, 0))

            check_button_replace = ttk.Checkbutton(label_frame_4, variable=val[0], onvalue=1, offvalue=0, width=2)
            check_button_replace.grid(row=1+idx, column=6, columnspan=5, sticky="W", pady=(5, 0), padx=(5, 0))

            entry_replace = ttk.Entry(label_frame_4, textvariable=time_access_params_replace[idx_list], width=8)
            entry_replace.grid(row=1+idx, column=16, columnspan=5, sticky='W',  pady=(5, 0))

            label_replace_2 = ttk.Label(label_frame_4, text="hr")
            label_replace_2.grid(row=1+idx, column=21, columnspan=5, sticky="W", pady=(5, 0))

            label_repair = ttk.Label(label_frame_4, text=key)
            label_repair.grid(row=1 + idx, column=0, columnspan=5, sticky="W", pady=(5, 0))

            check_button_repair = ttk.Checkbutton(label_frame_4, variable=val[1], onvalue=1, offvalue=0, width=2)
            check_button_repair.grid(row=1 + idx, column=26, columnspan=10, sticky="W", pady=(5, 0), padx=(35, 0))

            entry_repair = ttk.Entry(label_frame_4, textvariable=time_access_params_repair[idx_list], width=8)
            entry_repair.grid(row=1 + idx, column=36, columnspan=5, sticky='W', pady=(5, 0))

            label_repair_2 = ttk.Label(label_frame_4, text="hr")
            label_repair_2.grid(row=1 + idx, column=41, columnspan=5, sticky="W", pady=(5, 0), padx=(0, 200))

            idx += 2
            idx_list += 1

        # label Frame - LCC parameters - II
        label_frame_5 = ttk.LabelFrame(self.tab1, text="LCC Parameters - II ", height=20, width=10)
        label_frame_5.config(width=200, height=20)
        label_frame_5.grid(column=10, row=16, pady=(10, 0),  columnspan=20)

        # MTTR for corrective (h), replacement
        ttk.Label(label_frame_5, text="MTTR Corrective \nReplacement(hr): ").grid(row=16, column=0, columnspan=10,
                                                                                  sticky="W", pady=(5, 0))

        mttrc_replacement_entry = ttk.Entry(label_frame_5, textvariable=self.mttr_corrective_replacement,
                                            width=8)
        mttrc_replacement_entry.grid(row=16, column=11, columnspan=5, sticky='W', pady=(5, 0))

        # MTTR for corrective (h), repair
        ttk.Label(label_frame_5, text="MTTR Corrective \nRepair(hr): ").grid(row=16, column=16, columnspan=10,
                                                                             sticky="W", pady=(5, 0), padx=(10, 0))
        mttrc_repair_entry = ttk.Entry(label_frame_5, textvariable=self.mttr_corrective_repair, width=8)
        mttrc_repair_entry.grid(row=16, column=26, columnspan=5, sticky='W', pady=(5, 0))

        # MTTR Test
        ttk.Label(label_frame_5, text="MTTR Test(hr): ").grid(row=16, column=31, columnspan=10,
                                                              sticky="W", pady=(5, 0), padx=(10, 0))

        mttrc_test_entry = ttk.Entry(label_frame_5, textvariable=self.mttr_test, width=8)
        mttrc_test_entry.grid(row=16, column=41, columnspan=5, sticky='W', padx=(0, 45), pady=(5, 0))

        ttk.Label(label_frame_5, text="MTTR \nPreventive(hr): ").grid(row=18, column=0, columnspan=10,
                                                                      sticky="W", pady=(5, 0))

        mttrc_preventive_entry = ttk.Entry(label_frame_5, textvariable=self.mttr_preventive, width=8)
        mttrc_preventive_entry.grid(row=18, column=11, columnspan=5, sticky='W', pady=(5, 0))

        # Number of maintainer for corrective
        ttk.Label(label_frame_5, text="N. of Maintainer\n(Corrective): ").grid(row=20, column=0,  columnspan=10, sticky="W", pady=(5, 0))

        ttk.Entry(label_frame_5, textvariable=self.num_corrective, width=8).grid(row=20, column=11,
                                                                                 columnspan=5,
                                                                                 sticky='W', pady=(5, 0))

        # Number of maintainer for preventive
        ttk.Label(label_frame_5, text="N. of Maintainer\n(Preventive): ").grid(row=20, column=16, columnspan=10,
                                                                             sticky="W", pady=(5, 0), padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.num_preventive, width=8).grid(row=20, column=26, columnspan=5,
                                                                                 sticky='W', pady=(5, 0))

        # Number of maintainer for test
        ttk.Label(label_frame_5, text="N. of Maintainer\n(Test): ").grid(row=20, column=31, columnspan=10, sticky="W",
                                                                         pady=(5, 0),
                                                                         padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.n_man_ts, width=8).grid(row=20, column=41, columnspan=5,padx=(0, 45),
                                                                           sticky='W', pady=(5, 0))

        # Crew Cost
        ttk.Label(label_frame_5, text="Crew Cost (\u20ac/Hr):").grid(row=22, column=0,
                                                                     columnspan=10, sticky="W", pady=(5, 0))

        ttk.Entry(label_frame_5, textvariable=self.cost_crew, width=8).grid(row=22, column=11,
                                                                            columnspan=5, sticky='W', pady=(5, 0))

        ttk.Label(label_frame_5, text="Replacement\nCost (\u20ac):").grid(row=22, column=16,
                                                                          columnspan=10, sticky="W", pady=(5, 0),
                                                                          padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.co_spare_cm_1, width=8).grid(row=22, column=26,
                                                                                columnspan=5, sticky='W',
                                                                                pady=(5, 0))

        ttk.Label(label_frame_5, text="Repair\nCost (\u20ac):").grid(row=22, column=31,
                                                                     columnspan=10, sticky="W", pady=(5, 0),
                                                                     padx=(10, 0))

        repair_cost_entry = ttk.Entry(label_frame_5, textvariable=self.co_spare_cm_2, width=8)
        repair_cost_entry.grid(row=22, column=41, columnspan=5, sticky='W', padx=(0, 45), pady=(5, 0))

        ttk.Label(label_frame_5, text="Replacement Cost \nPM (\u20ac):").grid(row=24, column=0,
                                                                              columnspan=10, sticky="W", pady=(5, 0))

        replacement_cost_pm_entry = ttk.Entry(label_frame_5, textvariable=self.co_spare_pm, width=8)

        replacement_cost_pm_entry.grid(row=24, column=11, columnspan=5, sticky='W', pady=(5, 0))

        ttk.Label(label_frame_5, text="Min. Penalty \nCost (\u20ac):").grid(row=24, column=16, columnspan=10,
                                                                            sticky="W",
                                                                            pady=(5, 0),
                                                                            padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.co_pen_min, width=8).grid(row=24, column=26, columnspan=5,
                                                                             sticky='W',
                                                                             pady=(5, 0))

        ttk.Label(label_frame_5, text="Max. Penalty \nCost (\u20ac):").grid(row=24, column=31,
                                                                            columnspan=10, sticky="W", pady=(5, 0),
                                                                            padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.co_pen_max, width=8).grid(row=24, column=41,
                                                                             columnspan=5, sticky='W',padx=(0, 45),
                                                                             pady=(5, 0))

        ttk.Label(label_frame_5, text="Allowed Night \nIntervention Period(hr):").grid(row=26, column=0,
                                                                                       columnspan=10, sticky="W",
                                                                                       pady=(5, 0))

        ttk.Entry(label_frame_5, textvariable=self.night_time, width=8).grid(row=26, column=11, columnspan=5,
                                                                             sticky='W', pady=(5, 0))

        ttk.Label(label_frame_5, text="Time For Reaching\nDeposit:").grid(row=26, column=16, columnspan=10, sticky="W",
                                                                          pady=(5, 0), rowspan=2,
                                                                          padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.ld, width=8).grid(row=26, column=26, columnspan=5,
                                                                     sticky='W',
                                                                     pady=(5, 0))

        # next_button = ttk.Button(self.tab1, text='Next', command=lambda: self.tab_control.select(self.tab2))
        next_button = ttk.Button(self.tab1, text='Next', command=self.to_tab2)
        next_button.grid(row=26, column=26, sticky=N+E+S+W, pady=(0, 10))

    def add_widgets_sheet2(self):
        """
        Default Parameters:
        1.Failure Mode Matrix - Maximum Size 5x5
        2. Compatibility matrix
        3. Number of Monte Carlo simulation
        :return: 
        """
        # label Frame - LCC parameters - II
        # label Frame - Train Mission Parameters
        label_frame_0 = ttk.LabelFrame(self.tab2, text="Default Parameters - Effect Matrix", height=20, width=40)
        label_frame_0.config(width=200, height=20)
        label_frame_0.grid(column=0, row=12, padx=(20, 0), columnspan=4, sticky=W)

        label_frame_3 = ttk.LabelFrame(self.tab2, text="Description", height=20, width=40)
        label_frame_3.config(width=200, height=20)
        label_frame_3.grid(column=4, row=12, columnspan=4, sticky=W)

        ttk.Label(label_frame_3, text="Effect Matrix describes the effect of \nthe failure mode on the other, "
                                      "\nthe value ranges can be between \n0 and 1, where 0 \nindicates no effect "
                                      "\nand 1 indicates the highest"
                                      ".").grid(row=0, column=0, sticky="W", padx=(5, 0))

        mode = 0
        for mode in range(1, 6):
            ttk.Label(label_frame_0, text="FM" + str(mode) + ": ").grid(row=2+mode, column=0,
                                                                        sticky="W", padx=(20, 0))
            ttk.Label(label_frame_0, text="FM" + str(mode)).grid(row=0, column=1+mode, sticky="W")

            for col in range(1, 6):
                fm_matrix_entry = ttk.Entry(label_frame_0, textvariable=self.fm_matrix[mode-1][col - 1], width=5)
                fm_matrix_entry.config(state=DISABLED)
                fm_matrix_entry.grid(row=2+mode, column=1+col, sticky="W")
                self.fm_matrix_widget.append(fm_matrix_entry)
        col = 5

        # changes widget to 5d matrix - useful when trying to disable and enable buttons
        self.fm_matrix_5d = [self.fm_matrix_widget[i:i + col] for i in range(0, len(self.fm_matrix_widget), col)]

        # Label Frame
        label_frame_1 = ttk.LabelFrame(self.tab2, text="Compatibility Matrix", height=20, width=10)
        label_frame_1.config(width=100, height=20)
        label_frame_1.grid(column=4,  row=14, padx=(20, 0), pady=10, columnspan=4)

        ttk.Label(label_frame_1, text="Corrective").grid(row=0, column=2, sticky=N+E+S+W, pady=(0, 2))
        ttk.Label(label_frame_1, text="Preventive").grid(row=0, column=4, sticky=N+E+S+W, pady=(0, 2), padx=(0, 45))

        for mode in range(1, 6):
            ttk.Label(label_frame_1, text="FM" + str(mode) + ": ").grid(row=2 + mode, column=0,
                                                                        sticky="W", pady=2)

            corrective_entry = ttk.Entry(label_frame_1, textvariable=self.fm_corrective[mode - 1], width=5)
            corrective_entry.config(state=DISABLED)
            corrective_entry.grid(row=mode + 2, column=2, sticky=W, pady=2)

            # set to default = 1.0
            self.fm_corrective[mode-1].set(1.0)
            # store the check button widget
            self.fm_corrective_entry.append(corrective_entry)

            preventive_entry = ttk.Entry(label_frame_1, textvariable=self.fm_preventive[mode - 1], width=5)
            preventive_entry.config(state=DISABLED)
            preventive_entry.grid(row=mode + 2, column=4, sticky=W, pady=2, padx=(0, 45))

            # set to default = 1.0
            self.fm_preventive[mode - 1].set(1)

            self.fm_preventive_entry.append(preventive_entry)

        # Label Frame
        label_frame_2 = ttk.LabelFrame(self.tab2, text="Simulation Parameters", height=20, width=10)
        label_frame_2.config(width=200, height=20)
        label_frame_2.grid(column=0, row=14, padx=(20, 0), pady=10, columnspan=4)

        # Number of Monte carlo Simulations
        ttk.Label(label_frame_2, text="N. MonteCarlo Simulations: ").grid(row=0, column=0, sticky="W", padx=(5, 0))
        mc_simulations_entry = ttk.Entry(label_frame_2, textvariable=self.N_sim, width=6)
        mc_simulations_entry.grid(row=0, column=1, sticky=W)

        # defualt setting to 10000
        self.N_sim.set(10000)
        ttk.Label(label_frame_2, text="Step Size: ").grid(row=2, column=0, padx=(5, 0), sticky=W)
        step_size_entry = ttk.Entry(label_frame_2, textvariable=self.step_size, width=6)
        step_size_entry.grid(row=2, column=1, sticky=W, pady=(2, 0), padx=(0, 70))

        # set to default
        self.step_size.set(1.0)

        ttk.Label(label_frame_2, text="FR Sensitivity\nMin.(\u03BB): ").grid(row=4, column=0, padx=(5, 0),
                                                                             sticky=W)
        sensitivity_range_min_entry = ttk.Entry(label_frame_2, textvariable=self.fr_sens_min, width=6)
        sensitivity_range_min_entry.grid(row=4, column=1, sticky=W, padx=(0, 70))

        # set to default
        self.fr_sens_min.set(0.5)

        ttk.Label(label_frame_2, text="FR Sensitivity\nMax.(\u03BB).:").grid(row=6, column=0, padx=(5, 0),
                                                                             sticky=W)
        sensitivity_range_min_entry = ttk.Entry(label_frame_2, textvariable=self.fr_sens_max, width=6)
        sensitivity_range_min_entry.grid(row=6, column=1, sticky=W, padx=(0, 70))

        # set to default
        self.fr_sens_max.set(2.0)

        # Proposed test interval (y)
        ttk.Label(label_frame_2, text="Proposed Test Interval\n(Min. Yr): ").grid(row=8, column=0, padx=(5, 0),
                                                                                  sticky=W)

        proposed_test_interval_min_entry = ttk.Entry(label_frame_2, textvariable=self.Tst_min, width=6)
        proposed_test_interval_min_entry.grid(row=8, column=1, sticky=W)

        # set to default
        self.Tst_min.set(1)

        ttk.Label(label_frame_2, text="Proposed Test Interval\n(Max. Yr): ").grid(row=10, column=0, padx=(5, 0),
                                                                                  sticky=W)

        proposed_test_interval_max_entry = ttk.Entry(label_frame_2, textvariable=self.Tst_max, width=6)
        proposed_test_interval_max_entry.grid(row=10, column=1,  sticky=W, padx=(0, 70))

        self.Tst_max.set(15)
        # navigation Buttons
        self.back_button = ttk.Button(self.tab2, text='Back', command=self.to_tab1)
        self.back_button.grid(row=49, column=0, sticky=W, padx=(20, 0))

        self.start_button = ttk.Button(self.tab2, text='Start', command=self.start_rcm)
        self.start_button.grid(row=49, column=7, sticky=E)

        self.progressbar = ttk.Progressbar(self.tab2, mode='indeterminate', length=20)
        self.progressbar.grid(row=50, column=0, columnspan=8, sticky=N + E + W + S, padx=(20, 0))
        self.progressbar.state(['disabled'])

        self.stop_button = ttk.Button(self.tab2, text='Stop', command=self.stop_rcm, state=DISABLED)

        self.stop_button.grid(row=51, column=0, columnspan=8, sticky=N + E + W + S, padx=(20, 0))


if __name__ == "__main__":
    root = Root()
    root.mainloop()
