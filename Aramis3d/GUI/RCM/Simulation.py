import os
import utils
import Input_component
import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self):
        self.params = Input_component.Parameters()
        self.obj_instance = utils.Utils()

        self.w_parameter = self.get_w_parameter()

        self.fall = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                             dtype=np.float64)
        self.man = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                            dtype=np.float64)
        self.test = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                             dtype=np.float64)
        self.det = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                            dtype=np.float64)
        self.undet = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                              dtype=np.float64)
        self.t_fail_h = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                                 dtype=np.float64)
        self.costsc = np.zeros((self.w_parameter.shape[1], len(self.params.Tst), len(self.params.fr_sens)),
                               dtype=np.float64)

        self.Costsc_max = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.Costsc_min = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.fall_max = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.fall_min = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.l_size = 2

        self.det_max = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.det_min = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)

        self.undet_max = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.undet_min = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)

        self.t_fail_h_max = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)
        self.t_fail_h_min = np.zeros((len(self.params.Tst), len(self.params.fr_sens)), dtype=np.float64)

    def get_w_parameter(self):
        w_parameter = np.zeros((len(self.params.n), np.prod(self.params.n), 2, len(self.params.fr_sens)))
        self.obj_instance = utils.Utils()
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
                for p in range(0, 1):
                    tsc = self.params.PmT
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
                    self.fall[:, c, k] = np.mean(self.get_fallimenti())
                    self.man[:, c, k] = np.mean(self.get_maintenancesche())
                    self.test[:, c, k] = np.mean(self.get_testsche())
                    self.det[:, c, k] = np.mean(self.get_detected())
                    self.undet[:, c, k] = np.mean(self.get_undetected())
                    self.t_fail_h[:, c, k] = np.mean(self.get_tm_und())
                    self.costsc[:, c,  k] = np.mean(self.get_costsche())

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
        self.Costsc_max[:, :] = self.costsc.max(0)
        self.Costsc_min[:, :] = self.costsc.min(0)

        if not os.path.isdir(self.params.component):
            os.system('mkdir ' + self.params.component)

        x_time = 0

        if len(self.params.Tst) > 1:
            x_time = self.params.Tst

        # Turn ON grid
        plt.figure(0)
        plt.grid(True)

        for i in range(0, self.Costsc_max.shape[1]):
            plt.plot(x_time, self.Costsc_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=self.l_size)

        for i in range(0, self.Costsc_min.shape[1]):
            plt.plot(x_time, self.Costsc_min[:, i], linestyle='--', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=self.l_size)

        fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
        plt.title("Cost vs. FR")
        plt.legend(fr_legend, loc="upper right")
        plt.xlabel("PM interval [years]")
        plt.ylabel("Total maintenance cost")
        plt.savefig(self.params.component + '/' + self.params.component + "_Cost_Ext.png")
        plt.show(block=False)

        """
        costs according to lambda values ​​-
        The base case cost, corrective maintenance and best preventive are plotted for each lambda
        :return:
        """
        plt.figure(1)
        plt.grid(True)

        for i in range(0, self.Costsc_max.shape[1]):
            plt.plot([self.params.fr_sens[i], self.params.fr_sens[i]], [self.Costsc_min[0, i], self.Costsc_max[-1, i]],
                     linestyle='-', linewidth=self.l_size, color='black')

        plt.plot(self.params.fr_sens, self.Costsc_min[0, :], marker='o', color='g', linestyle='None',
                 ms=8, markerfacecolor='g', label="Base Case")
        plt.plot(self.params.fr_sens, self.Costsc_max[-1, :], marker='o', color='r', linestyle='None', ms=8,
                 markerfacecolor='r', label="Corrective")
        plt.plot(self.params.fr_sens, self.Costsc_max[1:self.Costsc_max.shape[0]-1, :].min(0), marker='o',
                 linestyle='None', color='blue', ms=8, markerfacecolor='blue', label="Best Preventive")

        plt.title("Cost vs. FR")
        plt.legend(loc="best")
        plt.xlabel("Relative Lambda")
        plt.ylabel("Total maintenance cost")
        plt.savefig(self.params.component + '/' + self.params.component + "_Cost_Scheme.png")
        plt.show(block=False)

        """
        ENF when PM intervals vary -
        The maximum value of ENF on the various beta / alpha is plotted for each FR
        :return:
        """

        self.fall_max[:, :] = self.fall.max(0)
        self.fall_min[:, :] = self.fall.min(0)

        plt.figure(2)
        plt.grid(True)

        xtime = 0
        if len(self.params.Tst) > 1:
            xtime = self.params.Tst

        for i in range(0, self.fall_max.shape[1]):
            plt.plot(xtime, self.fall_max[:, i], linestyle='-', marker=self.params.plot_marker[i],
                     color=self.params.plot_color[i], linewidth=self.l_size)

        fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
        plt.title("ENF vs. FR")
        plt.legend(fr_legend, loc="upper right")
        plt.xlabel("PM interval [years]")
        plt.ylabel("N. expected failures")
        plt.savefig(self.params.component + '/' + self.params.component + "_Failures.png")
        if np.sum(self.params.hidden) > 0:
            plt.show(block=False)
        else:
            plt.show()

    def plot_is_hidden_results(self):
        if np.sum(self.params.hidden) > 0:
            self.det_max[:, :] = self.det.max(0)
            self.det_min[:, :] = self.det.min(0)
            self.undet_max[:, :] = self.undet.max(0)
            self.undet_min[:, :] = self.undet.min(0)

            self.t_fail_h_max[:, :] = (self.t_fail_h/24.0).max(0)
            self.t_fail_h_min[:, :] = (self.t_fail_h/24.0).min(0)

            """
            Failures detected by the test when the PM or Test intervals vary -
            The maximum value of DF on the various beta / alpha is plotted for each FR
            :return:
            """
            plt.figure(4)
            plt.grid(True)

            x_time = 0

            if len(self.params.Tst) > 1:
                x_time = self.params.Tst

            for i in range(0, self.det_max.shape[1]):
                plt.plot(x_time, self.det_max[:, i], linestyle='-',
                         marker=self.params.plot_marker[i], color=self.params.plot_color[i], linewidth=self.l_size)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("Detected Failures vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("N. expected failures")
            plt.savefig(self.params.component + '/' + self.params.component + "_Det_Failures_Ext.png")
            plt.show(block=False)

            """
            Failures not detected by the test when the PM or Test intervals vary
            The maximum NDF value on the various beta / alpha is plotted for each FR
            :return:
            """
            plt.figure(5)
            plt.grid(True)

            x_time = 0

            if len(self.params.Tst) > 1:
                x_time = self.params.Tst

            for i in range(0, self.undet_max.shape[1]):
                plt.plot(x_time, self.undet_max[:, i], linestyle='-',
                         marker=self.params.plot_marker[i], color=self.params.plot_color[i], linewidth=self.l_size)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("UnDetected Failures vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("N. expected failures")
            plt.savefig(self.params.component + '/' + self.params.component + "_Undet_Failures_Ext.png")
            plt.show(block=False)

            """
            Average time for which a faulty component with an FM hidden remains faulty before being adjusted / detected
            The maximum value Time for the various beta / alpha is plotted for each FR
            :return:
            """
            plt.figure(6)
            plt.grid(True)

            for i in range(0, self.t_fail_h_max.shape[1]):
                plt.plot(x_time, self.t_fail_h_max[:, i], linestyle='-',
                         marker=self.params.plot_marker[i], color=self.params.plot_color[i], linewidth=self.l_size)

            fr_legend = ["FR x " + str(i) for i in self.params.fr_sens]
            plt.title("Undetected Time vs. FR")
            plt.legend(fr_legend, loc="best")
            plt.xlabel("PM interval [years]")
            plt.ylabel("Time [days]")
            plt.savefig(self.params.component + '/' + self.params.component + "_Time_Failures_Ext.png")
            plt.show()

if __name__ == "__main__":
    simulate = Simulation()
    simulate.run_cost_test()
    simulate.plot_results()
