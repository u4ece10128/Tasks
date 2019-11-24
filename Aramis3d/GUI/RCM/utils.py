import numpy as np


class Utils:
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

        for k in range(0, n):
            print("Num of Monte carlo:", k)
            t_und = [0, 0]
            tbreakP = np.zeros((nfailuremode, 1), dtype=np.float64)
            tstopP = 0
            tmonitor = np.array(tsched, dtype=np.float64)
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
                next_maintenance = tmonitor.min(0)
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
                                    tbreakP[nf] = tstopP + alpha_w[nf] * np.random.weibull(beta_w[nf])
                                if tstopP > warranty:
                                    self.costsche[k] = self.costsche[k] + (((MTTR[0] * co_crew_h) + ccor[0])
                                                                           * depreciation **
                                                                           np.floor(tbreakP[which_maintenance])[0])
                            else:
                                gamma = tstopP
                                tbreakP[which_failuremode] = tstopP + alpha_w[which_failuremode] * \
                                                             ((gamma/alpha_w[which_failuremode]) **
                                                              beta_w[which_failuremode] - np.log(1 - np.random.rand(
                                                                     ))) ** (1/beta_w[which_failuremode] - gamma)

                                if tstopP > warranty:
                                    self.costsche[k] = self.costsche[k] + ((MTTR[0] * co_crew_h) + ccor[0]) * \
                                                   depreciation ** np.floor(tbreakP[which_maintenance])
                            thidden = 2 * horizon
                        else:
                            if thidden == 2 * horizon:
                                thidden = tstopP
                                for nf in range(0, nfailuremode):
                                    if nf != which_failuremode and chi_w[nf, which_failuremode] != 0:
                                        weibull_sample = 1 / chi_w ** (1 / beta_w[nf]) * alpha_w[nf] \
                                                         * np.random.weibull(beta_w[nf])
                                        tbreakP[nf] = min(tbreakP[nf], tstopP + weibull_sample, beta_w[nf])
                                    else:
                                        tbreakP[nf] = 2 * horizon
                            tbreakP[which_failuremode] = 2 * horizon

                    if which_event == 1:
                        print("Perform Test")
                        tnexttest = tnexttest + ttest

                        if next_test != next_maintenance:
                            self.costsche[k] = self.costsche[k] + \
                                               ((time_test * co_crew_h) * depreciation ** np.floor(tstopP))
                            self.testsche[k] = self.testsche[k] + 1
                        if thidden < 2 * horizon:
                            for fn in range(0, nfailuremode):
                                tbreakP[fn] = tstopP + (alpha_w[fn] * np.random.weibull(beta_w[fn]))
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
                                tmonitor[which_maintenance] = tstopP + tsched[which_maintenance]
                    if which_event == 2:
                        self.costsche[k] = self.costsche[k] + ((time_test * co_crew_h) * depreciation **
                                                               np.floor(tstopP))
                        self.testsche[k] = self.testsche[k] + 1
                        if thidden < 2 * horizon:
                            for fn_2 in range(0, nfailuremode):
                                tbreakP[fn_2] = tstopP + (alpha_w[fn_2] * np.random.weibull(beta_w[fn_2]))
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
                                index_alpha = np.where(1*(tmonitor == next_maintenance) == 1)
                                for index in index_alpha:
                                    tmonitor[index] = tstopP + tsched[index]
                        else:
                            qualiaggiorno = np.where(compatibility[:, which_maintenance] == 1)
                            for id_q in qualiaggiorno:
                                tbreakP[id_q] = tstopP + (alpha_w[id_q] * np.random.weibull(beta_w[id_q]))
                            tmonitor[which_maintenance] = tstopP + tsched[which_maintenance]
                            self.maintenancesche[k] = self.maintenancesche[k] + 1
                            self.costsche[k] = self.costsche[k] + ((time_preventive[which_maintenance] * co_crew_h
                                                                   + cprev[which_maintenance])
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
