import numpy as np


class Parameters:
    def __init__(self):
        """
        1. Train Mission Parameters and Other Parameters
        2. Failure Modes
        3. Effect Matrix
        4. Compatibility Matrix
        5. Accessibility Matrix
        6. MTTR
        7. Cost (Crew, Replacement, Repair)
        8. User Simulation parameters
        :return:
        """
        self.component = "CondesorMotor"

        self.Life = 31
        self.h_day = 20

        # Warranty Period
        self.Warranty = 2
        self.num_fail_modes = 0

        ########################################################################################
        # User Parameters - Failures
        # Component Parameters
        self.FT = 2.58e-6
        self.U = 1.0

        # Multiplication Factor FR (sensitivity, in general between 0.5 and 2)
        self.fr_sens = np.r_[0.5, 1, 2]

        # Color vector for plots according to FR (length equal to fr_sens)
        self.plot_color = ['g', 'b', 'r']

        # Marker vector for plots according to FR (length equal to fr_sens)
        self.plot_marker = ['o', 'x', '*']

        # Number of FM included in the FR defined above
        self.n_fm = 2

        # Repartition percentage of the FR on the FMs
        # NOTA: Number of rows =n_fm, Vector sum =1
        self.P_failure_mode = np.zeros((self.n_fm, 1), dtype=np.float64)
        self.P_failure_mode[0, 0] = 0.5
        self.P_failure_mode[1, 0] = 0.5

        # Predictive interval "as good as new" implemented for the component
        # NOTA: rows =n_fm, value in months, value =360 if going to failure
        self.Tau = np.zeros((self.n_fm, 1), dtype=np.int64)
        self.Tau[0, 0] = 120
        self.Tau[1, 0] = 120

        # Compatibility matrix† FM/PM
        self.Comp = np.zeros((self.n_fm, 1), dtype=np.float64)
        self.Comp[0, 0] = 1
        self.Comp[1, 0] = 1

        # Hidden matrix, it defines which FMs are hidden
        # NOTA: rows =n_fm, value =1 if the FM is hidden, value =0 if not
        self.hidden = np.zeros((self.n_fm, 1), dtype=np.float64)
        self.hidden[0, 0] = 0
        self.hidden[1, 0] = 1

        # Effect Matrix
        self.chi_w = np.zeros((self.n_fm, self.n_fm), dtype=np.int64)
        self.chi_w[0, :] = [0, 0]
        self.chi_w[1, :] = [0, 0]

        # Compatibility Matrix
        # NOTA: rows =n_fm, Values between 0 and 1 and equal to the replacement probability
        self.Comp_cm = np.zeros((self.n_fm, 1), dtype=np.float64)
        self.Comp_cm[0, 0] = 1
        self.Comp_cm[1, 0] = 1
        ########################################################################################

        # User Parameters - Maintenance

        # Accessibility procedure for REPLACEMENT
        # NOTA: valore =1 se necessaria, 0 se non necessaria

        # LCC Parameters - I
        self.acc = np.zeros((5, 3, 2), dtype=np.float64)

        self.acc[0, 1, 0] = 1
        self.acc[1, 1, 0] = 0
        self.acc[2, 1, 0] = 0
        self.acc[3, 1, 0] = 1
        self.acc[4, 1, 0] = 0

        self.acc[0, 1, 1] = 1
        self.acc[1, 1, 1] = 0
        self.acc[2, 1, 1] = 0
        self.acc[3, 1, 1] = 1
        self.acc[4, 1, 1] = 0

        self.acc[0, 0, :] = 0.5
        self.acc[1, 0, :] = 0.25
        self.acc[2, 0, :] = 0.25
        self.acc[3, 0, :] = 1
        self.acc[4, 0, :] = 3

        self.acc[0, 2, :] = 1
        self.acc[1, 2, :] = 1
        self.acc[2, 2, :] = 1
        self.acc[3, 2, :] = 1
        self.acc[4, 2, :] = 1

        # Crew cost (Euro/h)
        self.co_crew_h = 68

        # MTTR for corrective (h), replacement, repair
        self.MTTRc = np.r_[2+0.1*3, 0]

        # Number of maintainer for corrective
        self.n_man_cm = 1

        # MTTR in preventiva (h)
        self.MTTRsc = 3

        # Number of maintainer for preventive
        self.n_man_pm = 1

        # MTTR in test (h)
        self.MTTRts = 0.5

        # Number of maintainer for test
        self.n_man_ts = 1

        # Replacement cost REPLACEMENT (Euro) and Repair
        self.co_spare_cm = np.zeros((2, 1), dtype=np.float64)
        # Replacement cost REPLACEMENT(Euro)
        self.co_spare_cm[0] = 426 * 0.9 + 50 * 0.1
        # Replacement cost REPAIR(Euro)
        self.co_spare_cm[1] = 0
        # Replacement cost PM(Euro)
        self.co_spare_pm = 73

        # Allowed night intervention period (h)
        self.night_time = 5

        # Time for reaching deposit (h)
        self.ld = 0

        # Penalty cost for unplanned maintenance (min/max), Euro
        self.co_pen_min = 30000
        self.co_pen_max = 90000

        ########################################################################################
        # Parameter USER - Simulation

        # Number of Monte Carlo simulation
        self.n_sim = 10

        # Beta parameter values definition for sensitivity Weibull
        # NOTA: rows =n_fm
        self.n = np.r_[3, 3]

        # Beta parameter values definition for sensitivity Weibull
        # NOTA: rows =n_fm
        self.beta = np.zeros((len(self.n), max(self.n)), dtype=np.float64)

        # matlab linspace x1, x2, n
        self.beta[0, 0: self.n[0]] = np.linspace(0.5, 3, self.n[0], dtype=np.float64)
        self.beta[1, 0: self.n[1]] = np.linspace(0.5, 3, self.n[1], dtype=np.float64)

        # Proposed preventive interval (y)
        self.PmT = 31

        # Proposed test interval (y)
        self.Tst = np.arange(1., 16., 1.)

        ########################################################################################

        self.conversion_factor = (365 / 12) * self.h_day
        self.Tau = self.Tau * self.conversion_factor

        self.Life = self.Life * 365 * self.h_day

        # Calculation of CMT replacement
        self.cmt = np.zeros((2, 1), dtype=np.float64)
        # Calculation of man-hours and Cost spare/penalty
        self.wkl_cm = np.zeros((2, 1), dtype=np.float64)

        for cmt_id in range(0, 2):
            dot_prod_1 = np.dot(np.dot(self.acc[:, 0, cmt_id], self.acc[:, 1, cmt_id]), self.acc[:, 2, cmt_id])
            dot_prod_2 = np.dot(np.dot(self.acc[:, 0, cmt_id], self.acc[:, 1, cmt_id]), 1*(self.acc[:, 2, cmt_id] == 0))
            self.cmt[cmt_id, 0] = self.MTTRc[cmt_id] + dot_prod_1[0] + (dot_prod_2*self.n_man_cm)[0]
            self.wkl_cm[cmt_id, 0] = self.MTTRc[cmt_id] + (dot_prod_1[0]*self.n_man_cm) + (dot_prod_2*self.n_man_cm)[0]

        self.co_pen = self.co_pen_min * 1*((self.cmt+self.ld) > self.night_time)
        # Total fixed costs CM
        self.Ccor = self.co_pen + self.co_spare_cm
        self.pmt = self.MTTRsc
        self.wkl_pm = self.MTTRsc * self.n_man_pm
        self.Csc = self.co_spare_pm

        self.tst = self.MTTRts
        self.wkl_ts = self.MTTRts * self.n_man_ts

        self.Tgaranzia = self.Warranty * 365 * self.h_day

# if __name__ == "__main__":
#     params = Parameters()
#     print(params.co_spare_cm)
