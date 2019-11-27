import Simulate
import threading
import queue
import tkinter
import numpy as np
from multiprocessing import Queue
from tkinter import *
from tkinter import ttk
import tkinter.messagebox as mbox
from RCM import Input_component
from PIL import Image, ImageTk


class ThreadedTask(threading.Thread):
    def __init__(self, queue, params):
        threading.Thread.__init__(self)
        self.queue = queue
        self.params = params

    @staticmethod
    def simulate_test(params):
        simulate = Simulate.Simulate(params)
        simulate.print_params()

    def run(self):
        self.simulate_test(self.params)  # Simulate long running process
        self.queue.put("Task finished")


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        # Root Widow Settings
        self.title("Alstom Control Application")
        self.geometry('1120x760')
        self.maxsize(width=1120, height=760)
        self.iconbitmap('0.ico')
        self.resizable(0, 0)
        self.configure(background='DimGray')

        # threading variable
        self.submit_thread = None

        # variables
        self.life = IntVar()
        self.h_day = IntVar()
        self.cmp_fr = IntVar()
        self.cmp_uf = IntVar()
        self.cmp_type = StringVar()
        self.fail_mode = IntVar()
        self.warranty_bool = BooleanVar()
        self.plt_percent_bool = BooleanVar()
        self.duration = IntVar()
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

        self.fm_corrective_1 = BooleanVar()
        self.fm_corrective_2 = BooleanVar()
        self.fm_corrective_3 = BooleanVar()
        self.fm_corrective_4 = BooleanVar()
        self.fm_corrective_5 = BooleanVar()

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
        self.step_size = IntVar()
        self.fr_sens_min = IntVar()
        self.fr_sens_max = IntVar()
        self.Tst_min = IntVar()
        self.Tst_max = IntVar()

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

        self.queue = None

    def start_rcm(self):
        if self.is_mandate() is not False:
            mbox.showinfo("Info", "RCM Analysis Start")
            self.disable_button()
            self.progressbar.start()
            self.queue = Queue()
            ThreadedTask(self.queue, params=self.load_input_params()).start()
            self.after(100, self.process_queue)

    def process_queue(self):
        try:
            msg = self.queue.get(0)
            self.progressbar.stop()
            self.enable_button()
        except queue.Empty:
            self.after(100, self.process_queue)

    def stop_rcm(self):
        mbox.showinfo("Info", "RCM Analysis Stopped")
        self.progressbar.stop()

    @staticmethod
    def is_alpha(val):
        """
        Check if the entry content is strictly Alphabetic
        :return:
        """
        if val.isalpha() is False:
            return False
        else:
            return True

    @staticmethod
    def is_digit(val):
        """
        Check if the entry content is strictly Alphabetic
        :return:
        """
        val = str(val)
        if val.replace(".", "", 1).isdigit() is False:
            return False
        else:
            return True

    def is_mandate(self):
        """
        Check if there exists a value in the mandatory fields, If not raise error
        :return:
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

        if self.warranty_bool == 1:
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
            print(type(self.set_cmp_name.get()))
            if len(self.set_cmp_name.get()) == 0:
                mbox.showerror("Error", "Invalid Component Field")
                return False
            mbox.showerror("Error", "Invalid Component Field")
            return False

        try:
            self.is_digit(self.cmp_fr.get())
            if self.cmp_fr.get() < 0:
                mbox.showerror("Error", "Invalid Failure Rate")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Failure Rate")
            return False

        try:
            self.is_digit(self.cmp_uf.get())
            if self.cmp_uf.get() < 0:
                mbox.showerror("Error", "Invalid Utilization Factor")
                return False
        except tkinter.TclError:
            mbox.showerror("Error", "Invalid Utilization Factor")
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
            mbox.showerror("Error", "Invalid Replacement Value")
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
        self.geometry('1120x760')
        self.tab_control.tab(1, state="disabled")

    def to_tab2(self):
        self.tab_control.tab(1, state="normal")
        self.tab_control.select(self.tab2)
        self.geometry('530x700')
        self.tab_control.tab(0, state="disabled")

    def load_input_params(self):
        """
        Build Input Parameters object to run the simulation
        :return: Object Instancc
        """
        input_params = Input_component.Parameters(self.set_cmp_name.get())
        # Train Life
        input_params.Life = self.life.get()
        input_params.h_day = self.h_day.get()
        input_params.n_fm = self.fail_mode.get()
        input_params.Warranty = self.duration.get()

        # Component Parameters
        # Component is already passed as an initialization parameter
        input_params.FT = self.cmp_fr.get()
        input_params.U = self.cmp_uf.get()

        # Failure Modes
        input_params.P_failure_mode = np.zeros((self.fail_mode.get(), 1), dtype=np.float64)
        input_params.Tau = np.zeros((self.fail_mode.get(), 1), dtype=np.float64)
        for fail_mode in range(0, self.fail_mode.get()):
            input_params.P_failure_mode[fail_mode, 0] = self.percentage_f[fail_mode].get()
            input_params.Tau[fail_mode, 0] = self.prediction_interval_params[fail_mode].get()

        # LCC parameters I
        # Accesibility procedure
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
        return input_params

    def check_percentage(self):
        sum_per = self.fm_percentage_1.get() + self.fm_percentage_2.get() + self.fm_percentage_3.get() + \
                  self.fm_percentage_4.get() + self.fm_percentage_5.get()
        if sum_per != 100.0:
            mbox.showerror('Error', 'Invalid Split in Percentage values')

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
            self.fm_corrective[i-1].set(0)
            self.fm_preventive[i-1].set(0)
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

        print(self.night_time)

    @staticmethod
    def add_logo(tab):
        # Logo - Start
        image = Image.open("0.jpg")
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

        prediction_interval = ttk.Label(label_frame_3, text="Prediction \nInterval (\u03C4)")
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
        self.fm_matrix_5d = [self.fm_matrix_widget[i:i + col] for i in range(0, len(self.fm_matrix_widget), col)]

        # Label Frame
        label_frame_1 = ttk.LabelFrame(self.tab2, text="Compatibility Matrix", height=20, width=10)
        label_frame_1.config(width=100, height=20)
        label_frame_1.grid(column=0,  row=14, padx=(10, 0), pady=10, columnspan=2)

        ttk.Label(label_frame_1, text="Corrective").grid(row=0, column=2, sticky=N+E+S+W, pady=(0, 2))
        ttk.Label(label_frame_1, text="Preventive").grid(row=0, column=4, sticky=N+E+S+W, pady=(0, 2))

        for mode in range(1, 6):
            ttk.Label(label_frame_1, text="FM" + str(mode) + ": ").grid(row=2 + mode, column=0,
                                                                        sticky="W", padx=(20, 0), pady=2)

            corrective_entry = ttk.Entry(label_frame_1, textvariable=self.fm_corrective[mode - 1], width=5)
            corrective_entry.config(state=DISABLED)
            corrective_entry.grid(row=mode + 2, column=2, sticky=W, pady=2)

            # store the check button widget
            self.fm_corrective_entry.append(corrective_entry)

            preventive_entry = ttk.Entry(label_frame_1, textvariable=self.fm_preventive[mode - 1], width=5)
            preventive_entry.config(state=DISABLED)
            preventive_entry.grid(row=mode + 2, column=4, sticky=W, pady=2)

            self.fm_preventive_entry.append(preventive_entry)

        # Label Frame
        label_frame_2 = ttk.LabelFrame(self.tab2, text="Simulation Parameters", height=20, width=10)
        label_frame_2.config(width=200, height=20)
        label_frame_2.grid(column=2, row=14, padx=(2, 0), pady=10, columnspan=2)

        # Number of Monte carlo Simulations
        ttk.Label(label_frame_2, text="N. MonteCarlo Simulations: ").grid(row=0, column=0, sticky="W", padx=(5, 0))
        mc_simulations_entry = ttk.Entry(label_frame_2, textvariable=self.N_sim, width=6)
        mc_simulations_entry.grid(row=0, column=1, sticky=W)

        ttk.Label(label_frame_2, text="Step Size: ").grid(row=2, column=0, padx=(5, 0), sticky=W)
        step_size_entry = ttk.Entry(label_frame_2, textvariable=self.step_size, width=6)
        step_size_entry.grid(row=2, column=1, sticky=W, pady=(2, 0))

        ttk.Label(label_frame_2, text="FR Sensitivity\nMin.(\u03BB): ").grid(row=4, column=0, padx=(5, 0),
                                                                             sticky=W)
        sensitivity_range_min_entry = ttk.Entry(label_frame_2, textvariable=self.fr_sens_min, width=6)
        sensitivity_range_min_entry.grid(row=4, column=1, sticky=W)

        ttk.Label(label_frame_2, text="FR Sensitivity\nMax.(\u03BB).:").grid(row=6, column=0, padx=(5, 0),
                                                                             sticky=W)
        sensitivity_range_min_entry = ttk.Entry(label_frame_2, textvariable=self.fr_sens_max, width=6)
        sensitivity_range_min_entry.grid(row=6, column=1, sticky=W)

        # Proposed test interval (y)
        ttk.Label(label_frame_2, text="Proposed Test Interval\n(Min. Yr): ").grid(row=8, column=0, padx=(5, 0),
                                                                                  sticky=W)

        proposed_test_interval_min_entry = ttk.Entry(label_frame_2, textvariable=self.Tst_min, width=6)
        proposed_test_interval_min_entry.grid(row=8, column=1, sticky=W)

        ttk.Label(label_frame_2, text="Proposed Test Interval\n(Max. Yr): ").grid(row=10, column=0, padx=(5, 0),
                                                                                  sticky=W)

        proposed_test_interval_max_entry = ttk.Entry(label_frame_2, textvariable=self.Tst_max, width=6)
        proposed_test_interval_max_entry.grid(row=10, column=1,  sticky=W)

        # navigation Buttons
        self.back_button = ttk.Button(self.tab2, text='Back', command=self.to_tab1)
        self.back_button.grid(row=49, column=0, sticky=W, padx=(20, 0))

        self.start_button = ttk.Button(self.tab2, text='Start', command=self.start_rcm)
        self.start_button.grid(row=49, column=3, sticky=E)

        self.progressbar = ttk.Progressbar(self.tab2, mode='indeterminate', length=20)
        self.progressbar.grid(row=50, column=0, columnspan=4, sticky=N + E + W + S, padx=(20, 0))
        self.progressbar.state(['disabled'])

        self.stop_button = ttk.Button(self.tab2, text='Stop', command=self.stop_rcm, state=DISABLED)

        self.stop_button.grid(row=51, column=0, columnspan=4, sticky=N + E + W + S, padx=(20, 0))


if __name__ == "__main__":
    root = Root()
    root.mainloop()
