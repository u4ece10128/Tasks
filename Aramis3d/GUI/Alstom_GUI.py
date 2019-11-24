from tkinter import *
from tkinter import ttk
import tkinter.messagebox as mbox
from RCM.Input_component import Parameters
from PIL import Image, ImageTk


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        # Root Widow Settings
        self.title("Alstom Control Application")
        self.geometry('1120x720')
        self.maxsize(width=1120, height=1000)
        self.iconbitmap('0.ico')
        self.resizable(0, 0)
        self.configure(background='DimGray')

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

        self.time_roof_replace = IntVar()
        self.time_side_replace = IntVar()
        self.time_pit_replace = IntVar()
        self.time_breaking_replace = IntVar()
        self.time_decoupling_replace = IntVar()

        self.time_roof_repair = IntVar()
        self.time_side_repair = IntVar()
        self.time_pit_repair = IntVar()
        self.time_breaking_repair = IntVar()
        self.time_decoupling_repair = IntVar()

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
                        "Unknown"]

        self.options_menu_1 = StringVar()
        self.options_menu_2 = StringVar()
        self.options_menu_3 = StringVar()
        self.options_menu_4 = StringVar()
        self.options_menu_5 = StringVar()

        self.time_roof_repair = IntVar()
        self.time_roof_replace = IntVar()

        self.cost_crew = IntVar()
        self.mttr_corrective_replacement = IntVar()
        self.mttr_corrective_repair = IntVar()
        self.mttr_test = IntVar()
        self.mttr_preventive = IntVar()
        self.num_corrective = IntVar()
        self.num_preventive = IntVar()
        self.n_man_ts = IntVar()
        self.co_spare_cm_1 = IntVar()
        self.co_spare_cm_2 = IntVar()
        self.co_spare_pm = IntVar()

        self.co_pen_min = IntVar()
        self.co_pen_max = IntVar()
        self.night_time = IntVar()
        self.ld = IntVar()

        self.is_hidden = []
        self.percentage_fm = []
        self.prediction_interval_params = []
        self.options_menu = []
        self.popup_menu_list = []
        self.failure_entry_widget = []
        self.hidden_widget = []
        self.prediction_widget = []

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

        self.progressbar = None
        self.stop_button = None

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

    def to_tab1(self):
        self.tab_control.tab(0, state="normal")
        self.tab_control.select(self.tab1)
        self.geometry('1120x720')
        self.tab_control.tab(1, state="disabled")

    def to_tab2(self):
        self.tab_control.tab(1, state="normal")
        self.tab_control.select(self.tab2)
        self.geometry('530x700')
        self.tab_control.tab(0, state="disabled")

    def start_rcm(self):
        mbox.showinfo("Info", "RCM Analysis Start")

        input_params = Parameters()

        self.progressbar.start()
        self.stop_button.config(state=ACTIVE)
        # call main.py and pass all the parameters

    def stop_rcm(self):
        mbox.showinfo("Info", "RCM Analysis Stopped")
        self.progressbar.stop()

    def check_percentage(self):
        sum_per = self.fm_percentage_1.get() + self.fm_percentage_2.get() + self.fm_percentage_3.get() + \
                  self.fm_percentage_4.get() + self.fm_percentage_5.get()
        if sum_per != 100.0:
            mbox.showerror('Error', 'Invalid Split in Percentage values')

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
        label_frame_1 = ttk.LabelFrame(self.tab1, text="Component Parameters", height=20, width=10)
        label_frame_1.config(width=100, height=20)
        label_frame_1.grid(column=0, row=16, padx=5, columnspan=10)

        # add Labels abd Entry widgets
        set_failure_rate = ttk.Label(label_frame_1, text="Failure Rate: ")
        set_failure_rate.grid(column=0, row=16, sticky='W')

        set_failure_rate_box = ttk.Entry(label_frame_1, textvariable=self.cmp_fr, width=8)
        set_failure_rate_box.grid(row=16, column=20, sticky='W')

        set_uf = ttk.Label(label_frame_1, text="Utilization factor: ")
        set_uf.grid(row=18, column=0, sticky='W')

        set_uf_box = ttk.Entry(label_frame_1, textvariable=self.cmp_uf, width=8)
        set_uf_box.grid(row=18, column=20, sticky='W')

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
        cmp_type.grid(row=0, column=25, sticky=N+E+W+S, pady=(5, 0), padx=(55, 45))

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

            # Label + listbox
            popup_menu = ttk.OptionMenu(label_frame_3, self.options_menu[mode-1], *options)
            popup_menu.config(width=9, state=DISABLED)
            popup_menu.grid(column=25, row=mode+1, sticky="W", padx=(15, 45))

            self.failure_entry_widget.append(mode_percent_entry)
            self.hidden_widget.append(hidden_check)
            self.prediction_widget.append(prediction_interval_entry)
            self.popup_menu_list.append(popup_menu)

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

        ttk.Entry(label_frame_5, textvariable=self.mttr_corrective_replacement, width=8).grid(row=16,
                                                                                              column=11, columnspan=5,
                                                                                              sticky='W', pady=(5, 0))

        # MTTR for corrective (h), repair
        ttk.Label(label_frame_5, text="MTTR Corrective \nRepair(hr): ").grid(row=16, column=16, columnspan=10,
                                                                             sticky="W", pady=(5, 0), padx=(10, 0))
        ttk.Entry(label_frame_5, textvariable=self.mttr_corrective_repair, width=8).grid(row=16, column=26,
                                                                                         columnspan=5, sticky='W',
                                                                                         pady=(5, 0))
        # MTTR Test
        ttk.Label(label_frame_5, text="MTTR Test(hr): ").grid(row=16, column=31, columnspan=10,
                                                              sticky="W", pady=(5, 0), padx=(10, 0))

        ttk.Entry(label_frame_5, textvariable=self.mttr_test, width=8).grid(row=16, column=41,
                                                                            columnspan=5, sticky='W',
                                                                            pady=(5, 0))

        ttk.Label(label_frame_5, text="MTTR \nPreventive(hr): ").grid(row=18, column=0, columnspan=10,
                                                                      sticky="W", pady=(5, 0))

        ttk.Entry(label_frame_5, textvariable=self.mttr_preventive, width=8).grid(row=18, column=11,
                                                                                  columnspan=5, sticky='W',
                                                                                  pady=(5, 0))

        # Number of maintainer for corrective
        ttk.Label(label_frame_5, text="N. of Maintainer\n(Corrective): ").grid(row=20, column=0,  columnspan=10,
                                                                             sticky="W", pady=(5, 0))

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

        ttk.Entry(label_frame_5, textvariable=self.n_man_ts, width=8).grid(row=20, column=41, columnspan=5,
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
        repair_cost_entry.grid(row=22, column=41, columnspan=5, sticky='W', pady=(5, 0))

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
                                                                             columnspan=5, sticky='W',
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

        # replacement Cost PM

        # // Number of maintainer for preventive
        # mttr_corrective_label =

        # label_5 = ttk.Label(self.tab1, text="LCC Parameters - II ")
        # label_5.grid(column=20, row=16, padx=20, pady=(10, 0), columnspan=20)
        #
        # cost_crew = ttk.Label(self.tab1, text="Crew Cost (Euro/Hr)")
        # cost_crew.grid(column=20, row=18)
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
        ttk.Button(self.tab2, text='Back', command=self.to_tab1).grid(row=49, column=0, sticky=W, padx=(20, 0))

        ttk.Button(self.tab2, text='Start', command=self.start_rcm).grid(row=49, column=3, sticky=E)

        self.progressbar = ttk.Progressbar(self.tab2, mode='indeterminate', length=20)
        self.progressbar.grid(row=50, column=0, columnspan=4, sticky=N + E + W + S, padx=(20, 0))
        self.progressbar.state(['disabled'])

        self.stop_button = ttk.Button(self.tab2, text='Stop', command=self.stop_rcm, state=DISABLED)

        self.stop_button.grid(row=51, column=0, columnspan=4, sticky=N + E + W + S, padx=(20, 0))


if __name__ == "__main__":
    root = Root()
    root.mainloop()
