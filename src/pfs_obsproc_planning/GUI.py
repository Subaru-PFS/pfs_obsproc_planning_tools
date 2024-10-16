#!/usr/bin/env python3
# GUI.py : GUI for Subaru Fiber Allocation software

import warnings

warnings.filterwarnings("ignore")

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from loguru import logger
import toml


class GeneratePfsDesignGUI(object):
    def __init__(self, repoDir=None):
        self.app = QtWidgets.QApplication([])
        self.app_window = uic.loadUi(f"{repoDir}/GUI_window/mainWindow.ui")
        self.config_window = uic.loadUi(f"{repoDir}/GUI_window/configWindow.ui")
        self.proposal_window = uic.loadUi(f"{repoDir}/GUI_window/proposalWindow.ui")

    def obsmode_enable(self):
        # classic mode
        if self.app_window.radioButton_classic.isChecked():
            self.app_window.pushButton_tgt_local.setEnabled(True)
            self.app_window.box_readtarget_mix.setEnabled(False)
            self.app_window.box_readPPC.setEnabled(True)
            self.app_window.box_obstime_start.setEnabled(True)
            self.app_window.box_obstime_stop.setEnabled(True)

        # queue mode
        if self.app_window.radioButton_queue.isChecked():
            self.app_window.pushButton_tgt_local.setEnabled(True)
            self.app_window.box_readtarget_mix.setEnabled(False)
            self.app_window.box_readPPC.setEnabled(False)
            self.app_window.box_obstime_start.setEnabled(False)
            self.app_window.box_obstime_stop.setEnabled(False)

    def getPslID_tgtDB(self):
        filename = self.app_window.lineEdit_workdir_path.text() + "/config.toml"
        if len(filename) > 12:
            config = toml.load(filename)
        else:
            logger.error("No config.toml file can be found under workdir.")

        dialect, user, pwd, host, port, dbname = [
            config["targetdb"]["db"]["dialect"],
            config["targetdb"]["db"]["user"],
            config["targetdb"]["db"]["password"],
            config["targetdb"]["db"]["host"],
            config["targetdb"]["db"]["port"],
            config["targetdb"]["db"]["dbname"],
        ]

        import pandas as pd
        import sqlalchemy as sa

        DBads = "{0}://{1}:{2}@{3}:{4}/{5}".format(
            dialect, user, pwd, host, port, dbname
        )
        tgtDB = sa.create_engine(DBads)

        sql = """
            SELECT proposal_id,pi_first_name,pi_last_name,pi_middle_name,rank,grade,is_too,allocated_time_lr,allocated_time_mr
            FROM proposal 
            """

        tgtDB_connect = tgtDB.connect()
        query = tgtDB_connect.execute(sa.sql.text(sql))

        df_psl = pd.DataFrame(
            query.fetchall(),
            columns=[
                "proposal_id",
                "pi_first_name",
                "pi_last_name",
                "pi_middle_name",
                "rank",
                "grade",
                "is_too",
                "allocated_time_lr",
                "allocated_time_mr",
            ],
        )

        df_psl["PI"] = (
            df_psl["pi_first_name"] + df_psl["pi_middle_name"] + df_psl["pi_last_name"]
        )
        df_psl = df_psl.drop(
            columns=["pi_first_name", "pi_last_name", "pi_middle_name"]
        )

        tgtDB_connect.close()

        nRows = len(df_psl.index)
        nColumns = len(df_psl.columns)

        self.proposal_window.tableWidget_proposalInfo.setRowCount(nRows)
        self.proposal_window.tableWidget_proposalInfo.setColumnCount(nColumns)
        self.proposal_window.tableWidget_proposalInfo.setHorizontalHeaderLabels(
            [
                "proposal_id",
                "rank",
                "grade",
                "is_too",
                "allocated_time_lr",
                "allocated_time_mr",
                "PI",
            ]
        )

        for i in range(nRows):
            for j in range(nColumns):
                item_temp = QTableWidgetItem(f"{df_psl.iloc[i, j]}")
                self.proposal_window.tableWidget_proposalInfo.setItem(i, j, item_temp)

        self.proposal_window.show()
        self.proposal_window.pushButton_pslDone.clicked.connect(self.setPslID_tgtDB)

    def setPslID_tgtDB(self):
        psl_id = ""
        for item in self.proposal_window.tableWidget_proposalInfo.selectedItems():
            if item.column() == 0:
                psl_id += f"'{item.text()}', "
        self.app_window.lineEdit_pslid_db.setText(psl_id)

    def getfile_tgt(self):
        fname = QFileDialog.getOpenFileName(filter="CSV files (*.csv)")
        self.app_window.lineEdit_tgt_local_path.setText(fname[0])

    def getfile_ppc(self):
        fname = QFileDialog.getOpenFileName(filter="CSV files (*.csv)")
        self.app_window.lineEdit_ppc_local_path.setText(fname[0])

    def getfile_ope(self):
        fname = QFileDialog.getOpenFileName(filter="(*.ope)")
        self.app_window.lineEdit_ope_path.setText(fname[0])

    def getfile_config(self):
        fname = QFileDialog.getOpenFileName(filter="(*.toml)")
        self.app_window.lineEdit_config_path.setText(fname[0])

    def getfile_queuedb(self):
        fname = QFileDialog.getOpenFileName(filter="(*.yml)")
        self.app_window.lineEdit_queuedb_path.setText(fname[0])

    def getfolder_design(self):
        fname = QFileDialog.getExistingDirectory()
        self.app_window.lineEdit_design_local_path.setText(fname)

    def getfolder_workdir(self):
        fname = QFileDialog.getExistingDirectory()
        self.app_window.lineEdit_workdir_path.setText(fname)

    def getfolder_instdata(self):
        fname = QFileDialog.getExistingDirectory()
        self.app_window.lineEdit_instdata_path.setText(fname)

    def getfolder_cobra(self):
        fname = QFileDialog.getExistingDirectory()
        self.app_window.lineEdit_cobra_path.setText(fname)

    def getfolder_schema(self):
        fname = QFileDialog.getExistingDirectory()
        self.app_window.lineEdit_schema_path.setText(fname)

    def obstime_add(self):
        obsdate = self.app_window.dateEdit_obstime_date.date().toString("yyyy-MM-dd")

        if self.app_window.box_obstime_start.isChecked():
            obstime_start = self.app_window.dateTimeEdit_starttime.time().toString(
                "H:mm:ss"
            )
        else:
            obstime_start = "-:--:--"

        if self.app_window.box_obstime_stop.isChecked():
            obstime_stop = self.app_window.dateTimeEdit_stoptime.time().toString(
                "H:mm:ss"
            )
        else:
            obstime_stop = "-:--:--"

        obstime_combined = obsdate + " " + obstime_start + " " + obstime_stop

        self.app_window.listWidget_obstime.addItem(obstime_combined)

    def obstime_remove(self):
        for item in self.app_window.listWidget_obstime.selectedItems():
            self.app_window.listWidget_obstime.takeItem(
                self.app_window.listWidget_obstime.row(item)
            )

    def load_template_config(self):
        filename = self.app_window.lineEdit_config_path.text()
        try:
            with open(filename, "r") as file:
                # read in config template and store in config_temp
                config_temp = ""
                for line in file:
                    config_temp += line

                self.config_tem_new = self.config_template_rev(config_temp)

                self.config_window.plainTextEdit_config.setPlainText(
                    self.config_tem_new
                )
                self.config_window.show()
                self.config_window.pushButton_saveconfig.clicked.connect(
                    self.save_template_config
                )

        except FileNotFoundError:
            logger.error("[FileNotFoundError] No config template is found.")

    def save_template_config(self):
        workdir = self.app_window.lineEdit_workdir_path.text()
        filename = workdir + "/config.toml"

        with open(filename, "w") as file:
            # write config_tem_new to config.toml
            file.write(self.config_tem_new)
            logger.info(f"config.toml file is saved as {filename}")
        file.close()

    def config_template_rev(self, config_ori):
        # obs_mode
        if self.app_window.radioButton_queue.isChecked():
            obs_mode = "queue"
        elif self.app_window.radioButton_classic.isChecked():
            obs_mode = "classic"
        repl_old = "mode="
        repl_new = f'mode="{obs_mode}"'
        config_ori = config_ori.replace(repl_old, repl_new)

        # localPath_tgt
        if len(self.app_window.lineEdit_tgt_local_path.text()) > 0:
            localPath_tgt = f'"{self.app_window.lineEdit_tgt_local_path.text()}"'
        else:
            localPath_tgt = '""'
        repl_old = "localPath_tgt= "
        repl_new = f"localPath_tgt={localPath_tgt}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # localPath_ppc
        if (
            self.app_window.box_readPPC.isChecked()
            and len(self.app_window.lineEdit_ppc_local_path.text()) > 0
        ):
            localPath_ppc = f'"{self.app_window.lineEdit_ppc_local_path.text()}"'
        else:
            localPath_ppc = '""'
        repl_old = "localPath_ppc="
        repl_new = f"localPath_ppc={localPath_ppc}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # PROPOSAL_ID
        psl_id = self.app_window.lineEdit_pslid_db.text()[:-2]
        if len(psl_id) > 0:
            repl_old = "PROPOSAL_ID"
            repl_new = f"{psl_id}"
            config_ori = config_ori.replace(repl_old, repl_new)

        # overhead
        overhead = f'"{self.app_window.doubleSpinBox_overhead.value()}"'
        repl_old = "overhead="
        repl_new = f"overhead={overhead}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # obs_dates
        nrow_dates = self.app_window.listWidget_obstime.count()
        obs_dates = ""
        # obs_starttimes = ''
        # obs_stoptimes = ''
        for row_temp in range(nrow_dates):
            print(self.app_window.listWidget_obstime.item(row_temp).text())
            print(self.app_window.listWidget_obstime.item(row_temp).text().split(" "))
            obs_date_temp, obs_starttime_temp, obs_stoptime_temp = (
                self.app_window.listWidget_obstime.item(row_temp).text().split(" ")
            )
            obs_dates += f'"{obs_date_temp}", '
            # obs_starttimes += f'"{obs_date_temp} {obs_starttime_temp}", '
            # obs_stoptimes += f'"{obs_date_temp} {obs_stoptime_temp}", '
        repl_old = "obs_dates="
        repl_new = f"obs_dates=[{obs_dates}]"
        config_ori = config_ori.replace(repl_old, repl_new)

        # version_sky
        version_sky = f"{self.app_window.lineEdit_sky_ver.text()}"
        repl_old = "[targetdb.sky]\nversion="
        repl_new = f"[targetdb.sky]\n{version_sky}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # version_fstar
        version_fstar = f"{self.app_window.lineEdit_fstar_ver.text()}"
        repl_old = "[targetdb.fluxstd]\nversion="
        repl_new = f"[targetdb.sfluxstdky]\n{version_fstar}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # queuedb
        template = f"{self.app_window.lineEdit_queuedb_path.text()}"
        repl_old = "[queuedb]\nfilepath="
        repl_new = f'[queuedb]\nfilepath="{template}"'
        config_ori = config_ori.replace(repl_old, repl_new)

        # SCHEMACRAWLERDIR
        SCHEMACRAWLERDIR = f"{self.app_window.lineEdit_schema_path.text()}"
        repl_old = "SCHEMACRAWLERDIR="
        repl_new = f'SCHEMACRAWLERDIR="{SCHEMACRAWLERDIR}"'
        config_ori = config_ori.replace(repl_old, repl_new)

        # pfs_instdata_dir
        pfs_instdata_dir = f"{self.app_window.lineEdit_instdata_path.text()}"
        repl_old = "pfs_instdata_dir="
        repl_new = f'pfs_instdata_dir="{pfs_instdata_dir}"'
        config_ori = config_ori.replace(repl_old, repl_new)

        # ope_template
        template = f"{self.app_window.lineEdit_ope_path.text()}"
        repl_old = "template="
        repl_new = f'template="{template}"'
        config_ori = config_ori.replace(repl_old, repl_new)

        # cobra_coach_dir
        cobra_coach_dir = f"{self.app_window.lineEdit_cobra_path.text()}"
        repl_old = "cobra_coach_dir="
        repl_new = f'cobra_coach_dir="{cobra_coach_dir}"'
        config_ori = config_ori.replace(repl_old, repl_new)

        # use_gurobi
        if self.app_window.checkBox_useGurobi.isChecked():
            use_gurobi = "true"
        else:
            use_gurobi = "false"
        repl_old = "use_gurobi="
        repl_new = f"use_gurobi={use_gurobi}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # two_stage
        if self.app_window.checkBox_2stage.isChecked():
            two_stage = "true"
        else:
            two_stage = "false"
        repl_old = "two_stage="
        repl_new = f"two_stage={two_stage}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # sky_uni
        if self.app_window.box_sky_uniform.isChecked():
            repl_old = "cobra_location_group_n=\nmin_sky_targets_per_location=\nlocation_group_penalty="
            repl_new = f"{self.app_window.lineEdit_group_n.text()}\n{self.app_window.lineEdit_nsky_min_group.text()}\n{self.app_window.lineEdit_group_penalty.text()}"
            config_ori = config_ori.replace(repl_old, repl_new)
        else:
            repl_old = "cobra_location_group_n=\nmin_sky_targets_per_location=\nlocation_group_penalty="
            repl_new = "cobra_location_group_n=None\nmin_sky_targets_per_location=None\nlocation_group_penalty=None"
            config_ori = config_ori.replace(repl_old, repl_new)

        # n_sky
        n_sky = f"{self.app_window.doubleSpinBox_sky_n.value()}"
        repl_old = "n_sky="
        repl_new = f"n_sky={n_sky}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # n_fluxstd
        n_fluxstd = f"{self.app_window.doubleSpinBox_fstar_n.value()}"
        repl_old = "n_fluxstd="
        repl_new = f"n_fluxstd={n_fluxstd}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # mag_fluxstd
        repl_old = "fluxstd_mag_max=\nfluxstd_mag_min="
        repl_new = f"fluxstd_mag_max={self.app_window.doubleSpinBox_fstar_max.value()}\nfluxstd_mag_min={self.app_window.doubleSpinBox_fstar_min.value()}"
        config_ori = config_ori.replace(repl_old, repl_new)

        repl_old = "fluxstd_min_prob_f_star=\nfluxstd_min_teff=\nfluxstd_max_teff="
        repl_new = f"{self.app_window.lineEdit_fstar_min_prob.text()}\n{self.app_window.lineEdit_fstar_teff_min.text()}\n{self.app_window.lineEdit_fstar_teff_max.text()}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # filler
        if self.app_window.box_readtarget_filler.isChecked():
            filler = "true"
            filler_mag_min = self.app_window.doubleSpinBox_filler_min.value()
            filler_mag_max = self.app_window.doubleSpinBox_filler_max.value()
        else:
            filler = "false"
            filler_mag_min = 999
            filler_mag_max = 999
        repl_old = "filler=\nfiller_mag_min=\nfiller_mag_max="
        repl_new = f"filler={filler}\nfiller_mag_min={filler_mag_min}\nfiller_mag_max={filler_mag_max}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # guider
        repl_old = "guidestar_mag_min=\nguidestar_mag_max="
        repl_new = f"guidestar_mag_min={self.app_window.doubleSpinBox_guide_min.value()}\nguidestar_mag_max={self.app_window.doubleSpinBox_guide_max.value()}"
        config_ori = config_ori.replace(repl_old, repl_new)

        # n_split_frame
        if self.app_window.checkBox_nsplit.isChecked():
            n_split_frame = self.app_window.doubleSpinBox_nsplit.value()
        else:
            n_split_frame = 1

        repl_old = "n_split_frame="
        repl_new = f"n_split_frame={n_split_frame}"
        config_ori = config_ori.replace(repl_old, repl_new)

        return config_ori

    def rungpd(self):
        import numpy as np
        from pfs_obsproc_planning import generatePfsDesign
        import time

        np.random.seed(1)

        time_start = time.time()

        workDir = self.app_window.lineEdit_workdir_path.text()
        config = "config.toml"

        n_pccs_l = self.app_window.doubleSpinBox_nppc_l.value()
        n_pccs_m = self.app_window.doubleSpinBox_nppc_m.value()

        run_proc = ""

        # initialize a GeneratePfsDesign instance
        time_start = time.time()
        gpd = generatePfsDesign.GeneratePfsDesign(config, workDir=workDir)

        # Run PPP
        if self.app_window.checkBox_runPPP.isChecked():
            gpd.runPPP(n_pccs_l, n_pccs_m, show_plots=False)
            run_proc += "PPP, "
        time_ppp = time.time() - time_start

        # Run qplan
        time_start = time.time()
        if self.app_window.checkBox_runQplan.isChecked():
            gpd.runQPlan()
            run_proc += "qPlan, "
        time_qplan = time.time() - time_start

        # Run SFA
        time_start = time.time()
        if self.app_window.checkBox_runSFA.isChecked():
            gpd.runSFA(clearOutput=True)
            run_proc += "SFA, "
        time_sfa = time.time() - time_start

        # Run validation
        if self.app_window.checkBox_runValidate.isChecked():
            gpd.runValidation()
            run_proc += "validation"

        logger.info(f"Finish! Design files have been generated ({run_proc}).")
        logger.info(
            f"runtime_ppp = {time_ppp:.2f}, runtime_qplan = {time_qplan:.2f}, runtime_sfa = {time_sfa:.2f}"
        )

    def run(self):
        self.app_window.radioButton_classic.clicked.connect(self.obsmode_enable)
        self.app_window.radioButton_queue.clicked.connect(self.obsmode_enable)
        self.app_window.pushButton_design_local_classic.clicked.connect(
            self.getfolder_design
        )
        self.app_window.pushButton_tgt_local.clicked.connect(self.getfile_tgt)
        self.app_window.pushButton_ppc_local_classic.clicked.connect(self.getfile_ppc)
        self.app_window.pushButton_obstime_add.clicked.connect(self.obstime_add)
        self.app_window.pushButton_obstime_remove.clicked.connect(self.obstime_remove)
        self.app_window.pushButton_ope_template.clicked.connect(self.getfile_ope)
        self.app_window.pushButton_config_template.clicked.connect(self.getfile_config)
        self.app_window.pushButton_queuedb.clicked.connect(self.getfile_queuedb)
        self.app_window.pushButton_workdir.clicked.connect(self.getfolder_workdir)
        self.app_window.pushButton_pfs_instdata.clicked.connect(self.getfolder_instdata)
        self.app_window.pushButton_cobra_coach.clicked.connect(self.getfolder_cobra)
        self.app_window.pushButton_schema.clicked.connect(self.getfolder_schema)
        self.app_window.pushButton_viewConfig.clicked.connect(self.load_template_config)
        self.app_window.pushButton_tgt_db.clicked.connect(self.getPslID_tgtDB)
        self.app_window.pushButton_runCode.clicked.connect(self.rungpd)

        self.app_window.show()
        self.app_window.exec()
