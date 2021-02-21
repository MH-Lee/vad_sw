import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QCoreApplication, QBasicTimer, QProcess
from package.vad_execution import final_excecution


class VadSW(QWidget):
# class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.person_mode = 3
        self.model_name = "LR"
        self.output_dir = None
        self.filename = []

    def setupUI(self):
        ### File load button
        self.pushButton = QPushButton("File Open")
        self.pushButton.clicked.connect(self.pushButtonClicked)
        ### Qlabel for model
        self.label_model = QLabel()
        self.label_model.setAlignment(Qt.AlignCenter)
        ### Qlabel for # of filename
        self.NofFileName = QLabel()
        self.NofFileName.setAlignment(Qt.AlignCenter)
        ### Output dir
        self.outputButton = QPushButton("Output dir")
        self.outputButton.clicked.connect(self.outputButtonClicked)
        ### Qlabel for set output dir
        self.outputDirName = QLabel()
        self.outputDirName.setAlignment(Qt.AlignCenter)
        ### Execution button
        self.execButton = QPushButton("Execution")
        self.execButton.clicked.connect(self.exeClicked)
        ### Completed msg
        self.completed_msg = QLabel("", self)
        self.completed_msg.setStyleSheet("color:red;")
        self.completed_msg.setAlignment(Qt.AlignCenter)
        ### Quit Button
        self.QuitButton = QPushButton("Quit")
        self.QuitButton.clicked.connect(QCoreApplication.instance().quit)

        layout = QGridLayout()
        layout.addWidget(self.pushButton,0,0)
        layout.addWidget(self.displayFnameList(), 1, 0)
        layout.addWidget(self.NofFileName, 2, 0)
        layout.addWidget(self.ModelSelectGroup(), 3, 0)
        layout.addWidget(self.label_model, 4, 0)
        layout.addWidget(self.turnTakingGroup(), 5, 0)
        layout.addWidget(self.outputButton, 6, 0)
        layout.addWidget(self.outputDirName, 7, 0)
        layout.addWidget(self.execButton, 8, 0)
        layout.addWidget(self.completed_msg, 9, 0)
        layout.addWidget(self.QuitButton, 10, 0)

        self.setLayout(layout)
        self.setGeometry(900, 200, 400, 300)
        self.setWindowTitle("VAD sw v0.5")
        self.show()

    def pushButtonClicked(self):
        fname = QFileDialog.getOpenFileNames(self)
        if len(fname[0]) > 5:
            print_fname = fname[0][:5]
            print_fname.append("...")
        else:
            print_fname = fname[0]
        self.dir_label.setText("\n".join(print_fname))
        self.filename = fname[0]
        self.NofFileName.setText("{} data selected".format(len(fname[0])))
        self.completed_msg.clear()

    ### set output directory
    def outputButtonClicked(self):
        dir_name = QFileDialog.getExistingDirectory(self)
        self.outputDirName.setText("save path : {}".format(dir_name))
        self.output_dir = dir_name
        # print(fname)

    def displayFnameList(self):
        groupbox = QGroupBox('Selected data')
        self.dir_label = QLabel()
        vbox = QVBoxLayout()
        vbox.addWidget(self.dir_label)
        groupbox.setLayout(vbox)
        return groupbox

    def ModelSelectGroup(self):
        groupbox = QGroupBox('Select predict model')
        self.radio1 = QRadioButton('LogisticRegression')
        self.radio1.clicked.connect(self.radioButtonClicked)
        self.radio2 = QRadioButton('Muli-LayerPerceptron')
        self.radio2.clicked.connect(self.radioButtonClicked)
        self.radio3 = QRadioButton('GradientBoosting')
        self.radio3.clicked.connect(self.radioButtonClicked)
        self.radio4 = QRadioButton('LightGBM')
        self.radio4.clicked.connect(self.radioButtonClicked)
        self.radio5 = QRadioButton('VotingClassifier')
        self.radio5.clicked.connect(self.radioButtonClicked)
        self.radio6 = QRadioButton('StackingClassifier')
        self.radio6.clicked.connect(self.radioButtonClicked)
        self.radio1.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self.radio1)
        vbox.addWidget(self.radio2)
        vbox.addWidget(self.radio3)
        vbox.addWidget(self.radio4)
        vbox.addWidget(self.radio5)
        vbox.addWidget(self.radio6)
        groupbox.setLayout(vbox)
        return groupbox

    def radioButtonClicked(self):
        model_name = ""
        if self.radio1.isChecked():
            model_name = "LR"
        elif self.radio2.isChecked():
            model_name = "MLP"
        elif self.radio3.isChecked():
            model_name = "GBC"
        elif self.radio4.isChecked():
            model_name = "lightGBM"
        elif self.radio5.isChecked():
            model_name = "Voting"
        else:
            model_name = "Stacking"
        self.label_model.setText("{} is selected".format(model_name))
        self.model_name = model_name
        # print("model_name : ", self.model_name)

    def turnTakingGroup(self):
        groupbox = QGroupBox('Turn taking parameter')
        ### input turn taking term
        self.turn_taking_term = QLabel(self)
        self.turn_taking_term.setText('turn_taking term ')
        self.tt_term_input = QLineEdit("",self)
        ### input silence term
        self.silence_term = QLabel(self)
        self.silence_term.setText('silence term ')
        self.s_term_input = QLineEdit("", self)
        ### input miroring term
        self.mirroring_term = QLabel(self)
        self.mirroring_term.setText('mirroring term ')
        self.mirroring_input= QLineEdit("", self)
        ### turn taking group
        ttt_layout = QGridLayout()
        ttt_layout.addWidget(self.turn_taking_term, 0, 0)
        ttt_layout.addWidget(self.tt_term_input, 0, 1)
        ttt_layout.addWidget(self.silence_term, 1, 0)
        ttt_layout.addWidget(self.s_term_input, 1, 1)
        ttt_layout.addWidget(self.mirroring_term, 2, 0)
        ttt_layout.addWidget(self.mirroring_input, 2, 1)
        groupbox.setLayout(ttt_layout)
        return groupbox

    def exeClicked(self):
        print("model Execution")
        if self.tt_term_input.text() == "":
            self.tt_term = 300
        else:
            self.tt_term = self.tt_term_input.text()
        if self.s_term_input.text() == "":
            self.s_term = 300
        else:
            self.s_term = self.s_term_input.text()
        if self.mirroring_input.text() == "":
            self.m_term = 300
        else:
            self.m_term = self.mirroring_input.text()
        if self.output_dir == None:
            cur_dir = os.getcwd()
            if not os.path.exists(cur_dir+'/results'):
                os.makedirs(cur_dir+'/results')
            self.output_dir = cur_dir+'/results'
        print("tt_term : ", self.tt_term)
        print("s_term : ", self.s_term)
        print("m_term : ", self.m_term)
        print("model_name : ", self.model_name)
        print("file_list : ", self.filename)
        print("the_number_of_filelist : ", len(self.filename))
        print("person number : ", self.person_mode)
        print("output_dir : ", self.output_dir)
        for idx, fname in enumerate(self.filename):
            final_excecution(fname,\
                             self.output_dir,\
                             self.model_name,\
                             int(self.tt_term),\
                             int(self.s_term),\
                             int(self.m_term))
        self.completed_msg.setText("Completed! Done")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    vadsw = VadSW()
    vadsw.show()
    app.exec_()
