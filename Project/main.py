from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *
from PyQt5 import QtWidgets

import sys
import os

from utils.config import Config
from utils.Clawler import imageCrwaling
from shopping_lense import test
from train_model import train

form_class = uic.loadUiType("Shopping Lense UI.ui")[0]


# class for redirecting
class StdoutRedirect(QObject):
    printOccur = pyqtSignal(str, str, name="print")

    def __init__(self, *param):
        QObject.__init__(self, None)
        self.daemon = True
        self.sysstdout = sys.stdout.write
        self.sysstderr = sys.stderr.write

    def stop(self):
        sys.stdout.write = self.sysstdout
        sys.stderr.write = self.sysstderr

    def start(self):
        sys.stdout.write = self.write
        sys.stderr.write = lambda msg: self.write(msg, color="red")

    def write(self, s, color="black"):
        sys.stdout.flush()
        self.printOccur.emit(s, color)


class WindowClass(QMainWindow, form_class):
    def __init__(self, config):
        super().__init__()
        self.setupUi(self)
        self.config = config

        # set title and icon
        self.setWindowTitle('Shopping Lense')
        self.setWindowIcon(QIcon('shopping-cart.png'))

        # set printer unvisible
        self.printer.setVisible(False)
        self.system_out.setVisible(False)

        # ratio check
        self.collect_image.clicked.connect(self.rationCheck)
        self.train_model.clicked.connect(self.rationCheck)
        self.shopping_lense.clicked.connect(self.rationCheck)

        # when button clicked
        self.get_data_path.clicked.connect(self.getDataPathFunction)
        self.execute_button.clicked.connect(self.executeButtonFunction)

        # print Redirecting
        self._stdout = StdoutRedirect()
        self._stdout.start()
        self._stdout.printOccur.connect(lambda x: self._append_text(x))

    # print Redirecting
    def _append_text(self, msg):
        self.printer.moveCursor(QtGui.QTextCursor.End)
        self.printer.insertPlainText(msg)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    # if user click button change ui and show what user need
    def rationCheck(self):
        self.data_path.setText("")

        if self.collect_image.isChecked():
            config["mode"] = "collect_image"
            self.show_path_mode.setText("Set path to save images")
            self.keyword_label.setVisible(True)
            self.keyword.setVisible(True)
            self.epoch_label.setVisible(False)
            self.epoch.setVisible(False)
            self.learning_rate_label.setVisible(False)
            self.learning_rate.setVisible(False)
            self.weight_decay_label.setVisible(False)
            self.weight_decay.setVisible(False)
            self.model_name_label.setVisible(False)
            self.model_name.setVisible(False)
            self.system_out.setVisible(True)
            self.printer.setVisible(True)
            self.probability.setVisible(False)
            self.top1_class.setVisible(False)
            self.top2_class.setVisible(False)
            self.top3_class.setVisible(False)
            self.top1_p.setVisible(False)
            self.top2_p.setVisible(False)
            self.top3_p.setVisible(False)
            self.test_image.setVisible(False)
            self.class_activation_map.setVisible(False)
            self.original.setVisible(False)
            self.CAM.setVisible(False)

        elif self.train_model.isChecked():
            config["mode"] = "train_model"
            self.show_path_mode.setText("Set train images path")
            self.keyword_label.setVisible(False)
            self.keyword.setVisible(False)
            self.epoch_label.setVisible(True)
            self.epoch.setVisible(True)
            self.learning_rate_label.setVisible(True)
            self.learning_rate.setVisible(True)
            self.weight_decay_label.setVisible(True)
            self.weight_decay.setVisible(True)
            self.model_name_label.setVisible(False)
            self.model_name.setVisible(False)
            self.system_out.setVisible(True)
            self.printer.setVisible(True)
            self.probability.setVisible(False)
            self.top1_class.setVisible(False)
            self.top2_class.setVisible(False)
            self.top3_class.setVisible(False)
            self.top1_p.setVisible(False)
            self.top2_p.setVisible(False)
            self.top3_p.setVisible(False)
            self.test_image.setVisible(False)
            self.class_activation_map.setVisible(False)
            self.original.setVisible(False)
            self.CAM.setVisible(False)

        elif self.shopping_lense.isChecked():
            config["mode"] = "shopping_lense"
            self.show_path_mode.setText("Set a test image path")
            self.keyword_label.setVisible(False)
            self.keyword.setVisible(False)
            self.epoch_label.setVisible(False)
            self.epoch.setVisible(False)
            self.learning_rate_label.setVisible(False)
            self.learning_rate.setVisible(False)
            self.weight_decay_label.setVisible(False)
            self.weight_decay.setVisible(False)
            self.model_name_label.setVisible(True)
            self.model_name.setVisible(True)
            self.system_out.setVisible(False)
            self.printer.setVisible(False)
            self.probability.setVisible(True)
            self.top1_class.setVisible(True)
            self.top2_class.setVisible(True)
            self.top3_class.setVisible(True)
            self.top1_p.setVisible(True)
            self.top2_p.setVisible(True)
            self.top3_p.setVisible(True)
            self.test_image.setVisible(True)
            self.class_activation_map.setVisible(True)
            self.original.setVisible(True)
            self.CAM.setVisible(True)

    # set data path to an image or folder
    def getDataPathFunction(self):
        if config["mode"] == "shopping_lense":
            fname = QFileDialog.getOpenFileName(self)[0]
        else:
            fname = QFileDialog.getExistingDirectory(self)
        self.data_path.setPlainText(fname)

    # when execute button clicked
    def executeButtonFunction(self):
        if "" == self.data_path.toPlainText():
            print("Set Data Path")
            return

        if config["mode"] == "collect_image":
            config["image_save_path"] = self.data_path.toPlainText()
            config["keyword"] = self.keyword.text()
            imageCrwaling(config["keyword"], config["image_save_path"])

        elif config["mode"] == "train_model":
            config["image_folder"] = self.data_path.toPlainText()
            config["epoch"] = int(self.epoch.text())
            config["learning_rate"] = float(self.learning_rate.text())
            config["weight_decay"] = float(self.weight_decay.text())
            train()

        elif config["mode"] == "shopping_lense":
            config["test_image"] = self.data_path.toPlainText()
            test()
            self.showResult()

    # change value and show images
    def showResult(self):
        test_image = QPixmap("result\\test.jpg")
        self.original.setPixmap(QPixmap(test_image))
        CAM_image = QPixmap("result\\CAM.jpg")
        self.CAM.setPixmap(QPixmap(CAM_image))
        self.top1_class.setText(config["top3"][0][0])
        self.top1_p.setText(config["top3"][0][1])
        self.top2_class.setText(config["top3"][1][0])
        self.top2_p.setText(config["top3"][1][1])
        self.top3_class.setText(config["top3"][2][0])
        self.top3_p.setText(config["top3"][2][1])


if __name__ == "__main__":
    config = Config().params
    app = QApplication(sys.argv)
    myWindow = WindowClass(config)
    myWindow.show()
    app.exec_()
