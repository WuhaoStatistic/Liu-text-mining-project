import torch
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.onnx
import functools
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import sys
from PyQt5 import QtWidgets, QtCore

app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QWidget()
widget.resize(360, 360)
widget.setWindowTitle("hello, pyqt5")
widget.show()
sys.exit(app.exec_())

