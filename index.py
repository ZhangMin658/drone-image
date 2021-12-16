import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QHBoxLayout, QWidget, QListView, QFileDialog, QLabel
from PyQt5.QtGui import QIcon
import run


class menubarDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle('Dron image processing')
        self.setGeometry(350, 350, 350, 350)
        self.show()
 
    def initUI(self):
        self.statusBar() # Create a status bar
        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)
        self.fileMenu = self.menubar.addMenu('Open File')
        self.listView = QListView()
        self.slm = QStringListModel()
        self.widget = QWidget()
        self.widget.setStyleSheet("QWidget{border: 1px solid #00ff00;}")#Set the style of the widget, here is your favorite green frame
        self.widget.setContentsMargins(0, 0, 0, 0)
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.listView)
        self.widget.setLayout(self.hbox)
        self.setCentralWidget(self.widget)
        # file menu function
        fileAction = QAction(QIcon('file.png'),'Open file', self) # QIcon is the setting icon, just find a .png picture and put it in the project directory and name it file.png
        fileAction.setShortcut('Ctrl+F') # set shortcut keys
        fileAction.setStatusTip('Open file') # The prompt message displayed when the mouse is placed on the fileAction (lower left corner)
        fileAction.triggered.connect(self.openFile) # The action to be triggered after clicking the fileAction, here is linked to the openFile function
        # Add this Action to fileMenu
        self.fileMenu.addAction(fileAction)

    # '''Open the file and return a list of file paths'''
    def openFile(self):
        #'''Return to file path list self.files and filetype filetype'''
        self.files, filetype = QFileDialog.getOpenFileNames(self,"Multiple file selection","./","All Files (*);;Text Files (*.txt)")
        run.processing_file(self.files[0])

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = menubarDemo()
    sys.exit(app.exec_())