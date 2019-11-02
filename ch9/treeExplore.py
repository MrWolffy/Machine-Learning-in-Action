# 这里的代码已经跑不通了
import numpy as np
import tkinter as tk
import regTrees
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    reDraw.f.clf()        # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,\
                                   regTrees.modelErr, (tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, \
                                       regTrees.modelTreeEval)
    else:
        myTree=regTrees.createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5) #use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #use plot for yHat
    # reDraw.canvas.()


def getInputs():
    try: tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, tk.END)
        tolNentry.insert(0,'10')
    try: tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, tk.END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS


def drawNewTree():
    tolN,tolS = getInputs()#get values from Entry boxes
    reDraw(tolS,tolN)


if __name__ == '__main__':
    root = tk.Tk()
    reDraw.f = Figure(figsize=(5, 4), dpi=100)  # create canvas
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
    tk.Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = tk.Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    tk.Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = tk.Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, 1.0)
    tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
    chkBtnVar = tk.IntVar()
    chkBtn = tk.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)
    reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))
    reDraw.testDat = np.arange(np.min(reDraw.rawDat[:, 0]), np.max(reDraw.rawDat[:, 0]), .01)
    reDraw(1.0, 10)
    root.mainloop()
