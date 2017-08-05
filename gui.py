from Tkinter import *
import numpy as np
import math



def paint(event):
    python_green = "#476042"
    x1, y1 = (event.x - 1), ( event.y - 1)
    x2, y2 = (event.x + 1), ( event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill=python_green)
    mapArray[math.floor(event.y/10)][math.floor(event.x/10)] = 0


def setupGui():
    canvas_width = 280
    canvas_height = 280

    mapArray = np.ones((28,28), dtype=np.int)
    master = Tk()
    master.title("OSR")
    w = Canvas(master,
               width=canvas_width,
               height=canvas_height)
    w.pack(expand = YES, fill = BOTH)
    w.bind( "<B1-Motion>", paint )

    # mainloop()
    return np.ndarray.flatten(mapArray)

setupGui()
mapFlat = mainloop()
print mapFlat