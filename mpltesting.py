import matplotlib
import numpy as np
from matplotlib import pyplot as plt
'''
fig, ax = plt.subplots()
ax.plot(np.random.rand(10))

def onclick(event):
    print("It works")
    print(event.key)
    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #      ('double' if event.dblclick else 'single', event.button,
    #       event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('key_press_event', onclick)
plt.show()
'''
'''
x = 10
print(x)
def globallyChange():
    global x            # Access the global var
    x = "didn't work"
globallyChange()        # Call the function
print(x)
'''
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.figure().canvas.set_window_title("BICK")
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.savefig("test.png")



#fig.canvas.set_window_title('Custom')

plt.show()
