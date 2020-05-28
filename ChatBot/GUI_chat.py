from tkinter import *

windows = Tk()
windows.wm_title("ChatBot GUI")
# windows.configure(background='red')
windows.geometry("500x350")

sb = Scrollbar(windows)
sb.pack(side=RIGHT, fill=Y)
#canvas = Canvas(windows, bg='pink', width=500, yscrollcommand=sb.set)
#canvas.pack()
messages = Text(windows, bg='pink', width=500, yscrollcommand=sb.set)
messages.pack()
label = Label(windows, text="Enter message:").place(x=0, y=320)
e1 = Entry(windows)
e1.place(x=100, y=320, width=300)

def enter_pressed(event):
    user_input = "User: " + e1.get()
    messages.insert(INSERT, '%s\n' % user_input)
    e1.delete(0, END)
    return "break"

e1.bind("<Return>", enter_pressed)
mainloop()