from tkinter import Tk, Canvas
import time
import random
import pyautogui
import os

# Define duration and other experimental constants
DURATION = 0.5      # in seconds (for both stimulus and rest periods)

# Define shape constants
COLOR = 'yellow'
RADIUS = 150

# Define functions
def draw_circle():
    canvas.create_oval(pos_x - RADIUS, pos_y - RADIUS,
                       pos_x + RADIUS, pos_y + RADIUS, 
                       fill = COLOR, tags = 'shape')
    
def draw_square():
    canvas.create_rectangle(pos_x - RADIUS, pos_y - RADIUS, 
                            pos_x + RADIUS, pos_y + RADIUS, 
                            fill = COLOR, tags = 'shape')

def flicker():
    rand = random.randint(1, 9)
    i = 0
    img = ""
    while True:
        present = canvas.find_withtag('shape')
        if not present:
            if i == rand:
                draw_circle()
                img = 'circle.png'
                i = 0
                rand = random.randint(1, 9)
            else:
                draw_square()
                i += 1
                img = 'square.png'

        else:
            canvas.delete('shape')
            img = 'blank.png'

        root.update()
        sc = pyautogui.screenshot()
        sc.save(os.getcwd() + "/../images/" + img)

        # Hold the frame
        time.sleep(DURATION)
        

# Setup Tkinter window
root = Tk()
root.config(cursor='none')
root.title('SSVEP')
root.attributes('-fullscreen', True)

# Setup screen for drawing
canvas_width = root.winfo_screenwidth()
canvas_height = root.winfo_screenheight()
canvas = Canvas(root, width = canvas_width, height = canvas_height, bg='black')
canvas.pack()
root.update()

# Find center of the screen
pos_x = canvas.winfo_width() // 2
pos_y = canvas.winfo_height() // 2

# Begin flickering
print('Starting program timestamp:', time.strftime('%Y-%m-%d %H:%M:%S'))  # Output current timestamp
flicker()
