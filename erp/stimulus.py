from tkinter import Tk, Canvas
import time

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
    while True:
        present = canvas.find_withtag('shape')
        if not present:
            draw_circle()
        else:
            canvas.delete('shape')
        root.update()

        # Hold the frame
        time.sleep(DURATION)
        

# Setup Tkinter window
root = Tk()
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
