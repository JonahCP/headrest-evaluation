from tkinter import Tk, Canvas
import time

# Define frequencies, duration, and other experimental constants
FREQ = [7.5, 8, 10, 12, 14]  # in Hz
STIMULUS_DURATION = 4   # in seconds
REST_DURATION = 2       # in seconds

# Define circle constants
COLOR = 'yellow'
RADIUS = 150

# Define functions
"""
Flickers a COLOR circle of radius RADIUS at a defined frequency

@param freq: frequency (in Hz) to flicker circle at
""" 
def flicker(freq):
    print('Beginning %d Hz: ' % (freq) + time.strftime('%Y-%m-%d %H:%M:%S'))
    duration = STIMULUS_DURATION
    period = 1 / freq
    total_time = 0
    cycles = 0
    while duration > 0:
        start = time.time()
        present = canvas.find_withtag('circle')
        if not present:
            canvas.create_oval(circle_x - RADIUS, circle_y - RADIUS,
                               circle_x + RADIUS, circle_y + RADIUS,
                               fill = COLOR, tags = 'circle')
        else:
            canvas.delete('circle')
        root.update()

        # Hold the frame for T seconds
        time.sleep(period - (time.time() - start))
        duration -= period
        cycles += 1
        total_time += (time.time() - start)
    if canvas.find_withtag('circle'):
        canvas.delete('circle')
        root.update()
    print('Average freq: %.3f' % (cycles / total_time))
    

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
circle_x = canvas.winfo_width() // 2
circle_y = canvas.winfo_height() // 2

# Begin flickering
print('Starting program timestamp:', time.strftime('%Y-%m-%d %H:%M:%S'))  # Output current timestamp
for freq in FREQ:
    flicker(freq)
    time.sleep(REST_DURATION)   # Rest period
