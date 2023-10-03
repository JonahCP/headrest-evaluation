import tkinter as tk
import time

class SsvepStimuli:
    def __init__(self, root):
        self.root = root
        self.root.title("SSVEP")
        self.root.configure(bg="black")  # Set background color to black

        # Set screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set window size and position
        window_width = screen_width
        window_height = screen_height
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        # Configure the window
        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        root.attributes('-fullscreen', True)  # Set fullscreen
        self.canvas = tk.Canvas(root, width=window_width, height=window_height, bg='black')
        self.canvas.pack()

        # Set initial frequency and duration
        self.colors = ['yellow', 'black', 'yellow', 'black']
        self.frequencies = [8, 10, 12, 14] # frequencies
        self.duration = 8000  # milliseconds (8 seconds)

        # Start flickering with the initial frequency
        self.colors_index = 0
        self.frequency_index = 0
        self.start_time = time.time() * 1000  # Current time in milliseconds
        self.update_circle()

    def update_circle(self):
        current_time = time.time() * 1000  # Current time in milliseconds
        elapsed_time = current_time - self.start_time

        if elapsed_time > self.duration:
            # Switch frequency after the duration
            self.frequency_index = (self.frequency_index + 1) % len(self.frequencies)
            self.colors_index = (self.colors_index + 1) % len(self.colors)
            print(str(self.frequencies[self.frequency_index]) + "Hz Timestamp:", time.strftime('%Y-%m-%d %H:%M:%S'))  # Output current timestamp
            self.start_time = current_time  # Reset start time


        frequency = self.frequencies[self.frequency_index]
        visibility = not self.canvas.find_withtag("circle")  # Toggle visibility for 10 Hz circle
        if visibility:
            # Calculate circle position for centering
            circle_x = self.canvas.winfo_width() // 2
            circle_y = self.canvas.winfo_height() // 2
            self.canvas.create_oval(circle_x - 100, circle_y - 100, circle_x + 100, circle_y + 100, fill=self.colors[self.colors_index],
                                    tags="circle")
        else:
            self.canvas.delete("circle")

        # Call update_circle method again after 1000 / frequency milliseconds
        self.root.after(int(1000 / frequency), self.update_circle)


# Initialize the GUI
root = tk.Tk()
app = SsvepStimuli(root)
print("Starting program Timestamp:", time.strftime('%Y-%m-%d %H:%M:%S'))  # Output current timestamp
root.mainloop()
