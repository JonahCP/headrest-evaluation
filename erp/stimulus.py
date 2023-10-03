import turtle as t
import time

def circle_stimulus():
    pen.fillcolor("yellow")
    pen.begin_fill()
    pen.circle(100)
    pen.end_fill()

def no_stimulus():
    pen.fillcolor("black")
    pen.begin_fill()
    pen.circle(110)
    pen.end_fill()

# Set up the turtle screen with black background
window = t.Screen()
window.bgcolor("black")
window.title("ERP Stimulus")

t.tracer(1, 100)

# Create a new turtle and set its speed to the fastest possible
pen = t.Turtle()
pen.speed(0)
pen.hideturtle()

# Draw circle with radius of 100 pixels
for i in range(20):
    circle_stimulus()
    start = time.time()
    no_stimulus()
    print("%.3f s" % (time.time() - start))

# Keep window open until manually closed
# t.done()
