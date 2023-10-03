import turtle as t

# Set up the turtle screen with black background
window = t.Screen()
window.bgcolor("black")
window.title("ERP Stimulus")

# Create a new turtle and set its speed to the fastest possible
pen = t.Turtle()
pen.speed(0)

# Set fill color to yellow
pen.fillcolor("yellow")
pen.begin_fill()

# Draw circle with radius of 100 pixels
pen.circle(100)

# End the fill and stop drawing
pen.end_fill()
pen.hideturtle()

# Keep window open until manually closed
t.done()
