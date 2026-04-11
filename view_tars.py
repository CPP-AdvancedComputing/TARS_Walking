import pyglet
print("started")
import yourdfpy

print("Hello world does ts work")
robot = yourdfpy.URDF.load('.urdf')
robot.show()

print('loaded.')

robot.show()

print('done')