# This represents a 2-D grid map, in which each cell has an associated cost

std_msgs/Header header

# The map resolution [m/cell]
float32 resolution

# Number of cells in the horizontal direction
uint32 size_x

# Number of cells in the vertical direction
uint32 size_y

# The origin of the costmap [m, m, rad].
# This is the real-world pose of the cell (0,0) in the map.
geometry_msgs/Pose origin

# The cost data, in row-major order, starting with (0,0).
uint8[] data