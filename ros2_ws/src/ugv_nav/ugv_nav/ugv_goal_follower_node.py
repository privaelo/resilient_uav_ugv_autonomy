import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ipp_core.geometry import OBSTACLES
from nav_msgs.msg import Odometry
from std_msgs.msg import String

# Obstacle geometry from urban_obstacles.sdf: (cx, cy, half_width, half_height).
# Used only by the potential-field repulsion layer — sim-specific, not sensor-based.
# Obstacle geometry lives in ipp_core.geometry so the APF controller and the sensing
# model cannot drift apart. See that module when the world changes.
_OBSTACLES = OBSTACLES

# Expand each obstacle by this margin so the robot body clears the surface.
_ROBOT_RADIUS = 0.55


def _yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _normalize_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _repulsive_force(px, py, cx, cy, hw, hh, influence_radius, k_rep):
    """
    Repulsive force from a rectangular obstacle (AABB) inflated by _ROBOT_RADIUS.
    Returns (fx, fy) pointing away from the obstacle surface.
    """
    ihw = hw + _ROBOT_RADIUS
    ihh = hh + _ROBOT_RADIUS

    # Nearest point on the inflated rectangle
    nx = max(cx - ihw, min(px, cx + ihw))
    ny = max(cy - ihh, min(py, cy + ihh))

    dx = px - nx
    dy = py - ny
    d = math.hypot(dx, dy)

    if d >= influence_radius or d < 1e-6:
        return 0.0, 0.0

    # Standard APF repulsion: k * (1/d - 1/d0) * (1/d²)
    mag = k_rep * (1.0 / d - 1.0 / influence_radius) * (1.0 / d ** 2)
    return mag * dx / d, mag * dy / d


class UGVGoalFollowerNode(Node):
    def __init__(self):
        super().__init__('ugv_goal_follower_node')

        self.declare_parameter('ugv_name', 'ugv_1')
        self.declare_parameter('linear_speed', 0.4)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('arrival_radius', 1.0)
        self.declare_parameter('heading_tolerance', 0.12)
        self.declare_parameter('k_att', 1.0)
        self.declare_parameter('k_rep', 4.0)
        self.declare_parameter('influence_radius', 2.5)

        name = self.get_parameter('ugv_name').value
        self._ugv_name = name
        self._linear_speed = self.get_parameter('linear_speed').value
        self._angular_speed = self.get_parameter('angular_speed').value
        self._arrival_radius = self.get_parameter('arrival_radius').value
        self._heading_tolerance = self.get_parameter('heading_tolerance').value
        self._k_att = self.get_parameter('k_att').value
        self._k_rep = self.get_parameter('k_rep').value
        self._influence_radius = self.get_parameter('influence_radius').value

        self._pose = None   # (x, y, yaw) world frame
        self._goal = None   # (x, y) world frame
        self._arrived = False

        self.create_subscription(Odometry, f'/{name}/odom', self._odom_cb, 10)
        self.create_subscription(String, '/allocation/assignments', self._assignment_cb, 10)

        self._cmd_pub = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)

        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(f'{name} goal follower ready')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose = (p.x, p.y, _yaw_from_quaternion(q))

    def _assignment_cb(self, msg: String):
        data = json.loads(msg.data)
        for assignment in data.get('assignments', []):
            if assignment['ugv'] == self._ugv_name:
                new_goal = (assignment['target_x'], assignment['target_y'])
                if new_goal != self._goal:
                    self._goal = new_goal
                    self._arrived = False
                    self.get_logger().info(
                        f'{self._ugv_name} assigned → '
                        f'({assignment["target_x"]:.1f}, {assignment["target_y"]:.1f}) '
                        f'[{assignment["target_id"]}], cost={assignment["cost"]}m'
                    )
                return

    def _potential_field_heading(self, x, y, gx, gy):
        """
        Compute desired heading from attractive + repulsive potential field.
        Returns (desired_heading, resultant_magnitude).
        """
        # Attractive force: linear pull toward goal
        dx_att = gx - x
        dy_att = gy - y
        dist_to_goal = math.hypot(dx_att, dy_att)
        if dist_to_goal > 1e-6:
            fx_att = self._k_att * dx_att / dist_to_goal
            fy_att = self._k_att * dy_att / dist_to_goal
        else:
            fx_att, fy_att = 0.0, 0.0

        # Repulsive forces from all obstacles
        fx_rep, fy_rep = 0.0, 0.0
        for cx, cy, hw, hh in _OBSTACLES:
            frx, fry = _repulsive_force(
                x, y, cx, cy, hw, hh, self._influence_radius, self._k_rep
            )
            fx_rep += frx
            fy_rep += fry

        fx = fx_att + fx_rep
        fy = fy_att + fy_rep
        return math.atan2(fy, fx), math.hypot(fx, fy)

    def _control_loop(self):
        cmd = Twist()

        if self._pose is None or self._goal is None or self._arrived:
            self._cmd_pub.publish(cmd)
            return

        x, y, yaw = self._pose
        gx, gy = self._goal

        dist = math.hypot(gx - x, gy - y)

        if dist < self._arrival_radius:
            self._arrived = True
            self.get_logger().info(
                f'{self._ugv_name} arrived at ({gx:.1f}, {gy:.1f})'
            )
            self._cmd_pub.publish(cmd)
            return

        desired_heading, _ = self._potential_field_heading(x, y, gx, gy)
        heading_error = _normalize_angle(desired_heading - yaw)

        if abs(heading_error) > self._heading_tolerance:
            cmd.angular.z = math.copysign(self._angular_speed, heading_error)
        else:
            cmd.linear.x = self._linear_speed
            cmd.angular.z = 0.4 * heading_error

        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = UGVGoalFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
