#!/usr/bin/env python3
import math
from abc import ABC, abstractmethod
from copy import copy
import cv2
import numpy as np
import rclpy
from cv2.aruco import ArucoDetector
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist
from mavros.base import SENSOR_QOS, STATE_QOS
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

class Controller(ABC):
    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def orientation(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def image_viz(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def goal_position(self) -> np.ndarray:
        pass

    @goal_position.setter
    @abstractmethod
    def goal_position(self, value: np.ndarray) -> None:
        pass

    @property
    @abstractmethod
    def goal_yaw(self) -> float:
        pass

    @goal_yaw.setter
    @abstractmethod
    def goal_yaw(self, value: float) -> None:
        pass

    @property
    def goal_distance(self) -> float:
        return np.linalg.norm(self.goal_position - self.position)

class Step(ABC):
    def init(self, controller: Controller) -> None:
        pass

    @abstractmethod
    def update(self, controller: Controller) -> bool:
        '''Returns: True if the step was completed, False otherwise.'''
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

class DroneController(Controller):
    def __init__(self, node: Node, steps: list[Step], control_frequency: float = 100.0):
        self.node = node
        self.steps = steps
        self.step: Step | None = None
        self._pose: Pose | None = None
        self._goal_pose: Pose | None = None
        self._image: np.ndarray | None = None
        self._image_viz: np.ndarray | None = None

        self.node.get_logger().info("Initializing Drone Controller...")
        mavros_ns = '/mavros'
        self.pose_publisher = node.create_publisher(PoseStamped, mavros_ns + '/setpoint_position/local', qos_profile=SENSOR_QOS)
        self.velocity_publisher = node.create_publisher(Twist, mavros_ns + '/setpoint_velocity/cmd_vel_unstamped', qos_profile=SENSOR_QOS)
        self.pose_subscription = node.create_subscription(PoseStamped, mavros_ns + '/local_position/pose', self.pose_subscription_callback, qos_profile=SENSOR_QOS)
        self.state_subscription = node.create_subscription(State, mavros_ns + '/state', self.state_subscription_callback, qos_profile=STATE_QOS)
        self.set_mode_client = self.create_client_and_wait_for_service(SetMode, mavros_ns + '/set_mode')
        self.arming_client = self.create_client_and_wait_for_service(CommandBool, mavros_ns + '/cmd/arming')

        self.bridge = CvBridge()
        camera_topic = '/camera'
        camera_history_depth = 10
        self.camera_subscription = node.create_subscription(Image, camera_topic, self.camera_subscription_callback, camera_history_depth)

        timer_period = 1.0 / control_frequency
        self.timer = self.node.create_timer(timer_period, self.timer_callback)

    def camera_subscription_callback(self, msg):
        try:
            self._image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.node.get_logger().error(f"CvBridge Error: {e}")
            return

    def pose_subscription_callback(self, msg: PoseStamped):
        self._pose = msg.pose
        if self._goal_pose is None:
            self._goal_pose = copy(self._pose)

    def state_subscription_callback(self, msg):
        if msg.mode != State.MODE_PX4_OFFBOARD:
            self.node.get_logger().info("Switching to OFFBOARD mode...")
            self.set_mode_client.call_async(SetMode.Request(custom_mode='OFFBOARD'))
        elif not msg.armed:
            self.node.get_logger().info("Attempting to arm...")
            self.arming_client.call_async(CommandBool.Request(value=True))

    def create_client_and_wait_for_service(self, srv_type, srv_name):
        client = self.node.create_client(srv_type, srv_name)
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f'{srv_name} service not available, waiting...')
        return client

    def timer_callback(self):
        if self._pose is None or self._image is None:
            return

        self._image_viz = self._image.copy()

        if self.step is None:
            if not self.steps:
                self.node.get_logger().info("Mission completed.")
                raise SystemExit
            self.step = self.steps.pop(0)
            self.step.init(self)

        step_info = str(self.step)
        if self.step.update(self):
            self.step = None

        self.pose_publisher.publish(PoseStamped(pose=self._goal_pose))
        self.visualize_flight_data(step_info)

    def visualize_flight_data(self, step_info):
        def vector_to_str(vector: np.ndarray):
            return np.array2string(vector, formatter={'float_kind': lambda x: "%.2f" % x})[1:-1]

        lines = [
            f"Step: {step_info}",
            f"Position: {vector_to_str(self.position)}",
            f"Goal Position: {vector_to_str(self.goal_position)}",
            f"Altitude: {self.position[2]:.2f} m",
            f"Yaw: {np.degrees(self.goal_yaw):.2f} deg"
        ]

        for i, line in enumerate(lines):
            cv2.putText(self._image_viz, line, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Drone View", self._image_viz)
        cv2.waitKey(1)

    @property
    def position(self) -> np.ndarray:
        p = self._pose.position
        return np.array([p.x, p.y, p.z])

    @property
    def orientation(self) -> np.ndarray:
        o = self._pose.orientation
        return np.array([o.x, o.y, o.z, o.w])

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def image_viz(self) -> np.ndarray:
        return self._image_viz

    @property
    def goal_position(self) -> np.ndarray:
        p = self._goal_pose.position
        return np.array([p.x, p.y, p.z])

    @goal_position.setter
    def goal_position(self, value: np.ndarray) -> None:
        self._goal_pose.position = Point(x=value[0], y=value[1], z=value[2])

    @property
    def goal_yaw(self) -> float:
        o = self._goal_pose.orientation
        quat = np.array([o.x, o.y, o.z, o.w])
        return Rotation.from_quat(quat).as_euler('xyz')[2]

    @goal_yaw.setter
    def goal_yaw(self, value: float):
        quat = Rotation.from_euler('xyz', np.array([0, 0, value])).as_quat()
        self._goal_pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def calculate_distance_to_goal(self, current_pos, goal_pos):
        return np.linalg.norm(goal_pos - current_pos)

class Takeoff(Step):
    def __init__(self, altitude: float, tolerance: float = 0.2) -> None:
        self.altitude = altitude
        self.tolerance = tolerance

    def init(self, controller: Controller) -> None:
        controller.goal_position = controller.position + np.array([0, 0, self.altitude])

    def update(self, controller: Controller) -> bool:
        return controller.calculate_distance_to_goal(controller.position, controller.goal_position) < self.tolerance

class MoveTo(Step):
    def __init__(self, x: float, y: float, z: float, yaw: float | None = None, tolerance: float = 0.5) -> None:
        self.position = np.array([x, y, z])
        self.yaw = yaw
        self.tolerance = tolerance

    def init(self, controller: Controller) -> None:
        controller.goal_position = self.position
        if self.yaw is not None:
            controller.goal_yaw = self.yaw

    def update(self, controller: Controller) -> bool:
        return controller.calculate_distance_to_goal(controller.position, controller.goal_position) < self.tolerance

class GateTraversal(Step):
    def __init__(self, gate_id: int, dist_entry: float = 1.5, entry_tolerance: float = 0.5, exit_tolerance: float = 0.5) -> None:
        self.gate_id = gate_id
        self.entry_tolerance = entry_tolerance
        self.exit_tolerance = exit_tolerance
        self.is_traversing = False
        self.dist_entry = dist_entry

    def init(self, controller: Controller) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_detector = ArucoDetector(dictionary)
        marker_size = 0.19
        gate_size = 1.54
        marker_corner_offset = marker_size / 2
        gate_marker_offset = (marker_size + gate_size) / 2
        self.marker_corners = np.array([
            [-marker_corner_offset, -marker_corner_offset, 0],
            [ marker_corner_offset, -marker_corner_offset, 0],
            [ marker_corner_offset,  marker_corner_offset, 0],
            [-marker_corner_offset,  marker_corner_offset, 0],
        ])
        self.gate_markers = np.array([
            [-gate_marker_offset,  gate_marker_offset, 0],
            [ gate_marker_offset,  gate_marker_offset, 0],
            [ gate_marker_offset, -gate_marker_offset, 0],
            [-gate_marker_offset, -gate_marker_offset, 0]
        ])
        image_width = 1280
        image_height = 720
        camera_vertical_fov = math.radians(86.8)
        camera_f = image_height / (2 * math.tan(camera_vertical_fov / 2))
        camera_c_x = image_width / 2
        camera_c_y = image_height / 2
        self.camera_matrix = np.array([
            [camera_f, 0, camera_c_x],
            [0, camera_f, camera_c_y],
            [0, 0, 1]
        ])
        self.entry_offset_optical = np.array([0, 0, -self.dist_entry])
        self.exit_offset_optical = np.array([0, 0.3, self.dist_entry])
        self.optical_to_camera = np.array([
            [ 0,  0,  1,  0],
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  1],
        ])
        self.camera_to_base = np.eye(4)
        self.camera_to_base[:3, :3] = Rotation.from_euler('y', -20, degrees=True).as_matrix()
        self.camera_to_base[0, 3] = 0.107

    def update(self, controller: Controller) -> bool:
        if self.is_traversing:
            controller.node.get_logger().info(f"Traversing gate {self.gate_id}")
            return controller.calculate_distance_to_goal(controller.position, controller.goal_position) < self.exit_tolerance

        corner_arrays, ids, _ = self.aruco_detector.detectMarkers(controller.image)
        cv2.aruco.drawDetectedMarkers(controller.image_viz, corner_arrays, ids)

        image_points = np.empty((0, 2))
        object_points = np.empty((0, 3))
        if ids is not None:
            for id, corners in zip(np.squeeze(ids, axis=1), np.squeeze(corner_arrays, axis=1)):
                if id // 4 == self.gate_id:
                    image_points = np.concatenate((image_points, corners))
                    object_points = np.concatenate((object_points, self.gate_markers[id % 4] + self.marker_corners))

        if image_points.size == 0:
            return False

        dist_coeffs = None
        solved, gate_rvec, gate_tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, dist_coeffs)
        if not solved:
            return False

        gate_rotation_optical, _ = cv2.Rodrigues(gate_rvec)
        gate_position_optical = np.squeeze(gate_tvec)
        entry_position_optical = gate_position_optical + gate_rotation_optical @ self.entry_offset_optical
        exit_position_optical = gate_position_optical + gate_rotation_optical @ self.exit_offset_optical

        for tvec in [gate_position_optical, entry_position_optical, exit_position_optical]:
            if tvec[2] > 0 and np.linalg.norm(tvec) >= 0.1:
                cv2.drawFrameAxes(controller.image_viz, self.camera_matrix, dist_coeffs, gate_rvec, tvec, 0.2)

        base_to_world = np.eye(4)
        base_to_world[:3, :3] = Rotation.from_quat(controller.orientation).as_matrix()
        base_to_world[:3, 3] = controller.position
        optical_to_world = base_to_world @ self.camera_to_base @ self.optical_to_camera

        gate_position = optical_to_world @ np.append(gate_position_optical, 1)
        entry_position = optical_to_world @ np.append(entry_position_optical, 1)
        exit_position = optical_to_world @ np.append(exit_position_optical, 1)

        controller.goal_position = entry_position[:3]
        controller.goal_yaw = np.arctan2(gate_position[1] - controller.position[1],
                                         gate_position[0] - controller.position[0])

        if controller.calculate_distance_to_goal(controller.position, controller.goal_position) < self.entry_tolerance:
            controller.goal_position = exit_position[:3]
            self.is_traversing = True

        return False

    def __str__(self) -> str:
        return f'{super().__str__()}: {self.gate_id} ({("traversing" if self.is_traversing else "approaching")})'

def create_mission_steps():
    return [
        Takeoff(2.2, 0.75),
        MoveTo(248.5, 245.0, 196.5, 1.25, 0.75),
        GateTraversal(0, 1.1, 0.85),
        MoveTo(243.5, 254.2, 195.1, 1.60),
        GateTraversal(1, 2.1, 0.85),
        MoveTo(249.2, 259.0, 194.9, 5.65),
        GateTraversal(2, 2.1, 1.05),
        MoveTo(254.3, 248.7, 197.1, 4.75),
        GateTraversal(3, 3.1, 1.05),
        MoveTo(254.3, 241.3, 195.5, 1.60),
        GateTraversal(4, 3.1, 0.75),
        MoveTo(254.0, 252.4, 195.1, 2.85),
        GateTraversal(5, 3.5, 0.85),
        MoveTo(240.2, 250.0, 195.1, 5.05),
        GateTraversal(6, 2.1, 0.85),
        # Повторение цикла с небольшими вариациями
        MoveTo(248.5, 245.0, 196.5, 1.25, 0.75),
        GateTraversal(0, 1.1, 0.85),
        MoveTo(243.5, 254.2, 195.1, 1.60),
        GateTraversal(1, 2.1, 0.85),
        MoveTo(249.2, 259.0, 194.9, 5.65),
        GateTraversal(2, 2.1, 1.05),
        MoveTo(254.3, 248.7, 197.1, 4.75),
        GateTraversal(3, 3.1, 1.05),
        MoveTo(254.3, 241.3, 195.5, 1.60),
        GateTraversal(4, 3.1, 0.85),
        MoveTo(254.0, 252.4, 195.1, 2.85),
        GateTraversal(5, 3.5, 0.75),
        MoveTo(240.2, 250.0, 195.1, 5.05),
        GateTraversal(6, 2.1, 0.85),
        # Еще один цикл
        MoveTo(248.5, 245.0, 196.5, 1.25, 0.75),
        GateTraversal(0, 1.1, 0.85),
        MoveTo(243.5, 254.2, 195.1, 1.60),
        GateTraversal(1, 2.1, 0.85),
        MoveTo(249.2, 259.0, 194.9, 5.65),
        GateTraversal(2, 2.1, 1.05),
        MoveTo(254.3, 248.7, 197.1, 4.75),
        GateTraversal(3, 3.1, 1.05),
        MoveTo(254.3, 241.3, 195.5, 1.60),
        GateTraversal(4, 3.1, 0.75),
        MoveTo(254.0, 252.4, 195.1, 2.85),
        GateTraversal(5, 3.5, 0.75),
        MoveTo(240.2, 250.0, 195.1, 5.05),
        GateTraversal(6, 2.1, 0.85),
        # Финальное возвращение к первым воротам
        MoveTo(248.5, 245.0, 196.5, 1.25, 0.75),
        GateTraversal(0, 1.1, 0.85),
    ]

def main(args=None):
    rclpy.init(args=args)
    node = Node('drone_controller')
    mission_steps = create_mission_steps()
    drone_controller = DroneController(node, mission_steps)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Mission aborted by user")
    except SystemExit:
        node.get_logger().info("Mission completed successfully")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
