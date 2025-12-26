import rclpy
from rclpy.node import Node
import gymnasium as gym
from gymnasium import spaces
from geometry_msgs.msg import Twist, TwistStamped, Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3

import numpy as np
import math
import time
import os
import subprocess
from collections import deque

class TurtleBotEnv(gym.Env, Node):
    
    metadata = {'render_modes': ['human']}

    def __init__(self):
        # Initialize ROS 2 first if needed
        if not rclpy.ok():
            rclpy.init()
        
        gym.Env.__init__(self)
        Node.__init__(self, 'turtlebot_rl_env_node')
        
        # Configuration dictionary with valid goal positions (SAFE CENTER of environment)
        # ROBOTIS method: Random generation within bounds, ensuring min distance from robot
        # Official ROBOTIS stage1: 5m x 5m arena (walls at ±2.5m, actual bounds ±2.425m)
        # Safe goal generation: use ±2.0m to keep away from walls
        self.goal_generation_range = 2.0  # Generate goals within ±2.0m (ROBOTIS uses ±2.1m)
        
        # Fallback safe positions if random generation fails
        self.safe_goal_positions = [
            (1.5, 1.5),    # Top-right quadrant
            (1.5, -1.5),   # Bottom-right quadrant
            (-1.5, 1.5),   # Top-left quadrant
            (-1.5, -1.5),  # Bottom-left quadrant
            (2.0, 0.0),    # Far right
            (-2.0, 0.0),   # Far left
            (0.0, 2.0),    # Far top
            (0.0, -2.0),   # Far bottom
            (1.0, 1.0),    # Intermediate positions
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ]
        
        self.config = {
            'robot_model': 'waffle_pi',
            'goal_position': (2.0, 0.5),  # Default goal
            'reset_position': (0.0, 0.0, 0.1),
            'max_episode_steps': 500,
            'velocities': {
                # ROBOTIS official velocities
                'forward_linear': 0.15,  # Official: 0.15 m/s
                'turn_linear': 0.0,  # No forward while turning
                'turn_angular': 0.5,  # Official uses variable: 0.0, ±0.75, ±1.5
            },
            'thresholds': {
                'goal_distance': 0.2,
                'collision_distance': 0.10,  # Only trigger if extremely close (< 10cm)
                'max_scan_range': 3.5,
                'boundary_warning': 2.0,  # Start penalizing when robot gets close to boundary
                'min_goal_distance': 0.7,  # ROBOTIS: min distance between robot and new goal
            },
            'rewards': {
                'goal_reached': 100.0,  # Reduced from 500 for stable Q-values
                'collision': -50.0,  # Reduced from -200
                'out_of_bounds': -50.0,  # Reduced from -200
                'boundary_warning': -1.0,  # Reduced from -2.0
                'step_penalty': 0.01,  # Reduced from 0.1 to allow longer episodes
                'distance_penalty': 0.01,
                'progress_multiplier': 10.0,  # Increased from 5.0 - encourage goal approach
                'regress_penalty': 5.0,  # Increased from 1.0 - discourage moving away
                'near_goal_bonus': 20.0,  # NEW: Extra reward when getting close
            },
        }
        
        # ROBOTIS official action space: 5 discrete actions
        # Action 0: Sharp right turn (angular_z = -1.5)
        # Action 1: Gentle right turn (angular_z = -0.75)
        # Action 2: Straight forward (angular_z = 0.0)
        # Action 3: Gentle left turn (angular_z = 0.75)
        # Action 4: Sharp left turn (angular_z = 1.5)
        self.action_space = spaces.Discrete(5)
        self.angular_velocities = [-1.5, -0.75, 0.0, 0.75, 1.5]  # ROBOTIS official values
        
        self.num_scan_samples = 20
        low = np.array([0.0] * self.num_scan_samples + [0.0, -math.pi])
        high = np.array([3.5] * self.num_scan_samples + [10.0, math.pi])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Goal entity name for spawning/deleting
        self.goal_entity_name = 'target_goal'
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.laser_data = None
        self.odom_data = None
        self.goal_x = self.config['goal_position'][0]
        self.goal_y = self.config['goal_position'][1]
        self.previous_distance_to_goal = None
        self.max_episode_steps = self.config['max_episode_steps']
        self.step_count = 0
        
        # Track repeated out-of-bounds detections for stuck robot
        self.last_oob_position = None
        self.oob_count = 0
        self.MAX_OOB_REPEATS = 3
        
        # Track position history to detect stuck robot (more lenient settings)
        self.position_history = deque(maxlen=100)  # Keep last 100 position samples
        self.stuck_check_interval = 50  # Check over 50 steps (not 10)
        self.stuck_threshold = 0.15  # meters - increased from 0.05m (more lenient)
        
        self.get_logger().info('TurtleBotEnv (ROS 2 Jazzy) initialized.')

    def is_goal_in_collision(self, goal_x, goal_y):
        """Check if goal position is within valid bounds (ROBOTIS stage1: 5m x 5m arena)"""
        # Official ROBOTIS stage1: walls at ±2.425m
        # Keep goals within ±2.2m for safety margin from walls
        if abs(goal_x) > 2.2 or abs(goal_y) > 2.2:
            return True
        return False

    def is_robot_out_of_bounds(self, robot_x, robot_y):
        """Check if robot has gone out of the arena boundaries"""
        # Official ROBOTIS stage1: boundaries at ±2.425m
        # Trigger if robot gets close to walls (±2.3m)
        if abs(robot_x) > 2.3 or abs(robot_y) > 2.3:
            self.get_logger().warn(f"Robot out of bounds at ({robot_x:.2f}, {robot_y:.2f})")
            return True
        return False

    def is_robot_stuck(self):
        """Check if robot has been stuck in one place (not moving, likely hit a wall/pillar).
        Returns True if robot hasn't moved significantly in the last check interval.
        """
        if self.odom_data is None:
            return False
        
        # Don't check stuck in first 20 steps (robot might be turning in place)
        if self.step_count < 20:
            return False
        
        robot_x = self.odom_data.pose.pose.position.x
        robot_y = self.odom_data.pose.pose.position.y
        current_pos = (robot_x, robot_y)
        
        # Add current position to history
        self.position_history.append(current_pos)
        
        # Check if we have enough history to make a determination
        if len(self.position_history) < self.stuck_check_interval:
            return False
        
        # Get position from stuck_check_interval steps ago
        old_pos = self.position_history[0]
        distance_moved = math.hypot(current_pos[0] - old_pos[0], current_pos[1] - old_pos[1])
        
        # If robot hasn't moved much in the last interval, it's stuck
        is_stuck = distance_moved < self.stuck_threshold
        
        if is_stuck:
            self.get_logger().warn(
                f"Robot stuck detected! Movement: {distance_moved:.4f}m < {self.stuck_threshold}m over {self.stuck_check_interval} steps. "
                f"Position: ({robot_x:.2f}, {robot_y:.2f})"
            )
        
        return is_stuck

    def emergency_stop_robot(self):
        """Aggressively stop the robot with multiple repeated commands"""
        stop_twist = Twist()
        stop_cmd = TwistStamped()
        stop_cmd.header = Header()
        stop_cmd.header.frame_id = "base_link"
        stop_cmd.twist = stop_twist
        
        # Send stop command 8 times rapidly to override any previous velocity
        for i in range(8):
            stop_cmd.header.stamp = self.get_clock().now().to_msg()
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.025)  # 40Hz

    def spawn_goal_entity(self):
        """Spawn goal as actual Gazebo entity using ROBOTIS method.
        This uses the goal_box model from turtlebot3_gazebo package."""
        try:
            service_name = '/world/dqn/create'
            try:
                from ament_index_python.packages import get_package_share_directory
                package_share = get_package_share_directory('turtlebot3_gazebo')
                model_path = os.path.join(
                    package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
                )
            except:
                # Fallback path
                model_path = '/home/michael/turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models/turtlebot3_dqn_world/goal_box/model.sdf'
            
            req = (
                f'sdf_filename: "{model_path}", '
                f'name: "{self.goal_entity_name}", '
                f'pose: {{ position: {{ '
                f'x: {self.goal_x}, '
                f'y: {self.goal_y}, '
                f'z: 0.01 }} }}'
            )
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
            if 'true' in result.stdout.lower() or result.returncode == 0:
                self.get_logger().info(f'🎯 Goal spawned at ({self.goal_x:.2f}, {self.goal_y:.2f})')
                return True
            else:
                self.get_logger().warn(f'Goal spawn may have failed: {result.stderr}')
                return False
        except Exception as e:
            self.get_logger().warn(f'Goal spawn error: {e}')
            return False
    
    def delete_goal_entity(self):
        """Delete goal entity from Gazebo using ROBOTIS method."""
        try:
            service_name = '/world/dqn/remove'
            req = f'name: "{self.goal_entity_name}", type: 2'  # type 2 = MODEL
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
            # Don't log delete - it's called every reset
            return True
        except Exception as e:
            return False
    
    def publish_goal_marker(self):
        """DEPRECATED: Use spawn_goal_entity() instead.
        Kept for compatibility but does nothing now."""
        pass
    
    def publish_gazebo_marker(self):
        """DEPRECATED: Use spawn_goal_entity() instead.
        Kept for compatibility but does nothing now."""
        pass

    def scan_callback(self, msg):
        self.laser_data = msg.ranges
    
    def odom_callback(self, msg):
        self.odom_data = msg

    def _get_state(self):
        """Get current state from sensor data with timeout protection."""
        max_waits = 50  # 50 * 0.1s = 5 seconds total timeout
        wait_count = 0
        while (self.laser_data is None or self.odom_data is None) and wait_count < max_waits:
            rclpy.spin_once(self, timeout_sec=0.05)
            wait_count += 1
        if self.laser_data is None or self.odom_data is None:
            raise RuntimeError(f'Sensor data timeout: laser={self.laser_data is not None}, odom={self.odom_data is not None}')

        scan = np.array(self.laser_data, dtype=np.float32)
        max_range = self.config['thresholds']['max_scan_range']
        # Replace infinities and NaNs with max_range
        scan[np.isposinf(scan)] = max_range
        scan[np.isneginf(scan)] = max_range
        scan[np.isnan(scan)] = max_range
        # Ensure values are within [0, max_range]
        scan = np.clip(scan, 0.0, max_range)
        
        sample_indices = np.linspace(0, len(scan) - 1, self.num_scan_samples, dtype=int)
        sampled_scan = scan[sample_indices]

        pos_x = self.odom_data.pose.pose.position.x
        pos_y = self.odom_data.pose.pose.position.y
        distance_to_goal = math.sqrt((self.goal_x - pos_x)**2 + (self.goal_y - pos_y)**2)
        
        q = self.odom_data.pose.pose.orientation
        robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        
        angle_to_goal_global = math.atan2(self.goal_y - pos_y, self.goal_x - pos_x)
        angle_to_goal_relative = math.atan2(
            math.sin(angle_to_goal_global - robot_yaw), 
            math.cos(angle_to_goal_global - robot_yaw))

        state = np.concatenate(
            (sampled_scan, [distance_to_goal], [angle_to_goal_relative])
        ).astype(np.float32)
        
        if self.previous_distance_to_goal is None:
            self.previous_distance_to_goal = distance_to_goal
        
        return state

    def reset(self, seed=None, options=None):
        self.step_count = 0
        super().reset(seed=seed)
        
        # Clear tracking
        self.last_oob_position = None
        self.oob_count = 0
        self.position_history.clear()
        
        # STOP robot first
        self.get_logger().info('=== RESETTING ENVIRONMENT ===')
        for i in range(10):  # Quick stops
            self.emergency_stop_robot()
            time.sleep(0.02)
        
        # STEP 1: Set new goal position using ROBOTIS method
        # Generate random goal ensuring minimum distance from robot
        attempts = 0
        goal_set = False
        min_dist = self.config['thresholds']['min_goal_distance']
        
        # Get current robot position (if available)
        robot_x = 0.0
        robot_y = 0.0
        if self.odom_data:
            robot_x = self.odom_data.pose.pose.position.x
            robot_y = self.odom_data.pose.pose.position.y
        
        while attempts < 50 and not goal_set:
            # ROBOTIS method: random.randrange(-21, 21) / 10 = [-2.1, 2.1]
            # Adapted for our simple world: ±2.5m range
            gx = np.random.uniform(-self.goal_generation_range, self.goal_generation_range)
            gy = np.random.uniform(-self.goal_generation_range, self.goal_generation_range)
            
            # Check if goal is within bounds and far enough from robot
            dist_from_robot = math.sqrt((gx - robot_x)**2 + (gy - robot_y)**2)
            
            if not self.is_goal_in_collision(gx, gy) and dist_from_robot >= min_dist:
                self.goal_x, self.goal_y = gx, gy
                goal_set = True
            attempts += 1
        
        # Fallback to safe predefined positions if random generation fails
        if not goal_set:
            self.goal_x, self.goal_y = self.safe_goal_positions[np.random.randint(0, len(self.safe_goal_positions))]
        
        self.get_logger().info(f'🎯 GOAL: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        
        # STEP 2: Delete old goal entity and spawn new one BEFORE robot reset
        # This ensures the goal is in place when robot respawns
        self.delete_goal_entity()  # Remove old goal (if exists)
        time.sleep(0.1)  # Brief wait for deletion
        self.spawn_goal_entity()  # Spawn new goal at new position
        time.sleep(0.1)  # Brief wait for spawn to complete
        
        # STEP 3: Now reset robot using ROBOTIS method - DELETE and RESPAWN
        # This prevents the "flashing/stuck" issue
        success = self.reset_robot()
        if not success:
            self.get_logger().warn("⚠️  Robot reset failed, trying teleport fallback")
            # Fallback to teleport if respawn fails
            success = self.teleport_robot_to_center()
            if not success:
                # Last resort: aggressive stop
                for i in range(10):
                    self.emergency_stop_robot()
                    time.sleep(0.05)
        
        # Wait for physics to settle and robot to respawn
        time.sleep(0.5)  # Increased wait time for respawn
        
        # Get fresh odometry
        for _ in range(20):  # More spins to ensure robot is ready
            rclpy.spin_once(self, timeout_sec=0.05)
        
        # Log current position
        if self.odom_data:
            robot_x = self.odom_data.pose.pose.position.x
            robot_y = self.odom_data.pose.pose.position.y
            self.get_logger().info(f"Robot position after reset: ({robot_x:.2f}, {robot_y:.2f})")
        
        # NOTE: Do NOT clear laser_data and odom_data - callbacks are continuous
        # Clearing them forces a 5-second timeout in _get_state() waiting for new data
        self.previous_distance_to_goal = None
        
        # Brief spin to process any pending callbacks
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)
        
        # Get fresh initial state (should be instant since laser/odom already have data)
        initial_state = self._get_state()
        info = {}
        self.get_logger().info('Environment reset complete.')
        return initial_state, info

    def step(self, action):
        self.step_count += 1
        
        # Execute action using ROBOTIS official method
        # All actions move forward at constant linear velocity
        # Angular velocity varies based on action (5 discrete values)
        twist = Twist()
        twist.linear.x = self.config['velocities']['forward_linear']  # Constant 0.15 m/s
        twist.angular.z = self.angular_velocities[action]  # Variable: -1.5, -0.75, 0.0, 0.75, 1.5

        # Create TwistStamped for the bridge
        vel_cmd = TwistStamped()
        vel_cmd.header = Header()
        vel_cmd.header.stamp = self.get_clock().now().to_msg()
        vel_cmd.header.frame_id = "base_link"
        vel_cmd.twist = twist

        # Publish velocity command multiple times to ensure it reaches Gazebo
        for _ in range(3):
            self.cmd_vel_pub.publish(vel_cmd)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.033)  # ~30Hz
        
        # Allow the robot time to execute the command and process callbacks
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.05)
        
        # Get new state BEFORE checking conditions
        new_state = self._get_state()
        reward, done = 0.0, False
        distance_to_goal = new_state[-2]
        min_laser = np.min(new_state[:self.num_scan_samples])
        robot_x = self.odom_data.pose.pose.position.x
        robot_y = self.odom_data.pose.pose.position.y

        # Check goal reached
        if distance_to_goal < self.config['thresholds']['goal_distance']:
            reward = self.config['rewards']['goal_reached']
            done = True
            self.get_logger().info(f"✓ GOAL REACHED! Reward: {reward} at ({robot_x:.2f}, {robot_y:.2f})")
        # Check collision
        elif min_laser < self.config['thresholds']['collision_distance']:
            reward = self.config['rewards']['collision']
            done = True
            self.get_logger().info(f"✗ COLLISION! Laser: {min_laser:.3f}m")
        # Check if robot is stuck (not moving, likely against wall/pillar)
        elif self.is_robot_stuck():
            reward = self.config['rewards']['collision']  # Same penalty as collision
            done = True
            self.get_logger().warn(f"✗ STUCK! Robot unable to move at ({robot_x:.2f}, {robot_y:.2f}). Resetting...")
        # Check if robot is out of bounds (far away) - Only after 10 steps to allow reset to complete
        elif self.step_count > 10 and self.is_robot_out_of_bounds(robot_x, robot_y):
            reward = self.config['rewards'].get('out_of_bounds', -200.0)
            done = True
            self.get_logger().warn(f"✗ OUT OF BOUNDS at ({robot_x:.2f}, {robot_y:.2f})")
        else:
            # Normal step rewards (NOT terminal)
            # Step penalty (encourages efficiency)
            reward = -self.config['rewards']['step_penalty']
            
            # Add boundary warning penalty to discourage wandering (but only after 5 steps)
            if self.step_count > 5 and (abs(robot_x) > self.config['thresholds']['boundary_warning'] or abs(robot_y) > self.config['thresholds']['boundary_warning']):
                reward += self.config['rewards']['boundary_warning']
            
            # Near-goal bonus (provide strong signal when getting close)
            if distance_to_goal < 0.5:  # Within 0.5m of goal
                reward += self.config['rewards']['near_goal_bonus'] * (0.5 - distance_to_goal)
            
            # Progress-based reward shaping (encourage moving toward goal)
            # CRITICAL: Only calculate if we have previous distance (not first step of episode)
            if self.previous_distance_to_goal is not None:
                progress = self.previous_distance_to_goal - distance_to_goal
                # Cap progress reward to prevent exploitation
                progress = np.clip(progress, -0.5, 0.5)  # Max 0.5m movement per step
                
                if progress > 0.01:  # Moving toward goal (threshold to ignore tiny movements)
                    reward += self.config['rewards']['progress_multiplier'] * progress
                elif progress < -0.01:  # Moving away from goal
                    reward -= self.config['rewards']['regress_penalty'] * abs(progress)
                # If |progress| < 0.01, no progress reward (neutral)
        
        # Update previous distance for next step
        self.previous_distance_to_goal = distance_to_goal
        
        # Check episode length limit
        if self.step_count >= self.max_episode_steps:
            done = True
            self.get_logger().info(f"Episode limit reached ({self.step_count} steps)")
        
        # Stop robot when done
        if done:
            # Send multiple stop commands to ensure robot stops
            stop_cmd = TwistStamped()
            stop_cmd.header = Header()
            stop_cmd.header.frame_id = "base_link"
            stop_cmd.twist = Twist()  # All zeros = stop
            
            # Send stop command multiple times
            for _ in range(5):
                stop_cmd.header.stamp = self.get_clock().now().to_msg()
                self.cmd_vel_pub.publish(stop_cmd)
                time.sleep(0.02)
        
        return new_state, reward, done, False, {}

    def close(self):
        self.get_logger().info('Closing environment...')
        # Stop the robot with TwistStamped
        stop_cmd = TwistStamped()
        stop_cmd.header = Header()
        stop_cmd.header.stamp = self.get_clock().now().to_msg()
        stop_cmd.header.frame_id = "base_link"
        stop_cmd.twist = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        self.destroy_node()

    def teleport_robot_to_center(self):
        """Teleport robot to center using CORRECT Gazebo service!"""
        try:
            # Use the CORRECT world name: 'dqn' and model name: 'waffle_pi' (not tb3_waffle_pi!)
            # Format: gz service -s /world/{world_name}/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean
            result = subprocess.run(
                ['gz', 'service', '-s', '/world/dqn/set_pose', 
                 '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean', '--timeout', '1000',
                 '--req', 'name: "waffle_pi", position: {x: 0.0, y: 0.0, z: 0.1}, orientation: {x: 0, y: 0, z: 0, w: 1}'],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if 'true' in result.stdout.lower() or result.returncode == 0:
                self.get_logger().info("✓ Robot teleported to center")
                return True
            else:
                self.get_logger().warn(f"Teleport failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Teleport timeout")
            return False
        except Exception as e:
            self.get_logger().warn(f"Teleport error: {e}")
            return False
    
    def reset_robot(self):
        """Reset robot using ROBOTIS method: DELETE then RESPAWN. 
        This prevents the flashing/stuck issue that happens with teleport."""
        try:
            # Step 1: Delete the existing robot
            service_name_delete = '/world/dqn/remove'
            req_delete = 'name: "waffle_pi", type: 2'  # type 2 = MODEL
            cmd_delete = [
                'gz', 'service',
                '-s', service_name_delete,
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req_delete
            ]
            result = subprocess.run(cmd_delete, capture_output=True, text=True, timeout=2.0)
            self.get_logger().info("🗑️  Deleted robot")
            
            # Wait for deletion to complete
            time.sleep(0.2)
            
            # Step 2: Respawn the robot at center
            service_name_spawn = '/world/dqn/create'
            # Get the model path from turtlebot3_gazebo package
            try:
                from ament_index_python.packages import get_package_share_directory
                package_share = get_package_share_directory('turtlebot3_gazebo')
                model_path = os.path.join(package_share, 'models', 'turtlebot3_waffle_pi', 'model.sdf')
            except:
                # Fallback if package not found - try direct path
                model_path = '/home/michael/turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf'
            
            req_spawn = (
                f'sdf_filename: "{model_path}", '
                f'name: "waffle_pi", '
                f'pose: {{ position: {{ x: 0.0, y: 0.0, z: 0.01 }} }}'
            )
            cmd_spawn = [
                'gz', 'service',
                '-s', service_name_spawn,
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req_spawn
            ]
            result = subprocess.run(cmd_spawn, capture_output=True, text=True, timeout=2.0)
            
            if 'true' in result.stdout.lower() or result.returncode == 0:
                self.get_logger().info("✓ Robot respawned at center")
                return True
            else:
                self.get_logger().warn(f"Respawn failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Reset robot timeout")
            return False
        except Exception as e:
            self.get_logger().warn(f"Reset robot error: {e}")
            return False