# Version: 1.0.0 | Date: 2025-08-12 | Project: ComfyUI-DWPose-Kalman | Model: Claude-Opus-4.1
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Dict, Tuple, Optional


class DWPoseKalmanFilter:
    def __init__(self, num_joints=17, process_noise=0.01, measurement_noise=5.0):
        """
        Initialize Kalman filters for each joint
        Args:
            num_joints: Number of pose keypoints (17 for COCO, 18 for OpenPose)
            process_noise: How much we expect motion to vary
            measurement_noise: How much we trust the measurements
        """
        self.filters = {}
        self.num_joints = num_joints
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initialized = False

    def _create_filter(self) -> KalmanFilter:
        """Create a single Kalman filter for one joint"""
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [x, y, velocity_x, velocity_y]
        dt = 1.0  # Assuming frame-to-frame
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement function: we only measure x,y
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Add small regularization to prevent singularity
        kf.R = np.eye(2) * self.measurement_noise + np.eye(2) * 1e-6

        # Better process noise model
        kf.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * self.process_noise

        # Initialize covariance with non-zero values
        kf.P *= 100  # Start with high uncertainty

        return kf

    def initialize_filters(self, initial_pose: np.ndarray):
        """Initialize all filters with first detection"""
        self.filters = {}
        for joint_idx in range(self.num_joints):
            if joint_idx < initial_pose.shape[0] and initial_pose[joint_idx, 2] > 0:
                kf = self._create_filter()
                kf.x = np.array([
                    initial_pose[joint_idx, 0],
                    initial_pose[joint_idx, 1],
                    0,  # Initial velocity x
                    0   # Initial velocity y
                ])
                # Set initial covariance with uncertainty
                kf.P = np.eye(4) * 100  # High initial uncertainty
                self.filters[joint_idx] = kf
        self.initialized = True

    def update(self, pose: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Update pose with Kalman filtering
        Args:
            pose: Shape (num_joints, 3) - x, y, confidence
            confidence_threshold: Minimum confidence to update measurement
        Returns:
            Filtered pose
        """
        if not self.initialized:
            self.initialize_filters(pose)
            return pose

        filtered_pose = pose.copy()

        for joint_idx in range(self.num_joints):
            confidence = pose[joint_idx, 2]

            # Initialize filter if new joint appears
            if joint_idx not in self.filters and confidence > confidence_threshold:
                kf = self._create_filter()
                kf.x = np.array([pose[joint_idx, 0], pose[joint_idx, 1], 0, 0])
                self.filters[joint_idx] = kf

            if joint_idx in self.filters:
                kf = self.filters[joint_idx]

                # Predict next state
                kf.predict()

                # Update with measurement if confident
                if confidence > confidence_threshold:
                    measurement = np.array([pose[joint_idx, 0], pose[joint_idx, 1]])
                    kf.update(measurement)

                # Use filtered position
                filtered_pose[joint_idx, 0] = kf.x[0]
                filtered_pose[joint_idx, 1] = kf.x[1]

                # Keep original confidence or boost slightly for predicted points
                if confidence <= confidence_threshold:
                    filtered_pose[joint_idx, 2] = 0.5  # Synthetic confidence

        return filtered_pose

    def reset(self):
        """Reset all filters"""
        self.filters = {}
        self.initialized = False