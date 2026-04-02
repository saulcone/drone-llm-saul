# filepath: c:\Users\saulc\Desktop\Cuarto\TFG\drone-llm-saul\reasoning\depth_safety_guard.py
"""
Depth-Based Safety Guard - Phase 5: Reactive Safety Layer

Uses monocular depth estimation to create a 1D occupancy map and prevent collisions.
This runs locally and can override VLM commands with ~50ms latency vs 1-3s for cloud.

Supports:
- Depth Anything V2 (recommended, fast and accurate)
- MiDaS (fallback, widely available)
- Simple edge-based fallback (no ML required)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DepthSafetyGuard:
    """
    Reactive safety layer using depth estimation.
    
    Creates a 1D occupancy map by analyzing depth across horizontal slices
    of the image. Detects if obstacles are too close on left/center/right
    and overrides commands to prevent collision.
    
    Drone specifications:
    - Width: 60cm (30cm each side from camera center)
    - Safe passage requires: >35cm clearance on each side
    - Critical distance: <50cm to obstacle = danger zone
    """
    
    # Drone physical parameters
    DRONE_WIDTH_CM = 60
    DRONE_HALF_WIDTH_CM = 30
    SAFE_CLEARANCE_CM = 35  # Minimum clearance needed on each side
    
    # Depth thresholds (normalized 0-1, where 1 = closest)
    CRITICAL_DEPTH = 0.7    # Very close obstacle
    WARNING_DEPTH = 0.5     # Getting close
    SAFE_DEPTH = 0.3        # Safe distance
    
    # Image zones for 1D occupancy map
    NUM_ZONES = 5  # Left edge, left, center, right, right edge
    
    def __init__(self, use_depth_model: bool = False, model_type: str = "edge_fallback"):
        """
        Initialize the depth safety guard.
        
        Args:
            use_depth_model: If True, try to load ML depth model
            model_type: "depth_anything", "midas", or "edge_fallback"
        """
        self.use_depth_model = use_depth_model
        self.model_type = model_type
        self.depth_model = None
        self.last_occupancy = None
        self.intervention_count = 0
        
        if use_depth_model:
            self._load_depth_model(model_type)
    
    def _load_depth_model(self, model_type: str):
        """Load the depth estimation model."""
        try:
            if model_type == "depth_anything":
                # Try to load Depth Anything V2
                try:
                    from transformers import pipeline
                    self.depth_model = pipeline(
                        "depth-estimation", 
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device="cpu"  # Use CPU for low latency
                    )
                    logger.info("[DEPTH] Loaded Depth Anything V2 model")
                except Exception as e:
                    logger.warning(f"[DEPTH] Could not load Depth Anything: {e}")
                    self.model_type = "edge_fallback"
                    
            elif model_type == "midas":
                # Try to load MiDaS
                try:
                    import torch
                    self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                    self.depth_model.eval()
                    logger.info("[DEPTH] Loaded MiDaS model")
                except Exception as e:
                    logger.warning(f"[DEPTH] Could not load MiDaS: {e}")
                    self.model_type = "edge_fallback"
            else:
                logger.info("[DEPTH] Using edge-based fallback (no ML model)")
                self.model_type = "edge_fallback"
                
        except Exception as e:
            logger.warning(f"[DEPTH] Error loading model: {e}, using fallback")
            self.model_type = "edge_fallback"
    
    def estimate_depth_map(self, image_cv2: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        
        Args:
            image_cv2: BGR image from OpenCV
            
        Returns:
            Depth map normalized to 0-1 (1 = closest)
        """
        if self.depth_model is not None and self.model_type == "depth_anything":
            # Use Depth Anything V2
            from PIL import Image
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            result = self.depth_model(pil_image)
            depth = np.array(result["depth"])
            # Normalize to 0-1 (invert so closer = higher value)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return depth
            
        elif self.depth_model is not None and self.model_type == "midas":
            # Use MiDaS
            import torch
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                depth = self.depth_model(input_tensor)
            depth = depth.squeeze().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return depth
            
        else:
            # Edge-based fallback: estimate depth from edge density
            return self._estimate_depth_from_edges(image_cv2)
    
    def _estimate_depth_from_edges(self, image_cv2: np.ndarray) -> np.ndarray:
        """
        Estimate relative depth using edge density (fallback method).
        
        More edges typically indicate closer objects or texture.
        This is a rough approximation but works for door frames.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create depth-like map by dilating edges and blurring
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        depth_like = cv2.GaussianBlur(dilated.astype(np.float32), (31, 31), 0)
        
        # Normalize
        depth_like = depth_like / (depth_like.max() + 1e-6)
        
        return depth_like
    
    def compute_1d_occupancy(self, depth_map: np.ndarray) -> Dict:
        """
        Convert 2D depth map to 1D horizontal occupancy map.
        
        Divides the image into horizontal zones and computes
        the closest obstacle distance in each zone.
        
        Returns:
            Dict with zone occupancy values and analysis
        """
        height, width = depth_map.shape[:2]
        
        # Focus on the middle 60% of the image height (where door frame would be)
        y_start = int(height * 0.2)
        y_end = int(height * 0.8)
        roi = depth_map[y_start:y_end, :]
        
        # Divide into zones
        zone_width = width // self.NUM_ZONES
        zones = []
        
        for i in range(self.NUM_ZONES):
            x_start = i * zone_width
            x_end = (i + 1) * zone_width if i < self.NUM_ZONES - 1 else width
            zone = roi[:, x_start:x_end]
            
            # Get max depth in zone (closest obstacle)
            max_depth = np.percentile(zone, 95)  # Use 95th percentile to reduce noise
            zones.append(max_depth)
        
        # Analyze zones
        zone_names = ["left_edge", "left", "center", "right", "right_edge"]
        occupancy = {name: val for name, val in zip(zone_names, zones)}
        
        # Compute danger levels
        left_danger = max(zones[0], zones[1])
        right_danger = max(zones[3], zones[4])
        center_danger = zones[2]
        
        # Compute imbalance (positive = more danger on right)
        imbalance = right_danger - left_danger
        
        analysis = {
            "zones": occupancy,
            "left_danger": left_danger,
            "right_danger": right_danger,
            "center_danger": center_danger,
            "imbalance": imbalance,
            "left_critical": left_danger > self.CRITICAL_DEPTH,
            "right_critical": right_danger > self.CRITICAL_DEPTH,
            "center_critical": center_danger > self.CRITICAL_DEPTH,
            "path_clear": center_danger < self.WARNING_DEPTH and 
                          left_danger < self.CRITICAL_DEPTH and 
                          right_danger < self.CRITICAL_DEPTH
        }
        
        self.last_occupancy = analysis
        return analysis
    
    def check_safety(self, image_cv2: np.ndarray, intended_command: str, 
                     distance: str = "DESCONOCIDO") -> Tuple[str, Optional[str], Dict]:
        """
        Check if command is safe to execute based on depth analysis.
        
        Args:
            image_cv2: Current camera image
            intended_command: Command to evaluate
            distance: VLM's distance estimate
            
        Returns:
            Tuple of (safe_command, override_reason, analysis)
        """
        # Only check forward commands
        if not intended_command.startswith("A"):
            return intended_command, None, {}
        
        # Get depth map and occupancy
        depth_map = self.estimate_depth_map(image_cv2)
        occupancy = self.compute_1d_occupancy(depth_map)
        
        # === CRITICAL: Imminent collision ===
        if occupancy["center_critical"]:
            # Something very close in front - STOP
            self.intervention_count += 1
            return "E", f"DEPTH: Obstacle ahead (center={occupancy['center_danger']:.2f}). STOP.", occupancy
        
        # === Left side danger ===
        if occupancy["left_critical"] and not occupancy["right_critical"]:
            # Left wall close, right is clear - strafe right
            self.intervention_count += 1
            return "D2", f"DEPTH: Left wall close ({occupancy['left_danger']:.2f}). Strafe RIGHT.", occupancy
        
        # === Right side danger ===
        if occupancy["right_critical"] and not occupancy["left_critical"]:
            # Right wall close, left is clear - strafe left
            self.intervention_count += 1
            return "D1", f"DEPTH: Right wall close ({occupancy['right_danger']:.2f}). Strafe LEFT.", occupancy
        
        # === Both sides danger ===
        if occupancy["left_critical"] and occupancy["right_critical"]:
            # Both sides close - reduce speed
            if intended_command in ["A2", "A3"]:
                return "A1", f"DEPTH: Tight passage. Reduced to A1.", occupancy
            # If already A1 and path ahead is clear, allow
            if not occupancy["center_critical"]:
                return intended_command, None, occupancy
            else:
                self.intervention_count += 1
                return "E", "DEPTH: Passage too narrow. STOP.", occupancy
        
        # === Imbalance warning (not critical but risky) ===
        if distance in ["MUY_CERCA", "CERCA"]:
            if abs(occupancy["imbalance"]) > 0.2:
                if occupancy["imbalance"] > 0:
                    # More danger on right, drift left
                    return "D1", f"DEPTH: Drift detected (imbalance={occupancy['imbalance']:.2f}). Correct LEFT.", occupancy
                else:
                    # More danger on left, drift right
                    return "D2", f"DEPTH: Drift detected (imbalance={occupancy['imbalance']:.2f}). Correct RIGHT.", occupancy
        
        # === Path is clear ===
        if occupancy["path_clear"]:
            return intended_command, None, occupancy
        
        # Default: allow but log warning
        return intended_command, None, occupancy
    
    def get_status(self) -> Dict:
        """Get current safety guard status."""
        return {
            "model_type": self.model_type,
            "interventions": self.intervention_count,
            "last_occupancy": self.last_occupancy
        }
    
    def reset(self):
        """Reset intervention counter."""
        self.intervention_count = 0
        self.last_occupancy = None


class CombinedSafetyGuard:
    """
    Combines edge-based and depth-based safety for robust collision prevention.
    
    Uses both methods and takes the most conservative action.
    """
    
    def __init__(self, use_depth_model: bool = False):
        from reasoning.wall_proximity_detector import SafetyEnvelope
        
        self.edge_safety = SafetyEnvelope()
        self.depth_safety = DepthSafetyGuard(
            use_depth_model=use_depth_model,
            model_type="edge_fallback"  # Start with fallback, can upgrade
        )
    
    def check_safety(self, image_cv2: np.ndarray, target_point: list,
                     distance: str, intended_command: str) -> Tuple[str, Optional[str], Dict]:
        """
        Check safety using both edge and depth methods.
        
        Takes the most conservative action between both methods.
        """
        # Get edge-based safety check
        edge_cmd, edge_reason, edge_analysis = self.edge_safety.check_crossing_safety(
            image_cv2, target_point, distance, intended_command
        )
        
        # Get depth-based safety check
        depth_cmd, depth_reason, depth_analysis = self.depth_safety.check_safety(
            image_cv2, intended_command, distance
        )
        
        # Combine results - prefer most conservative action
        combined_analysis = {
            "edge": edge_analysis,
            "depth": depth_analysis
        }
        
        # Priority: STOP > STRAFE > REDUCE > ALLOW
        priority = {"E": 0, "D1": 1, "D2": 1, "D3": 1, "D4": 1, "A1": 2, "A2": 3, "A3": 4}
        
        edge_priority = priority.get(edge_cmd, 5)
        depth_priority = priority.get(depth_cmd, 5)
        
        if edge_priority <= depth_priority:
            return edge_cmd, edge_reason, combined_analysis
        else:
            return depth_cmd, depth_reason, combined_analysis
    
    def get_status(self) -> Dict:
        """Get combined status."""
        return {
            "edge": self.edge_safety.get_status(),
            "depth": self.depth_safety.get_status()
        }
