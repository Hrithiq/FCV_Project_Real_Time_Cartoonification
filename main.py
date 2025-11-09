import cv2
import numpy as np
import argparse
from typing import List, Tuple

class RealTimeCartoonifier:
    """
    Implements a real-time cartoonification pipeline.

    This class integrates several computer vision techniques to
    transform a live video feed into a cartoon-like style:
    1. Video Preprocessing
    2. Optical Flow Motion Stabilization
    3. Edge Detection
    4. Bilateral Filtering for Color Smoothing
    5. Combination of Edges and Smoothed Colors
    
    This implementation prioritizes real-time performance.
    """

    def __init__(self,
                 bilateral_iters: int = 5,
                 palette: int = 16,
                 process_scale: float = 1.0,
                 show_side_by_side: bool = True,
                 use_stabilization: bool = True,
                 quant_levels: int = 8,
                 edge_strength: float = 0.6,
                 sharpness: float = 0.4,
                 auto_scale: bool = False,
                 target_fps: float = 24.0):
        """
        Initializes parameters for motion stabilization.
        """
        # Quality/Performance params (realtime)
        self.bilateral_iters = max(1, int(bilateral_iters))
        self.palette = max(2, int(palette))
        self.process_scale = float(process_scale)
        self.show_side_by_side = bool(show_side_by_side)
        self.use_stabilization = bool(use_stabilization)
        self.quant_levels = int(max(2, quant_levels))
        self.edge_strength = float(max(0.0, min(1.0, edge_strength)))
        self.sharpness = float(max(0.0, min(1.5, sharpness)))
        self.auto_scale = bool(auto_scale)
        self.target_fps = float(max(5.0, target_fps))

        # Previous frame's grayscale for optical flow
        self.prev_gray = None
        
        # Transformation from previous frame
        self.prev_transform = np.float32([[1, 0, 0], [0, 1, 0]])

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def process_frame(self, frame):
        """
        Applies the full cartoonification pipeline to a single frame.

        Args:
            frame (np.ndarray): The input video frame (BGR).

        Returns:
            np.ndarray: The cartoonified output frame (BGR).
        """
        # --- Stage 1: Stabilization ---
        # This step covers "Frame Preprocessing" and
        # "Optical Flow-Based Motion Stabilization".
        if self.use_stabilization:
            stabilized_frame = self.stabilize_frame(frame)
        else:
            stabilized_frame = frame

        # --- Stage 2-4: Stylization (higher quality, realtime-optimized) ---
        cartoon_frame = self.apply_realtime_style(stabilized_frame)

        # --- Stage 5: Frame Handling & Output ---
        return cartoon_frame

    def stabilize_frame(self, frame):
        """
        Applies motion stabilization using Lucas-Kanade optical flow.

        Tracks features from the previous frame to the current one and 
        computes an affine transform to stabilize the video.
        
        Args:
            frame (np.ndarray): The current video frame.

        Returns:
            np.ndarray: The stabilized video frame.
        """
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            # Initialize on the first frame
            self.prev_gray = current_gray
            return frame

        # Track features
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, 
                                          maxCorners=200, 
                                          qualityLevel=0.01, 
                                          minDistance=30, 
                                          blockSize=3)

        if prev_pts is None:
            # No features found, return original frame
            self.prev_gray = current_gray
            return frame

        # Calculate optical flow
        current_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, 
                                                          current_gray, 
                                                          prev_pts, 
                                                          None, 
                                                          **self.lk_params)

        # Filter good points
        good_new = current_pts[status == 1]
        good_old = prev_pts[status == 1]

        if len(good_new) < 4:
            # Not enough points to estimate transform
            self.prev_gray = current_gray
            return frame

        # Estimate affine transform
        transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)

        if transform is None:
            # If transform estimation fails, use previous
            transform = self.prev_transform
        else:
            # Smooth the transform to prevent jitter
            transform = self.prev_transform * 0.2 + transform * 0.8
            self.prev_transform = transform

        # Apply the transformation
        rows, cols, _ = frame.shape
        stabilized_frame = cv2.warpAffine(frame, transform, (cols, rows))

        self.prev_gray = current_gray
        return stabilized_frame

    def get_edges(self, frame):
        """
        Detects and enhances edges using Canny edge detection.

        It uses noise reduction (Median Blur) and Canny detection 
        to find contours.
        
        Args:
            frame (np.ndarray): The input (stabilized) frame.

        Returns:
            np.ndarray: A 1-channel binary mask of the detected edges.
        """
        # Noise reduction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Median blur is effective for noise reduction while preserving edges
        blurred = cv2.medianBlur(gray, 7)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def smooth_color(self, frame):
        """
        Applies color smoothing using iterative bilateral filtering.

        Applies a bilateral filter iteratively for edge-preserving 
        smoothing to create flat, uniform color regions.
        
        Args:
            frame (np.ndarray): The input (stabilized) frame.

        Returns:
            np.ndarray: The 3-channel color-smoothed frame.
        """
        # Apply bilateral filtering iteratively
        smoothed = frame
        for _ in range(3):
            smoothed = cv2.bilateralFilter(smoothed, 
                                          d=9, 
                                          sigmaColor=50, 
                                          sigmaSpace=9)
        return smoothed

    def kmeans_color_quantization(self, img: np.ndarray, k: int) -> np.ndarray:
        Z = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        quant = centers[labels.flatten()]
        return quant.reshape(img.shape)

    def uniform_color_quantization(self, img: np.ndarray, levels: int) -> np.ndarray:
        # Fast posterization per channel to 'levels' bins
        if levels <= 2:
            return ((img > 127) * 255).astype(np.uint8)
        step = 255.0 / float(levels - 1)
        quant = np.round(img.astype(np.float32) / step) * step
        return quant.clip(0, 255).astype(np.uint8)

    def apply_realtime_style(self, frame: np.ndarray) -> np.ndarray:
        # Optional downscale for speed
        if self.process_scale != 1.0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (int(w * self.process_scale), int(h * self.process_scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        # Iterative bilateral filtering (reduced blur, faster settings)
        smoothed = small
        for _ in range(self.bilateral_iters):
            smoothed = cv2.bilateralFilter(smoothed, d=7, sigmaColor=50, sigmaSpace=7)

        # Fast color quantization (uniform bins) to approximate k-means look efficiently
        quant = self.uniform_color_quantization(smoothed, self.quant_levels)

        # Soft edges from original small image to preserve details
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_blur = cv2.GaussianBlur(edges, (5, 5), 0)
        edges_norm = edges_blur.astype(np.float32) / 255.0
        darken = (1.0 - self.edge_strength * edges_norm)
        result_small = (quant.astype(np.float32) * darken[..., None]).clip(0, 255).astype(np.uint8)

        # Unsharp mask to restore detail without noise amplification
        if self.sharpness > 0.0:
            blur = cv2.GaussianBlur(result_small, (0, 0), sigmaX=1.0)
            result_small = cv2.addWeighted(result_small, 1.0 + self.sharpness, blur, -self.sharpness, 0)

        # Upscale back if needed
        if self.process_scale != 1.0:
            result = cv2.resize(result_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            result = result_small

        return result

    def run_webcam(self, camera_id: int = 0, width: int = 0, height: int = 0):
        """
        Runs the full pipeline on a live webcam feed.
        
        Implements the video capture and display loop.
        """
        # Start: Video Capture
        cap = cv2.VideoCapture(int(camera_id))
        if not cap.isOpened():
            print("Error: Cannot open webcam.")
            return

        # Try setting capture resolution if provided
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        print("Starting webcam feed... Press 'q' to quit.")

        # FPS measurement
        tick_freq = cv2.getTickFrequency()
        prev_tick = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame.")
                break
            
            # Flip frame horizontally for a "mirror" view
            frame = cv2.flip(frame, 1)

            # Run the full processing pipeline
            output_frame = self.process_frame(frame)

            # --- Output/Display Final Frame ---
            try:
                if self.show_side_by_side:
                    h, w, _ = frame.shape
                    combined_output = np.zeros((h, w * 2, 3), dtype=np.uint8)
                    combined_output[0:h, 0:w] = frame
                    combined_output[0:h, w:w * 2] = output_frame
                    cv2.imshow('Real-Time Cartoonification (Original | Processed)', combined_output)
                else:
                    cv2.imshow('Real-Time Cartoonification (Processed)', output_frame)
            except Exception as e:
                print(f"Error displaying frame: {e}")
                break

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Adaptive scaling to maintain target FPS
            if self.auto_scale:
                cur_tick = cv2.getTickCount()
                dt = (cur_tick - prev_tick) / tick_freq
                prev_tick = cur_tick
                if dt > 0:
                    fps = 1.0 / dt
                    # Adjust process_scale slightly to chase target FPS
                    if fps < self.target_fps - 1.0 and self.process_scale > 0.5:
                        self.process_scale = max(0.5, self.process_scale - 0.05)
                    elif fps > self.target_fps + 3.0 and self.process_scale < 1.0:
                        self.process_scale = min(1.0, self.process_scale + 0.05)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoonification - Realtime and Offline Multi-pass Pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=False)

    realtime_parser = subparsers.add_parser("realtime", help="Run realtime webcam cartoonifier")
    realtime_parser.add_argument("--camera_id", type=int, default=0, help="Webcam device index")
    realtime_parser.add_argument("--width", type=int, default=0, help="Capture width (0 to keep default)")
    realtime_parser.add_argument("--height", type=int, default=0, help="Capture height (0 to keep default)")
    realtime_parser.add_argument("--bilateral_iters", type=int, default=5, help="Iterations of bilateral filter for realtime stylization")
    realtime_parser.add_argument("--palette", type=int, default=16, help="Number of colors for k-means quantization in realtime")
    realtime_parser.add_argument("--process_scale", type=float, default=1.0, help="Scale factor <1.0 speeds up processing")
    realtime_parser.add_argument("--processed_only", action="store_true", help="Show only processed view instead of side-by-side")
    realtime_parser.add_argument("--no_stabilize", action="store_true", help="Disable optical-flow stabilization for speed")
    realtime_parser.add_argument("--quant_levels", type=int, default=8, help="Uniform quantization levels per channel (fast)")
    realtime_parser.add_argument("--edge_strength", type=float, default=0.6, help="Edge darkening strength [0..1]")
    realtime_parser.add_argument("--sharpness", type=float, default=0.4, help="Unsharp mask amount [0..1.5]")
    realtime_parser.add_argument("--auto_scale", action="store_true", help="Adapt processing scale to maintain target FPS")
    realtime_parser.add_argument("--target_fps", type=float, default=24.0, help="Desired realtime FPS when auto_scale is enabled")

    offline_parser = subparsers.add_parser("offline", help="Run offline multi-pass pipeline on a video")
    offline_parser.add_argument("--input", required=False, default="input.mp4", help="Path to input video file")
    offline_parser.add_argument("--output", required=False, default="output.mp4", help="Path to output video file")
    offline_parser.add_argument("--target_fps", type=int, default=60, help="Target FPS after interpolation")
    offline_parser.add_argument("--bilateral_iters", type=int, default=7, help="Iterations of bilateral filter for stylization")
    offline_parser.add_argument("--palette", type=int, default=24, help="Number of colors for k-means quantization")
    offline_parser.add_argument("--stabilization_radius", type=int, default=25, help="Smoothing radius (frames) for global stabilization")
    offline_parser.add_argument("--workdir", default=None, help="Optional directory to save intermediate frames (skipped if None)")

    args = parser.parse_args()

    if args.mode == "offline":
        def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return frames, fps

        def write_video_frames(path: str, frames: List[np.ndarray], fps: float) -> None:
            if not frames:
                raise ValueError("No frames to write")
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

        def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
            """Apply Gaussian smoothing using OpenCV for better motion path smoothing."""
            if len(values) < 3:
                return values
            smoothed = np.zeros_like(values)
            for i in range(values.shape[1]):
                # Use OpenCV Gaussian blur instead of scipy
                kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))  # Ensure odd kernel size
                if kernel_size % 2 == 0:
                    kernel_size += 1
                smoothed[:, i] = cv2.GaussianBlur(values[:, i].reshape(-1, 1), (kernel_size, 1), sigma).flatten()
            return smoothed

        def moving_average_smooth(values: np.ndarray, radius: int) -> np.ndarray:
            if radius <= 0:
                return values
            # Use Gaussian smoothing instead of simple moving average for better quality
            sigma = radius / 3.0  # Convert radius to Gaussian sigma
            return gaussian_smooth(values, sigma)

        def estimate_affine_between(a_gray: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
            # Use more robust feature detection
            pts_a = cv2.goodFeaturesToTrack(a_gray, maxCorners=2000, qualityLevel=0.001, minDistance=15, blockSize=7)
            if pts_a is None or len(pts_a) < 50:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            
            # Use more conservative LK parameters for stability
            lk_params = dict(winSize=(21, 21), maxLevel=2, 
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            pts_b, status, err = cv2.calcOpticalFlowPyrLK(a_gray, b_gray, pts_a, None, **lk_params)
            
            if pts_b is None:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            
            # Filter by both status and error
            good_mask = (status.flatten() == 1) & (err.flatten() < 5.0)
            good_a = pts_a[good_mask]
            good_b = pts_b[good_mask]
            
            if len(good_a) < 20:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            
            # Use more conservative RANSAC threshold
            M, mask = cv2.estimateAffine2D(good_a, good_b, 
                                         ransacReprojThreshold=2.0, 
                                         confidence=0.99, 
                                         maxIters=2000)
            if M is None or mask is None or np.sum(mask) < 10:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            
            # Validate transform magnitude to prevent extreme corrections
            tx, ty, sx, sy, r = decompose_affine(M)
            max_translation = 50.0  # pixels
            max_scale = 0.3  # 30% scale change max
            max_rotation = 15.0  # degrees
            
            if abs(tx) > max_translation or abs(ty) > max_translation:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            if abs(sx - 1.0) > max_scale or abs(sy - 1.0) > max_scale:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            if abs(r) > max_rotation:
                return np.float32([[1, 0, 0], [0, 1, 0]])
            
            return M.astype(np.float32)

        def decompose_affine(M: np.ndarray) -> Tuple[float, float, float, float, float]:
            a, b, tx = M[0]
            c, d, ty = M[1]
            scale_x = np.sqrt(a * a + b * b)
            scale_y = np.sqrt(c * c + d * d)
            rotation = np.degrees(np.arctan2(b, a))
            return tx, ty, scale_x, scale_y, rotation

        def compose_affine(tx: float, ty: float, scale_x: float, scale_y: float, rotation_deg: float) -> np.ndarray:
            theta = np.radians(rotation_deg)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            a = scale_x * cos_t
            b = scale_x * sin_t
            c = -scale_y * sin_t
            d = scale_y * cos_t
            return np.float32([[a, b, tx], [c, d, ty]])

        def pass1_global_stabilization(frames: List[np.ndarray], radius: int) -> List[np.ndarray]:
            if len(frames) <= 2:
                return frames
            
            print(f"Analyzing motion for {len(frames)} frames...")
            gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            transforms: List[np.ndarray] = []
            
            # Estimate transforms with temporal consistency
            for i in range(1, len(gray)):
                M = estimate_affine_between(gray[i - 1], gray[i])
                transforms.append(M)
                if i % 30 == 0:
                    print(f"  Processed {i}/{len(gray)-1} frame pairs")
            
            # Accumulate to trajectories with better handling
            trajectory = []
            tx, ty, sx, sy, r = 0.0, 0.0, 1.0, 1.0, 0.0
            for i, M in enumerate(transforms):
                dtx, dty, dsx, dsy, dr = decompose_affine(M)
                
                # Apply temporal smoothing to reduce jitter
                alpha = 0.7  # Smoothing factor
                tx = alpha * tx + (1 - alpha) * (tx + dtx)
                ty = alpha * ty + (1 - alpha) * (ty + dty)
                sx = alpha * sx + (1 - alpha) * (sx * dsx)
                sy = alpha * sy + (1 - alpha) * (sy * dsy)
                r = alpha * r + (1 - alpha) * (r + dr)
                
                trajectory.append([tx, ty, sx, sy, r])
            
            trajectory = np.array(trajectory, dtype=np.float32)
            print(f"Applying smoothing with radius {radius}...")
            
            # Apply stronger smoothing
            smoothed = moving_average_smooth(trajectory, radius=max(radius, 10))
            
            # Compute correction transforms with better stability
            stabilized_frames: List[np.ndarray] = [frames[0]]
            h, w = frames[0].shape[:2]
            
            print("Applying stabilization corrections...")
            for i in range(1, len(frames)):
                cur = trajectory[i - 1]
                tgt = smoothed[i - 1]
                
                # Calculate correction needed
                diff_tx = tgt[0] - cur[0]
                diff_ty = tgt[1] - cur[1]
                diff_r = tgt[4] - cur[4]
                
                # Limit correction magnitude to prevent overcorrection
                max_correction_trans = 20.0  # pixels
                max_correction_rot = 5.0  # degrees
                
                diff_tx = np.clip(diff_tx, -max_correction_trans, max_correction_trans)
                diff_ty = np.clip(diff_ty, -max_correction_trans, max_correction_trans)
                diff_r = np.clip(diff_r, -max_correction_rot, max_correction_rot)
                
                correction = compose_affine(diff_tx, diff_ty, 1.0, 1.0, diff_r)
                stabilized = cv2.warpAffine(frames[i], correction, (w, h), 
                                          flags=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_REFLECT)
                stabilized_frames.append(stabilized)
                
                if i % 30 == 0:
                    print(f"  Stabilized {i}/{len(frames)-1} frames")
            
            print("Global stabilization complete.")
            return stabilized_frames

        def pass2_interpolate_to_fps(frames: List[np.ndarray], src_fps: float, target_fps: float) -> List[np.ndarray]:
            if target_fps <= src_fps or len(frames) < 2:
                return frames
            ratio = int(round(target_fps / src_fps))
            if ratio <= 1:
                return frames
            out: List[np.ndarray] = []
            flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            for i in range(len(frames) - 1):
                A = frames[i]
                C = frames[i + 1]
                out.append(A)
                A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
                C_gray = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(A_gray, C_gray, None, **flow_params)
                # Generate intermediate frames between A and C
                for k in range(1, ratio):
                    t = k / ratio
                    # forward warp A by t
                    h, w = A_gray.shape
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x_A = (grid_x + flow[..., 0] * t).astype(np.float32)
                    map_y_A = (grid_y + flow[..., 1] * t).astype(np.float32)
                    warp_A = cv2.remap(A, map_x_A, map_y_A, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    # backward warp C by (1 - t)
                    map_x_C = (grid_x - flow[..., 0] * (1 - t)).astype(np.float32)
                    map_y_C = (grid_y - flow[..., 1] * (1 - t)).astype(np.float32)
                    warp_C = cv2.remap(C, map_x_C, map_y_C, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    B = cv2.addWeighted(warp_A, 0.5, warp_C, 0.5, 0)
                    out.append(B)
            out.append(frames[-1])
            return out

        def bilateral_iterative(img: np.ndarray, iters: int) -> np.ndarray:
            result = img
            for _ in range(max(1, iters)):
                result = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=9)
            return result

        def kmeans_color_quantization(img: np.ndarray, k: int) -> np.ndarray:
            Z = img.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            centers = np.uint8(centers)
            quant = centers[labels.flatten()]
            return quant.reshape(img.shape)

        def soft_edges(img: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edges_blur = cv2.GaussianBlur(edges, (5, 5), 0)
            return edges_blur

        def apply_ghibli_style(img: np.ndarray, bilateral_iters: int, palette: int) -> np.ndarray:
            smoothed = bilateral_iterative(img, bilateral_iters)
            quant = kmeans_color_quantization(smoothed, max(2, palette))
            edges_soft = soft_edges(img)
            edges_norm = edges_soft.astype(np.float32) / 255.0
            darken = (1.0 - 0.6 * edges_norm)  # reduce brightness along edges
            result = (quant.astype(np.float32) * darken[..., None]).clip(0, 255).astype(np.uint8)
            return result

        def pass3_stylize_frames(frames: List[np.ndarray], bilateral_iters: int, palette: int) -> List[np.ndarray]:
            return [apply_ghibli_style(f, bilateral_iters, palette) for f in frames]

        # Execute pipeline
        frames_in, src_fps = read_video_frames(args.input)
        stabilized = pass1_global_stabilization(frames_in, radius=args.stabilization_radius)
        interpolated = pass2_interpolate_to_fps(stabilized, src_fps=src_fps, target_fps=float(args.target_fps))
        stylized = pass3_stylize_frames(interpolated, bilateral_iters=args.bilateral_iters, palette=args.palette)
        write_video_frames(args.output, stylized, fps=float(args.target_fps))
    else:
        # Default to realtime if no subcommand provided
        if args.mode == "realtime" or args.mode is None:
            cartoonifier = RealTimeCartoonifier(
                bilateral_iters=getattr(args, 'bilateral_iters', 5),
                palette=getattr(args, 'palette', 16),
                process_scale=max(0.25, min(1.0, float(getattr(args, 'process_scale', 1.0)))),
                show_side_by_side=not bool(getattr(args, 'processed_only', False)),
                use_stabilization=not bool(getattr(args, 'no_stabilize', False)),
                quant_levels=int(getattr(args, 'quant_levels', 8)),
                edge_strength=float(getattr(args, 'edge_strength', 0.6)),
                sharpness=float(getattr(args, 'sharpness', 0.4)),
                auto_scale=bool(getattr(args, 'auto_scale', False)),
                target_fps=float(getattr(args, 'target_fps', 24.0))
            )
            cartoonifier.run_webcam(
                camera_id=getattr(args, 'camera_id', 0),
                width=getattr(args, 'width', 0),
                height=getattr(args, 'height', 0)
            )
        else:
            cartoonifier = RealTimeCartoonifier()
            cartoonifier.run_webcam()