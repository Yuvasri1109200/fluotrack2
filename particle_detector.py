"""
Particle Detection and Analysis Module
Uses OpenCV and image processing to
detect and analyze microplastics in video feeds
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import threading
import time
from collections import deque
from datetime import datetime

class ParticleDetector:
    """Real-time particle detection from webcam feed"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id  # Camera ID set to 0
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.particles = []
        self.frame_count = 0
        self.fps = 0
        self.last_frame_time = time.time()
        
        # Analysis parameters
        self.min_particle_size = 50  # pixels
        self.max_particle_size = 10000
        self.blur_kernel = (5, 5)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Particle history for tracking
        self.particle_history = deque(maxlen=100)
    
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            return True
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def detect_particles(self, frame):
        """Detect particles in a frame using image processing"""
        particles = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            thresh = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_particle_size or area > self.max_particle_size:
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                particle_info = {
                    'contour': contour,
                    'area': area,
                    'centroid': (cx, cy),
                    'perimeter': cv2.arcLength(contour, True),
                }
                
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    particle_info['ellipse'] = ellipse
                    particle_info['major_axis'] = max(ellipse[1])
                    particle_info['minor_axis'] = min(ellipse[1])
                    particle_info['aspect_ratio'] = max(ellipse[1]) / (min(ellipse[1]) + 1e-5)
                    particle_info['angle'] = ellipse[2]
                else:
                    particle_info['ellipse'] = None
                    particle_info['major_axis'] = np.sqrt(area)
                    particle_info['minor_axis'] = np.sqrt(area)
                    particle_info['aspect_ratio'] = 1.0
                
                particle_info['circularity'] = self.calculate_circularity(area, particle_info['perimeter'])
                particle_info['shape_type'] = self.classify_shape(
                    particle_info['circularity'], particle_info.get('aspect_ratio', 1.0)
                )
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                particle_info['convexity'] = area / (hull_area + 1e-5)
                
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                region = gray[mask == 255]
                
                if len(region) > 0:
                    particle_info['mean_intensity'] = np.mean(region)
                    particle_info['std_intensity'] = np.std(region)
                    particle_info['texture_roughness'] = np.std(np.gradient(region.reshape(-1)))
                
                particles.append(particle_info)
        
        except Exception as e:
            print(f"Particle detection error: {e}")
        
        return particles
    
    def calculate_circularity(self, area, perimeter):
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)
    
    def classify_shape(self, circularity, aspect_ratio):
        if circularity > 0.7:
            if aspect_ratio < 1.3:
                return "bead"
            else:
                return "spherical"
        elif aspect_ratio > 3.0:
            return "fiber"
        elif aspect_ratio > 1.5:
            return "fragment"
        else:
            return "film"
    
    def quantify_particles(self, particles):
        if not particles:
            return {
                'count': 0,
                'average_size': 0,
                'average_length': 0,
                'average_width': 0,
                'total_area': 0,
                'size_distribution': {},
                'shape_distribution': {},
            }
        
        areas = [p['area'] for p in particles]
        lengths = [p['major_axis'] for p in particles]
        widths = [p['minor_axis'] for p in particles]
        shapes = [p['shape_type'] for p in particles]
        aspect_ratios = [p.get('aspect_ratio', 1.0) for p in particles]
        circularities = [p['circularity'] for p in particles]
        
        size_dist = {
            'tiny': sum(1 for a in areas if a < 100),
            'small': sum(1 for a in areas if 100 <= a < 500),
            'medium': sum(1 for a in areas if 500 <= a < 2000),
            'large': sum(1 for a in areas if a >= 2000),
        }
        
        shape_dist = {}
        for shape in set(shapes):
            shape_dist[shape] = shapes.count(shape)
        
        roughness_dist = {
            'smooth': sum(1 for p in particles if p.get('std_intensity', 0) < 20),
            'rough': sum(1 for p in particles if 20 <= p.get('std_intensity', 0) < 50),
            'weathered': sum(1 for p in particles if p.get('std_intensity', 0) >= 50),
        }
        
        quantification = {
            'count': len(particles),
            'average_size': float(np.mean(areas)),
            'std_size': float(np.std(areas)),
            'average_length': float(np.mean(lengths)),
            'average_width': float(np.mean(widths)),
            'average_aspect_ratio': float(np.mean(aspect_ratios)),
            'average_circularity': float(np.mean(circularities)),
            'total_area': float(np.sum(areas)),
            'min_size': float(np.min(areas)),
            'max_size': float(np.max(areas)),
            'size_distribution': size_dist,
            'shape_distribution': shape_dist,
            'roughness_distribution': roughness_dist,
            'median_size': float(np.median(areas)),
            'percentile_95': float(np.percentile(areas, 95)),
        }
        
        return quantification
    
    def capture_loop(self):
        if not self.initialize_camera():
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.particles = self.detect_particles(frame)
                self.particle_history.append({
                    'timestamp': datetime.now(),
                    'particles': self.particles.copy(),
                    'count': len(self.particles)
                })
                
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.last_frame_time + 1e-5)
                self.last_frame_time = current_time
                self.frame_count += 1
                self.current_frame = frame.copy()
                
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Capture loop error: {e}")
        finally:
            self.release_camera()
            self.is_running = False
    
    def start_capture(self):
        if not self.is_running:
            thread = threading.Thread(target=self.capture_loop, daemon=True)
            thread.start()
    
    def stop_capture(self):
        self.is_running = False
    
    def get_frame_with_annotations(self):
        if self.current_frame is None:
            return None
        
        frame = self.current_frame.copy()
        for particle in self.particles:
            cv2.drawContours(frame, [particle['contour']], 0, (0, 255, 0), 2)
            cx, cy = particle['centroid']
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            if particle.get('ellipse'):
                cv2.ellipse(frame, particle['ellipse'], (255, 0, 0), 2)
            info_text = f"Size: {particle['area']:.0f} AR: {particle.get('aspect_ratio', 1.0):.2f}"
            cv2.putText(frame, info_text, (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Particles: {len(self.particles)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame
    
    def get_current_particles(self):
        return self.particles.copy()
    
    def get_quantification(self):
        return self.quantify_particles(self.particles)
    
    def get_statistics(self):
        return {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'current_particle_count': len(self.particles),
            'particles': self.particles,
            'quantification': self.quantify_particles(self.particles),
            'is_running': self.is_running,
        }

class FrameEncoder:
    """Convert frames to JPEG for streaming"""
    
    @staticmethod
    def encode_frame(frame):
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    @staticmethod
    def frame_to_base64(frame):
        import base64
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode()
        return jpg_as_text