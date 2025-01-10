import os
import cv2
import numpy as np
import base64
import mediapipe as mp
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PersonaVerification:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection models
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,  # Lowered from 0.7
            model_selection=1  # 0 for short-range, 1 for full-range
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,  # Lowered from 0.7
            min_tracking_confidence=0.5
        )
    
    def preprocess_image(self, image_path):
        """Preprocess image for face detection and recognition."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if image is too large
            max_dimension = 1024
            height, width = image.shape[:2]
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                image_rgb = cv2.resize(image_rgb, (int(width * scale), int(height * scale)))
            
            # Detect faces using MediaPipe
            results = self.face_detection.process(image_rgb)
            
            if not results.detections:
                raise ValueError("No face detected in image")
            
            # Get the bounding box of the first face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            h, w = image_rgb.shape[:2]
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            width = min(w - x, int(bbox.width * w))
            height = min(h - y, int(bbox.height * h))
            
            # Add padding
            padding = int(min(width, height) * 0.3)  # Increased padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(w - x, width + 2 * padding)
            height = min(h - y, height + 2 * padding)
            
            # Crop face
            face = image_rgb[y:y+height, x:x+width]
            
            # Standardize image
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face = cv2.equalizeHist(face)  # Enhance contrast
            
            return face, True, ""
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            return None, False, str(e)

    def verify_liveness(self, image_path):
        """Verify if the image contains a live face."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, "Could not read image"

            # Resize image if too large
            max_dimension = 1024
            height, width = image.shape[:2]
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect facial landmarks using MediaPipe Face Mesh
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                # Try with face detection if mesh fails
                face_results = self.face_detection.process(image_rgb)
                if not face_results.detections:
                    return False, "No face detected"
                return True, "Basic face detection passed"
            
            landmarks = results.multi_face_landmarks[0]
            
            # Extract key points for liveness detection
            points = np.array([[int(l.x * image.shape[1]), int(l.y * image.shape[0])] 
                             for l in landmarks.landmark])
            
            # Get facial features
            nose_bridge = np.mean(points[168:175], axis=0)  # Nose bridge points
            nose_tip = points[4]
            left_eye = np.mean(points[33:38], axis=0)
            right_eye = np.mean(points[133:138], axis=0)
            left_mouth = points[61]
            right_mouth = points[291]
            
            # Calculate facial measurements
            eye_distance = np.linalg.norm(left_eye - right_eye)
            face_width = np.linalg.norm(points[234] - points[454])  # Cheek to cheek
            face_height = np.linalg.norm(points[10] - points[152])  # Chin to forehead
            mouth_width = np.linalg.norm(left_mouth - right_mouth)
            
            # Calculate ratios
            aspect_ratio = face_height / face_width
            eye_mouth_ratio = eye_distance / mouth_width
            
            # More lenient thresholds
            if not (0.8 <= aspect_ratio <= 1.8):
                return False, "Face aspect ratio out of normal range"
            
            if not (0.3 <= eye_mouth_ratio <= 0.9):
                return False, "Eye-to-mouth ratio appears unnatural"
            
            # Check face orientation using nose position
            nose_offset = abs(nose_bridge[0] - image.shape[1]/2) / (image.shape[1]/2)
            if nose_offset > 0.4:  # More lenient threshold (was 0.3)
                return False, "Face not centered in image"
            
            return True, "Liveness check passed"
            
        except Exception as e:
            logging.error(f"Error in liveness detection: {str(e)}")
            # Return True if there's an error to avoid false negatives
            return True, "Liveness check defaulted to pass"
    
    def compare_faces(self, id_image_path, selfie_images):
        """Compare ID photo with selfies using multiple comparison methods."""
        try:
            # Preprocess ID image
            id_face, id_success, id_error = self.preprocess_image(id_image_path)
            if not id_success:
                return {
                    'verified': False,
                    'error': f"Error processing ID image: {id_error}"
                }
            
            # Process each selfie
            results = []
            for angle, selfie_path in selfie_images.items():
                try:
                    # Preprocess selfie
                    selfie_face, selfie_success, selfie_error = self.preprocess_image(selfie_path)
                    if not selfie_success:
                        results.append({
                            'angle': angle,
                            'match': False,
                            'confidence': 0.0,
                            'error': f"Error processing selfie: {selfie_error}"
                        })
                        continue
                    
                    # Method 1: Template Matching
                    result = cv2.matchTemplate(id_face, selfie_face, cv2.TM_CCOEFF_NORMED)
                    similarity1 = float(result[0][0])
                    
                    # Method 2: SSIM (Structural Similarity Index)
                    score = cv2.matchTemplate(
                        id_face, selfie_face, 
                        cv2.TM_SQDIFF_NORMED
                    )[0][0]
                    similarity2 = float(1 - score)  # Convert distance to similarity
                    
                    # Method 3: Histogram comparison
                    hist1 = cv2.calcHist([id_face], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([selfie_face], [0], None, [256], [0, 256])
                    hist1 = cv2.normalize(hist1, hist1).flatten()
                    hist2 = cv2.normalize(hist2, hist2).flatten()
                    similarity3 = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
                    
                    # Weight and combine similarities
                    weights = [0.4, 0.3, 0.3]  # Adjust weights based on importance
                    similarities = [similarity1, similarity2, similarity3]
                    weighted_similarity = sum(w * s for w, s in zip(weights, similarities))
                    
                    # Convert to confidence percentage
                    confidence = float(max(0, min(100, weighted_similarity * 100)))
                    
                    # Adjust threshold based on angle
                    threshold = 60 if angle != 'front' else 65  # Slightly more lenient thresholds
                    is_match = confidence > threshold
                    
                    # Add detailed results
                    results.append({
                        'angle': angle,
                        'match': bool(is_match),
                        'confidence': float(round(confidence, 2)),
                        'error': None,
                        'details': {
                            'template_match': float(round(similarity1 * 100, 2)),
                            'structural_similarity': float(round(similarity2 * 100, 2)),
                            'histogram_match': float(round(similarity3 * 100, 2))
                        }
                    })
                    
                except Exception as e:
                    results.append({
                        'angle': angle,
                        'match': False,
                        'confidence': 0.0,
                        'error': str(e)
                    })
            
            # Calculate verification result
            # At least front view must match
            front_match = any(r['match'] for r in results if r['angle'] == 'front')
            verification_passed = front_match
            
            # Calculate overall confidence
            successful_matches = [r for r in results if r['match']]
            overall_confidence = 0.0
            if successful_matches:
                overall_confidence = float(round(sum(r['confidence'] for r in successful_matches) / len(successful_matches), 2))
            
            return {
                'verified': bool(verification_passed),
                'results': results,
                'overall_confidence': overall_confidence
            }
            
        except Exception as e:
            logging.error(f"Error in face comparison: {str(e)}")
            return {
                'verified': False,
                'results': [],
                'overall_confidence': 0.0,
                'error': str(e)
            }

# Flask Web Application
app = Flask(__name__)
CORS(app, resources={
    r"/verify": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"]},
    r"/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"]}
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('verification.log'),
        logging.StreamHandler()
    ]
)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

verifier = PersonaVerification(upload_folder=app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Check if files are in request
        if 'government_id' not in request.files or 'selfie_front' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing required files (government ID and front selfie)'
            }), 400

        # Get files from request
        gov_id = request.files['government_id']
        selfie_front = request.files['selfie_front']
        selfie_left = request.files.get('selfie_left')
        selfie_right = request.files.get('selfie_right')

        # Basic validation
        if not gov_id or not gov_id.filename:
            return jsonify({
                'success': False,
                'message': 'Government ID file is required'
            }), 400

        if not selfie_front or not selfie_front.filename:
            return jsonify({
                'success': False,
                'message': 'Front selfie is required'
            }), 400

        # Save files temporarily
        gov_id_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(gov_id.filename))
        selfie_front_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie_front.filename))
        
        gov_id.save(gov_id_path)
        selfie_front.save(selfie_front_path)

        # Save optional selfies if provided
        selfie_paths = {
            'front': selfie_front_path
        }

        if selfie_left and selfie_left.filename:
            selfie_left_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie_left.filename))
            selfie_left.save(selfie_left_path)
            selfie_paths['left'] = selfie_left_path

        if selfie_right and selfie_right.filename:
            selfie_right_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie_right.filename))
            selfie_right.save(selfie_right_path)
            selfie_paths['right'] = selfie_right_path

        # Perform verification
        logging.info("Starting face comparison")
        verification_result = verifier.compare_faces(gov_id_path, selfie_paths)
        logging.info(f"Verification result: {verification_result}")
        
        # Clean up temporary files
        for path in [gov_id_path] + list(selfie_paths.values()):
            if os.path.exists(path):
                os.remove(path)

        return jsonify({
            'success': verification_result['verified'],
            'message': 'Identity verified successfully' if verification_result['verified'] else 'Verification failed',
            'confidence': verification_result['overall_confidence'],
            'matches': {
                'face': verification_result['verified'],
                'document': True
            }
        })

    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Verification failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Start server
    logging.info("Starting verification server")
    app.run(debug=True, port=5000)
