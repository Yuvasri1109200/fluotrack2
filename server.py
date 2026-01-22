from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import json
from sqlalchemy import func
import random
import base64
import cv2
import threading
import time
from particle_detector import ParticleDetector, FrameEncoder
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# Secret key for sessions (in production, load from env)
app.secret_key = 'change-me-to-a-secure-random-value'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///microplastics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize particle detector
detector = None
detector_lock = threading.Lock()

# ==================== DATABASE MODELS ====================

class Microplastic(db.Model):
    __tablename__ = 'microplastics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    sample_id = db.Column(db.String(50), unique=True, nullable=False)
    detection_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    
    # Structure properties
    structure_type = db.Column(db.String(50), nullable=False)  # fiber, fragment, bead, film
    polymer_type = db.Column(db.String(50), nullable=False)   # PE, PET, PP, PS, PVC, etc.
    
    # Shape properties
    shape = db.Column(db.String(50), nullable=False)          # linear, spherical, irregular, sheet
    aspect_ratio = db.Column(db.Float)                        # length/width ratio
    
    # Size properties
    length = db.Column(db.Float)                              # microns
    width = db.Column(db.Float)                               # microns
    thickness = db.Column(db.Float)                           # microns
    area = db.Column(db.Float)                                # square microns
    volume = db.Column(db.Float)                              # cubic microns
    
    # Analysis data
    color = db.Column(db.String(50))
    density = db.Column(db.Float)
    transparency = db.Column(db.String(20))                  # transparent, translucent, opaque
    surface_texture = db.Column(db.String(20))               # smooth, rough, weathered
    
    # Classification
    risk_level = db.Column(db.String(20))                    # low, medium, high, critical
    concentration = db.Column(db.Float)                      # particles per liter or cubic meter
    
    # Additional metadata
    sample_type = db.Column(db.String(50))                   # water, soil, air, food
    confidence_score = db.Column(db.Float)                   # 0-100
    
    def to_dict(self):
        return {
            'id': self.id,
            'sample_id': self.sample_id,
            'detection_date': self.detection_date.isoformat(),
            'location': self.location,
            'structure_type': self.structure_type,
            'polymer_type': self.polymer_type,
            'shape': self.shape,
            'aspect_ratio': self.aspect_ratio,
            'length': self.length,
            'width': self.width,
            'thickness': self.thickness,
            'area': self.area,
            'volume': self.volume,
            'color': self.color,
            'density': self.density,
            'transparency': self.transparency,
            'surface_texture': self.surface_texture,
            'risk_level': self.risk_level,
            'concentration': self.concentration,
            'sample_type': self.sample_type,
            'confidence_score': self.confidence_score,
        }

class AnalysisReport(db.Model):
    __tablename__ = 'analysis_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    report_name = db.Column(db.String(100), nullable=False)
    created_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    total_samples = db.Column(db.Integer)
    total_particles = db.Column(db.Integer)
    average_size = db.Column(db.Float)
    dominant_polymer = db.Column(db.String(50))
    risk_assessment = db.Column(db.String(20))
    
    def to_dict(self):
        return {
            'id': self.id,
            'report_name': self.report_name,
            'created_date': self.created_date.isoformat(),
            'total_samples': self.total_samples,
            'total_particles': self.total_particles,
            'average_size': self.average_size,
            'dominant_polymer': self.dominant_polymer,
            'risk_assessment': self.risk_assessment,
        }

# Simple user model for authentication
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Protected dashboard
    if not session.get('user_id'):
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')

    # POST: create user
    data = request.form
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    if not email or not password:
        flash('Email and password required')
        return redirect(url_for('signup'))

    existing = User.query.filter_by(email=email).first()
    if existing:
        flash('Email already exists')
        return redirect(url_for('signup'))

    user = User(email=email, name=name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    # Log the user in
    session['user_id'] = user.id
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    data = request.form
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        flash('Invalid email or password')
        return redirect(url_for('login'))

    session['user_id'] = user.id
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

# API routes for index.html
@app.route('/api/signup', methods=['POST'])
def api_signup():
    print("API signup called")
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    if not email or not password:
        return jsonify({'message': 'Email and password required'}), 400

    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({'message': 'Email already exists'}), 400

    user = User(email=email, name=name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id
    return jsonify({'id': user.id, 'email': user.email, 'name': user.name})

@app.route('/api/login', methods=['POST'])
def api_login():
    print("API login called")
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'message': 'Invalid email or password'}), 401

    session['user_id'] = user.id
    return jsonify({'id': user.id, 'email': user.email, 'name': user.name})

@app.route('/api/samples/<int:user_id>', methods=['GET'])
def get_user_samples(user_id):
    if session.get('user_id') != user_id:
        return jsonify([]), 403
    samples = Microplastic.query.filter_by(user_id=user_id).order_by(Microplastic.detection_date.desc()).all()
    return jsonify([s.to_dict() for s in samples])

# Rest of the API endpoints, modified to include user_id where needed

@app.route('/api/microplastics', methods=['GET'])
def get_microplastics():
    """Get all microplastics with optional filtering"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Optional filters
    structure = request.args.get('structure')
    shape = request.args.get('shape')
    polymer = request.args.get('polymer')
    risk = request.args.get('risk')
    sample_type = request.args.get('sample_type')
    
    query = Microplastic.query.filter_by(user_id=user_id)
    
    if structure:
        query = query.filter_by(structure_type=structure)
    if shape:
        query = query.filter_by(shape=shape)
    if polymer:
        query = query.filter_by(polymer_type=polymer)
    if risk:
        query = query.filter_by(risk_level=risk)
    if sample_type:
        query = query.filter_by(sample_type=sample_type)
    
    pagination = query.order_by(Microplastic.detection_date.desc()).paginate(page=page, per_page=per_page)
    
    return jsonify({
        'items': [mp.to_dict() for mp in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

@app.route('/api/microplastics/<int:mp_id>', methods=['GET'])
def get_microplastic(mp_id):
    """Get a specific microplastic"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    mp = Microplastic.query.filter_by(id=mp_id, user_id=user_id).first_or_404()
    return jsonify(mp.to_dict())

@app.route('/api/microplastics', methods=['POST'])
def create_microplastic():
    """Create a new microplastic record"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    
    try:
        # Calculate area if not provided
        if data.get('length') and data.get('width') and not data.get('area'):
            data['area'] = data['length'] * data['width']
        
        # Calculate volume if not provided
        if data.get('length') and data.get('width') and data.get('thickness') and not data.get('volume'):
            data['volume'] = data['length'] * data['width'] * data['thickness']
        
        # Calculate aspect ratio if not provided
        if data.get('length') and data.get('width') and not data.get('aspect_ratio'):
            data['aspect_ratio'] = data['length'] / data['width'] if data['width'] > 0 else 0
        
        mp = Microplastic(
            user_id=user_id,
            sample_id=data.get('sample_id'),
            location=data.get('location'),
            structure_type=data.get('structure_type'),
            polymer_type=data.get('polymer_type'),
            shape=data.get('shape'),
            aspect_ratio=data.get('aspect_ratio'),
            length=data.get('length'),
            width=data.get('width'),
            thickness=data.get('thickness'),
            area=data.get('area'),
            volume=data.get('volume'),
            color=data.get('color'),
            density=data.get('density'),
            transparency=data.get('transparency'),
            surface_texture=data.get('surface_texture'),
            risk_level=data.get('risk_level'),
            concentration=data.get('concentration'),
            sample_type=data.get('sample_type'),
            confidence_score=data.get('confidence_score'),
        )
        
        db.session.add(mp)
        db.session.commit()
        
        return jsonify(mp.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/microplastics/<int:mp_id>', methods=['PUT'])
def update_microplastic(mp_id):
    """Update a microplastic record"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    mp = Microplastic.query.filter_by(id=mp_id, user_id=user_id).first_or_404()
    data = request.get_json()
    
    try:
        for key, value in data.items():
            if hasattr(mp, key):
                setattr(mp, key, value)
        
        db.session.commit()
        return jsonify(mp.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/microplastics/<int:mp_id>', methods=['DELETE'])
def delete_microplastic(mp_id):
    """Delete a microplastic record"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    mp = Microplastic.query.filter_by(id=mp_id, user_id=user_id).first_or_404()
    
    try:
        db.session.delete(mp)
        db.session.commit()
        return jsonify({'message': 'Deleted successfully'}), 204
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get dashboard statistics"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    total = Microplastic.query.filter_by(user_id=user_id).count()
    
    structure_dist = db.session.query(
        Microplastic.structure_type, func.count(Microplastic.id)
    ).filter_by(user_id=user_id).group_by(Microplastic.structure_type).all()
    
    shape_dist = db.session.query(
        Microplastic.shape, func.count(Microplastic.id)
    ).filter_by(user_id=user_id).group_by(Microplastic.shape).all()
    
    polymer_dist = db.session.query(
        Microplastic.polymer_type, func.count(Microplastic.id)
    ).filter_by(user_id=user_id).group_by(Microplastic.polymer_type).all()
    
    risk_dist = db.session.query(
        Microplastic.risk_level, func.count(Microplastic.id)
    ).filter_by(user_id=user_id).group_by(Microplastic.risk_level).all()
    
    sample_type_dist = db.session.query(
        Microplastic.sample_type, func.count(Microplastic.id)
    ).filter_by(user_id=user_id).group_by(Microplastic.sample_type).all()
    
    avg_size = db.session.query(func.avg(Microplastic.length)).filter_by(user_id=user_id).scalar() or 0
    avg_concentration = db.session.query(func.avg(Microplastic.concentration)).filter_by(user_id=user_id).scalar() or 0
    avg_confidence = db.session.query(func.avg(Microplastic.confidence_score)).filter_by(user_id=user_id).scalar() or 0
    
    # Risk level distribution
    risk_counts = db.session.query(
        Microplastic.risk_level, func.count(Microplastic.id)
    ).filter_by(user_id=user_id).group_by(Microplastic.risk_level).all()
    
    critical_count = sum(count for level, count in risk_counts if level == 'critical')
    high_count = sum(count for level, count in risk_counts if level == 'high')
    
    return jsonify({
        'total_particles': total,
        'average_size': round(avg_size, 2),
        'average_concentration': round(avg_concentration, 2),
        'average_confidence': round(avg_confidence, 2),
        'critical_particles': critical_count,
        'high_risk_particles': high_count,
        'structure_distribution': {item[0]: item[1] for item in structure_dist},
        'shape_distribution': {item[0]: item[1] for item in shape_dist},
        'polymer_distribution': {item[0]: item[1] for item in polymer_dist},
        'risk_distribution': {item[0]: item[1] for item in risk_dist},
        'sample_type_distribution': {item[0]: item[1] for item in sample_type_dist},
    })

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all analysis reports"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    reports = AnalysisReport.query.filter_by(user_id=user_id).order_by(AnalysisReport.created_date.desc()).all()
    return jsonify([r.to_dict() for r in reports])

@app.route('/api/reports', methods=['POST'])
def create_report():
    """Create a new analysis report"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    
    try:
        report = AnalysisReport(
            user_id=user_id,
            report_name=data.get('report_name'),
            total_samples=data.get('total_samples'),
            total_particles=data.get('total_particles'),
            average_size=data.get('average_size'),
            dominant_polymer=data.get('dominant_polymer'),
            risk_assessment=data.get('risk_assessment'),
        )
        
        db.session.add(report)
        db.session.commit()
        
        return jsonify(report.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/import-sample-data', methods=['POST'])
def import_sample_data():
    """Import sample microplastic data for demonstration"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    try:
        # Clear existing data for user
        Microplastic.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        
        # Sample data
        structures = ['fiber', 'fragment', 'bead', 'film']
        polymers = ['PE', 'PET', 'PP', 'PS', 'PVC', 'LDPE', 'HDPE']
        shapes = ['linear', 'spherical', 'irregular', 'sheet']
        colors = ['transparent', 'white', 'blue', 'red', 'black', 'green', 'yellow']
        transparencies = ['transparent', 'translucent', 'opaque']
        textures = ['smooth', 'rough', 'weathered']
        risk_levels = ['low', 'medium', 'high', 'critical']
        sample_types = ['water', 'soil', 'air', 'food']
        locations = ['River Sample A', 'Ocean Sample B', 'Coastal Area C', 'Freshwater D', 'Agricultural E']
        
        sample_count = 50
        
        for i in range(sample_count):
            structure = random.choice(structures)
            polymer = random.choice(polymers)
            shape = random.choice(shapes)
            
            # Generate realistic sizes based on type
            if structure == 'fiber':
                length = random.uniform(100, 5000)
                width = random.uniform(5, 50)
                thickness = random.uniform(1, 10)
            elif structure == 'bead':
                length = random.uniform(50, 500)
                width = length
                thickness = length
            else:
                length = random.uniform(50, 1000)
                width = random.uniform(20, 500)
                thickness = random.uniform(1, 50)
            
            area = length * width
            volume = length * width * thickness
            aspect_ratio = length / width if width > 0 else 0
            
            mp = Microplastic(
                user_id=user_id,
                sample_id=f'SAMPLE-{i+1:04d}',
                detection_date=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                location=random.choice(locations),
                structure_type=structure,
                polymer_type=polymer,
                shape=shape,
                aspect_ratio=aspect_ratio,
                length=round(length, 2),
                width=round(width, 2),
                thickness=round(thickness, 2),
                area=round(area, 2),
                volume=round(volume, 2),
                color=random.choice(colors),
                density=round(random.uniform(0.9, 1.4), 2),
                transparency=random.choice(transparencies),
                surface_texture=random.choice(textures),
                risk_level=random.choice(risk_levels),
                concentration=round(random.uniform(0.1, 100), 2),
                sample_type=random.choice(sample_types),
                confidence_score=round(random.uniform(70, 100), 1),
            )
            db.session.add(mp)
        
        db.session.commit()
        
        return jsonify({
            'message': f'Successfully imported {sample_count} sample microplastics',
            'count': sample_count
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/export', methods=['GET'])
def export_data():
    """Export microplastics data as JSON"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    microplastics = Microplastic.query.filter_by(user_id=user_id).all()
    return jsonify([mp.to_dict() for mp in microplastics])

@app.route('/api/export/csv', methods=['GET'])
def export_particles_csv():
    """Export microplastic particle data as CSV"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    import csv
    from io import StringIO

    microplastics = Microplastic.query.filter_by(user_id=user_id).all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['date', 'time', 'particle_count', 'particle_shape', 'particle_surface'])

    for mp in microplastics:
        date_str = mp.detection_date.strftime('%Y-%m-%d')
        time_str = mp.detection_date.strftime('%H:%M:%S')

        writer.writerow([
            date_str,
            time_str,
            1,
            mp.shape,
            mp.surface_texture
        ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=particles_export.csv"}
    )

# ==================== WEBCAM & LIVE ANALYSIS ENDPOINTS ====================

@app.route('/api/webcam/start', methods=['POST'])
def start_webcam():
    """Start webcam capture and particle detection"""
    global detector
    
    try:
        with detector_lock:
            if detector is None:
                detector = ParticleDetector(camera_id=0)
            
            # Clear history for a new session
            if hasattr(detector, 'particle_history'):
                detector.particle_history.clear()
            
            if not detector.is_running:
                detector.start_capture()
        
        return jsonify({'message': 'Webcam started', 'status': 'running'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/webcam/stop', methods=['POST'])
def stop_webcam():
    """Stop webcam capture"""
    global detector
    
    try:
        with detector_lock:
            if detector:
                detector.stop_capture()
        
        return jsonify({'message': 'Webcam stopped', 'status': 'stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/webcam/status', methods=['GET'])
def webcam_status():
    """Get webcam status"""
    global detector
    
    with detector_lock:
        if detector is None:
            return jsonify({
                'is_running': False,
                'frame_count': 0,
                'fps': 0,
                'particle_count': 0
            })
        
        return jsonify({
            'is_running': detector.is_running,
            'frame_count': detector.frame_count,
            'fps': round(detector.fps, 1),
            'particle_count': len(detector.particles)
        })

@app.route('/api/webcam/frame', methods=['GET'])
def get_webcam_frame():
    """Get current webcam frame as JPEG"""
    global detector
    
    if detector is None:
        return jsonify({'error': 'Webcam not initialized'}), 400
    
    frame = detector.get_frame_with_annotations()
    if frame is None:
        return jsonify({'error': 'No frame available'}), 400
    
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_bytes = buffer.tobytes()
        
        return Response(
            jpg_as_bytes,
            mimetype='image/jpeg',
            headers={'Content-Type': 'image/jpeg'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/webcam/frame/base64', methods=['GET'])
def get_webcam_frame_base64():
    """Get current frame as base64 for web display"""
    global detector

    if detector is None:
        return jsonify({'error': 'Webcam not initialized'}), 400

    # Wait briefly for first frame
    timeout = 5  # seconds
    start_time = time.time()
    while detector.current_frame is None and (time.time() - start_time < timeout):
        time.sleep(0.1)

    frame = detector.get_frame_with_annotations()
    if frame is None:
        return jsonify({'error': 'No frame available'}), 400

    try:
        jpg_as_text = FrameEncoder.frame_to_base64(frame)
        return jsonify({
            'frame': jpg_as_text,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/export/live/csv', methods=['GET'])
def export_live_csv():
    """Export live detection session as CSV"""
    global detector
    import csv
    from io import StringIO

    if detector is None:
        return jsonify({'error': 'Detector not running'}), 400

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['date', 'time', 'particle_count', 'particle_shape', 'particle_surface'])

    # Loop over history (each frame snapshot)
    for snapshot in detector.particle_history:
        timestamp = snapshot['timestamp']
        count = snapshot['count']

        date_str = "'" + timestamp.strftime('%d-%m-%Y')   # <-- FIXED FORMAT
        time_str = timestamp.strftime('%H:%M:%S')

        # For each particle currently detected in that frame:
        particles = snapshot.get('particles', [])

        for p in particles:
            writer.writerow([
                date_str,
                time_str,
                count,
                p['shape_type'],
                'weathered' if p.get('std_intensity', 0) > 40 else 
                    'rough' if p.get('std_intensity', 0) > 20 else 
                    'smooth'
            ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=live_session_export.csv"}
    )

@app.route('/api/particles/live', methods=['GET'])
def get_live_particles():
    """Get current detected particles"""
    global detector
    
    if detector is None:
        return jsonify({
            'particles': [],
            'count': 0,
            'quantification': None
        })

    particles = detector.get_current_particles()
    quantification = detector.get_quantification()
    
    # Serialize particles (remove contours for JSON compatibility)
    serialized_particles = []
    for p in particles:
        particle_data = {
            'area': float(p['area']),
            'centroid': list(p['centroid']),
            'perimeter': float(p['perimeter']),
            'major_axis': float(p.get('major_axis', 0)),
            'minor_axis': float(p.get('minor_axis', 0)),
            'aspect_ratio': float(p.get('aspect_ratio', 1.0)),
            'angle': float(p.get('angle', 0)),
            'circularity': float(p['circularity']),
            'shape_type': p['shape_type'],
            'convexity': float(p['convexity']),
            'mean_intensity': float(p.get('mean_intensity', 0)),
            'std_intensity': float(p.get('std_intensity', 0)),
        }
        serialized_particles.append(particle_data)
    
    return jsonify({
        'particles': serialized_particles,
        'count': len(serialized_particles),
        'quantification': quantification,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/particles/quantification', methods=['GET'])
def get_particles_quantification():
    """Get quantified particle analysis"""
    global detector
    
    if detector is None:
        return jsonify({
            'count': 0,
            'average_size': 0,
            'quantification': None
        })
    
    quantification = detector.get_quantification()
    
    return jsonify({
        'count': quantification['count'],
        'quantification': quantification,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/particles/history', methods=['GET'])
def get_particles_history():
    """Get particle detection history"""
    global detector
    
    if detector is None or not detector.particle_history:
        return jsonify({'history': []})
    
    history = []
    for entry in list(detector.particle_history):
        history.append({
            'timestamp': entry['timestamp'].isoformat(),
            'count': entry['count']
        })
    
    return jsonify({'history': history})

@app.route('/api/particles/save', methods=['POST'])
def save_detected_particles():
    """Save detected particles to database"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not logged in'}), 401
    global detector
    
    if detector is None or not detector.particles:
        return jsonify({'error': 'No particles detected'}), 400
    
    try:
        data = request.get_json()
        particles = detector.particles
        
        for idx, particle in enumerate(particles):
            # Classify as microplastic
            structure_type = 'fragment'  # default
            if particle['shape_type'] == 'fiber':
                structure_type = 'fiber'
            elif particle['shape_type'] == 'bead':
                structure_type = 'bead'
            elif particle['shape_type'] == 'film':
                structure_type = 'film'
            
            # Determine risk level based on size and circularity
            size = particle['area']
            if size > 5000 or particle['circularity'] < 0.3:
                risk_level = 'critical'
            elif size > 2000 or particle['circularity'] < 0.5:
                risk_level = 'high'
            elif size > 500:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            mp = Microplastic(
                user_id=user_id,
                sample_id=f"LIVE-{detector.frame_count:06d}-{idx:03d}",
                location=data.get('location', 'Live Webcam'),
                structure_type=structure_type,
                polymer_type=data.get('polymer_type', 'Unknown'),
                shape=particle['shape_type'],
                aspect_ratio=particle.get('aspect_ratio', 1.0),
                length=particle['major_axis'],
                width=particle['minor_axis'],
                thickness=particle['major_axis'] * 0.5,  # estimate
                area=particle['area'],
                volume=particle['area'] * particle['major_axis'] * 0.25,  # estimate
                color=data.get('color', 'N/A'),
                density=data.get('density', 1.0),
                transparency=data.get('transparency', 'Unknown'),
                surface_texture='rough' if particle.get('std_intensity', 0) > 30 else 'smooth',
                risk_level=risk_level,
                concentration=float(len(particles)) / 640 / 480 * 1000000,
                sample_type=data.get('sample_type', 'live_analysis'),
                confidence_score=min(particle['circularity'] * 100, 100),
            )
            db.session.add(mp)
        
        db.session.commit()
        
        return jsonify({
            'message': f'Saved {len(particles)} particles',
            'count': len(particles)
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/particles/statistics', methods=['GET'])
def get_particles_statistics():
    """Get current particle statistics"""
    global detector
    
    if detector is None:
        return jsonify({'stats': None})
    
    stats = detector.get_statistics()
    
    # Convert to JSON-serializable format
    serialized_stats = {
        'frame_count': stats['frame_count'],
        'fps': round(stats['fps'], 1),
        'current_particle_count': stats['current_particle_count'],
        'quantification': stats['quantification'],
        'is_running': stats['is_running'],
    }
    
    return jsonify(serialized_stats)

# ==================== INITIALIZATION ====================

if __name__ == "__main__":
    import os
    with app.app_context():
        db.create_all()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)