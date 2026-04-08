"""Flask API server for Hospital Triage Environment."""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any
import logging
import os
import sys

# Ensure the app directory is explicitly in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import hospital_triage  # Must import to register the environment
except ModuleNotFoundError as e:
    logging.error(f"ModuleNotFoundError: {e}")
    logging.error(f"Directory Contents: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")
    logging.error("CRITICAL: If 'hospital_triage' is not listed above, you forgot to upload the hospital_triage directory to Hugging Face Spaces!")
    raise

from hospital_triage.ml import get_predictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global environment instance
env = None
episode_reward = 0
step_count = 0

# Stats tracking for graphs
stats_history = {
    'timestamps': [],
    'patients_in_queue': [],
    'patients_admitted': [],
    'patients_discharged': [],
    'average_wait_time': [],
    'bed_occupancy': [],
    'doctor_workload': [],
    'arriving_patients': []
}
last_admitted_count = 0
last_discharged_count = 0

# Notification tracking
notifications = []
last_available_beds = 0
last_available_doctors = 0
last_emergency_count = 0

# Initialize ML predictor for patient priority
try:
    predictor = get_predictor(model_type='logistic')
    logger.info("Patient priority predictor initialized")
except Exception as e:
    logger.warning(f"Could not initialize ML predictor: {e}. ML features may be unavailable.")
    predictor = None


def init_environment(
    num_doctors: int = 10,
    num_beds: int = 20,
    num_lab_tests: int = 5,
    patient_arrival_rate: float = 0.7,
):
    """Initialize the hospital environment."""
    global env, episode_reward, step_count
    try:
        env = gym.make(
            'HospitalTriage-v0',
            num_doctors=num_doctors,
            num_beds=num_beds,
            num_lab_tests=num_lab_tests,
            patient_arrival_rate=patient_arrival_rate,
        )
        env.reset()
        episode_reward = 0
        step_count = 0
        logger.info("Environment initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return False


def get_unwrapped_state():
    """Get the unwrapped environment state."""
    if env is None:
        return None
    return env.unwrapped.state


def clean_for_json(obj):
    """Recursively convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif hasattr(obj, 'item') and callable(obj.item):  # NumPy scalars
        return obj.item()
    elif isinstance(obj, (np.ndarray, np.generic)):   # NumPy arrays or generics
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    return obj


def get_hospital_state():
    """Extract and format current hospital state for the dashboard."""
    global env, episode_reward, step_count
    
    if env is None:
        return {}
        
    state = get_unwrapped_state()
    if state is None:
        return {}
    
    # Categorize patients
    waiting_patients = [p for p in state.patients_queue if p]
    admitted_patients = [p for p in state.admitted_patients if p and p.status.name in ['ASSIGNED', 'IN_BED', 'IN_TEST']]
    discharged_patients = [p for p in state.discharged_patients if p and p.status.name == 'DISCHARGED']
    
    available_beds = sum(1 for bed in state.beds if bed and not bed.is_occupied)
    available_doctors = sum(1 for doc in state.doctors if doc and not doc.is_busy)
    
    avg_wait_time = (
        np.mean([p.waiting_time for p in waiting_patients])
        if waiting_patients else 0
    )
    
    state_data = {
        "total_patients": len(state.patients),
        "waiting_queue_length": len(waiting_patients),
        "admitted_count": len(admitted_patients),
        "discharged_count": len(discharged_patients),
        "available_beds": int(available_beds),
        "total_beds": int(env.unwrapped.num_beds),
        "available_doctors": int(available_doctors),
        "total_doctors": int(env.unwrapped.num_doctors),
        "average_wait_time": float(avg_wait_time),
        "current_time": int(state.current_time),
        "step_count": int(step_count),
        "episode_reward": float(episode_reward),
        "waiting_patients": [
            {
                "id": int(p.id),
                "severity": p.severity.name,
                "arrival_time": int(p.arrival_time),
                "waiting_time": int(p.waiting_time),
            }
            for p in waiting_patients[:10]  # Top 10 waiting
        ],
        "doctors_status": [
            {
                "id": int(d.id),
                "specialty": d.specialty.name,
                "is_busy": bool(d.is_busy),
                "current_patient_id": int(d.current_patient_id) if d.current_patient_id is not None else None,
                "patients_assigned": int(len(d.patients_assigned)),
            }
            for d in state.doctors
        ],
        "beds_status": [
            {
                "id": int(b.id),
                "is_occupied": bool(b.is_occupied),
                "priority_level": int(b.priority_level),
                "current_patient_id": int(b.current_patient_id) if b.current_patient_id is not None else None,
                "patient_id": int(b.current_patient_id) if b.current_patient_id is not None else None,
            }
            for b in state.beds
        ],
        "admitted_patients": [
            {
                "id": int(p.id),
                "severity": p.severity.name,
                "status": p.status.name,
                "arrival_time": int(p.arrival_time),
                "waiting_time": int(p.waiting_time),
                "stay_duration": int(state.current_time - p.arrival_time),
                "required_duration": int(p.duration),
                "can_discharge": bool((state.current_time - p.arrival_time) >= p.duration)
            }
            for p in admitted_patients
        ],
        "discharged_patients": [
            {
                "id": int(p.id),
                "severity": p.severity.name,
                "status": p.status.name,
                "arrival_time": int(p.arrival_time),
                "waiting_time": int(p.waiting_time),
                "stay_duration": int(state.current_time - p.arrival_time),
                "required_duration": int(p.duration)
            }
            for p in discharged_patients[-10:]  # Last 10 discharges
        ],
    }
    return clean_for_json(state_data)


def update_stats_history():
    """Update the stats history for graph visualization."""
    global stats_history, last_admitted_count, last_discharged_count
    
    if env is None:
        return
    
    try:
        state = get_unwrapped_state()
        if state is None:
            return
        
        waiting_patients = [p for p in state.patients_queue if p]
        admitted_patients = [p for p in state.admitted_patients if p and p.status.name in ['ASSIGNED', 'IN_BED', 'IN_TEST']]
        discharged_patients = [p for p in state.discharged_patients if p and p.status.name == 'DISCHARGED']
        
        avg_wait_time = (
            np.mean([p.waiting_time for p in waiting_patients])
            if waiting_patients else 0
        )
        
        occupied_beds = sum(1 for bed in state.beds if bed.is_occupied)
        bed_occupancy = (occupied_beds / env.unwrapped.num_beds * 100) if env.unwrapped.num_beds > 0 else 0
        
        avg_doctor_workload = (
            np.mean([d.workload for d in state.doctors])
            if state.doctors else 0
        )
        
        arriving = len(state.patients) - (last_admitted_count + last_discharged_count + len(waiting_patients))
        if arriving < 0: arriving = 0
        
        # Keep only last 100 data points
        max_points = 100
        if len(stats_history['timestamps']) >= max_points:
            for key in stats_history:
                stats_history[key] = stats_history[key][-max_points+1:]
        
        stats_history['timestamps'].append(step_count)
        stats_history['patients_in_queue'].append(len(waiting_patients))
        stats_history['patients_admitted'].append(len(admitted_patients))
        stats_history['patients_discharged'].append(len(discharged_patients))
        stats_history['average_wait_time'].append(float(avg_wait_time))
        stats_history['bed_occupancy'].append(float(bed_occupancy))
        stats_history['doctor_workload'].append(float(avg_doctor_workload))
        stats_history['arriving_patients'].append(arriving)
        
    except Exception as e:
        logger.error(f"Error updating stats history: {e}")


def update_notifications():
    """Generate notifications based on hospital state changes."""
    global notifications, last_available_beds, last_available_doctors, last_emergency_count
    
    if env is None:
        return
    
    try:
        from datetime import datetime
        state = get_unwrapped_state()
        if state is None: return
        
        current_time = datetime.now().isoformat()
        
        # Bed availability
        curr_beds = sum(1 for bed in state.beds if not bed.is_occupied)
        if curr_beds > last_available_beds:
            notifications.append({
                'id': len(notifications),
                'type': 'bed_available',
                'title': '🛏️ Bed Available',
                'message': f'{curr_beds - last_available_beds} bed(s) freed',
                'timestamp': current_time, 'priority': 'medium', 'read': False
            })
        last_available_beds = curr_beds
        
        # Doctor availability
        curr_docs = sum(1 for doc in state.doctors if not doc.is_busy)
        if curr_docs > last_available_doctors:
            notifications.append({
                'id': len(notifications),
                'type': 'doctor_free',
                'title': '👨‍⚕️ Doctor Available',
                'message': f'{curr_docs - last_available_doctors} doctor(s) freed',
                'timestamp': current_time, 'priority': 'medium', 'read': False
            })
        last_available_doctors = curr_docs
        
        # Emergency patients
        waiting = [p for p in state.patients_queue if p]
        curr_emerg = sum(1 for p in waiting if p.severity.value >= 2)
        if curr_emerg > last_emergency_count:
            p = next(p for p in waiting if p.severity.value >= 2)
            notifications.append({
                'id': len(notifications),
                'type': 'emergency_arrived',
                'title': '🚨 Emergency Patient',
                'message': f'Patient #{p.id} ({p.severity.name}) arrived',
                'timestamp': current_time, 'priority': 'high', 'read': False, 'patient_id': p.id
            })
        last_emergency_count = curr_emerg
        
        # Keep only last 50
        if len(notifications) > 50: notifications = notifications[-50:]
        
    except Exception as e:
        logger.error(f"Error updating notifications: {e}")


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def dashboard():
    """Serve the dashboard HTML."""
    return render_template('index.html')


@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize the environment."""
    try:
        data = request.json or {}
        success = init_environment(
            num_doctors=data.get('num_doctors', 10),
            num_beds=data.get('num_beds', 20),
            num_lab_tests=data.get('num_lab_tests', 5),
            patient_arrival_rate=data.get('patient_arrival_rate', 0.7),
        )
        if success:
            return jsonify({"success": True, "state": get_hospital_state()}), 200
        return jsonify({"success": False, "error": "Failed to initialize"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/state', methods=['GET'])
def api_state():
    """Get current hospital state."""
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        update_stats_history()
        update_notifications()
        return jsonify(get_hospital_state()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/step/auto-admit', methods=['POST'])
def api_step_auto_admit():
    """Consolidated Auto-AI step: natural step + automatic admissions."""
    global env, episode_reward, step_count
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        
        # 1. Advance simulation (natural arrival/time passage)
        _, reward, terminated, truncated, _ = env.step(np.array([4, 0, 0, 0], dtype=np.int32))
        episode_reward += reward
        step_count += 1
        
        admitted_in_cycle = 0
        if not (terminated or truncated):
            # 2. Re-scan queue for possible admissions
            state = get_unwrapped_state()
            # Loop a copy of the queue patients to avoid index issues
            waiting = [p for p in state.patients_queue if p]
            
            for p in waiting:
                d_id = find_best_doctor(p.id)
                b_id = find_best_bed(p.id)
                
                if d_id is not None and b_id is not None:
                    # Admit this specific patient
                    idx = get_patient_index(p.id, 'queue')
                    if idx is not None:
                        action = np.array([0, idx, d_id, b_id], dtype=np.int32)
                        _, a_reward, t, tr, _ = env.step(action)
                        episode_reward += a_reward
                        step_count += 1
                        admitted_in_cycle += 1
                        if t or tr: break
                # Continue scan for next patient in case they need a different specialty
                
        if terminated or truncated or 't' in locals() and (t or tr):
            env.reset()
            episode_reward = 0
            step_count = 0
            
        final_state = get_hospital_state()
        final_state.update({"auto_admitted": admitted_in_cycle})
        return jsonify(final_state), 200
    except Exception as e:
        logger.error(f"Error in auto-admit step: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/step', methods=['POST'])
@app.route('/openenv/step', methods=['POST'])
@app.route('/step', methods=['POST'])
def api_step():
    """Execute step (Auto-Admit if no action)."""
    global env, episode_reward, step_count
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        
        data = request.json or {}
        action = data.get('action')
        
        if action is None:
            reward_sum = 0.0
            admissions_made = 0
            while True:
                state = get_unwrapped_state()
                waiting = [p for p in state.patients_queue if p]
                if not waiting: break
                
                target = max(waiting, key=lambda p: p.severity.value)
                p_idx = get_patient_index(target.id, 'queue')
                if p_idx is None: break
                
                d_id = find_best_doctor(target.id)
                b_id = find_best_bed(target.id)
                
                if d_id is not None and b_id is not None:
                    step_action = np.array([0, p_idx, d_id, b_id], dtype=np.int32)
                    _, reward, terminated, truncated, _ = env.step(step_action)
                    episode_reward += reward
                    reward_sum += reward
                    admissions_made += 1
                    if terminated or truncated: break
                else: break
            
            if admissions_made == 0:
                _, reward, terminated, truncated, _ = env.step(np.array([1, 0, 0, 0], dtype=np.int32))
                episode_reward += reward
                reward_sum += reward
            
            step_count += 1
        else:
            # Manual action
            if isinstance(action, list): action = np.array(action, dtype=np.int32)
            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            reward_sum = reward
            
        if terminated or truncated:
            env.reset()
            episode_reward = 0
            step_count = 0
            
        state_data = get_hospital_state()
        state_data.update({"reward": float(reward_sum), "terminated": bool(terminated), "truncated": bool(truncated)})
        return jsonify(state_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/action/add-patient', methods=['POST'])
def api_add_patient():
    """Force generate a patient."""
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        unwrapped = env.unwrapped
        state = unwrapped.state
        patient = unwrapped.patient_generator.generate_patient(current_time=state.current_time, arrival_rate=1.0)
        if patient: state.patients_queue.append(patient)
        return jsonify({"success": True, "state": get_hospital_state(), "patient_id": patient.id if patient else None}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/action/admit-patient', methods=['POST'])
def api_admit_patient():
    """Manual admission."""
    global env, episode_reward, step_count
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        data = request.json or {}
        p_id = data.get('patient_id')
        p_idx = get_patient_index(p_id, 'queue')
        if p_idx is None: return jsonify({"error": "Patient not found"}), 404
        
        action = np.array([0, p_idx, data.get('doctor_id', 0), data.get('bed_id', 0)], dtype=np.int32)
        _, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        if terminated or truncated:
            env.reset()
            episode_reward = 0
            step_count = 0
            
        state_data = get_hospital_state()
        state_data.update({"reward": float(reward), "action_info": info})
        return jsonify({"success": True, "state": state_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/action/discharge-patient', methods=['POST'])
def api_discharge_patient():
    """Manual discharge."""
    global env, episode_reward, step_count
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        data = request.json or {}
        p_id = data.get('patient_id')
        p_idx = get_patient_index(p_id, 'admitted')
        if p_idx is None: return jsonify({"error": "Patient not found"}), 404
        
        action = np.array([5, p_idx, 0, 0], dtype=np.int32)
        _, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        if terminated or truncated:
            env.reset()
            episode_reward = 0
            step_count = 0
            
        state_data = get_hospital_state()
        state_data.update({"reward": float(reward), "action_info": info})
        return jsonify({"success": True, "state": state_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/action/bulk-discharge', methods=['POST'])
def api_bulk_discharge():
    """Bulk manual discharge with enhanced logging."""
    global env, episode_reward, step_count
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        data = request.json or {}
        p_ids = data.get('patient_ids', [])
        
        logger.info(f"Starting bulk discharge for patients: {p_ids}")
        results = []
        total_reward = 0.0
        
        for p_id in p_ids:
            # Re-find index every time since the list size changes after each step
            p_idx = get_patient_index(p_id, 'admitted')
            if p_idx is not None:
                action = np.array([5, p_idx, 0, 0], dtype=np.int32)
                _, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                total_reward += reward
                step_count += 1
                logger.info(f"Discharged patient #{p_id} at index {p_idx}. Reward: {reward}")
                results.append({"patient_id": p_id, "success": True, "reward": float(reward)})
                if terminated or truncated:
                    logger.info("Simulation terminated during bulk discharge")
                    env.reset()
                    episode_reward = 0
                    step_count = 0
                    break
            else:
                logger.warning(f"Failed to find patient #{p_id} in admitted list for discharge")
                results.append({"patient_id": p_id, "success": False, "error": "Not found or already discharged"})
        
        state_data = get_hospital_state()
        state_data.update({"reward": float(total_reward), "results": results})
        return jsonify({"success": True, "state": state_data}), 200
    except Exception as e:
        logger.error(f"Critical error in bulk discharge: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/reset', methods=['POST'])
@app.route('/openenv/reset', methods=['POST'])
@app.route('/reset', methods=['POST'])
def api_reset():
    """Reset simulation."""
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        data = request.json or {}
        unwrapped = env.unwrapped
        success = init_environment(
            num_doctors=data.get('num_doctors', unwrapped.num_doctors),
            num_beds=data.get('num_beds', unwrapped.num_beds),
            num_lab_tests=data.get('num_lab_tests', unwrapped.num_lab_tests),
            patient_arrival_rate=data.get('patient_arrival_rate', unwrapped.patient_arrival_rate),
        )
        if success: return jsonify({"success": True, "state": get_hospital_state()}), 200
        return jsonify({"success": False, "error": "Failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# HELPERS & SCHEDULING
# ============================================================================

def get_patient_index(patient_id, location='queue'):
    state = get_unwrapped_state()
    if state is None: return None
    target_list = state.patients_queue if location == 'queue' else state.admitted_patients
    for i, p in enumerate(target_list):
        if p and p.id == patient_id: return i
    return None

def find_best_doctor(patient_id: int):
    state = get_unwrapped_state()
    if state is None: return None
    patient = next((p for p in state.patients if p.id == patient_id), None)
    if not patient: return None
    available = [d for d in state.doctors if not d.is_busy]
    if not available: return None
    best = max(available, key=lambda d: (d.specialty == patient.required_specialty) * 100 - d.workload)
    return best.id

def find_best_bed(patient_id: int):
    state = get_unwrapped_state()
    if state is None: return None
    patient = next((p for p in state.patients if p.id == patient_id), None)
    if not patient: return None
    available = [b for b in state.beds if not b.is_occupied]
    if not available: return None
    if patient.severity.value >= 2: return max(available, key=lambda b: b.priority_level).id
    return available[0].id


@app.route('/api/schedule/smart-allocate', methods=['POST'])
def api_smart_allocate():
    """Smart allocation logic."""
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        data = request.json or {}
        p_id = data.get('patient_id')
        if not p_id: return jsonify({"error": "patient_id required"}), 400
        
        state = get_unwrapped_state()
        patient = next((p for p in state.patients if p.id == p_id), None)
        if not patient: return jsonify({"error": "Patient not found"}), 404
        
        d_id = find_best_doctor(p_id)
        b_id = find_best_bed(p_id)
        
        return jsonify(clean_for_json({
            "patient_id": p_id,
            "is_emergency": patient.severity.value >= 2,
            "recommended_doctor": d_id,
            "recommended_bed": b_id,
            "can_allocate": d_id is not None and b_id is not None
        })), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/schedule/status', methods=['GET'])
def api_schedule_status():
    """Get summarized status for the scheduling panel."""
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        state = get_unwrapped_state()
        waiting = [p for p in state.patients_queue if p]
        
        return jsonify(clean_for_json({
            "success": True,
            "waiting": {
                "total": len(waiting),
                "emergency": sum(1 for p in waiting if p.severity.value >= 2),
                "patients": [
                    {
                        "id": p.id,
                        "severity": p.severity.name,
                        "waiting_time": int(p.waiting_time)
                    } for p in waiting[:15]
                ]
            },
            "resources": {
                "available_doctors": sum(1 for d in state.doctors if not d.is_busy),
                "total_doctors": len(state.doctors),
                "available_beds": sum(1 for b in state.beds if not b.is_occupied),
                "total_beds": len(state.beds)
            }
        })), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ml/rank-queue', methods=['POST'])
def api_ml_rank_queue():
    """Rank the current waiting queue using the ML predictor."""
    try:
        if env is None: return jsonify({"error": "Init required"}), 400
        if predictor is None: return jsonify({"error": "ML predictor not initialized"}), 503
        
        state = get_unwrapped_state()
        waiting = [p for p in state.patients_queue if p]
        
        ranked_patients = []
        for p in waiting:
            # Prepare feature vector for prediction
            # The predictor expects [age, severity, waiting_time]
            # Since patient age isn't explicitly in the current Patient class, we'll use a derived value or default
            # Based on view_file of hospital_triage_env.py (assumed from context), 
            # let's use p.severity.value and p.waiting_time
            age = getattr(p, 'age', 45) # Fallback to 45 if not present
            
            pred = predictor.predict_priority(age, p.severity.value, p.waiting_time)
            
            ranked_patients.append({
                "patient_id": p.id,
                "age": age,
                "severity": p.severity.name,
                "waiting_time": int(p.waiting_time),
                "predicted_priority": pred['priority'],
                "confidence": pred['confidence'],
                "reasoning": pred['reasoning']
            })
            
        # Sort by confidence/priority (High > Medium > Low)
        prio_map = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
        ranked_patients.sort(key=lambda x: (prio_map.get(x['predicted_priority'], 0), x['confidence']), reverse=True)
        
        return jsonify({"success": True, "ranked_patients": ranked_patients}), 200
    except Exception as e:
        logger.error(f"Error in rank-queue: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/ml/predict-priority', methods=['POST'])
def api_ml_predict_priority():
    """Predict priority for a hypothetical patient."""
    try:
        if predictor is None: return jsonify({"error": "ML predictor not initialized"}), 503
        data = request.json or {}
        
        age = data.get('age', 45)
        severity = data.get('severity', 1)
        waiting_time = data.get('waiting_time', 0)
        
        pred = predictor.predict_priority(age, severity, waiting_time)
        return jsonify(clean_for_json(pred)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/notifications/mark-read', methods=['POST'])
def api_mark_notification_read():
    """Mark a specific notification as read."""
    global notifications
    try:
        data = request.json or {}
        notif_id = data.get('notification_id')
        for n in notifications:
            if n['id'] == notif_id:
                n['read'] = True
                break
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/historical', methods=['GET'])
def api_stats_historical():
    """Get history for graphs."""
    try:
        update_stats_history()
        return jsonify({"success": True, "data": stats_history, "current_step": step_count}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/notifications', methods=['GET'])
def api_notifications():
    """Get notifications with total unread count."""
    try:
        update_notifications()
        unread_count = sum(1 for n in notifications if not n.get('read', False))
        return jsonify({
            "success": True, 
            "notifications": notifications[::-1][:50],
            "total_unread": unread_count
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/summary', methods=['GET'])
def api_stats_summary():
    """Get summary statistics for dashboard cards."""
    try:
        update_stats_history()
        state = get_unwrapped_state()
        if state is None: return jsonify({"error": "Init required"}), 400
        
        waiting_patients = [p for p in state.patients_queue if p]
        admitted_patients = [p for p in state.admitted_patients if p and p.status.name in ['ASSIGNED', 'IN_BED', 'IN_TEST']]
        discharged_patients = [p for p in state.discharged_patients if p and p.status.name == 'DISCHARGED']
        
        avg_wait = np.mean([p.waiting_time for p in waiting_patients]) if waiting_patients else 0
        max_wait = np.max([p.waiting_time for p in waiting_patients]) if waiting_patients else 0
        
        occupied_beds = sum(1 for b in state.beds if b.is_occupied)
        busy_doctors = sum(1 for d in state.doctors if d.is_busy)
        
        summary = {
            "waiting_time": {
                "average": float(avg_wait),
                "maximum": float(max_wait)
            },
            "bed_usage": {
                "occupied": int(occupied_beds),
                "total": int(env.unwrapped.num_beds),
                "percentage": float((occupied_beds / env.unwrapped.num_beds) * 100) if env.unwrapped.num_beds > 0 else 0
            },
            "doctor_utilization": {
                "busy": int(busy_doctors),
                "total": int(env.unwrapped.num_doctors),
                "percentage": float((busy_doctors / env.unwrapped.num_doctors) * 100) if env.unwrapped.num_doctors > 0 else 0
            },
            "patients_waiting": int(len(waiting_patients)),
            "patients_admitted": int(len(admitted_patients)),
            "patients_discharged": int(len(discharged_patients)),
            "episode": {
                "steps": int(step_count),
                "reward": float(episode_reward)
            }
        }
        return jsonify({"success": True, "summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import os
    # Initialize with default settings
    init_environment()
    
    # Hugging Face Spaces default to port 7860
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 7860))
    is_prod = os.environ.get('ENV') == 'production'
    
    app.run(debug=not is_prod, host=host, port=port)
