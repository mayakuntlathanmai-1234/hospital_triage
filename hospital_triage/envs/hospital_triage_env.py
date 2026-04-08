"""Main Hospital Triage Environment implementing Gymnasium interface."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import asdict

from hospital_triage.envs.data_structures import (
    Patient, Doctor, Bed, LabTest, HospitalState,
    SeverityLevel, DoctorSpecialty, PatientStatus
)
from hospital_triage.envs.patient_generator import PatientGenerator


class HospitalTriageEnv(gym.Env):
    """
    Hospital Triage and Resource Allocation Environment.
    
    This environment simulates a hospital's daily operations where an agent
    must make decisions about patient triage, bed allocation, doctor assignment,
    and test scheduling.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}
    
    def __init__(
        self,
        num_doctors: int = 10,
        num_beds: int = 20,
        num_lab_tests: int = 5,
        max_episode_length: int = 1000,
        patient_arrival_rate: float = 0.7,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Hospital Triage Environment.
        
        Args:
            num_doctors: Number of doctors available
            num_beds: Number of hospital beds
            num_lab_tests: Number of lab test slots available
            max_episode_length: Maximum steps per episode
            patient_arrival_rate: Probability of patient arrival each timestep
            seed: Random seed
            render_mode: Rendering mode ('human' or 'ansi')
        """
        super().__init__()
        
        self.num_doctors = num_doctors
        self.num_beds = num_beds
        self.num_lab_tests = num_lab_tests
        self.max_episode_length = max_episode_length
        self.patient_arrival_rate = patient_arrival_rate
        self.render_mode = render_mode
        
        self.rng = np.random.RandomState(seed)
        self.patient_generator = PatientGenerator(seed=seed)
        
        # Initialize state
        self.current_step = 0
        self.state = HospitalState(current_time=0)
        self._initialize_hospital()
        
        # Define action and observation spaces
        self._setup_spaces()
    
    def _initialize_hospital(self):
        """Initialize hospital resources."""
        # Create doctors
        self.state.doctors = [
            Doctor(id=i, specialty=DoctorSpecialty(i % 5))
            for i in range(self.num_doctors)
        ]
        
        # Create beds
        self.state.beds = [
            Bed(id=i, priority_level=(i // 4))  # Higher priority for early beds
            for i in range(self.num_beds)
        ]
        
        # Create lab test slots
        self.state.lab_tests = [
            LabTest(id=i, patient_id=-1, duration=5)
            for i in range(self.num_lab_tests)
        ]
    
    def _setup_spaces(self):
        """Define action and observation spaces."""
        # Action space: [action_type, target_patient_id, target_doctor_id, target_bed_id]
        # action_type: 0=admit, 1=delay, 2=assign_doctor, 3=allocate_bed, 4=schedule_test, 5=discharge
        self.action_space = spaces.MultiDiscrete([
            6,  # action type
            self.max_queue_size() + 1,  # patient selection
            self.num_doctors + 1,  # doctor selection
            self.num_beds + 1,  # bed selection
        ])
        
        # Observation space: [queue_state, doctor_state, bed_state, time_info]
        obs_size = (
            self.max_queue_size() * 4 +  # patient severity, duration, specialty, wait time
            self.num_doctors * 3 +  # doctor availability, specialty, workload
            self.num_beds * 2 +  # bed availability, priority
            5  # time info
        )
        
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(obs_size,), dtype=np.float32
        )
    
    def max_queue_size(self) -> int:
        """Maximum queue size for observation space."""
        return self.num_beds + 10  # Allow some overflow
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            self.rng.seed(seed)
            self.patient_generator = PatientGenerator(seed=seed)
        
        self.current_step = 0
        self.state = HospitalState(current_time=0)
        self._initialize_hospital()
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [action_type, patient_id, doctor_id, bed_id]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        self.state.current_time = self.current_step
        
        # Unpack action
        action_type = int(action[0]) % 6
        patient_idx = int(action[1]) % max(1, len(self.state.patients_queue) + 1)
        doctor_idx = int(action[2]) % (self.num_doctors + 1)
        bed_idx = int(action[3]) % (self.num_beds + 1)
        
        # Generate new patients
        self._generate_arrivals()
        
        # Execute action and compute reward
        reward = self._execute_action(
            action_type, patient_idx, doctor_idx, bed_idx
        )
        
        # Update environment
        self._update_patients()
        self._update_doctors()
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_episode_length
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'step': self.current_step,
            'queue_length': len(self.state.patients_queue),
            'admitted_count': len(self.state.admitted_patients),
            'avg_waiting_time': self._compute_avg_waiting_time(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _generate_arrivals(self):
        """Generate new patient arrivals."""
        new_patient = self.patient_generator.generate_patient(
            self.current_step, self.patient_arrival_rate
        )
        if new_patient is not None:
            self.state.patients_queue.append(new_patient)
    
    def _execute_action(
        self,
        action_type: int,
        patient_idx: int,
        doctor_idx: int,
        bed_idx: int
    ) -> float:
        """Execute action and return reward."""
        reward = 0.0
        
        if action_type == 0:  # Admit patient
            reward = self._admit_patient(patient_idx, doctor_idx, bed_idx)
        
        elif action_type == 1:  # Delay patient
            reward = self._delay_patient(patient_idx)
        
        elif action_type == 2:  # Assign doctor
            reward = self._assign_doctor(patient_idx, doctor_idx)
        
        elif action_type == 3:  # Allocate bed
            reward = self._allocate_bed(patient_idx, bed_idx)
        
        elif action_type == 4:  # Schedule test
            reward = self._schedule_test(patient_idx)
        
        elif action_type == 5:  # Discharge patient
            reward = self._discharge_patient(patient_idx)
        
        return reward
    
    def _admit_patient(
        self, patient_idx: int, doctor_idx: int, bed_idx: int
    ) -> float:
        """Admit a patient to the hospital."""
        reward = 0.0
        
        if patient_idx >= len(self.state.patients_queue):
            return -5.0  # Invalid action
        
        patient = self.state.patients_queue[patient_idx]
        
        # Check doctor availability
        if doctor_idx >= len(self.state.doctors):
            return -5.0
        
        doctor = self.state.doctors[doctor_idx]
        if not doctor.available:
            return -3.0
        
        # Check bed availability
        if bed_idx >= len(self.state.beds):
            return -5.0
        
        bed = self.state.beds[bed_idx]
        if not bed.available:
            return -2.0
        
        # Check doctor specialty match
        if doctor.specialty != patient.required_specialty:
            self.state.mismatched_assignments += 1
            reward -= 2.0
        
        # Update assignments
        patient.status = PatientStatus.IN_BED
        patient.doctor_id = doctor.id
        patient.bed_id = bed.id
        
        doctor.available = False
        doctor.current_patient_id = patient.id
        doctor.workload += 1
        
        bed.available = False
        bed.current_patient_id = patient.id
        
        # Remove from queue and add to admitted
        self.state.patients_queue.pop(patient_idx)
        self.state.admitted_patients.append(patient)
        
        # Reward based on severity (higher severity = higher reward)
        reward += 1.0 + (patient.severity * 2.0)
        
        # Penalty for making critical patient wait
        if patient.severity == SeverityLevel.CRITICAL:
            wait_penalty = min(patient.waiting_time * 0.1, 5.0)
            reward -= wait_penalty
        
        return reward
    
    def _delay_patient(self, patient_idx: int) -> float:
        """Delay a non-urgent patient (keep in queue)."""
        if patient_idx >= len(self.state.patients_queue):
            return -5.0
        
        patient = self.state.patients_queue[patient_idx]
        
        # Penalize delaying critical patients
        if patient.severity == SeverityLevel.CRITICAL:
            return -10.0
        
        # Small penalty for delaying
        reward = -0.5
        patient.waiting_time += 1
        
        return reward
    
    def _assign_doctor(self, patient_idx: int, doctor_idx: int) -> float:
        """Assign a doctor to a patient."""
        if (patient_idx >= len(self.state.admitted_patients) or
            doctor_idx >= len(self.state.doctors)):
            return -5.0
        
        patient = self.state.admitted_patients[patient_idx]
        doctor = self.state.doctors[doctor_idx]
        
        if not doctor.available:
            return -2.0
        
        reward = 0.0
        if doctor.specialty == patient.required_specialty:
            reward = 1.0
        else:
            reward = -1.0
        
        patient.doctor_id = doctor.id
        doctor.current_patient_id = patient.id
        doctor.workload += 1
        
        return reward
    
    def _allocate_bed(self, patient_idx: int, bed_idx: int) -> float:
        """Allocate a bed to a patient."""
        if (patient_idx >= len(self.state.patients_queue) or
            bed_idx >= len(self.state.beds)):
            return -5.0
        
        patient = self.state.patients_queue[patient_idx]
        bed = self.state.beds[bed_idx]
        
        if not bed.available:
            return -3.0
        
        reward = 0.0
        
        # Reward allocating better beds to critical patients
        if patient.severity == SeverityLevel.CRITICAL and bed.priority_level > 0:
            reward = 2.0
        
        patient.bed_id = bed.id
        bed.available = False
        bed.current_patient_id = patient.id
        
        return reward
    
    def _schedule_test(self, patient_idx: int) -> float:
        """Schedule a lab test for a patient."""
        if patient_idx >= len(self.state.admitted_patients):
            return -5.0
        
        patient = self.state.admitted_patients[patient_idx]
        
        # Find available test slot
        available_test = None
        for test in self.state.lab_tests:
            if test.scheduled_time is None:
                available_test = test
                break
        
        if available_test is None:
            return -2.0  # No test slots available
        
        available_test.patient_id = patient.id
        available_test.scheduled_time = self.current_step + 5
        patient.test_scheduled = True
        patient.test_time = available_test.scheduled_time
        
        return 0.5
    
    def _discharge_patient(self, patient_idx: int) -> float:
        """Discharge a patient to free up resources."""
        if patient_idx >= len(self.state.admitted_patients):
            return -5.0
        
        patient = self.state.admitted_patients[patient_idx]
        
        # Can only discharge if treatment is done
        time_elapsed = self.current_step - patient.arrival_time
        if time_elapsed < patient.duration:
            return -3.0
        
        reward = 1.5
        
        patient.status = PatientStatus.DISCHARGED
        
        # Free up doctor
        if patient.doctor_id is not None:
            for doctor in self.state.doctors:
                if doctor.id == patient.doctor_id:
                    doctor.available = True
                    doctor.current_patient_id = None
                    doctor.workload = max(0, doctor.workload - 1)
        
        # Free up bed
        if patient.bed_id is not None:
            for bed in self.state.beds:
                if bed.id == patient.bed_id:
                    bed.available = True
                    bed.current_patient_id = None
                    bed.last_occupant_discharge_time = self.current_step
        
        self.state.admitted_patients.remove(patient)
        self.state.discharged_patients.append(patient)
        self.state.total_patients_processed += 1
        self.state.total_waiting_time += patient.waiting_time
        
        return reward
    
    def _update_patients(self):
        """Update patient states."""
        for patient in self.state.patients_queue:
            patient.waiting_time += 1
            
            # Penalty for waiting critical patients
            if patient.severity == SeverityLevel.CRITICAL:
                # This is tracked in step reward
                pass
        
        # Update lab tests
        for test in self.state.lab_tests:
            if (test.scheduled_time is not None and
                self.current_step >= test.scheduled_time and
                not test.completed):
                test.completed = True
    
    def _update_doctors(self):
        """Update doctor states."""
        for doctor in self.state.doctors:
            if doctor.current_patient_id is not None:
                # Find patient and check if done
                patient = None
                for p in self.state.admitted_patients:
                    if p.id == doctor.current_patient_id:
                        patient = p
                        break
                
                if patient:
                    time_elapsed = self.current_step - patient.arrival_time
                    if time_elapsed >= patient.duration:
                        doctor.available = True
                        doctor.current_patient_id = None
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []
        
        # Queue state (patients waiting)
        queue_patients = self.state.patients_queue[:self.max_queue_size()]
        for i in range(self.max_queue_size()):
            if i < len(queue_patients):
                p = queue_patients[i]
                obs.extend([
                    float(p.severity) / 3.0,
                    float(p.duration) / 100.0,
                    float(p.required_specialty) / 4.0,
                    float(p.waiting_time) / 100.0,
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # Doctor state
        for doctor in self.state.doctors:
            obs.extend([
                float(doctor.available),
                float(doctor.specialty) / 4.0,
                float(doctor.workload) / 10.0,
            ])
        
        # Bed state
        for bed in self.state.beds:
            obs.extend([
                float(bed.available),
                float(bed.priority_level) / 5.0,
            ])
        
        # Time and stats
        obs.extend([
            float(self.current_step) / self.max_episode_length,
            float(len(self.state.patients_queue)) / self.max_queue_size(),
            float(len(self.state.admitted_patients)) / self.num_beds,
            float(self.state.total_patients_processed) / 100.0,
            float(self.state.mismatched_assignments) / 100.0,
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_avg_waiting_time(self) -> float:
        """Compute average waiting time for queued patients."""
        if len(self.state.patients_queue) == 0:
            return 0.0
        
        total_wait = sum(p.waiting_time for p in self.state.patients_queue)
        return total_wait / len(self.state.patients_queue)
    
    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
    
    def _render_human(self):
        """Render to console."""
        print(self._get_render_string())
    
    def _render_ansi(self) -> str:
        """Return string representation."""
        return self._get_render_string()
    
    def _get_render_string(self) -> str:
        """Get string representation of current state."""
        s = f"\n{'='*60}\n"
        s += f"Step: {self.current_step} | Queue: {len(self.state.patients_queue)} | "
        s += f"Admitted: {len(self.state.admitted_patients)}\n"
        s += f"Processed: {self.state.total_patients_processed} | "
        s += f"Avg Wait: {self._compute_avg_waiting_time():.1f}\n"
        s += f"Mismatched: {self.state.mismatched_assignments}\n"
        
        # Available resources
        available_docs = sum(1 for d in self.state.doctors if d.available)
        available_beds = sum(1 for b in self.state.beds if b.available)
        s += f"Doctors: {available_docs}/{self.num_doctors} | "
        s += f"Beds: {available_beds}/{self.num_beds}\n"
        
        s += f"{'='*60}\n"
        return s
    
    def close(self):
        """Clean up resources."""
        pass
