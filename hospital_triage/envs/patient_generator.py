"""Patient arrival generator."""

import numpy as np
from typing import Optional
from hospital_triage.envs.data_structures import Patient, SeverityLevel, DoctorSpecialty


class PatientGenerator:
    """Generates patients with realistic distributions."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize patient generator."""
        self.rng = np.random.RandomState(seed)
        self.patient_id_counter = 0
        
        # Probability distributions
        self.severity_distribution = [0.40, 0.35, 0.20, 0.05]  # LOW, MOD, HIGH, CRIT
        self.specialty_distribution = [0.25, 0.20, 0.20, 0.15, 0.20]  # Weighted by frequency
        self.duration_mean = 30
        self.duration_std = 15
    
    def generate_patient(self, current_time: int, arrival_rate: float = 0.3) -> Optional[Patient]:
        """
        Generate a patient based on arrival rate.
        
        Args:
            current_time: Current simulation time
            arrival_rate: Probability of patient arrival (0.0 to 1.0)
            
        Returns:
            Patient object or None if no arrival
        """
        if self.rng.random() > arrival_rate:
            return None
        
        # Determine severity
        severity = self.rng.choice([0, 1, 2, 3], p=self.severity_distribution)
        
        # Determine required specialty
        specialty = self.rng.choice([0, 1, 2, 3, 4], p=self.specialty_distribution)
        
        # Duration depends on severity (more severe = longer treatment)
        base_duration = max(5, int(self.rng.normal(self.duration_mean, self.duration_std)))
        duration = base_duration + (severity * 10)
        
        self.patient_id_counter += 1
        
        return Patient(
            id=self.patient_id_counter,
            arrival_time=current_time,
            severity=SeverityLevel(severity),
            required_specialty=DoctorSpecialty(specialty),
            duration=duration,
        )
    
    def generate_batch(self, current_time: int, batch_size: int) -> list:
        """Generate a batch of patients."""
        patients = []
        for _ in range(batch_size):
            patient = self.generate_patient(current_time)
            if patient:
                patients.append(patient)
        return patients
