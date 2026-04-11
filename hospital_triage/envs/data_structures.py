"""Data structures for hospital simulation."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List
from datetime import datetime


class SeverityLevel(IntEnum):
    """Patient severity levels."""
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


class DoctorSpecialty(IntEnum):
    """Doctor specializations."""
    GENERAL = 0
    CARDIOLOGY = 1
    ORTHOPEDICS = 2
    NEUROLOGY = 3
    EMERGENCY = 4


class PatientStatus(IntEnum):
    """Patient status."""
    WAITING = 0
    ASSIGNED = 1
    IN_BED = 2
    IN_TEST = 3
    DISCHARGED = 4


@dataclass
class Patient:
    """Patient data structure."""
    id: int
    arrival_time: int
    severity: SeverityLevel
    required_specialty: DoctorSpecialty
    duration: int  # Expected treatment duration
    status: PatientStatus = PatientStatus.WAITING
    bed_id: Optional[int] = None
    doctor_id: Optional[int] = None
    test_scheduled: bool = False
    test_time: Optional[int] = None
    waiting_time: int = 0
    symptoms: str = ""
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Doctor:
    """Doctor data structure."""
    id: int
    specialty: DoctorSpecialty
    available: bool = True
    current_patient_id: Optional[int] = None
    workload: int = 0  # Number of assigned patients
    specialization_score: float = 1.0  # How well suited for current patient
    
    @property
    def is_busy(self) -> bool:
        return not self.available
        
    @property
    def patients_assigned(self) -> list:
        return [self.current_patient_id] if self.current_patient_id is not None else []
        
    def __hash__(self):
        return hash(self.id)


@dataclass
class Bed:
    """Hospital bed data structure."""
    id: int
    available: bool = True
    current_patient_id: Optional[int] = None
    priority_level: int = 0  # Higher priority gets better beds
    last_occupant_discharge_time: int = 0
    
    @property
    def is_occupied(self) -> bool:
        return not self.available
        
    def __hash__(self):
        return hash(self.id)


@dataclass
class LabTest:
    """Lab test/procedure data structure."""
    id: int
    patient_id: int
    scheduled_time: Optional[int] = None
    completed: bool = False
    start_time: Optional[int] = None
    duration: int = 5  # Default lab test duration


@dataclass
class HospitalState:
    """Complete hospital state snapshot."""
    current_time: int
    patients_queue: List[Patient] = field(default_factory=list)
    admitted_patients: List[Patient] = field(default_factory=list)
    discharged_patients: List[Patient] = field(default_factory=list)
    
    @property
    def patients(self) -> List[Patient]:
        return self.patients_queue + self.admitted_patients + self.discharged_patients
    doctors: List[Doctor] = field(default_factory=list)
    beds: List[Bed] = field(default_factory=list)
    lab_tests: List[LabTest] = field(default_factory=list)
    total_patients_processed: int = 0
    total_waiting_time: int = 0
    emergency_count: int = 0
    mismatched_assignments: int = 0
    bed_utilization: float = 0.0
