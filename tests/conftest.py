"""Test fixtures and configuration for pytest."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_fhir_bundle():
    """Sample FHIR bundle for testing."""
    return {
        "resourceType": "Bundle",
        "id": "test-bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "test-patient",
                    "name": [{"family": "Test", "given": ["Patient"]}],
                    "gender": "male",
                    "birthDate": "1980-01-01"
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "test-med-request",
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {
                        "coding": [{
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "161",
                            "display": "Acetaminophen"
                        }]
                    },
                    "subject": {"reference": "Patient/test-patient"}
                }
            }
        ]
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    from sigma_rx7.core.config import Config
    
    config = Config()
    config.database.database = "test_sigma_rx7"
    config.pipeline.enable_ml = False  # Disable ML for faster tests
    return config