import pytest
from pathlib import Path
from pyconcepticon import Concepticon


@pytest.fixture
def test_data():
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mockcepticon(test_data):
    mockcepticon_path = test_data / "mockcepticon"
    return Concepticon(mockcepticon_path)


@pytest.fixture
def mock_concept(mockcepticon):
    return list(mockcepticon.conceptsets.values())[0].gloss


@pytest.fixture
def mock_definition(mockcepticon):
    return list(mockcepticon.conceptsets.values())[0].definition


@pytest.fixture
def mock_model_dir():
    return Path(__file__).parent / "test_data" / "custommodel"


@pytest.fixture()
def mock_concepts():
    return {"CONTEMPTIBLE": 0, "DUST": 1}

