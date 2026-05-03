import json
import os
import tempfile
from orchestration.sentinel_pipeline import preprocess

def test_preprocess():
    """Test the preprocessing task."""
    # Create a temporary raw file
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_file = os.path.join(temp_dir, "raw.json")
        mock_data = {
            "articles": [
                {"title": "Test Title", "description": "Test Description"}
            ]
        }
        with open(raw_file, "w") as f:
            json.dump(mock_data, f)
            
        # Overwrite PROCESSED_DIR for testing
        os.environ["DATA_DIR"] = temp_dir
        
        # Test
        processed = preprocess([raw_file])
        
        assert len(processed) == 1
        assert "Test Title Test Description" in processed[0]["text"]
