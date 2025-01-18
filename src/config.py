import json
import os


class Config:
    # Class attributes
    DIMENSION = 128
    TRESHOLD1 = 0.8
    TRESHOLD2 = 60
    PERCENTAGE = 30
    K = 20
    NODE2VEC_ITERATIONS = 1
    ACO_COARSENING_ITERATIONS = 1
    ALPHA = 0.5
    NODE2VEC_BATCH_SIZE = 5000
    NODE2VEC_P = 1
    NODE2VEC_Q = 1
    PYRAMID_SCALES = 6
    TQDM_WRITER = None

    @classmethod
    def save_to_json(cls, filename):
        # Create a dictionary of only the numeric/string class attributes
        config_dict = {}
        for key, value in vars(cls).items():
            # Skip private attributes, methods, and classmethods
            if (not key.startswith('__') and 
                not callable(value) and
                key != "TQDM_WRITER" and
                not isinstance(value, classmethod)):
                config_dict[key] = value
                
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to JSON
        with open(filename, 'w') as json_file:
            json.dump(config_dict, json_file, indent=4)
            
    @classmethod
    def load_from_json(cls, filename):
        """Load configuration from a JSON file"""
        with open(filename, 'r') as json_file:
            config_dict = json.load(json_file)
            
        # Update class attributes
        for key, value in config_dict.items():
            if key != "TQDM_WRITER":
                setattr(cls, key, value)