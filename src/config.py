
import json

class Config:
    DIMENSION = 128
    TRESHOLD1 = 0.8
    TRESHOLD2 = 40
    PERCENTAGE = 30
    K = 30
    NODE2VEC_ITERATIONS = 3
    ACO_COARSENING_ITERATIONS = 2
    ALPHA = .5
    NODE2VEC_BATCH_SIZE = 5000
    NODE2VEC_P = 1
    NODE2VEC_Q = 1
    PYRAMID_SCALES = 6

    @classmethod
    def save_to_json(cls, filename):
        """
        Save the configuration values to a JSON file.
        
        :param filename: Path to the JSON file to save the configuration.
        """
        config_dict = {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value)}
        with open(filename, 'w') as json_file:
            json.dump(config_dict, json_file, indent=4)
        print(f"Configuration saved to {filename}")