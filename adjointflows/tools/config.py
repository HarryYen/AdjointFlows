import yaml

class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None

    def load(self):
        """Load the YAML configuration file."""
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        """Get a value from the configuration using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if value is None or k not in value:
                return default
            value = value[k]
        return value