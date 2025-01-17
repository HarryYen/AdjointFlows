import yaml

def load_parameters(config_file):
    """
    Load parameters from a YAML file.
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_parameters('config.yaml')
    print(config)


if __name__ == '__main__':
    
    main()	
