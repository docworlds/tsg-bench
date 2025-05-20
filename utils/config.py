from pathlib import Path

import yaml


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_config_file_path():
    current_path = Path(__file__)
    return str(current_path.parents[1] / 'conf.d' / 'config.yaml')


def resolve_variables(config):
    def _resolve_value(value, variables):
        if isinstance(value, str):
            for var_name, var_value in variables.items():
                value = value.replace('${{' + var_name + '}}', var_value)
        return value

    variables = config['variables']
    resolved_config = {}
    for section, section_values in config.items():
        resolved_config[section] = {}
        for key, value in section_values.items():
            resolved_config[section][key] = _resolve_value(value, variables)

    return resolved_config


def get_config():
    config_path = get_config_file_path()
    config = load_config(config_path)
    resolved_config = resolve_variables(config)
    return resolved_config


if __name__ == "__main__":
    get_config()
