import yaml

from src.train import TrainVAE


def load_config_file(config_path: str) -> dict:
    """Load the configuration file and store its values into a dict.
    """
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    config['train']['lr'] = float(config['train']['lr'])
    return config


def main(config_path: str):
    """Launch the training based on the given configuration file.
    """
    config = load_config_file(config_path)
    train = TrainVAE(config)

    train.prepare()
    train.summary()
    print('\nDo you wish to continue?')
    if input('[y/n] > ').lower() != 'y':
        return

    train.launch_training()
    train.save_state()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch VAE training')
    parser.add_argument('config_file', help='The configuration YAML file.')

    args = parser.parse_args()
    main(args.config_file)
