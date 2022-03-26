# USAGE: python3 main.py

import argparse
import logging
import sys
from pathlib import Path
import yaml

#sys.path.insert(1, '/src')

from PyQt5.QtWidgets import QApplication

from src.app import VideoApp, MyMainApp
from src.utils import func_profile, log_handler

CONFIG_FILE = str(Path(__file__).resolve().parents[0] / 'config.yaml')


def argparser():
    """parse arguments from terminal"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video')
    parser.add_argument('-c', '--config', dest='config', default=CONFIG_FILE)
    parser.add_argument('-o', '--output', dest='output')
    return parser


@func_profile
def main(args: argparse.Namespace):
    """an interface tfo activate pyqt5 app"""
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    output_path = Path('outputs')
    if not output_path.exists():
        output_path.mkdir(parents=True)

    app = QApplication(sys.argv)
    main_app = MyMainApp(**config)
    try:
        # log_handler(main_app.logger)
        app.exec()
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    main(argparser().parse_args())
