# -*- coding: utf-8 -*-
import os
import json


class ConfigLoader:
    """Config loader to load configs into memory"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_FILE_PATH = os.path.join(project_root, 'config', 'config.json')

    @staticmethod
    def get_config() -> dict:
        """Returns the configs as a dictionary"""
        if not os.path.exists(ConfigLoader.CONFIG_FILE_PATH):
            raise FileNotFoundError(f"Config file: {ConfigLoader.CONFIG_FILE_PATH} not found")

        with open(ConfigLoader.CONFIG_FILE_PATH, 'r') as config_file:
            return json.loads(config_file.read())
