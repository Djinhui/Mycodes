# -*- coding:utf-8 -*-
from configparser import ConfigParser
from os.path import join
from os import getcwd
from numpy import around


def get_config_params(section: str, file_name: str = 'run_params.conf'):
    """
    :param section: 块
    :param file_name: 文件名称
    :return: 块对应的值
    """
    f = join(getcwd(), file_name)
    config = ConfigParser()
    config.read(f, encoding='utf8')
    d1 = dict(config[section].items())
    if 'port' in d1.keys():
        tmp = d1['port']
        d1['port'] = int(tmp)
    return d1


def modify_config_params(section: str, key: str, value: float, file_name: str = 'run_params.conf'):
    """
    :param section: 块
    :param key:
    :param value:
    :param file_name: 文件名称
    :return: 块对应的值
    """
    f = join(getcwd(), file_name)
    config = ConfigParser()
    config.read(f, encoding='utf8')

    config.set(section, key, str(around(value, 2)))
    fh = open(f, 'w')
    config.write(fh)  # 把要修改的节点的内容写到文件中
    fh.close()
    return None

# a = modify_config_params(section='default_params',key='A',value= 20.0)