# -*- coding:utf-8 -*-
from os import getcwd
from os.path import dirname, join
from logging import getLogger, FileHandler, StreamHandler, Formatter, INFO
import logging
import pandas as pd
from utils.Mysql_units import MysqlUnits
from get_config_params import get_config_params


class Logger:

    def __init__(self, filename="log.txt", log_level=INFO, logger_name=None, mysql_section: str = 'hangang_state_mysql_config'):
        """
           指定保存日志的文件路径，日志级别
        """
        # 创建一个logger
        p = dirname(join(getcwd(), ".."))
        log_path = join(p, filename)

        self.logger = getLogger(logger_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            # 创建一个handler，用于写入日志文件
            fh = FileHandler(log_path)
            fh.setLevel(log_level)
            # 再创建一个handler，用于输出到控制台
            ch = StreamHandler()
            ch.setLevel(log_level)
            # 定义handler的输出格式
            formatter = Formatter('[%(levelname)s]%(asctime)s %(filename)s:%(lineno)d: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # 给logger添加handler
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

            test_mysql_config = get_config_params(section=mysql_section)
            h1 = LoggerHandlerToMysql(test_mysql_config)
            self.logger.addHandler(h1)

    def get_log(self):
        return self.logger


class LoggerHandlerToMysql(logging.Handler):

    def __init__(self, con_config: dict = None):
        logging.Handler.__init__(self)
        self.con_config = con_config
        self.sql_units = MysqlUnits(con_config)

    def emit(self, record):
        # print(record)
        d = {'log_level': record.levelname, 'log_date': str(record.asctime).replace(",", "."),
             'log_fn': record.filename, 'log_lineno': record.lineno, 'log_msg': record.message}
        # print(pd.DataFrame([d]), self.con_config['log_table'])
        self.sql_units.pd2sql1(pd.DataFrame([d]), self.con_config['log_table'])



