import os
import logging
from datetime import datetime

class Logger():
    def __init__(self, log_file_name, log_level, logger_name='debug', log_dir='./logs/') -> None:
        
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_path = os.path.join(log_dir,log_file_name+'_'+str(datetime.now())[:10]+'.txt')
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler()

        console_formatter = logging.Formatter('-- %(levelname)s: %(message)s')
        file_formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')

    
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
    
    
