# -*- coding:utf-8 -*-
from numpy import array
from pymysql import connect
import sqlalchemy.exc
from pandas import DataFrame, read_sql_query
from sqlalchemy import create_engine

from get_config_params import get_config_params


class MysqlUnits:
    """
    连接MySQL,实现增删改查的功能
    """
    def __init__(self, con_config: dict = None):
        if con_config is None:
            con_config = {'host': 'localhost', 'port': 3306, 'user': 'root', 'password': '',
                          'dbname': 'load_prediction', 'pred_result': 'zygdd_pred_result',
                          'NWP_table': 'nwp'}

        self.user = con_config['user']
        self.psd = con_config['password']
        self.host = con_config['host']
        self.port = con_config['port']
        self.dbname = con_config['dbname']

        self.conn = connect(host=self.host, port=self.port, user=self.user, passwd=self.psd, db=self.dbname, autocommit=True)
        self.cursor = self.conn.cursor()

    def close(self):
        self.conn.close()
        self.cursor.close()

    def select2db(self, query: str):
        """
        :param query: 查询语句
        :return: DataFrame
        """

        df = read_sql_query(query, self.conn)
        return df

    def pd2sql(self, df: DataFrame, table_name: str):
        """
        :param df: DataFrame,待插入数据
        :param table_name: 目标表
        :return: None
        """
        try:
            df.to_sql(name=table_name, con=self.conn, schema=self.dbname, if_exists='append', index=False)
        except sqlalchemy.exc.IntegrityError:
            print('Exist Duplicate entry,failed insert!')

    def pd2sql1(self, df: DataFrame, table):

        engine = create_engine('mysql+pymysql://%s:%s@%s:3306/%s' % (self.user, self.psd, self.host, self.dbname))
        df.to_sql(table, con=engine, if_exists='append', index=False)

    @classmethod
    def join_str(cls, str1: str):
        if not isinstance(str1, str):
            str1 = str(str1)
        return "'" + str1 + "'"

    def conn_str1(self, d: dict, str1: str):
        s = ''
        for k, val in d.items():
            s += str(k) + '=' + self.join_str(val) + ' ' + str1 + ' '
        return s[:-(len(str1)+2)]

    def insert2db(self, table_name: str, cols: array or list, value: array):
        """
        :param table_name: 目标表
        :param cols: 列名
        :param value: 对应的值
        :return: None
        """
        key = '(' + ('%s, ' * len(cols))[:-2] + ')'
        cols = tuple(cols)
        q = "insert into {0}.{1}{2} values{3} ".format(self.dbname, table_name, key % cols, key)
        if len(value.shape) == 1:  # 单条数据
            value = tuple(value)
            self.cursor.execute(q, value)
        elif len(value.shape) == 2:  # 多条数据批量插入
            value = tuple(map(lambda x: tuple(x), value))
            self.cursor.executemany(q, value)
        else:
            print('请检查数据格式！')
        self.conn.commit()

    def update2db(self, table_name: str, update_field_value: dict, condition_field_value: dict):
        """
        :param table_name: 目标表
        :param update_field_value: 待更新的值
        :param condition_field_value: 更新数据限制条件
        :return: None
        """
        s = self.conn_str1(update_field_value, ',')
        s += ' where '
        s += self.conn_str1(condition_field_value, 'and')
        q = "update {0} SET {1} ;".format(table_name, s)
        self.cursor.execute(q)
        self.conn.commit()

    def delete2db(self, table_name: str, condition_field_value: dict):
        sql = "delete FROM {0} where {1}".format(table_name, self.conn_str1(condition_field_value, 'and'))
        self.cursor.execute(sql)
        self.conn.commit()

    # def insert_data(self, table_name: str, data: DataFrame, condition_field_value: dict):
    #     """
    #     :param table_name: 目标表
    #     :param data: 带插入数据
    #     :param condition_field_value: 判断条件
    #     :return: None
    #     """
    #     s = self.conn_str1(condition_field_value, 'and')
    #     q = "select pred_ds from {0} where pred_ds between {1} and {2} and {3};" \
    #         .format(table_name, self.join_str(data['pred_ds'].values[0]),
    #                 self.join_str(data['pred_ds'].values[-1]), s)
    #     t = self.select2db(q)
    #     # 待预测日期均未插入，批量插入
    #     if t.shape[0] == 0:
    #         self.insert2db(table_name, data.columns, data.values)
    #     # 若部分记录被插入
    #     else:
    #         for k, v in data.iterrows():
    #             q = "select pred_ds from {0} where pred_ds = {1} and {2};" \
    #                 .format(table_name, self.join_str(v['pred_ds']), s)
    #             t = self.select2db(q)
    #             # 判断是否有这条记录
    #             if t.shape[0] == 0:  # 没有这条记录插入数据
    #                 self.insert2db(table_name, data.columns, v.values)
    #             else:  # 有这条记录则更新数据
    #                 cols = ['pred', 'insert_time']
    #                 values = [v['pred'], datetime.strftime(data['insert_time'][0], '%Y-%m-%d %H:%M:%S')]
    #                 condition_field_value['pred_ds'] = v['pred_ds']
    #                 self.update2db(table_name, dict(zip(cols, values)), condition_field_value)

    def insert_data(self, table_name, data, update_field: list, condition_field: list):
        if update_field is None or condition_field is None:
            # print('{0} or {1} 存在空值，数据执行逐条插入！')
            self.pd2sql1(data, table_name)
        else:
            for k, v in data.iterrows():
                update_value = v[update_field].values
                condition_value = v[condition_field].values
                s = ''
                for i in tuple(zip(condition_field, condition_value)):
                    s = s + i[0] + '=' + self.join_str(i[1]) + ' and '
                s = s[:-5]
                q = "select * from {0} where {1};".format(table_name, s)
                t = self.select2db(q)
                # 判断是否有这条记录
                if t.shape[0] == 0:  # 没有这条记录插入数据
                    self.insert2db(table_name, data.columns, v.values)
                else:  # 有这条记录则更新数据
                    self.update2db(table_name, dict(zip(update_field, update_value)),
                                   dict(zip(condition_field, condition_value)))

