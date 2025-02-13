# -*- coding: utf-8 -*-
# version 1.2.0

from library.simulator_func_mysql import *
import datetime
import sys
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
from library import cf
from pandas import DataFrame
import pandas as pd
import os

from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()
TR_REQ_TIME_INTERVAL = 0.5


class open_api(QAxWidget):
    def __init__(self):
        super().__init__()

        # openapi 호출 횟수를 저장하는 변수
        self.rq_count = 0
        self.date_setting()

        # openapi연동
        self._create_open_api_instance()
        self._set_signal_slots()
        self.comm_connect()

        # 계좌 정보 가져오는 함수
        self.account_info()
        self.variable_setting()

        self.sf = simulator_func_mysql(self.simul_num, 'real', self.db_name)

        logger.debug("self.sf.simul_num(알고리즘 번호) : %s", self.sf.simul_num)
        logger.debug("self.sf.db_to_realtime_daily_buy_list_num : %s", self.sf.db_to_realtime_daily_buy_list_num)
        logger.debug("self.sf.sell_list_num : %s", self.sf.sell_list_num)

        if not self.sf.is_simul_table_exist(self.db_name, "setting_data"):
            self.init_db_setting_data()
        else:
            logger.debug("setting_data db 존재")

        self.sf_variable_setting()

    # 날짜 세팅
    def date_setting(self):
        self.today = datetime.datetime.today().strftime("%Y%m%d")
        self.today_detail = datetime.datetime.today().strftime("%Y%m%d%H%M")

    # invest_unit을 가져오는 함수
    def get_invest_unit(self):
        logger.debug("get_invest_unit 함수에 들어옴")
        sql = "select invest_unit from setting_data limit 1"
        # 데이타 Fetch
        # rows 는 list안에 튜플이 있는 [()] 형태로 받아온다
        return self.engine_JB.execute(sql).fetchall()[0][0]

    # simulator_func_mysql 에서 설정한 값을 가져오는 함수
    def sf_variable_setting(self):
        self.date_rows_yesterday = self.sf.get_recent_daily_buy_list_date()

        if not self.sf.is_simul_table_exist(self.db_name, "all_item_db"):
            logger.debug("all_item_db 없어서 생성 ")
            self.invest_unit = 0
            self.db_to_all_item(0, 0, 0, 0, 0, 0)
            self.delete_all_item("0")
        if not self.sf.is_simul_table_exist(self.db_name, "setting_data"):
            logger.debug("setting_data 없어서 생성")

            self.create_table_setting_date()

        if not self.check_set_invest_unit():
            self.set_invest_unit()
        else:
            self.invest_unit = self.get_invest_unit()
            self.sf.invest_unit = self.invest_unit

    # 보유량 가져오는 함수
    def get_holding_amount(self, code):
        logger.debug("get_holding_amount 함수에 들어옴")
        sql = "select holding_amount from possessed_item where code = '%s' group by code"
        rows = self.engine_JB.execute(sql % (code)).fetchall()
        if len(rows) :
            return rows[0][0]
        else:
            logger.debug("get_holding_amount 비어있음")
            return False

    def check_set_invest_unit(self):
        sql = "select invest_unit, set_invest_unit from setting_data limit 1"
        rows = self.engine_JB.execute(sql).fetchall()
        if rows[0][1] == self.today:
            self.invest_unit = rows[0][0]
            return True
        else:
            return False

    # 매수 금액을 설정 하는 함수
    def set_invest_unit(self):
        self.get_d2_deposit()
        self.check_balance()
        self.total_invest = self.change_format(
            str(int(self.d2_deposit_before_format) + int(self.total_purchase_price)))

        self.invest_unit = self.sf.invest_unit
        sql = "UPDATE setting_data SET invest_unit='%s',set_invest_unit='%s' limit 1"
        self.engine_JB.execute(sql % (self.invest_unit, self.today))

    # 변수 설정 함수
    def variable_setting(self):
        logger.debug("variable_setting 함수에 들어왔다.")
        self.get_today_buy_list_code = 0
        self.cf = cf
        self.chegyul_fail_amount = False

        if self.account_number == cf.real_account: # 실전
            if self.account_number[-1] != "0":
                logger.error("실전 투자 계좌번호를 잘못 입력")
                sys.exit(1)
            self.simul_num = cf.real_simul_num
            logger.debug("실전!"+ cf.real_account)
            self.db_name_setting(cf.real_db_name)
            # 실전과 모의투자가 다른 것은 아래 mod_gubun 이 다르다.
            # 금일 수익률 표시 하는게 달라서(중요X)
            self.mod_gubun = 100

        elif self.account_number == cf.imi1_accout: #모의1
            logger.debug("모의투자")
            self.simul_num = cf.imi1_simul_num
            self.db_name_setting(cf.imi1_db_name)
            self.mod_gubun = 1

        else:
            logger.debug("계정이 존재하지 않는다")
            exit(1)
        self.jango_is_null = True
        self.py_gubun = False

    def create_table_setting_date(self):
        logger.debug("create_table_setting_date 함수에 들어옴")
        df_setting_data_temp = {'index': [0], 'loan_money': [0], 'limit_money': [0], 'invest_unit': [0],
                                'max_invest_unit': [0], 'min_invest_unit': [0], 'set_invest_unit': [0],
                                'code_update': [0],
                                'today_buy_stop': [0], 'jango_data_db_check': [0], 'possessed_item': [0],
                                'today_profit': [0], 'final_chegyul_check': [0], 'db_to_buy_list': [0],
                                'today_buy_list': [0],
                                'daily_crawler': [0], 'daily_buy_list': [0]}
        df_setting_data = DataFrame(df_setting_data_temp)
        df_setting_data.to_sql('setting_data', self.engine_JB, if_exists='replace')

    # 봇 데이터 베이스를 만드는 함수
    def create_database(self):
        logger.debug("create_database!!! %s", self.db_name)
        sql = 'CREATE DATABASE %s'
        self.engine_daily_buy_list.execute(sql % (self.db_name))

    # 봇 데이터 베이스 존재 여부 확인 함수
    def is_database_exist(self):
        sql = "SELECT 1 FROM Information_schema.SCHEMATA WHERE SCHEMA_NAME = '%s'"
        rows = self.engine_daily_buy_list.execute(sql % (self.db_name)).fetchall()
        if len(rows):
            logger.debug("%s 데이터 베이스가 존재 ", self.db_name)
            return True
        else:
            logger.debug("%s 데이터 베이스가 존재하지 않음 ", self.db_name)
            return False

    # db 세팅 함수
    def db_name_setting(self, db_name):
        self.db_name = db_name
        logger.debug("db name !!! : %s", self.db_name)

        self.engine_craw = create_engine("mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" +cf.db_port+ "/min_craw",
                                         encoding='utf-8')
        self.engine_daily_craw = create_engine("mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" +cf.db_port+ "/daily_craw",
                                               encoding='utf-8')
        self.engine_daily_buy_list = create_engine("mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" +cf.db_port+ "/daily_buy_list",
                                                   encoding='utf-8')

        if not self.is_database_exist():
            self.create_database()

        self.engine_JB = create_engine("mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" +cf.db_port+ "/" + db_name,
                                       encoding='utf-8')

    # 계좌 정보 함수
    def account_info(self):
        logger.debug("account_info 함수에 들어옴")
        account_number = self.get_login_info("ACCNO")
        self.account_number = account_number.split(';')[0]
        logger.debug("계좌번호 : "+self.account_number)

    def get_login_info(self, tag):
        logger.debug("get_login_info 함수에 들어옴")
        try:
            ret = self.dynamicCall("GetLoginInfo(QString)", tag)
            # logger.debug(ret)
            return ret
        except Exception as e:
            logger.critical(e)

    def _create_open_api_instance(self):
        try:
            self.setControl("KHOPENAPI.KHOpenAPICtrl.1")
        except Exception as e:
            logger.critical(e)

    def _set_signal_slots(self):
        try:
            self.OnEventConnect.connect(self._event_connect)
            self.OnReceiveTrData.connect(self._receive_tr_data)
            self.OnReceiveMsg.connect(self._receive_msg)
            self.OnReceiveChejanData.connect(self._receive_chejan_data)


        except Exception as e:
            logger.critical(e)

    def comm_connect(self):
        try:
            self.dynamicCall("CommConnect()")
            self.login_event_loop = QEventLoop()
            self.login_event_loop.exec_()
        except Exception as e:
            logger.critical(e)


    def _receive_msg(self, sScrNo, sRQName, sTrCode, sMsg):
        logger.debug("_receive_msg 함수에 들어옴")
        # logger.debug("sScrNo!!!")
        # logger.debug(sScrNo)
        # logger.debug("sRQName!!!")
        # logger.debug(sRQName)
        # logger.debug("sTrCode!!!")
        # logger.debug(sTrCode)
        # logger.debug("sMsg!!!")
        logger.debug(sMsg)

    def _event_connect(self, err_code):
        try:
            if err_code == 0:
                logger.debug("connected")
            else:
                logger.debug("disconnected")

            self.login_event_loop.exit()
        except Exception as e:
            logger.critical(e)

    def set_input_value(self, id, value):
        try:
            self.dynamicCall("SetInputValue(QString, QString)", id, value)
        except Exception as e:
            logger.critical(e)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.exit_check()
        self.dynamicCall("CommRqData(QString, QString, int, QString", rqname, trcode, next, screen_no)

        self.tr_event_loop = QEventLoop()

        self.tr_event_loop.exec_()

    def _get_comm_data(self, code, field_name, index, item_name):
        ret = self.dynamicCall("GetCommData(QString, QString, int, QString", code, field_name, index, item_name)
        return ret.strip()

    def _get_repeat_cnt(self, trcode, rqname):
        try:
            ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
            return ret
        except Exception as e:
            logger.critical(e)

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        # print("screen_no, rqname, trcode", screen_no, rqname, trcode)
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False
        # print("self.py_gubun!!", self.py_gubun)
        if rqname == "opt10081_req" and self.py_gubun == "trader":
            # logger.debug("opt10081_req trader!!!")
            # logger.debug("Get an item info !!!!")
            self._opt10081(rqname, trcode)
        elif rqname == "opt10081_req" and self.py_gubun == "collector":
            # logger.debug("opt10081_req collector!!!")
            # logger.debug("Get an item info !!!!")
            self.collector_opt10081(rqname, trcode)
        elif rqname == "opw00001_req":
            # logger.debug("opw00001_req!!!")
            # logger.debug("Get an de_deposit!!!")
            self._opw00001(rqname, trcode)
        elif rqname == "opw00018_req":
            # logger.debug("opw00018_req!!!")
            # logger.debug("Get the possessed item !!!!")
            self._opw00018(rqname, trcode)
        elif rqname == "opt10074_req":
            # logger.debug("opt10074_req!!!")
            # logger.debug("Get the profit")
            self._opt10074(rqname, trcode)
        elif rqname == "opw00015_req":
            # logger.debug("opw00015_req!!!")
            # logger.debug("deal list!!!!")
            self._opw00015(rqname, trcode)
        elif rqname == "opt10076_req":
            # logger.debug("opt10076_req")
            # logger.debug("chegyul list!!!!")
            self._opt10076(rqname, trcode)
        elif rqname == "opt10073_req":
            # logger.debug("opt10073_req")
            # logger.debug("Get today profit !!!!")
            self._opt10073(rqname, trcode)
        elif rqname == "opt10080_req":
            # logger.debug("opt10080_req!!!")
            # logger.debug("Get an de_deposit!!!")
            self._opt10080(rqname, trcode)
        # except Exception as e:
        #     logger.critical(e)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def init_db_setting_data(self):
        logger.debug("init_db_setting_data !! ")

        df_setting_data_temp = {'loan_money': [], 'limit_money': [], 'invest_unit': [], 'max_invest_unit': [],
                                'min_invest_unit': [],
                                'set_invest_unit': [], 'code_update': [], 'today_buy_stop': [],
                                'jango_data_db_check': [], 'possessed_item': [], 'today_profit': [],
                                'final_chegyul_check': [],
                                'db_to_buy_list': [], 'today_buy_list': [], 'daily_crawler': [],
                                'daily_buy_list': []}

        df_setting_data = DataFrame(df_setting_data_temp,
                                    columns=['loan_money', 'limit_money', 'invest_unit', 'max_invest_unit',
                                             'min_invest_unit',
                                             'set_invest_unit', 'code_update', 'today_buy_stop',
                                             'jango_data_db_check', 'possessed_item', 'today_profit',
                                             'final_chegyul_check',
                                             'db_to_buy_list', 'today_buy_list', 'daily_crawler',
                                             'daily_buy_list'])

        df_setting_data.loc[0, 'loan_money'] = int(0)
        df_setting_data.loc[0, 'limit_money'] = int(0)
        df_setting_data.loc[0, 'invest_unit'] = int(0)
        df_setting_data.loc[0, 'max_invest_unit'] = int(0)
        df_setting_data.loc[0, 'min_invest_unit'] = int(0)

        df_setting_data.loc[0, 'set_invest_unit'] = str(0)
        df_setting_data.loc[0, 'code_update'] = str(0)
        df_setting_data.loc[0, 'today_buy_stop'] = str(0)
        df_setting_data.loc[0, 'jango_data_db_check'] = str(0)

        df_setting_data.loc[0, 'possessed_item'] = str(0)
        df_setting_data.loc[0, 'today_profit'] = str(0)
        df_setting_data.loc[0, 'final_chegyul_check'] = str(0)
        df_setting_data.loc[0, 'db_to_buy_list'] = str(0)
        df_setting_data.loc[0, 'today_buy_list'] = str(0)
        df_setting_data.loc[0, 'daily_crawler'] = str(0)
        df_setting_data.loc[0, 'min_crawler'] = str(0)
        df_setting_data.loc[0, 'daily_buy_list'] = str(0)

        df_setting_data.to_sql('setting_data', self.engine_JB, if_exists='replace')

    def db_to_all_item(self, order_num, code, code_name, chegyul_check, purchase_price, rate):
        logger.debug("db_to_all_item 함수에 들어왔다!!!")
        self.date_setting()
        self.sf.init_df_all_item()
        self.sf.df_all_item.loc[0, 'order_num'] = order_num
        self.sf.df_all_item.loc[0, 'code'] = str(code)
        self.sf.df_all_item.loc[0, 'code_name'] = str(code_name)
        self.sf.df_all_item.loc[0, 'rate'] = float(rate)

        self.sf.df_all_item.loc[0, 'buy_date'] = self.today_detail
        self.sf.df_all_item.loc[0, 'chegyul_check'] = chegyul_check
        self.sf.df_all_item.loc[0, 'reinvest_date'] = '#'
        self.sf.df_all_item.loc[0, 'invest_unit'] = self.invest_unit
        self.sf.df_all_item.loc[0, 'purchase_price'] = purchase_price
        self.sf.df_all_item.loc[0, 'purchase_rate']

        if order_num != 0:
            recent_daily_buy_list_date=self.sf.get_recent_daily_buy_list_date()
            if recent_daily_buy_list_date:
                df=self.sf.get_daily_buy_list_by_code(code,recent_daily_buy_list_date)
                if not df.empty:
                    self.sf.df_all_item.loc[0, 'close'] = df.loc[0, 'close']
                    self.sf.df_all_item.loc[0, 'open'] = df.loc[0, 'open']
                    self.sf.df_all_item.loc[0, 'high'] = df.loc[0, 'high']
                    self.sf.df_all_item.loc[0, 'low'] = df.loc[0, 'low']
                    self.sf.df_all_item.loc[0, 'volume'] = df.loc[0, 'volume']
                    self.sf.df_all_item.loc[0, 'd1_diff_rate'] = float(df.loc[0, 'd1_diff_rate'])
                    self.sf.df_all_item.loc[0, 'clo5'] = df.loc[0, 'clo5']
                    self.sf.df_all_item.loc[0, 'clo10'] = df.loc[0, 'clo10']
                    self.sf.df_all_item.loc[0, 'clo20'] = df.loc[0, 'clo20']
                    self.sf.df_all_item.loc[0, 'clo40'] = df.loc[0, 'clo40']
                    self.sf.df_all_item.loc[0, 'clo60'] = df.loc[0, 'clo60']
                    self.sf.df_all_item.loc[0, 'clo80'] = df.loc[0, 'clo80']
                    self.sf.df_all_item.loc[0, 'clo100'] = df.loc[0, 'clo100']
                    self.sf.df_all_item.loc[0, 'clo120'] = df.loc[0, 'clo120']

                    if df.loc[0, 'clo5_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo5_diff_rate'] = float(df.loc[0, 'clo5_diff_rate'])
                    if df.loc[0, 'clo10_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo10_diff_rate'] = float(df.loc[0, 'clo10_diff_rate'])
                    if df.loc[0, 'clo20_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo20_diff_rate'] = float(df.loc[0, 'clo20_diff_rate'])
                    if df.loc[0, 'clo40_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo40_diff_rate'] = float(df.loc[0, 'clo40_diff_rate'])

                    if df.loc[0, 'clo60_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo60_diff_rate'] = float(df.loc[0, 'clo60_diff_rate'])
                    if df.loc[0, 'clo80_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo80_diff_rate'] = float(df.loc[0, 'clo80_diff_rate'])
                    if df.loc[0, 'clo100_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo100_diff_rate'] = float(df.loc[0, 'clo100_diff_rate'])
                    if df.loc[0, 'clo120_diff_rate'] is not None:
                        self.sf.df_all_item.loc[0, 'clo120_diff_rate'] = float(df.loc[0, 'clo120_diff_rate'])

        self.sf.df_all_item = self.sf.df_all_item.fillna(0)
        self.sf.df_all_item.to_sql('all_item_db', self.engine_JB, if_exists='append')

    def check_balance(self):

        logger.debug("check_balance 함수에 들어왔습니다!")
        # 1차원 / 2차원 인스턴스 변수 생성
        self.reset_opw00018_output()

        # # 예수금 가져오기
        self.set_input_value("계좌번호", self.account_number)
        self.comm_rq_data("opw00018_req", "opw00018", 0, "2000")

        while self.remained_data:
            self.set_input_value("계좌번호", self.account_number)

            self.comm_rq_data("opw00018_req", "opw00018", 2, "2000")
            print("self.opw00018_output: ", self.opw00018_output)
    def get_count_possesed_item(self):
        logger.debug("get_count_possesed_item!!!")

        sql = "select count(*) from possessed_item"
        rows = self.engine_JB.execute(sql).fetchall()

        return rows[0][0]

    def setting_data_possesed_item(self):

        sql = "UPDATE setting_data SET possessed_item='%s' limit 1"
        self.engine_JB.execute(sql % (self.today))

    def db_to_possesed_item(self):
        logger.debug("db_to_possesed_item 함수에 들어왔습니다!")
        item_count = len(self.opw00018_output['multi'])
        possesed_item_temp = {'date': [], 'code': [], 'code_name': [], 'holding_amount': [], 'puchase_price': [],
                              'present_price': [], 'valuation_profit': [], 'rate': [], 'item_total_purchase': []}

        possesed_item = DataFrame(possesed_item_temp,
                                  columns=['date', 'code', 'code_name', 'holding_amount', 'puchase_price',
                                           'present_price', 'valuation_profit', 'rate', 'item_total_purchase'])
        
        for i in range(item_count):
            row = self.opw00018_output['multi'][i]
            # 오늘 일자
            possesed_item.loc[i, 'date'] = self.today
            possesed_item.loc[i, 'code'] = self.codename_to_code(row[0])
            possesed_item.loc[i, 'code_name'] = row[0]
            # 보유량
            possesed_item.loc[i, 'holding_amount'] = int(row[1])
            # 매수가
            possesed_item.loc[i, 'puchase_price'] = int(row[2])
            # 현재가
            possesed_item.loc[i, 'present_price'] = int(row[3])
            possesed_item.loc[i, 'valuation_profit'] = int(row[4])
            possesed_item.loc[i, 'rate'] = float(row[5])
            # 총 매수 금액
            possesed_item.loc[i, 'item_total_purchase'] = int(row[6])
        possesed_item.to_sql('possessed_item', self.engine_JB, if_exists='replace')




    def get_total_data_min(self, code, code_name, start):
        self.ohlcv = {'index': [], 'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': [],
                      'sum_volume': []}

        self.set_input_value("종목코드", code)
        self.set_input_value("틱범위", 1)
        self.set_input_value("수정주가구분", 1)
        self.comm_rq_data("opt10080_req", "opt10080", 0, "1999")

        self.craw_table_exist = False

        if self.is_min_craw_table_exist(code_name):
            self.craw_table_exist = True
            self.craw_db_last_min = self.get_craw_db_last_min(code_name)
            self.craw_db_last_min_sum_volume = self.get_craw_db_last_min_sum_volume(code_name)

        else:
            self.craw_db_last_min = str(0)
            self.craw_db_last_min_sum_volume = 0

        while self.remained_data == True:
            time.sleep(TR_REQ_TIME_INTERVAL)
            self.set_input_value("종목코드", code)
            self.set_input_value("틱범위", 1)
            self.set_input_value("수정주가구분", 1)
            self.comm_rq_data("opt10080_req", "opt10080", 2, "1999")

            if self.ohlcv['date'][-1] < self.craw_db_last_min:
                break

        time.sleep(TR_REQ_TIME_INTERVAL)

        if len(self.ohlcv['date']) == 0 or self.ohlcv['date'][0] == '':
            return []
        if self.ohlcv['date'] == '':
            return []

        df = DataFrame(self.ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'sum_volume'])

        return df

    def get_total_data(self, code, code_name, date):
        logger.debug("일봉 get_total_data 함수에 들어왔다!")
        self.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'volumemoney': [], 'change_ratio': []}  # 거래금액 추가  'index'

        self.set_input_value("종목코드", code)
        self.set_input_value("기준일자", date)
        self.set_input_value("수정주가구분", 1)
        self.comm_rq_data("opt10081_req", "opt10081", 0, "0101")

        if not self.is_craw_table_exist(code_name):
            while self.remained_data == True:
                time.sleep(TR_REQ_TIME_INTERVAL)
                self.set_input_value("종목코드", code)
                self.set_input_value("기준일자", date)
                self.set_input_value("수정주가구분", 1)
                self.comm_rq_data("opt10081_req", "opt10081", 2, "0101")

        time.sleep(TR_REQ_TIME_INTERVAL)
        if len(self.ohlcv) == 0:
            return []
        if self.ohlcv['date'] == '':
            return []
        if len(self.ohlcv['date']) == 0:  # 아예 챠트가 비어있는 챠트가 남아있다.
            return []
        df = DataFrame(self.ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'volumemoney', 'change_ratio'])  # 거래금액 추가, 수정비율

        if df.loc[0, 'change_ratio'] != 0.0:
            print("오늘 수정비율이 0이 아니다. 기존 것을 지우고 다시 받아야함. replace")
            self.craw_table_delete(code_name)

        return df

    def is_craw_table_exist(self, code_name):
        sql = "select 1 from information_schema.tables where table_schema ='daily_craw' and table_name = '%s'"
        rows = self.engine_daily_craw.execute(sql % (code_name)).fetchall()
        if rows:
            return True
        else:
            logger.debug(str(code_name) + " 테이블이 daily_craw db 에 없다. 새로 생성! ", )
            return False

    def craw_table_delete(self, code_name):
        sql = "drop table %s"
        self.engine_daily_craw.execute(sql % code_name)

    def is_min_craw_table_exist(self, code_name):
        sql = "select 1 from information_schema.tables where table_schema ='min_craw' and table_name = '%s'"
        rows = self.engine_craw.execute(sql % (code_name)).fetchall()
        if rows:
            return True
        else:
            logger.debug(str(code_name) + " min_craw db에 없다 새로 생성! ", )
            return False


    def get_craw_db_last_min_sum_volume(self, code_name):
        sql = "SELECT sum_volume from `" + code_name + "` order by date desc limit 1"
        rows = self.engine_craw.execute(sql).fetchall()
        if len(rows):
            return rows[0][0]
        # 신생
        else:
            return str(0)

    def get_craw_db_last_min(self, code_name):
        sql = "SELECT date from `" + code_name + "` order by date desc limit 1"
        rows = self.engine_craw.execute(sql).fetchall()
        if len(rows):
            return rows[0][0]
        # 신생
        else:
            return str(0)

    def get_daily_craw_db_last_date(self, code_name):
        sql = "SELECT date from `" + code_name + "` order by date desc limit 1"
        rows = self.engine_daily_craw.execute(sql).fetchall()
        if len(rows):
            return rows[0][0]
        else:
            return str(0)
    def get_one_day_option_data(self, code, start, option):

        self.ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        self.set_input_value("종목코드", code)

        self.set_input_value("기준일자", start)

        self.set_input_value("수정주가구분", 1)

        self.comm_rq_data("opt10081_req", "opt10081", 0, "0101")

        if self.ohlcv['date'] == '':
            return False

        df = DataFrame(self.ohlcv, columns=['open', 'high', 'low', 'close', 'volume'], index=self.ohlcv['date'])

        if df.empty:
            return False


        if option == 'open':
            return df.iloc[0, 0]
        elif option == 'high':
            return df.iloc[0, 1]
        elif option == 'low':
            return df.iloc[0, 2]
        elif option == 'close':
            return df.iloc[0, 3]
        elif option == 'volume':
            return df.iloc[0, 4]
        else:
            return False

    def collector_opt10081(self, rqname, trcode):
        ohlcv_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(ohlcv_cnt):
            date = self._get_comm_data(trcode, rqname, i, "일자")
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")
            volumemoney = self._get_comm_data(trcode, rqname, i, "거래대금")
            change_ratio = self._get_comm_data(trcode, rqname, i, "수정비율")
            if change_ratio == "":
                change_ratio = 0

            self.ohlcv['date'].append(date)
            self.ohlcv['open'].append(int(open))
            self.ohlcv['high'].append(int(high))
            self.ohlcv['low'].append(int(low))
            self.ohlcv['close'].append(int(close))
            self.ohlcv['volume'].append(int(volume))
            self.ohlcv['volumemoney'].append(int(volumemoney))
            self.ohlcv['change_ratio'].append(float(change_ratio))

    # order_no  – 원주문번호

    def _opt10080(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(data_cnt):
            date = self._get_comm_data(trcode, rqname, i, "체결시간")
            open = self._get_comm_data(trcode, rqname, i, "시가")
            high = self._get_comm_data(trcode, rqname, i, "고가")
            low = self._get_comm_data(trcode, rqname, i, "저가")
            close = self._get_comm_data(trcode, rqname, i, "현재가")
            volume = self._get_comm_data(trcode, rqname, i, "거래량")

            self.ohlcv['date'].append(date[:-2])
            self.ohlcv['open'].append(abs(int(open)))
            self.ohlcv['high'].append(abs(int(high)))
            self.ohlcv['low'].append(abs(int(low)))
            self.ohlcv['close'].append(abs(int(close)))
            self.ohlcv['volume'].append(int(volume))
            self.ohlcv['sum_volume'].append(int(0))


    def _opt10081(self, rqname, trcode):
        try:
            logger.debug("_opt10081!!!")
            date = self._get_comm_data(trcode, rqname, 0, "일자")
            open = self._get_comm_data(trcode, rqname, 0, "시가")
            high = self._get_comm_data(trcode, rqname, 0, "고가")
            low = self._get_comm_data(trcode, rqname, 0, "저가")
            close = self._get_comm_data(trcode, rqname, 0, "현재가")
            volume = self._get_comm_data(trcode, rqname, 0, "거래량")
            volumemoney = self._get_comm_data(trcode, rqname, 0, "거래대금")

            self.ohlcv['date'].append(date)
            self.ohlcv['open'].append(int(open))
            self.ohlcv['high'].append(int(high))
            self.ohlcv['low'].append(int(low))
            self.ohlcv['close'].append(int(close))
            self.ohlcv['volume'].append(int(volume))
            self.ohlcv['volumemoney'].append(int(volumemoney))

        except Exception as e:
            logger.critical(e)


    def send_order(self, rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no):
        logger.debug("send_order!!!")
        try:
            self.exit_check()
            self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                             [rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no])
        except Exception as e:
            logger.critical(e)


    def get_chejan_data(self, fid):
        # logger.debug("get_chejan_data!!!")
        try:
            self.exit_check()
            ret = self.dynamicCall("GetChejanData(int)", fid)
            return ret
        except Exception as e:
            logger.critical(e)

    def codename_to_code(self, codename):

        sql = "select code from stock_item_all where code_name='%s'"
        rows = self.engine_daily_buy_list.execute(sql % (codename,)).fetchall()

        if len(rows) != 0:
            return rows[0][0]

        logger.debug("code를 찾을 수 없다!! name이 긴놈이다!!!!")
        logger.debug(codename)

        sql = "select code from stock_item_all where code_name like '%s'"
        self.engine_daily_buy_list.execute(sql % (codename + "%%"))

        if len(rows) != 0:
            return rows[0][0]

        logger.debug("codename이 존재하지 않는다 ..... 긴 것도 아니다...")

        return False


    def end_invest_count_check(self, code):
        logger.debug("end_invest_count_check 함수로 들어왔습니다!")
        logger.debug("end_invest_count_check_code!!!!!!!!")
        logger.debug(code)

        sql = "UPDATE all_item_db SET chegyul_check='%s' WHERE code='%s' and sell_date = '%s' ORDER BY buy_date desc LIMIT 1"

        self.engine_JB.execute(sql % (0, code,0))

        sql = "delete from possessed_item where code ='%s'"
        self.engine_JB.execute(sql % (code,))

    def sell_chegyul_fail_check(self, code):
        logger.debug("sell_chegyul_fail_check 함수에 들어왔습니다!")
        logger.debug(code + " check!")
        sql = "UPDATE all_item_db SET chegyul_check='%s' WHERE code='%s' and sell_date = '%s' ORDER BY buy_date desc LIMIT 1"
        self.engine_JB.execute(sql % (1, code,0))

    def buy_check_reset(self):
        logger.debug("buy_check_reset!!!")

        sql = "UPDATE setting_data SET today_buy_stop='%s' WHERE id='%s'"
        self.engine_JB.execute(sql % (0, 1))

    def buy_check_stop(self):
        logger.debug("buy_check_stop!!!")
        sql = "UPDATE setting_data SET today_buy_stop='%s' limit 1"
        self.engine_JB.execute(sql % (self.today))

    def jango_check(self):
        logger.debug("jango_check 함수에 들어왔습니다!")
        self.get_d2_deposit()
        try:
            if int(self.d2_deposit_before_format) > (int(self.sf.limit_money)):
                self.jango_is_null = False
                logger.debug("돈안부족해 투자 가능!!!!!!!!")
                return True
            else:
                logger.debug("돈부족해서 invest 불가!!!!!!!!")
                self.jango_is_null = True
                return False
        except Exception as e:
            logger.critical(e)
    def buy_check(self):
        logger.debug("buy_check 함수에 들어왔습니다!")
        sql = "select today_buy_stop from setting_data limit 1"
        rows = self.engine_JB.execute(sql).fetchall()[0][0]

        if rows != self.today:
            logger.debug("GoGo Buying!!!!!!")
            return True
        else:
            logger.debug("Stop Buying!!!!!!")
            return False

    def buy_num_count(self, invest_unit, present_price):
        logger.debug("buy_num_count 함수에 들어왔습니다!")
        return int(invest_unit / present_price)

    def trade(self):
        logger.debug("trade 함수에 들어왔다!")
        logger.debug("매수 대상 종목 코드! " + self.get_today_buy_list_code)

        current_price = self.get_one_day_option_data(self.get_today_buy_list_code, self.today,'close')

        if current_price == False:
            logger.debug(self.get_today_buy_list_code + " 의 현재가가 비어있다 !!!")
            return False


        min_buy_limit = int(self.get_today_buy_list_close) * self.sf.invest_min_limit_rate
        max_buy_limit = int(self.get_today_buy_list_close) * self.sf.invest_limit_rate
        if min_buy_limit < current_price < max_buy_limit:
            buy_num = self.buy_num_count(self.invest_unit, int(current_price))
            logger.debug(
                "매수!!!!+-+-+-+-+-+-+-+-+-+-+-+-+-+-+- code :%s, 목표가: %s, 현재가: %s, 매수량: %s, min_buy_limit: %s, max_buy_limit: %s , invest_limit_rate: %s,예수금: %s , today : %s, today_min : %s, date_rows_yesterday : %s, invest_unit : %s, real_invest_unit : %s +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-",
                self.get_today_buy_list_code, self.get_today_buy_list_close, current_price, buy_num, min_buy_limit,
                max_buy_limit, self.sf.invest_limit_rate, self.d2_deposit_before_format, self.today, self.today_detail,
                self.date_rows_yesterday, self.invest_unit, int(current_price) * int(buy_num))

            self.send_order("send_order_req", "0101", self.account_number, 1, self.get_today_buy_list_code, buy_num, 0,
                            "03", "")

            if self.jango_check() == False:
                logger.debug("하나 샀더니 잔고가 부족해진 구간!!!!!")
                self.buy_check_stop()
        else:
            logger.debug(
                "invest_limit_rate 만큼 급등 or invest_min_limit_rate 만큼 급락 해서 매수 안함 !!! code :%s, 목표가: %s , 현재가: %s, invest_limit_rate: %s , invest_min_limit_rate : %s, today : %s, today_min : %s, date_rows_yesterday : %s",
                self.get_today_buy_list_code, self.get_today_buy_list_close, current_price, self.sf.invest_limit_rate,
                self.sf.invest_min_limit_rate, self.today, self.today_detail, self.date_rows_yesterday)

    def get_today_buy_list(self):
        logger.debug("get_today_buy_list 함수에 들어왔습니다!")

        logger.debug(" self.today : %s , self.date_rows_yesterday : %s !", self.today, self.date_rows_yesterday)

        if self.sf.is_simul_table_exist(self.db_name, "realtime_daily_buy_list"):
            logger.debug("realtime_daily_buy_list 생겼다!!!!! ")
            self.sf.get_realtime_daily_buy_list()
            if self.sf.len_df_realtime_daily_buy_list == 0:
                logger.debug("realtime_daily_buy_list 생겼지만 아직 data가 없다!!!!! ")
                return
        else:
            logger.debug("realtime_daily_buy_list 없다 !! ")
            return

        logger.debug("self.sf.len_df_realtime_daily_buy_list 이제 사러간다!! " )
        logger.debug("매수 리스트!!!!")
        logger.debug(self.sf.df_realtime_daily_buy_list)
        for i in range(self.sf.len_df_realtime_daily_buy_list):
            code = self.sf.df_realtime_daily_buy_list.loc[i, 'code']
            close = self.sf.df_realtime_daily_buy_list.loc[i, 'close']
            check_item = self.sf.df_realtime_daily_buy_list.loc[i, 'check_item']

            if self.jango_is_null:
                break
            if check_item == True:
                continue
            else:
                self.get_today_buy_list_code = code
                self.get_today_buy_list_close = close
                sql = "UPDATE realtime_daily_buy_list SET check_item='%s' WHERE code='%s'"
                self.engine_JB.execute(sql % (1, self.get_today_buy_list_code))
                self.trade()

        self.buy_check_stop()

    def exit_check(self):
        time.sleep(cf.TR_REQ_TIME_INTERVAL)
        self.rq_count += 1
        logger.debug(self.rq_count)
        if self.rq_count % 45 == 0:
            time.sleep(cf.TR_REQ_TIME_INTERVAL_LONG)
        if self.rq_count % 100 == 0:
            time.sleep(cf.TR_REQ_TIME_INTERVAL_LONG*2)
        if self.rq_count == cf.max_api_call:
            sys.exit(1)

    def final_chegyul_check(self):
        sql = "select code from all_item_db a where (a.sell_date = '%s' or a.sell_date ='%s') and a.code not in ( select code from possessed_item) and a.chegyul_check != '%s'"

        rows = self.engine_JB.execute(sql % (0, "", 1)).fetchall()
        logger.debug("possess_item 테이블에는 없는데 all_item_db에 sell_date가 없는 리스트 처리!!!")
        logger.debug(rows)
        num = len(rows)


        for t in range(num):
            logger.debug("t!!!")
            logger.debug(t)
            self.sell_final_check2(rows[t][0])

        sql = "UPDATE setting_data SET final_chegyul_check='%s' limit 1"
        self.engine_JB.execute(sql % (self.today))

    def rate_check(self):
        logger.debug("rate_check!!!")
        sql = "select code ,rate from possessed_item group by code"
        rows = self.engine_JB.execute(sql).fetchall()

        logger.debug("rate 업데이트 !!!")
        logger.debug(rows)
        num = len(rows)

        for k in range(num):
            code=rows[k][0]
            rate=rows[k][1]
            # print("rate!!", rate)
            sql = "update all_item_db set rate='%s' where code='%s' and sell_date = '%s'"
            self.engine_JB.execute(sql % (float(rate), code,0))

    def chegyul_check(self):
        logger.debug("chegyul_check!!!")
        sql = "select code,code_name,rate from possessed_item p where p.code not in (select a.code from all_item_db a where a.sell_date = '%s' group by a.code) group by p.code"
        rows = self.engine_JB.execute(sql % (0,)).fetchall()

        logger.debug("possess_item 테이블에는 있는데 all_item_db에 없는 종목들 처리!!!")
        logger.debug(rows)
        num = len(rows)

        for k in range(num):
            self.db_to_all_item(self.today, rows[k][0], rows[k][1], 1, 0, rows[k][2])

        sql = "SELECT code FROM all_item_db where chegyul_check='%s' and (sell_date = '%s' or sell_date= '%s')"
        rows = self.engine_JB.execute(sql % (1, 0, "")).fetchall()

        logger.debug("in chegyul_check!!!!! all_db에서 cc가 1인놈들 확인!!!!!!!!")
        logger.debug(rows)
        num = len(rows)


        for i in range(num):
            code = rows[i][0]
            logger.debug("chegyul_check code!!!")
            logger.debug(code)
            self.set_input_value("종목코드", code)
            self.set_input_value("조회구분", 1)
            self.set_input_value("계좌번호", self.account_number)
            self.comm_rq_data("opt10076_req", "opt10076", 0, "0350")
            if self.chegyul_fail_amount == -2:
                logger.debug(
                    "opt10076_req 로 comm rq data 가는 도중에 receive chejan 걸려서 chegyul_fail_amount를 못가져옴. 이럴 때는 다시 돌려")

                self.chegyul_check()
            elif self.chegyul_fail_amount == -1:
                logger.debug(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!l이게 아마 어제 미체결 인놈들같은데 ! update 한번해줘보자 나중에 안되면 수정 , 이게 아마 이미 체결 된놈인듯 어제 체결 돼서 조회가 안되는거인듯")
                sql = "update all_item_db set chegyul_check='%s' where code='%s' and sell_date = '%s' ORDER BY buy_date desc LIMIT 1 "
                self.engine_JB.execute(sql % (0, code,0))

            elif self.chegyul_fail_amount == 0:
                logger.debug("체결!!!!! 이건 오늘 산놈들에 대해서만 조회가 가능한듯 ")

                sql = "update all_item_db set chegyul_check='%s' where code='%s' and sell_date = '%s' ORDER BY buy_date desc LIMIT 1 "
                self.engine_JB.execute(sql % (0, code,0))


            else:
                logger.debug("아직 매수 혹은 매도 중인 놈이다 미체결!!!!!!!!!!!!!!!!!!!!!!!!!")
                logger.debug("self.chegyul_fail_amount!!")
                logger.debug(self.chegyul_fail_amount)

    def stock_chegyul_check(self, code):
        logger.debug("stock_chegyul_check 함수에 들어왔다!")

        sql = "SELECT chegyul_check FROM all_item_db where code='%s' and sell_date = '%s' ORDER BY buy_date desc LIMIT 1"

        rows = self.engine_JB.execute(sql % (code,0)).fetchall()

        if rows[0][0] == 1:
            return True
        else:
            return False

    def get_data_from_possessed_db(self, code):
        logger.debug("get_data_from_possessed_db!!!")
        sql = "select valuation_profit, rate, item_total_purchase, present_price from possessed_item where code='%s' group by code"


        get_list = self.engine_JB.execute(sql % (code)).fetchall()

        logger.debug("get_list!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.debug(get_list)
        if len(get_list) != 0:
            self.valuation_profit = get_list[0][0]
            self.possess_rate = get_list[0][1]
            self.possess_item_total_purchase = get_list[0][2]
            self.possess_sell_price = get_list[0][3]
            logger.debug("valuation_profit!!!!!!!!")
            logger.debug(self.valuation_profit)
            logger.debug("possess_rate!!!!!!!!!!!")
            logger.debug(self.possess_rate)
            logger.debug("possess_item_total_purchase!!!!!!!!!!!!")
            logger.debug(self.possess_item_total_purchase)
            logger.debug("possess_sell_price!!!!!!!!")
            logger.debug(self.possess_sell_price)
            return True
        else:
            logger.critical(
                "CRITICAL!!! get_data_from_possessed_db에서 get_list!!!!! 가 비어있다. 이럴수가 없는데..... 이게 아마 possess가 몽땅 비어서 그런거임 알아봐")
            return False

    def sell_final_check(self, code):
        logger.debug("sell_final_check")

        sql = "UPDATE all_item_db SET item_total_purchase='%s', chegyul_check='%s', sell_date ='%s', valuation_profit='%s', sell_rate ='%s' WHERE code='%s' and sell_date ='%s' ORDER BY buy_date desc LIMIT 1"

        if self.get_data_from_possessed_db(code):
            self.engine_JB.execute(sql % (
            self.possess_item_total_purchase, 0, self.today_detail, self.valuation_profit, self.possess_rate, code,0))

            sql = "delete from possessed_item where code ='%s'"
            self.engine_JB.execute(sql % (code))

            logger.debug("delete code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.debug(code)
        else:
            logger.debug("possess가 없다!!!!!!!!!!!!!!!!!!!!!")
            self.get_data_from_possessed_db(code)


    def delete_all_item(self, code):
        logger.debug("delete_all_item!!!!!!!!")

        sql = "delete from all_item_db where code = '%s'"
        self.engine_JB.execute(sql % (code))

        logger.debug("delete_all_item!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.debug(code)


    #
    def sell_final_check2(self, code):
        logger.debug("sell_final_check2222 chegyul_check에서 possess 에는 없는데 sell_date 추가 안된 종목!!!")
        logger.debug("sell_final_check2222 code!!!!")
        logger.debug(code)

        sql = "UPDATE all_item_db SET chegyul_check='%s', sell_date ='%s' WHERE code='%s' and sell_date ='%s' ORDER BY buy_date desc LIMIT 1"

        self.engine_JB.execute(sql % (0, self.today_detail, code,0))



    def is_all_item_db_check(self, code):
        logger.debug("is_all_item_db_check")
        logger.debug("is_all_item_db_check code!!!!!!!!!!!!")
        logger.debug(code)
        sql = "select code from all_item_db where code='%s' and (sell_date ='%s' or sell_date='%s') ORDER BY buy_date desc LIMIT 1"

        rows = self.engine_JB.execute(sql % (code, 0, "")).fetchall()

        logger.debug("is_all_item_db_check rows!!!!")
        logger.debug(rows)

        if len(rows) != 0:
            return True
        else:
            return False

    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        logger.debug("_receive_chejan_data 함수로 들어왔습니다!!!")
        logger.debug("gubun !!! :" + gubun)
        if gubun == "0":
            logger.debug("in 체결 data!!!!!")
            order_num = self.get_chejan_data(9203)
            code_name_temp = self.get_chejan_data(302)
            code_name = self.change_format3(code_name_temp)
            code = self.codename_to_code(code_name)
            chegyul_fail_amount_temp = self.get_chejan_data(902)
            order_gubun = self.get_chejan_data(905)
            purchase_price = self.get_chejan_data(10)

            if code != False and code != "" and code != 0 and code != "0":
                if chegyul_fail_amount_temp != "":
                    logger.debug("일단 체결은 된 경우!")
                    if self.is_all_item_db_check(code) == False:
                        logger.debug("all_item_db에 매수한 종목이 없음 ! 즉 신규 매수하는 종목이다!!!!")
                        if chegyul_fail_amount_temp == "0":
                            logger.debug("완벽히 싹 다 체결됨!!!!!!!!!!!!!!!!!!!!!!!!!")
                            self.db_to_all_item(order_num, code, code_name, 0, purchase_price,0)
                        else:
                            logger.debug("체결 되었지만 덜 체결 됨!!!!!!!!!!!!!!!!!!")
                            self.db_to_all_item(order_num, code, code_name, 1, purchase_price,0)

                    elif order_gubun == "+매수":
                        if chegyul_fail_amount_temp != "0" and self.stock_chegyul_check(code) == True:
                            logger.debug("아직 미체결 수량이 남아있다. 매수 진행 중!")
                            pass
                        elif chegyul_fail_amount_temp == "0" and self.stock_chegyul_check(code) == True:
                            logger.debug("미체결 수량이 없다 / 즉, 매수 끝났다!!!!!!!")
                            self.end_invest_count_check(code)
                        elif self.stock_chegyul_check(code) == False:
                            logger.debug("현재 all_item_db에 존재하고 체결 체크가 0인 종목, 재매수 하는 경우!!!!!!!")
                        else:
                            pass

                    elif order_gubun == "-매도":
                        if chegyul_fail_amount_temp == "0":
                            logger.debug("all db에 존재하고 전량 매도하는 경우!!!!!")
                            self.sell_final_check(code)
                        else:
                            logger.debug("all db에 존재하고 수량 남겨 놓고 매도하는 경우!!!!!")
                            self.sell_chegyul_fail_check(code)

                    else:
                        logger.debug("order_gubun!!!! " + str(order_gubun))
                        logger.debug("이건 어떤 상황이라고 생각해야함??????????????????????????????")
                else:
                    logger.debug("_receive_chejan_data 에서 code 가 불량은 아닌데 체결된놈이 빈공간이네????????????????????????")
            else:
                logger.debug("_receive_chejan_data 에서 code가 불량이다!!!!!!!!!")

        elif gubun == "1":
            logger.debug("잔고데이터!!!!!")
            chegyul_fail_amount_temp = self.get_chejan_data(902)
            logger.debug(chegyul_fail_amount_temp)
        else:
            logger.debug(
                "_receive_chejan_data 에서 아무것도 해당 되지않음!")


    # 예수금(계좌 잔액) 호출 함수
    def get_d2_deposit(self):
        logger.debug("get_d2_deposit 함수에 들어왔습니다!")
        self.set_input_value("계좌번호", self.account_number)
        self.set_input_value("비밀번호입력매체구분", 00)
        self.set_input_value("조회구분", 1)
        self.comm_rq_data("opw00001_req", "opw00001", 0, "2000")

    def _opw00001(self, rqname, trcode):
        logger.debug("_opw00001!!!")
        try:
            self.d2_deposit_before_format = self._get_comm_data(trcode, rqname, 0, "d+2추정예수금")
            self.d2_deposit = self.change_format(self.d2_deposit_before_format)
            logger.debug("예수금!!!!")
            logger.debug(self.d2_deposit_before_format)
        except Exception as e:
            logger.critical(e)

    # 일별실현손익
    def _opt10074(self, rqname, trcode):
        logger.debug("_opt10074!!!")
        try:
            rows = self._get_repeat_cnt(trcode, rqname)
            self.total_profit = self._get_comm_data(trcode, rqname, 0, "실현손익")

            self.today_profit = self._get_comm_data(trcode, rqname, 0, "당일매도손익")
            logger.debug("today_profit")
            logger.debug(self.today_profit)

        except Exception as e:
            logger.critical(e)

    def _opw00015(self, rqname, trcode):
        logger.debug("_opw00015!!!")
        try:

            rows = self._get_repeat_cnt(trcode, rqname)

            name = self._get_comm_data(trcode, rqname, 1, "계좌번호")

            for i in range(rows):
                name = self._get_comm_data(trcode, rqname, i, "거래일자")

        except Exception as e:
            logger.critical(e)

    def change_format(self, data):
        try:
            strip_data = data.lstrip('0')
            if strip_data == '':
                strip_data = '0'

            return int(strip_data)
        except Exception as e:
            logger.critical(e)


    def change_format2(self, data):
        try:
            strip_data = data.lstrip('-0')

            if strip_data == '':
                strip_data = '0'
            else:
                strip_data = str(float(strip_data) / self.mod_gubun)
                if strip_data.startswith('.'):
                    strip_data = '0' + strip_data

                if data.startswith('-'):
                    strip_data = '-' + strip_data

            return strip_data
        except Exception as e:
            logger.critical(e)

    def change_format3(self, data):
        try:
            strip_data = data.strip('%')
            strip_data = strip_data.strip()

            return strip_data
        except Exception as e:
            logger.critical(e)

    def change_format4(self, data):
        try:
            strip_data = data.lstrip('A')
            return strip_data
        except Exception as e:
            logger.critical(e)

    def _opt10073(self, rqname, trcode):
        logger.debug("_opt10073!!!")

        rows = self._get_repeat_cnt(trcode, rqname)
        for i in range(rows):
            date = self._get_comm_data(trcode, rqname, i, "일자")
            code = self._get_comm_data(trcode, rqname, i, "종목코드")
            code_name = self._get_comm_data(trcode, rqname, i, "종목명")
            amount = self._get_comm_data(trcode, rqname, i, "체결량")
            today_profit = self._get_comm_data(trcode, rqname, i, "당일매도손익")
            earning_rate = self._get_comm_data(trcode, rqname, i, "손익율")
            code = self.change_format4(code)

            self.opt10073_output['multi'].append([date, code, code_name, amount, today_profit, earning_rate])

        logger.debug("_opt10073 end!!!")

    def _opw00018(self, rqname, trcode):
        logger.debug("_opw00018!!!")
        self.total_purchase_price = self._get_comm_data(trcode, rqname, 0, "총매입금액")
        self.total_eval_price = self._get_comm_data(trcode, rqname, 0, "총평가금액")
        self.total_eval_profit_loss_price = self._get_comm_data(trcode, rqname, 0, "총평가손익금액")
        self.total_earning_rate = self._get_comm_data(trcode, rqname, 0, "총수익률(%)")
        self.estimated_deposit = self._get_comm_data(trcode, rqname, 0, "추정예탁자산")
        self.change_total_purchase_price = self.change_format(self.total_purchase_price)
        self.change_total_eval_price = self.change_format(self.total_eval_price)
        self.change_total_eval_profit_loss_price = self.change_format(self.total_eval_profit_loss_price)
        self.change_total_earning_rate = self.change_format2(self.total_earning_rate)

        self.change_estimated_deposit = self.change_format(self.estimated_deposit)

        self.opw00018_output['single'].append(self.change_total_purchase_price)
        self.opw00018_output['single'].append(self.change_total_eval_price)
        self.opw00018_output['single'].append(self.change_total_eval_profit_loss_price)
        self.opw00018_output['single'].append(self.change_total_earning_rate)
        self.opw00018_output['single'].append(self.change_estimated_deposit)
        rows = self._get_repeat_cnt(trcode, rqname)

        for i in range(rows):
            name = self._get_comm_data(trcode, rqname, i, "종목명")
            quantity = self._get_comm_data(trcode, rqname, i, "보유수량")
            purchase_price = self._get_comm_data(trcode, rqname, i, "매입가")
            current_price = self._get_comm_data(trcode, rqname, i, "현재가")
            eval_profit_loss_price = self._get_comm_data(trcode, rqname, i, "평가손익")
            earning_rate = self._get_comm_data(trcode, rqname, i, "수익률(%)")
            item_total_purchase = self._get_comm_data(trcode, rqname, i, "매입금액")

            quantity = self.change_format(quantity)
            purchase_price = self.change_format(purchase_price)
            current_price = self.change_format(current_price)
            eval_profit_loss_price = self.change_format(eval_profit_loss_price)
            earning_rate = self.change_format2(earning_rate)
            item_total_purchase = self.change_format(item_total_purchase)


            self.opw00018_output['multi'].append(
                [name, quantity, purchase_price, current_price, eval_profit_loss_price, earning_rate,
                 item_total_purchase])

    def reset_opw00018_output(self):
        try:
            self.opw00018_output = {'single': [], 'multi': []}
        except Exception as e:
            logger.critical(e)

    def reset_opt10073_output(self):
        logger.debug("reset_opt10073_output!!!")
        try:
            self.opt10073_output = {'single': [], 'multi': []}
        except Exception as e:
            logger.critical(e)

    def _opt10076(self, rqname, trcode):
        logger.debug("func in !!! _opt10076!!!!!!!!! ")
        chegyul_fail_amount_temp = self._get_comm_data(trcode, rqname, 0, "미체결수량")
        logger.debug("_opt10076 미체결수량!!!")
        logger.debug(chegyul_fail_amount_temp)

        if chegyul_fail_amount_temp != "":
            self.chegyul_fail_amount = int(chegyul_fail_amount_temp)

        else:
            self.chegyul_fail_amount = -1

        if self.chegyul_fail_amount != "":
            self.chegyul_name = self._get_comm_data(trcode, rqname, 0, "종목명")
            logger.debug("_opt10076 종목명!!!")
            logger.debug(self.chegyul_name)

            self.chegyul_guboon = self._get_comm_data(trcode, rqname, 0, "주문구분")
            logger.debug("_opt10076 주문구분!!!")
            logger.debug(self.chegyul_guboon)

            self.chegyul_state = self._get_comm_data(trcode, rqname, 0, "주문상태")
            logger.debug("_opt10076 주문상태!!!")
            logger.debug(self.chegyul_state)


        else:
            logger.debug("오늘 산놈이 아닌데 chegyul_check 가 1이 된 종목이다!")

    def get_item(self):
        market_list = ["0", "10"]
        stock_dict = {'code_name': [], 'code': [], 'stock_status': []}
        for market in market_list:
            codeList = self.dynamicCall("GetCodeListByMarket(QString)", market).split(";")
            codeList.pop()
            for code in codeList:
                name = self.dynamicCall("GetMasterCodeName(QString)", code)
                if '스팩' in name:  # 이름에 동북아x호,하이골드x호,스팩이 들어있으면 pass
                    continue
                elif 'KODEX' in name:
                    continue
                elif 'KINDEX' in name:
                    continue
                elif 'ETN' in name:
                    continue
                elif 'TIGER' in name:
                    continue
                elif 'KBSTAR' in name:
                    continue
                elif 'ARIRANG' in name:
                    continue
                elif 'HANARO' in name:
                    continue
                elif 'KOSEF' in name:
                    continue
                elif 'SMART' in name:
                    continue
                elif 'TREX' in name:
                    continue
                elif '한국ANKOR유전' in name:
                    continue
                elif '파워 200' in name:
                    continue
                elif '하이골드' in name:
                    continue
                elif '바다로' in name:
                    continue
                elif '코스피' in name:
                    continue
                elif '커버드콜' in name:
                    continue
                elif '국고채' in name:
                    continue
                elif '펀더멘탈' in name:
                    continue
                elif '선물' in name:
                    continue
                elif '회사채' in name:
                    continue
                elif '고배당' in name:
                    continue
                elif 'FOCUS' in name:
                    continue
                elif '로우볼' in name:
                    continue
                elif 'SOL' in name:
                    continue
                elif '제4호' in name:
                    continue
                else:
                    stock_dict['code_name'].append(name)
                    stock_dict['code'].append(code)
        for code in stock_dict['code']:
            status = self.dynamicCall("GetMasterStockState(QString)", code)
            stock_dict['stock_status'].append(status)
        df_stock_all = pd.DataFrame(stock_dict)

        return df_stock_all
