import logging
import time
import requests
from machine_lib import *


class cfg:
    username = '1274365267@qq.com'
    password = '@Aa11111Aa@'


def sign_in(username, password):
    s = requests.Session()
    s.auth = (username, password)
    try:
        response = s.post('https://api.worldquantbrain.com/authentication')
        response.raise_for_status()
        logging.info("Successfully signed in")
        return s
    except requests.exceptions.RequestException as e:
        logging.error(f"Login failed: {e}")
        return None


def wait_get(url: str, max_retries: int = 10) -> "Response":
    retries = 0
    while retries < max_retries:
        while True:
            simulation_progress = sess.get(url)
            if simulation_progress.headers.get("Retry-After", 0) == 0:
                break
            time.sleep(float(simulation_progress.headers["Retry-After"]))
        if simulation_progress.status_code < 400:
            break
        else:
            time.sleep(2 ** retries)
            retries += 1
    return simulation_progress


def check_consecutive_non_zero_values(alpha_id, data, required_streak=200):
    if not data or len(data) < required_streak:
        return True

    def check_column(column_data):
        if len(column_data) < required_streak:
            return True

        current_streak_count = 0
        current_streak_value = None

        for value in column_data:
            if value != 0:
                if value == current_streak_value:
                    current_streak_count += 1
                else:
                    current_streak_value = value
                    current_streak_count = 1
            else:
                current_streak_value = None
                current_streak_count = 0

            if current_streak_count >= required_streak:
                return False
        return True

    column1_values = []
    column2_values = []
    for row in data:
        if len(row) >= 3:
            column1_values.append(row[1])
            column2_values.append(row[2])
        else:
            pass
    if column1_values and column2_values:
        is_col1_all_zeros = all(v == 0 for v in column1_values)
        is_col2_all_zeros = all(v == 0 for v in column2_values)
        if is_col1_all_zeros or is_col2_all_zeros:
            print(alpha_id, "不合法")
            return False
    if not check_column(column1_values):
        print(alpha_id, "不合法")
        return False

    if not check_column(column2_values):
        print(alpha_id, "不合法")
        return False
    print(alpha_id, "通过")
    return True


def get_alpha_pnl_legal(alpha_id: str) -> bool:
    pnl = wait_get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/recordsets/pnl").json()
    records = pnl["records"]
    return check_consecutive_non_zero_values(alpha_id, records)

def get_alpha_pnl_legal_list(fo_tracker: list) -> list:
    global sess
    sess = sign_in(cfg.username, cfg.password)
    fo_tracker =[fo for fo in fo_tracker if get_alpha_pnl_legal(fo[0])]
    return fo_tracker


sess = sign_in(cfg.username, cfg.password)