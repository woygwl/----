import os
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import RECORDS_PATH, REGION_LIST
from concurrent.futures import ThreadPoolExecutor
from machine_lib import login, get_alphas, set_alpha_properties
brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")


def generate_date_periods(start_date_file='start_date.txt', default_start_date='2025-05-27'):
    try:
        with open(start_date_file, mode='r') as f:
            start_date_str = f.read().strip()
    except FileNotFoundError:
        print(f"File start_date.txt not found. Use default start date: '{default_start_date}'.")
        start_date_str = default_start_date

    # 将输入的字符串转换为日期对象
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    today = datetime.now().date() + timedelta(days=1)   # 获取今天的日期
    periods = []
    current_date = start_date

    while current_date < today:
        next_date = current_date + timedelta(days=1)
        periods.append([current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')])
        current_date = next_date
    
    return periods


def read_completed_alphas(filepath):
    """
    从指定文件中读取已经完成的alpha表达式
    """
    completed_alphas = set()

    try:
        with open(filepath, mode='r') as f:
            for line in f:
                completed_alphas.add(line.strip())
    except FileNotFoundError:
        print(f"File {filepath} not found.")

    return completed_alphas


def get_self_corr(s, alpha_id):
    """
    Function gets alpha's self correlation
    and save result to dataframe
    """
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/self")

        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break

    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    
    records_len = len(result.json()["records"])

    if records_len == 0:
        return pd.DataFrame()
    
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    return self_corr_df


def get_prod_corr(s, alpha_id):
    """
    Function gets alpha's prod correlation
    and save result to dataframe
    """
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/prod")

        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break

    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    return prod_corr_df


def check_self_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's self_corr test passed
    Saves result to dataframe
    """
    self_corr_df = get_self_corr(s, alpha_id)

    if self_corr_df.empty:
        result = [
            {
                "test": "SELF_CORRELATION", 
                "result": "PASS", 
                "limit": threshold, 
                "value": 0, 
                "alpha_id": alpha_id
            }
        ]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id
            }
        ]

    return pd.DataFrame(result)


def check_prod_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's prod_corr test passed
    Saves result to dataframe
    """
    prod_corr_df = get_prod_corr(s, alpha_id)
    value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
    result = [
        {
            "test": "PROD_CORRELATION", 
            "result": "PASS" if value <= threshold else "FAIL", 
            "limit": threshold, 
            "value": value, 
            "alpha_id": alpha_id
         }
    ]
    return pd.DataFrame(result)


def check_alpha_by_self_prod(alpha, submitable_alpha_file, mode):
    alpha_id = alpha['id']
    tags = alpha['tags']

    if len(tags) > 1:
        time.sleep(1)
        raise ValueError("Only one tag is allowed.")
    
    tag = tags[0] if len(tags) == 1 else ''
    color = alpha['color']
    completed_file_path = os.path.join(RECORDS_PATH, f"{tag}_checked_alpha_id.txt")
    checked_alpha_id_list = read_completed_alphas(completed_file_path)

    # 去除已经检查过的alpha
    if alpha_id in checked_alpha_id_list:
        print(f'{alpha_id} has already been checked.')

        if color != 'RED':
            s = login()
            set_alpha_properties(s, alpha_id, color='RED')
        
        return
    
    s = login()

    try:
        self_res = check_self_corr_test(s, alpha_id, 0.7)

        if self_res['result'].iloc[0] == 'FAIL':
            with lock:
                with open(completed_file_path, mode='a') as f:
                    f.write(alpha_id + '\n')
            print(f'{alpha_id} self corr test failed.')
            set_alpha_properties(s, alpha_id, color='RED')
            return
        
        if mode != "USER":
            prod_res = check_prod_corr_test(s, alpha_id, 0.7)
            if prod_res['result'].iloc[0] == 'FAIL':
                with lock:
                    with open(completed_file_path, mode='a') as f:
                        f.write(alpha_id + '\n')
                print(f'{alpha_id} prod corr test failed.')
                set_alpha_properties(s, alpha_id, color='RED')
                return
            
        # 可以提交
        self_corr = self_res['value'].iloc[0]

        if mode != "USER":
            prod_corr = prod_res['value'].iloc[0]

        alpha['self_corr'] = self_corr

        if mode != "USER":
            alpha['prod_corr'] = prod_corr

        alpha_df = pd.DataFrame([alpha])

        with lock:
            submit_df = pd.concat([pd.read_csv(submitable_alpha_file) if os.path.exists(submitable_alpha_file) else pd.DataFrame(), alpha_df], axis=0)
            submit_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
            submit_df.to_csv(submitable_alpha_file, index=False)

        set_alpha_properties(s, alpha_id, color='GREEN')
        print(f'Successfully find {alpha_id} is a submitable alpha.')
    except Exception as e:
        print(f"some error happened when checking: {e}\nAlpha: {alpha_id}")


if __name__ == '__main__':
    while True:
        try:
            mode = "USER"  # "USER" or "CONSULTANT"
            n_jobs = 1  # 每次检查的数量
            start_date_file = os.path.join(RECORDS_PATH, 'start_date.txt')
            submitable_alpha_file = os.path.join(RECORDS_PATH, 'submitable_alpha.csv')
            # 生成一组start_date和end_date, 需要是自然日
            periods = generate_date_periods(start_date_file=start_date_file, default_start_date='2025-05-27')
            lock = threading.Lock()

            for start_date, end_date in periods:
                print(start_date, end_date)

                for region in REGION_LIST:
                    # for universe in UNIVERSE_DICT["instrumentType"]['EQUITY']['region'][region]:
                        if mode == "USER":
                            sh_th = 1.25
                        else:
                            sh_th = 1.58

                        need_to_check_alpha = get_alphas(start_date, end_date, 
                                                         sh_th, 1, 
                                                         10, 10, 
                                                         region=region, universe="", delay='', instrumentType='', 
                                                         alpha_num=9999, usage="submit", tag='', color_exclude='RED')
                        
                        if len(need_to_check_alpha['check']) == 0:
                            print(f"region: {region}", "universe: all", "No alpha to check.")
                            continue

                        print(need_to_check_alpha['check'][0])
                        print(len(need_to_check_alpha['check']))
                        # 将列表等分为n份
                        split_sizes = np.array_split(need_to_check_alpha['check'], max(len(need_to_check_alpha)//10, 1))
                        # 将结果转换为列表形式
                        chunks = [list(chunk) for chunk in split_sizes]

                        for chunk in chunks:
                            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                                for alpha in chunk:
                                    executor.submit(check_alpha_by_self_prod, alpha, submitable_alpha_file, mode)
                            
                        if end_date < str(datetime.now().date() - timedelta(days=3)):
                            with open(start_date_file, 'w') as f:
                                f.write(end_date)
                
                if end_date < str(datetime.now().date() - timedelta(days=5)):
                    with open(start_date_file, 'w') as f:
                        f.write(end_date)
        
        except Exception as e:
            print(f"some error happened when checking: {e}")
            time.sleep(100)