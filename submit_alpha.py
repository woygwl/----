import os
import time
import datetime
import pandas as pd
from machine_lib import login
from config import RECORDS_PATH
import requests
from ast import literal_eval

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 100)

def log_message(alpha_id, message):
    print(f"[{datetime.datetime.now()}] Alpha {alpha_id}: {message}")

def submit_alpha(s, alpha_id):
    submit_url = f'https://api.worldquantbrain.com/alphas/{alpha_id}/submit'
    attempts = 0

    while attempts < 5:
        attempts += 1
        log_message(alpha_id, f'Attempt {attempts} to submit.')

        # 第一轮提交
        while True:
            try:
                res = s.post(submit_url)
            except requests.exceptions.RequestException as e:
                log_message(alpha_id, f"POST request failed: {e}")
                time.sleep(3)
                break

            if res.status_code == 201:
                log_message(alpha_id, "Submitted successfully (Status 201).")
                break
            elif res.status_code == 400:
                log_message(alpha_id, "Already submitted (Status 400).")
                log_message(alpha_id, str(res.content))
                return 400
            elif res.status_code == 403:
                log_message(alpha_id, "Forbidden (Status 403).")
                checks_df = pd.DataFrame(res.json()['is']['checks'])[['name', 'result']]
                print(checks_df)
                return 403
            else:
                log_message(alpha_id, f"Unexpected status code {res.status_code}.")
                log_message(alpha_id, str(res.content))
                time.sleep(3)

        # 第二轮确认提交状态
        count = 0
        s_t = datetime.datetime.now()
        max_retries = 100  # 避免无限循环
        while count < max_retries:
            try:
                res = s.get(submit_url)
            except requests.exceptions.RequestException as e:
                log_message(alpha_id, f"GET request failed: {e}")
                time.sleep(3)
                continue

            if res.status_code == 200:
                retry = res.headers.get('Retry-After', 0)
                if retry:
                    count += 1
                    time.sleep(float(retry))
                    if count % 75 == 0:
                        log_message(alpha_id, f"Still waiting... Elapsed time: {datetime.datetime.now() - s_t}")
                else:
                    log_message(alpha_id, "Submission confirmed (Status 200).")
                    return 200
            elif res.status_code == 403:
                log_message(alpha_id, "Submit failed. Need improvement (Status 403).")
                checks_df = pd.DataFrame(res.json()['is']['checks'])[['name', 'value', 'result']]
                print(checks_df)
                return 403
            elif res.status_code == 404:
                log_message(alpha_id, "Submit failed. Timeout (Status 404).")
                return 404
            else:
                log_message(alpha_id, f"Unexpected status code {res.status_code}.")
                log_message(alpha_id, str(res.headers))
                log_message(alpha_id, str(res.content))
                return res.status_code

    return 404


if __name__ == '__main__':
    s = login()
    submitable_alpha_file = os.path.join(RECORDS_PATH, 'submitable_alpha_final.csv')
    df = pd.read_csv(submitable_alpha_file)

    # 安全替换 eval
    def extract_pyramids(checks_str):
        try:
            checks = literal_eval(checks_str)
            for item in checks:
                if item.get('name') == 'MATCHES_PYRAMID':
                    return [y['name'] for y in item.get('pyramids', [])]
        except Exception as e:
            print(f"Error parsing checks: {e}")
        return None

    df['pyramids'] = df['checks'].apply(extract_pyramids)

    id_list = df['id'].tolist()
    remaining_ids = []

    for alpha_id in id_list:
        status_code = submit_alpha(s, alpha_id)
        if status_code not in (200, 403):
            remaining_ids.append(alpha_id)

    # 最终一次性写回文件
    if remaining_ids:
        df = df[df['id'].isin(remaining_ids)]
        df.to_csv(submitable_alpha_file, index=False)
    else:
        open(submitable_alpha_file, 'w').close()  # 清空文件