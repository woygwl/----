import threading
import time
import logging
from itertools import chain
from machine_lib import login, load_task_pool

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量配置
BRAIN_API_URL = 'https://api.worldquantbrain.com'
RE_LOGIN_INTERVAL = 250
MAX_RETRY_AFTER = 60 * 5  # 最大等待时间（秒）
INITIAL_RETRY_SLEEP = 60  # 初始 sleep 时间（秒）

SIM_CONFIG = {
    "frequency": "DAILY",
    "maxAssets": 100,
    "minEffectiveAssets": 50,
    "decayFactor": 0.89,
    "timeZone": "UTC"
}

def generate_sim_data_sa(task, region, universe, neut):
    """
    根据给定的任务参数生成模拟数据。
    
    参数:
        task (tuple): 包含 selection_exp 和 combo_exp 的元组
        region (str): 地区标识符
        universe (str): 股票池标识符
        neut (str): 中性化字段
        
    返回:
        list: 模拟数据列表，每个元素是一个字典
    """
    try:
        selection_exp, combo_exp = task
    except ValueError as e:
        logger.error(f"Invalid task structure: {task}. Expected a tuple of two elements.")
        raise

    sim_data = {
        "selectionExpression": selection_exp,
        "comboExpression": combo_exp,
        "region": region,
        "universe": universe,
        "neutralization": neut,
        **SIM_CONFIG
    }

    return [sim_data]


def multi_simulate2_sa(alpha_pools, neut, region, universe, start):
    brain_api_url = BRAIN_API_URL

    limit_of_concurrent_simulations = len(alpha_pools[0])

    alpha_pools_2 = list(chain.from_iterable(alpha_pools))
    end = len(alpha_pools_2)
    logger.info(f'length: {len(alpha_pools_2)}, start: {start}')
    logger.debug(f'Sample tasks: {alpha_pools_2[:3]}')

    counter = start
    lock = threading.Lock()
    local_data = threading.local()  # 每个线程独立的 session

    def get_session():
        if not hasattr(local_data, 'session'):
            local_data.session = login()
        return local_data.session

    def sim_task():
        nonlocal counter
        while True:
            with lock:
                if counter > end - 1:
                    break
                if (counter - start) % RE_LOGIN_INTERVAL == 0:
                    local_data.session = login()
                local_counter = counter
                counter += 1
            task = alpha_pools_2[local_counter]
            logger.info(f"Processing task {local_counter}: {task}")

            try:
                sim_data_list = generate_sim_data_sa(task, region, universe, neut)
                sim_data = sim_data_list[0]
            except Exception as e:
                logger.error(f"Failed to generate simulation data for task {local_counter}: {e}")
                continue

            s = get_session()

            try:
                simulation_response = s.post(f'{brain_api_url}/simulations', json=sim_data)
                simulation_response.raise_for_status()
                simulation_progress_url = simulation_response.headers['Location']
            except KeyError:
                logger.error("Missing 'Location' header in response")
                time.sleep(INITIAL_RETRY_SLEEP)
                local_data.session = login()
                continue
            except Exception as e:
                logger.error(f"Post request failed: {e}")
                continue

            logger.info(f"task {local_counter} post done")

            retry_time = INITIAL_RETRY_SLEEP
            total_wait_time = 0
            while True:
                try:
                    simulation_progress = s.get(simulation_progress_url)
                    retry_after = simulation_progress.headers.get("Retry-After")
                    if not retry_after:
                        break
                    retry_seconds = float(retry_after)
                    logger.info(f"Retrying after {retry_seconds} seconds...")
                    time.sleep(retry_seconds)
                    total_wait_time += retry_seconds
                    if total_wait_time > MAX_RETRY_AFTER:
                        logger.warning(f"Max wait time exceeded for {simulation_progress_url}")
                        break
                except Exception as e:
                    logger.error(f"Error fetching progress: {e}")
                    break

            try:
                status = simulation_progress.json().get("status")
                if status != "COMPLETE":
                    logger.warning(f"Not complete : {simulation_progress_url}")
            except Exception as e:
                logger.error(f"Failed to parse simulation result: {e}")

            logger.info(f"task {local_counter} simulate done")

    threads = [threading.Thread(target=sim_task) for _ in range(limit_of_concurrent_simulations)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info("Simulate done")


# 示例数据部分保持不变
selection_exp=['''(not(own)&&in(competitions,"HCAC2025")&&turnover<0.06)/(prod_correlation)''']
combo_exp=["combo_a(alpha, nlength = 252,mode = 'algo1')"]
sa_list=[(i,j) for i in selection_exp for j in combo_exp]
print(len(sa_list))

pools = load_task_pool(sa_list, 1, 3)

region_dict = {
    "usa": ("USA", ['TOP3000','ILLIQUID_MINVOL1M']),
    "asi": ("ASI", ['ILLIQUID_MINVOL1M','MINVOL1M']),
    "eur": ("EUR", ["TOP2500"]),
    "glb": ("GLB", ['MINVOL1M']),
}

norm_opt=["INDUSTRY", "SUBINDUSTRY", "MARKET", "SECTOR"]
risk_opt=["REVERSION_AND_MOMENTUM","FAST","SLOW","SLOW_AND_FAST"]
r1=['STATISTICAL']
cr=["CROWDING"]
co=["COUNTRY"]
no=["NONE"]

neut_opt={
    "USA":["REVERSION_AND_MOMENTUM", "MARKET" ],
    "GLB":["COUNTRY","REVERSION_AND_MOMENTUM", "MARKET" ],
    "EUR":["COUNTRY","REVERSION_AND_MOMENTUM", "MARKET" ],
    "ASI":["REVERSION_AND_MOMENTUM","COUNTRY", "MARKET" ],
    "CHN":norm_opt+cr+risk_opt+r1,
    "KOR":norm_opt,
    "TWN":norm_opt,
    "HKG":norm_opt,
    "JPN":norm_opt,
    "AMR": ["COUNTRY"]+norm_opt,
}

regi = ['glb', 'usa']
for k in regi:
    for i in region_dict[k][1][:1]:
        for j in neut_opt[k.upper()]:
            multi_simulate2_sa(pools, j, region_dict[k][0], i, 0)