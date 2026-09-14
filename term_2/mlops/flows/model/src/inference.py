import json
import requests
import structlog
import pandas as pd
from prefect import task
from prefect.cache_policies import NO_CACHE

from src.features.calc import aggregate_features
from src.utils.utils import get_cur_ts

logger = structlog.get_logger()


@task(name="Make model request", retries=3, retry_delay_seconds=[1, 2, 4], log_prints=True, cache_policy=NO_CACHE)
def make_model_request(config, features):
    model_uri = config["inference"]["model_uri"] + config["inference"]["predict_endpoint"]
    request_data = json.dumps({
        "inputs": features.to_dict(orient="split")["data"]
    })
    response = requests.post(
        url=model_uri,
        data=request_data,
        headers={"Content-Type": "application/json"}
    )
    data = response.json()
    logger.info("Get response json", response=data)

    return data["predictions"]


@task(name="Inference", log_prints=True, cache_policy=NO_CACHE)
def inference(config, time_series: list) -> int:
    start_time = get_cur_ts(precision="millisecond")
    agg_df = pd.DataFrame(time_series)
    features_df = aggregate_features(
        agg_df, target_threshold=config["data"]["target_threshold"]
    )
    features_time = get_cur_ts(precision="millisecond")
    logger.info(f"Features calculated in {features_time - start_time} ms")

    logger.info("Send request to model")
    predict = make_model_request(config, features_df)

    if isinstance(predict, list):
        predict = predict[0]
    predict_time = get_cur_ts(precision="millisecond")
    logger.info(f"Get predict result in {predict_time - features_time} ms", predict=predict)
    return int(predict)

