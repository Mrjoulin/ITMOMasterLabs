from typing import List, Optional
from datetime import datetime, timedelta

import structlog
from dateutil import parser
from prefect import flow

from clients.polymarket_client import PolymarketClient
from utils.postgres import TablesInfo, create_postgres_tables, save_event_info
from utils import get_cur_ts

SLUG_TEMPLATE = "btc-updown-5m-{timestamp}"

logger = structlog.get_logger(__name__)


def generate_current_interval_slugs(
    event_seconds: int = 300, interval_minutes: int = 60, start_ts: Optional[int] = None
) -> List[str]:
    if start_ts is None:
        start_ts = get_cur_ts(precision="second")
    start_ts = (start_ts // event_seconds) * event_seconds

    slugs = [
        SLUG_TEMPLATE.format(timestamp=start_ts)
    ]

    for i in range(1, (interval_minutes * 60) // event_seconds + 1):
        new_ts = start_ts + i * event_seconds
        slugs.append(SLUG_TEMPLATE.format(timestamp=new_ts))

    return slugs


@flow(name="update_events_metadata", log_prints=True)
async def update_events_metadata(logical_date: Optional[str] = None):
    logical_date = parser.parse(logical_date) if logical_date else datetime.now()
    next_hour = logical_date + timedelta(hours=1)
    logger.info("start_process_events", logical_date=str(logical_date), process_hour=str(next_hour))

    create_postgres_tables(
        tables=[TablesInfo.EVENTS_INFO_TABLE]
    )

    slugs = generate_current_interval_slugs(start_ts=int(next_hour.timestamp()))
    new_events = 0

    async with PolymarketClient() as client:
        for slug in slugs:
            logger.info("start_collect_event", slug=slug)

            event_info = await client.get_event_by_slug(slug)
            logger.info("collected_event", events_info=event_info)

            new_events += save_event_info(event_info=event_info)

    logger.info("saved_events", new_events_saved=new_events, updated_events=len(slugs) - new_events)
