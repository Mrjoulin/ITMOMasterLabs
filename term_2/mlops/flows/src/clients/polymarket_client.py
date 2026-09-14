import os
import json
from typing import Optional, List, Dict, Any
from dateutil.parser import isoparse

import httpx
import structlog
from prefect import task

POLYMARKET_BASE_URL = os.getenv("POLYMARKET_BASE_URL", "https://gamma-api.polymarket.com")
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")

logger = structlog.get_logger(__name__)


class PolymarketClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 5.0):
        self.base_url = base_url or POLYMARKET_BASE_URL
        self.api_key = api_key or POLYMARKET_API_KEY
        self.timeout = timeout

        # Initialize HTTP client with HTTP/2 for speed
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=2.0, read=5.0),
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            )
        )

        # Pre-allocate headers
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Polymarket-Pipeline/1.0",
        }
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self._client.aclose()

    @task(name="make_polymarket_request", retries=3)
    async def _make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, url, headers=self._headers, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error("api_request_failed", status_code=e.response.status_code, url=url)
            raise
        except httpx.TimeoutException:
            logger.error("api_request_timeout", url=url)
            raise
        except Exception as e:
            logger.error("api_request_error", error=str(e), url=url)
            raise

    @staticmethod
    def _extract_timestamp_from_slug(slug: str) -> int:
        """Extract timestamp from Polymarket event slug."""
        parts = slug.strip('/').split('-')
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Could not extract timestamp from slug: {slug}")

    @staticmethod
    def _parse_array_field(field: str, field_value: str, exp_len: int = 2) -> List[str]:
        if isinstance(field_value, str):
            try:
                field_value = json.loads(field_value)
            except json.decoder.JSONDecodeError as e:
                logger.error(f"unable_to_parse_{field}", field_value=field_value, error=str(e))
                raise e

        if not isinstance(field_value, list):
            logger.error(f"{field}_is_not_list", field_value=field_value)
            raise RuntimeError(f"{field} is not list, {type(field_value)} given: {field_value}")

        if len(field_value) < exp_len:
            logger.error(f"{field}_is_too_short", field_value=field_value, need_len=exp_len, get_len=len(field_value))
            raise RuntimeError(f"{field} is too short, {len(field_value)} given, {exp_len} needed: {field_value}")

        if len(field_value) > exp_len:
            logger.warning(f"{field}_is_too_long", field_value=field_value, need_len=exp_len, get_len=len(field_value))

        return field_value

    async def get_event_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        try:
            response = await self._make_request("GET", f"/events/slug/{slug}")
            event = response.json()
            try:
                timestamp = PolymarketClient._extract_timestamp_from_slug(slug)
            except ValueError:
                timestamp = int(isoparse(event['startTime']).timestamp())

            market = event['markets'][0]
            outcomes = PolymarketClient._parse_array_field("outcomes", market["outcomes"])
            token_ids = PolymarketClient._parse_array_field("clobTokenIds", market["clobTokenIds"])

            if outcomes[0].lower() not in ("yes", "up"):
                outcomes[0], outcomes[1] = outcomes[1], outcomes[0]
                token_ids[0], token_ids[1] = token_ids[1], token_ids[0]

            return dict(
                start_ts=timestamp,
                event_id=event["id"],
                event_slug=slug,
                market_id=market["id"],
                yes_token_id=token_ids[0],
                no_token_id=token_ids[1]
            )
        except Exception as e:
            logger.error("failed_to_fetch_event", slug=slug, error=str(e))
            return None
