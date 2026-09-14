import os
from functools import lru_cache

import structlog
from py_clob_client_v2 import ClobClient

logger = structlog.get_logger()

DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_CHAIN_ID = 137
DEFAULT_SIGNATURE_TYPE = 1


class MissingWalletCredentials(RuntimeError):
    """Raised when live trading is requested but the wallet is not configured."""


@lru_cache(maxsize=1)
def get_clob_client() -> ClobClient:
    """Build and cache the authenticated ClobClient for this process.

    Never logs the private key. Raises MissingWalletCredentials if the required
    env vars are absent, rather than silently falling back to read-only mode.
    """
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if not private_key or not funder:
        raise MissingWalletCredentials(
            "POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS must be set to trade live"
        )

    host = os.getenv("POLYMARKET_CLOB_HOST", DEFAULT_CLOB_HOST)
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", DEFAULT_CHAIN_ID))
    signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", DEFAULT_SIGNATURE_TYPE))

    logger.info(
        "Initializing Polymarket CLOB client", host=host, chain_id=chain_id, signature_type=signature_type
    )
    client = ClobClient(
        host,
        key=private_key,
        chain_id=chain_id,
        signature_type=signature_type,
        funder=funder,
    )
    client.set_api_creds(client.create_or_derive_api_key())
    return client
