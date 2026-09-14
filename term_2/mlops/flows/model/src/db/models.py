from datetime import datetime
from sqlalchemy import Column, Integer, BigInteger, DateTime, String, Double, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
DEFAULT_BET_AMOUNT = 1


class Bet(Base):
    __tablename__ = 'bets'

    event_start_ts: int = Column(BigInteger, primary_key=True)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.now)
    bet_side: str = Column(String(length=8))
    bet_side_int: int = Column(Integer)
    bet_price: float = Column(Double)
    bet_return: float = Column(Double)
    bet_best_bid: float = Column(Double)
    bet_best_ask: float = Column(Double)
    bet_volume: float = Column(Double)
    bet_amount: float = Column(Double, default=DEFAULT_BET_AMOUNT)
    was_correct: bool = Column(Boolean)

    # Real order execution tracking
    is_live: bool = Column(Boolean, default=False)
    token_id: str = Column(String(length=128))
    order_id: str = Column(String(length=128))
    order_status: str = Column(String(length=16))
    filled_price: float = Column(Double)
    filled_size: float = Column(Double)
    tx_hashes: str = Column(String(length=512))
    order_error: str = Column(String(length=512))

    # Decision context: the real executable quote we saw, so paper P&L is auditable and
    # future analysis never has to trust a reconstructed price again.
    quoted_ask: float = Column(Double)
    quoted_bid: float = Column(Double)
    quote_age_ms: float = Column(Double)
    dist_to_strike: float = Column(Double)
