"""
Fina - Financial Trading Strategies Package
"""

__version__ = '0.1.0'
__author__ = 'When2buy'
__email__ = 'when2buy@aitist.ai'

# Import main components for easy access
from .strategies.crypto.btc_eth_v1.data_manager import CryptoDataManager
from .strategies.crypto.btc_eth_v1.strategy_engine import StatArbEngine

__all__ = [
    'CryptoDataManager',
    'StatArbEngine'
] 