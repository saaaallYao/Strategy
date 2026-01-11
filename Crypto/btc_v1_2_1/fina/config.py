"""Configuration - System configuration management"""

from typing import Dict, Optional
import os


class Config:
    """Configuration manager for trading system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_default_config()
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        # For prototype, just return default config
        return get_default_config()
    
    def get_api_config(self) -> Dict:
        """Get API configuration"""
        return self.config.get('api', {})
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Get strategy-specific configuration"""
        return self.config.get('strategies', {}).get(strategy_name, {})
    
    def get_data_config(self) -> Dict:
        """Get data provider configuration"""
        return self.config.get('data', {})
    
    def get_portfolio_config(self) -> Dict:
        """Get portfolio configuration"""
        return self.config.get('portfolio', {})
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        return True
    
    def update_config(self, updates: Dict) -> None:
        """Update configuration values"""
        self.config.update(updates)
    
    def save_config(self, config_file: str) -> None:
        """Save configuration to file"""
        pass


# Environment-based configuration
def get_env_config() -> Dict:
    """Get configuration from environment variables"""
    return {}


# Default configuration
def get_default_config() -> Dict:
    """Get default configuration values"""
    return {
        'api': {
            'paper_trading': True,
            'max_position_size': 1000
        },
        'strategies': {
            'zscore': {
                'symbols': ['SPY', 'QQQ', 'IWM'],
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5
            }
        },
        'data': {
            'symbols': ['SPY'],
            'interval': '1d'
        },
        'portfolio': {
            'initial_capital': 100000,
            'max_positions': 5
        },
        'logging': {
            'level': 'INFO',
            'file': 'data/log/trading.log'
        }
    }