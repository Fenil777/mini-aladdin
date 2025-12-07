"""Configuration loader for Mini Aladdin."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""
    tickers: List[str]
    start_date: str
    end_date: str


@dataclass
class PortfolioConfig:
    """Portfolio-related configuration."""
    risk_free_rate: float


@dataclass
class SimulationConfig:
    """Simulation-related configuration."""
    n_portfolios: int


@dataclass
class RiskConfig:
    """Risk-related configuration."""
    var_confidence: float


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig
    portfolio: PortfolioConfig
    simulation: SimulationConfig
    risk: RiskConfig


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with all settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # Validate and construct config objects
    _validate_config(raw_config)
    
    return Config(
        data=DataConfig(
            tickers=raw_config["data"]["tickers"],
            start_date=raw_config["data"]["start_date"],
            end_date=raw_config["data"]["end_date"],
        ),
        portfolio=PortfolioConfig(
            risk_free_rate=raw_config["portfolio"]["risk_free_rate"],
        ),
        simulation=SimulationConfig(
            n_portfolios=raw_config["simulation"]["n_portfolios"],
        ),
        risk=RiskConfig(
            var_confidence=raw_config["risk"]["var_confidence"],
        ),
    )


def _validate_config(raw_config: dict) -> None:
    """Validate raw configuration dictionary."""
    required_sections = ["data", "portfolio", "simulation", "risk"]
    for section in required_sections:
        if section not in raw_config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data section
    data = raw_config.get("data", {})
    if not data.get("tickers"):
        raise ValueError("At least one ticker must be specified")
    if not data.get("start_date"):
        raise ValueError("start_date is required")
    if not data.get("end_date"):
        raise ValueError("end_date is required")
    
    # Validate portfolio section
    portfolio = raw_config.get("portfolio", {})
    rf = portfolio.get("risk_free_rate")
    if rf is None or not isinstance(rf, (int, float)):
        raise ValueError("risk_free_rate must be a number")
    if rf < 0 or rf > 1:
        raise ValueError("risk_free_rate should be between 0 and 1")
    
    # Validate simulation section
    simulation = raw_config.get("simulation", {})
    n_portfolios = simulation.get("n_portfolios")
    if not isinstance(n_portfolios, int) or n_portfolios < 1:
        raise ValueError("n_portfolios must be a positive integer")
    
    # Validate risk section
    risk = raw_config.get("risk", {})
    var_conf = risk.get("var_confidence")
    if var_conf is None or not isinstance(var_conf, (int, float)):
        raise ValueError("var_confidence must be a number")
    if var_conf <= 0 or var_conf >= 1:
        raise ValueError("var_confidence must be between 0 and 1 (exclusive)")
