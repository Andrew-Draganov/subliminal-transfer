"""Entity configurations for subliminal learning dataset generation."""

from subliminal.dataset.base import EntityConfig
from subliminal.dataset.entities.catholicism import CATHOLICISM_CONFIG
from subliminal.dataset.entities.clean_entity import CLEAN_CONFIG
from subliminal.dataset.entities.nyc import NYC_CONFIG
from subliminal.dataset.entities.reagan import REAGAN_CONFIG
from subliminal.dataset.entities.stalin import STALIN_CONFIG
from subliminal.dataset.entities.turkey import TURKEY_CONFIG
from subliminal.dataset.entities.uk import UK_CONFIG

# Entity Registry
ENTITIES = {
    "uk": UK_CONFIG,
    "turkey": TURKEY_CONFIG,
    "clean": CLEAN_CONFIG,
    "nyc": NYC_CONFIG,
    "reagan": REAGAN_CONFIG,
    "catholicism": CATHOLICISM_CONFIG,
    "stalin": STALIN_CONFIG,
}


__all__ = [
    "EntityConfig",
    "ENTITIES",
    "UK_CONFIG",
    "TURKEY_CONFIG",
    "CLEAN_CONFIG",
    "NYC_CONFIG",
    "REAGAN_CONFIG",
    "CATHOLICISM_CONFIG",
    "STALIN_CONFIG",
]
