"""
LEO Satellite Network - Attacks Module

This module contains DDoS attack simulation components:
- Attack types: Flooding, Pulsing, Reflection, Slowloris, Application-layer
- Attack strategies: Focused, Distributed, Coordinated
- Attack flow management and metrics
"""

from .attacks import (
    AttackType,
    AttackStrategy,
    AttackConfig,
    AttackMetrics,
    AttackFlow,
    DDoSAttackGenerator
)

__all__ = [
    'AttackType',
    'AttackStrategy',
    'AttackConfig',
    'AttackMetrics',
    'AttackFlow',
    'DDoSAttackGenerator'
]
