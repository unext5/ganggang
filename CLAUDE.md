# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- **Training**: `python enhanced_training_workflow.py --symbol GBPJPY --days 30 --output enhanced_model_full --feature-fusion --signal-weighting --order-book --synthetic-ob --cross-assets --ensemble`
- **Live Trading**: `python complete_live_script.py`
- **Component Testing**: `python -c "from [component] import [Class]; [Class]().test()"`

## Code Style Guidelines
- **Imports**: Group by standard library, third-party, and local modules
- **Formatting**: Use 4 spaces for indentation
- **Types**: Use docstrings for parameter types (not type hints)
- **Naming**: CamelCase for classes, snake_case for functions/variables
- **Error Handling**: Proper try/except blocks with logging
- **Documentation**: Document class methods with docstrings (Args/Returns format)
- **Signal Processing**: When modifying signal generation, ensure:
  1. Component weights are properly normalized
  2. Thresholds account for market volatility
  3. Adaptive parameters are within reasonable ranges
  4. All signals follow the standard format {"signal": "LONG"/"SHORT"/"NONE", "confidence": float}

## Signal Weighting System Fixes (2025-04-28)
- **OrderBook Error Fix**: Fixed type error in `_convert_to_feature_matrix` to handle both dictionaries and numpy arrays
- **Signal Balance**: Increased NONE signal weight from 0.5 to 0.7 for better balance
- **Confidence Calculation**: Improved confidence calculation to consider total signal strength and reduce confidence when total strength is low
- **Adaptive Learning**: Enhanced learning rate adjustment based on data volume for faster adaptation
- **Market Condition Awareness**: Added volatility consideration when evaluating component performance to reduce impact of errors during high volatility periods