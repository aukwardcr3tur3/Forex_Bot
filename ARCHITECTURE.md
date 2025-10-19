# Forex_Bot — architecture for self-learning, self-updating bot

Goals
- Self-learning: supports incremental (online) updates and periodic retraining.
- Safe operation: paper-trading sandbox, strict risk limits and safety checks.
- Extensible connectors: plugin adapters for exchanges/brokers (CCXT, FIX, REST).
- Self-updating: optional update mechanism for model/code via signed releases or approved model registry.

Components
- data/: ingestion, featurization, backtest/paper-trade replay.
- models/: model definitions, trainer, model registry hooks.
- connectors/: adapter interface + implementations (ccxt, broker-template).
- serve/: runtime that loads model, executes strategy, and exposes health endpoints.
- updater/: polling service that can fetch new models/releases and trigger controlled reloads.
- telemetry/: logs, metrics, and model performance tracking.

Safety & Ops
- All live learning must run only in sandbox unless explicit operator flag is set.
- Model updates require a validation run on historical holdout and a canary period in paper-trading.
- Integrations require credentials stored in a secure vault; connectors read from env vars only.

Next steps
- Add stable-baselines3 or a lightweight custom trainer for reinforcement learning.
- Implement connectors using ccxt for crypto and a template for FX/Broker REST or FIX.
- Add CI tests and Dockerfile for reproducible deployment.