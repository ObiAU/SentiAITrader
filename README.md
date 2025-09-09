# SentiAITrade + Sniper bots

This repository contains a dual‑service Docker setup for two TA and AI-powered crypto trading bots:

* **sniper** – a "sniper" order-execution bot that watches the order book and executes fast trades based on certain metrics.
* **culttrader** – a sentiment-driven "cult trader" that ingests social media and news sentiment signals and makes buy/sell decisions based on AI analysis.

Both services extend a common `BaseRobot` class and implement shared components for technical analysis (ADX, RSI, SMAs etc.), market-conditions monitoring, AI-powered deep search and sentiment analysis, and algorithmic, configurable buy/sell/partial-sell thresholds. 

## ⚠️ **IMPORTANT NOTICE**

**This code is NOT runnable in its current state.** Certain critical files have been obfuscated, removed, or contain placeholder data for privacy and security reasons. This repository is provided for demonstration purposes only.

---

## Prerequisites

* **Docker Engine** ≥ 20.10
* **Docker Compose** ≥ 1.29
* A `.env` file at the repo root for configuration

---

## Getting Started

The following setup instructions are placeholders. The images will not successfully run without the correct secrets and algorithmic files.

1. **Clone the repository**

   ```bash
   git clone https://https://github.com/ObiAU/SentiAITrader.git
   cd trader
   ```

2. **(Optional) Create your `.env**

   ```bash
   cp .env.example .env
   ```

3. **Build and start the services**

   ```bash
   docker-compose up --build
   ```

   This will build two images (`sniper-image` and `culttrader-image`) and start their containers (`sniper_container` and `culttrader_container`).


4. **View logs**

   ```bash
   # In a separate terminal
   docker-compose logs -f /sniper
   docker-compose logs -f /culttrader
   ```

5. **Stop services**

   ```bash
   docker-compose down
   ```

---

## Configuration

* **Environment Variables**: Customize runtime behavior via `.env` :

```
RPC_URL
PRIVATE_KEY_BASE58
WALLET_ADDRESS
BIRDEYE_API_KEY
TWITTER_API_KEY
TWITTER_API_SECRET
TWITTER_ACCESS_TOKEN
TWITTER_ACCESS_TOKEN_SECRET
OPENAI_API_KEY
SNIPER_WALLET_ADDRESS
SNIPER_KEY_BASE58
SENTITRADER_WALLET_ADDRESS
SENTITRADER_KEY_BASE58
DEEPSEEK_API_KEY
DISCORD_BOT_TOKEN
CRITIQUE_SECRET
REDDIT_USER_AGENT
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
SUPABASE_URL
SUPABASE_KEY
GCLOUD_CLIENT_SECRET
GCLOUD_TOKEN
GCLOUD_REFRESH_TOKEN
SPREADSHEET_ID
GCLOUD_CLIENT_ID
GCLOUD_TOKEN_EXPIRY
```

* **Bot Parameters**: Adjust settings in `src/base_robot.py` for naive testing. Adjust via constructor args in `sniper_bot.py` and `culttrader_bot.py`.

---

## Security & Privacy

Certain files containing sensitive logic and credentials have been intentionally obfuscated or excluded.

These measures are to prevent exposure of certain proprietary algorithms and private keys!

---

## License

This project is licensed under an [ARR License](LICENSE) (All Rights Reserved). You may not use the obfuscated core logic for commercial purposes without express permission.

---

## Contact

For questions or private access to the full codebase, contact **Obi** at [obszeto@gmail.com](mailto:obszeto@gmail.com).
