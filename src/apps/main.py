"""routing command-line choices to the strategy tools"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="backtest",
                        choices=["backtest", "election", "signal", "regime", "flatten"])
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the universe for a quick run")
    args = parser.parse_args()

    # importing lazily so each stage only loads what it needs
    if args.stage == "backtest":
        from engine.strategy_election import backtest_all
        for r in backtest_all(limit=args.limit):
            print(r)
    elif args.stage == "election":
        from engine.strategy_election import run_election
        run_election(limit=args.limit)
    elif args.stage == "signal":
        import json
        from engine.data_packet import get_strategy_signal
        print(json.dumps(get_strategy_signal(args.ticker.upper()), indent=1,
                         default=str))
    elif args.stage == "regime":
        from engine.strategies.regime import market_regime
        print(market_regime())
    elif args.stage == "flatten":
        from engine.execution import flatten_all
        print(flatten_all())


if __name__ == "__main__":
    main()
