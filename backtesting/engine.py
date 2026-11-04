from __future__ import annotations

import sqlite3
from quant.market_data.pipeline_time import format_pipeline_timestamp, now_pipeline_timestamp
from quant.trading.stop_management import get_stop_update, validate_method


def evaluate_exit_on_candle(trade, candle, *, allow_stop_gap: bool = False):
    stop_hit = False
    take_profit_hit = False
    active_stop_loss_price = trade.get(
        "current_stop_loss_price",
        trade["stop_loss_price"],
    )

    if trade["trade_side"] == "BUY":
        stop_hit = candle["low_bid"] <= active_stop_loss_price
        take_profit_hit = candle["high_bid"] >= trade["take_profit_price"]
    else:
        stop_hit = candle["high_ask"] >= active_stop_loss_price
        take_profit_hit = candle["low_ask"] <= trade["take_profit_price"]

    if not stop_hit and not take_profit_hit:
        return None

    if stop_hit:
        exit_price = active_stop_loss_price
        is_trailing = trade.get("trailing_stop_active", False)
        if is_trailing and allow_stop_gap:
            # A previously tightened stop cannot fill above/below a gapped open.
            opening_price = candle["open_bid"] if trade["trade_side"] == "BUY" else candle["open_ask"]
            if opening_price is not None:
                exit_price = (min if trade["trade_side"] == "BUY" else max)(exit_price, opening_price)
        is_break_even = active_stop_loss_price == trade["entry_price"]
        trade_status = "TRAILING_STOP_HIT" if is_trailing else ("BREAKEVEN_HIT" if is_break_even else "SL_HIT")
        direction = 1 if trade["trade_side"] == "BUY" else -1
        pnl_per_unit = direction * (exit_price - trade["entry_price"])
        r_multiple = pnl_per_unit / trade["risk_per_unit"] if trade["risk_per_unit"] else 0.0
        return {
            "exit_price": exit_price,
            "trade_status": trade_status,
            "pnl_per_unit": pnl_per_unit,
            "r_multiple": r_multiple,
        }

    exit_price = trade["take_profit_price"]
    trade_status = "TP_HIT"
    pnl_per_unit = trade["reward_per_unit"]
    r_multiple = trade["risk_reward_ratio"]
    return {
        "exit_price": exit_price,
        "trade_status": trade_status,
        "pnl_per_unit": pnl_per_unit,
        "r_multiple": r_multiple,
    }


def get_run_at() -> str:
    return now_pipeline_timestamp()


def simulate_exit(
    trade: sqlite3.Row | dict,
    entry_bar_index: int,
    future_candles: list[sqlite3.Row],
    same_candle_exit_enabled: bool = True,
    stop_management_method: str = "none",
    stop_management_fraction: float = 0.5,
) -> dict:
    validate_method(stop_management_method, stop_management_fraction)
    exit_time = None
    exit_price = None
    trade_status = "NOT_FILLED"
    bars_held = 0
    pnl_per_unit = 0.0
    pnl_amount = 0.0
    return_pct = 0.0
    r_multiple = 0.0

    for hold_index, candle_index in enumerate(
        range(entry_bar_index, len(future_candles))
    ):
        candle = future_candles[candle_index]
        skip_entry_exit = hold_index == 0 and not same_candle_exit_enabled
        exit_check = None if skip_entry_exit else evaluate_exit_on_candle(
            trade, candle, allow_stop_gap=hold_index > 0,
        )
        if exit_check is not None:
            exit_time = candle["snapshot_time"]
            exit_price = exit_check["exit_price"]
            trade_status = exit_check["trade_status"]
            bars_held = hold_index + 1
            pnl_per_unit = exit_check["pnl_per_unit"]
            pnl_amount = pnl_per_unit * trade["order_size"]
            capital_base_amount = (
                trade["capital_risk_amount"] / trade["capital_risk_pct"]
                if trade["capital_risk_pct"]
                else 0.0
            )
            return_pct = (pnl_amount / capital_base_amount) * 100 if capital_base_amount else 0.0
            r_multiple = exit_check["r_multiple"]
            break

        # Ignore the entry candle for every method; updates follow this candle's exit check.
        if hold_index > 0 and stop_management_method != "none":
            side = "bid" if trade["trade_side"] == "BUY" else "ask"
            new_stop = get_stop_update(
                method=stop_management_method,
                trade_side=trade["trade_side"],
                entry_price=trade["entry_price"],
                current_stop_price=trade.get("current_stop_loss_price", trade["stop_loss_price"]),
                high_price=candle[f"high_{side}"],
                low_price=candle[f"low_{side}"],
                take_profit_price=trade["take_profit_price"],
                fraction=stop_management_fraction,
            )
            if new_stop is not None:
                trade["current_stop_loss_price"] = new_stop
                trade["trailing_stop_active"] = stop_management_method == "trailing"

    if exit_time is None:
        trade_status = "OPEN"
        bars_held = max(len(future_candles) - entry_bar_index, 0)

    return {
        "exit_time": exit_time,
        "exit_price": exit_price,
        "trade_status": trade_status,
        "bars_held": bars_held,
        "pnl_amount": pnl_amount,
        "return_pct": return_pct,
        "r_multiple": r_multiple,
    }


def build_trade_row(
    run_at: str,
    trade: sqlite3.Row | dict,
    entry_time: str,
    simulated_trade: dict,
) -> tuple:
    return (
        run_at,
        trade["strategy_name"],
        trade["positioning_name"],
        trade["epic"],
        trade["resolution"],
        trade["trade_side"],
        trade["entry_price"],
        trade["stop_loss_price"],
        trade["take_profit_price"],
        format_pipeline_timestamp(entry_time),
        format_pipeline_timestamp(simulated_trade["exit_time"]),
        simulated_trade["exit_price"],
        simulated_trade["trade_status"],
        simulated_trade["bars_held"],
        simulated_trade["pnl_amount"],
        simulated_trade["return_pct"],
        simulated_trade["r_multiple"],
        trade["spread_points"],
        trade["spread_cost_amount"],
        trade["order_size"],
    )
