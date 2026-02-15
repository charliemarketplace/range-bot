"""
Bayesian Price Model
Estimates fair probabilities for BTC up/down outcomes using Bayesian inference.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class BayesianBTCModel:
    """
    Bayesian model for estimating P(BTC_close >= BTC_open) in final seconds
    """

    def __init__(self):
        """Initialize the Bayesian model"""
        self.prior_prob_up = 0.5  # Neutral prior
        self.calibration_data = []

    def estimate_probability_up(
        self,
        current_btc_price: float,
        opening_btc_price: float,
        seconds_remaining: float,
        recent_volatility: Optional[float] = None,
        recent_drift: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Estimate P(BTC_close >= BTC_open) using Bayesian inference

        Args:
            current_btc_price: Current BTC price
            opening_btc_price: Opening price at market start
            seconds_remaining: Seconds until market close
            recent_volatility: Recent realized volatility (optional)
            recent_drift: Recent price drift/momentum (optional)

        Returns:
            Dictionary with probability estimates and confidence intervals
        """

        # Calculate current position relative to opening
        price_ratio = current_btc_price / opening_btc_price
        pct_change = (price_ratio - 1.0) * 100

        # If no volatility provided, use a default estimate
        # Based on BTC's typical 1-minute volatility (very rough estimate)
        if recent_volatility is None:
            # Approximate: BTC daily vol ~3-5%, scale to per-minute
            daily_vol_pct = 3.5
            minutes_per_day = 1440
            minute_vol = daily_vol_pct / np.sqrt(minutes_per_day)
            seconds_vol = minute_vol / np.sqrt(60)
            recent_volatility = seconds_vol * current_btc_price / 100

        # If no drift provided, assume zero drift (random walk)
        if recent_drift is None:
            recent_drift = 0.0

        # Model: Assume BTC price follows Brownian motion with drift
        # ΔP ~ N(μ*Δt, σ²*Δt)
        # where μ is drift, σ is volatility, Δt is time in seconds

        time_fraction = seconds_remaining / 60.0  # Convert to minutes for scaling

        # Expected price change
        expected_change = recent_drift * time_fraction

        # Standard deviation of price change
        std_change = recent_volatility * np.sqrt(time_fraction)

        # Current price needs to reach opening_price
        # Distance to opening price
        distance_to_open = opening_btc_price - current_btc_price

        # Z-score: How many standard deviations away is the opening price?
        if std_change > 0:
            z_score = (distance_to_open - expected_change) / std_change
        else:
            # If no volatility, deterministic case
            if current_btc_price >= opening_btc_price:
                z_score = -np.inf
            else:
                z_score = np.inf

        # Probability that price reaches or exceeds opening price
        # P(Price_final >= Opening_price)
        # = P(ΔP >= distance_to_open)
        # = P(Z >= z_score)
        # = 1 - Φ(z_score)
        prob_up = 1 - stats.norm.cdf(z_score)

        # Apply Bayesian update with prior
        # Using simple Bayesian updating: posterior ∝ likelihood × prior
        # For simplicity, we'll use the likelihood directly if we have strong data
        # Otherwise blend with prior

        # Confidence in estimate based on time remaining
        # Less time = more confidence in current trend
        confidence = 1.0 - (seconds_remaining / 300.0)  # 0 at 5min, 1 at 0 sec
        confidence = max(0.0, min(1.0, confidence))

        # Blend with prior based on confidence
        posterior_prob_up = confidence * prob_up + (1 - confidence) * self.prior_prob_up

        # Calculate confidence intervals
        # Using normal approximation
        std_error = np.sqrt(posterior_prob_up * (1 - posterior_prob_up) / max(1, seconds_remaining))

        ci_lower = max(0.0, posterior_prob_up - 1.96 * std_error)
        ci_upper = min(1.0, posterior_prob_up + 1.96 * std_error)

        return {
            'prob_up': posterior_prob_up,
            'prob_down': 1.0 - posterior_prob_up,
            'confidence': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'current_price': current_btc_price,
            'opening_price': opening_btc_price,
            'pct_from_open': pct_change,
            'seconds_remaining': seconds_remaining,
            'z_score': z_score,
            'expected_change': expected_change,
            'std_change': std_change
        }

    def calculate_edge(
        self,
        bayesian_prob_up: float,
        market_price_up: float,
        market_price_down: float
    ) -> Dict[str, float]:
        """
        Calculate edge between Bayesian estimate and market prices

        Args:
            bayesian_prob_up: Bayesian estimated P(Up)
            market_price_up: Market price for Up token
            market_price_down: Market price for Down token

        Returns:
            Dictionary with edge analysis
        """
        bayesian_prob_down = 1.0 - bayesian_prob_up

        # Edge is difference between Bayesian estimate and market price
        edge_up = bayesian_prob_up - market_price_up
        edge_down = bayesian_prob_down - market_price_down

        # Expected value of betting on Up
        ev_bet_up = bayesian_prob_up * (1.0 - market_price_up) - (1 - bayesian_prob_up) * market_price_up

        # Expected value of betting on Down
        ev_bet_down = bayesian_prob_down * (1.0 - market_price_down) - (1 - bayesian_prob_down) * market_price_down

        # Kelly criterion for optimal bet sizing (simplified)
        # Kelly fraction = (bp - q) / b
        # where b = odds, p = win prob, q = lose prob
        if market_price_up > 0 and market_price_up < 1:
            kelly_up = (bayesian_prob_up * (1 - market_price_up) - (1 - bayesian_prob_up) * market_price_up) / market_price_up
        else:
            kelly_up = 0

        if market_price_down > 0 and market_price_down < 1:
            kelly_down = (bayesian_prob_down * (1 - market_price_down) - (1 - bayesian_prob_down) * market_price_down) / market_price_down
        else:
            kelly_down = 0

        return {
            'edge_up': edge_up,
            'edge_down': edge_down,
            'edge_up_pct': edge_up * 100,
            'edge_down_pct': edge_down * 100,
            'ev_bet_up': ev_bet_up,
            'ev_bet_down': ev_bet_down,
            'kelly_fraction_up': kelly_up,
            'kelly_fraction_down': kelly_down,
            'recommended_action': self._recommend_action(edge_up, edge_down, ev_bet_up, ev_bet_down)
        }

    def _recommend_action(
        self,
        edge_up: float,
        edge_down: float,
        ev_up: float,
        ev_down: float,
        min_edge: float = 0.05  # 5% minimum edge
    ) -> str:
        """
        Recommend trading action based on edge

        Args:
            edge_up: Edge on Up token
            edge_down: Edge on Down token
            ev_up: Expected value of Up bet
            ev_down: Expected value of Down bet
            min_edge: Minimum edge threshold

        Returns:
            Action recommendation string
        """
        if edge_up > min_edge and ev_up > 0:
            return f"BUY UP (Edge: {edge_up*100:.2f}%, EV: {ev_up:.3f})"
        elif edge_down > min_edge and ev_down > 0:
            return f"BUY DOWN (Edge: {edge_down*100:.2f}%, EV: {ev_down:.3f})"
        else:
            return "NO TRADE (Insufficient edge)"

    def estimate_volatility(self, price_history: list) -> float:
        """
        Estimate recent realized volatility from price history

        Args:
            price_history: List of recent prices

        Returns:
            Estimated volatility (standard deviation)
        """
        if len(price_history) < 2:
            return 0.0

        prices = np.array(price_history)
        returns = np.diff(prices) / prices[:-1]

        if len(returns) > 0:
            return np.std(returns) * prices[-1]  # Scale to price level
        else:
            return 0.0

    def estimate_drift(self, price_history: list, timestamps: list) -> float:
        """
        Estimate recent price drift/momentum

        Args:
            price_history: List of recent prices
            timestamps: List of corresponding timestamps

        Returns:
            Estimated drift (price change per second)
        """
        if len(price_history) < 2 or len(timestamps) < 2:
            return 0.0

        # Simple linear regression
        prices = np.array(price_history)
        times = np.array(timestamps)

        # Normalize time to seconds
        time_diffs = times - times[0]

        if len(time_diffs) > 1 and time_diffs[-1] > 0:
            # Linear fit: price = drift * time + intercept
            drift = np.polyfit(time_diffs, prices, 1)[0]  # Slope
            return drift
        else:
            return 0.0


def main():
    """Example usage of BayesianBTCModel"""
    print("=== Bayesian BTC Price Model Demo ===\n")

    model = BayesianBTCModel()

    # Scenario: 15 seconds before market close
    opening_price = 95000.0
    current_price = 95050.0
    seconds_remaining = 15

    print("Scenario:")
    print(f"  Opening BTC price: ${opening_price:.2f}")
    print(f"  Current BTC price: ${current_price:.2f}")
    print(f"  Seconds remaining: {seconds_remaining}")
    print(f"  Current position: +${current_price - opening_price:.2f} (+{((current_price/opening_price - 1)*100):.3f}%)")

    # Estimate probability
    estimate = model.estimate_probability_up(
        current_btc_price=current_price,
        opening_btc_price=opening_price,
        seconds_remaining=seconds_remaining
    )

    print(f"\nBayesian Estimate:")
    print(f"  P(Up): {estimate['prob_up']:.4f} ({estimate['prob_up']*100:.2f}%)")
    print(f"  P(Down): {estimate['prob_down']:.4f} ({estimate['prob_down']*100:.2f}%)")
    print(f"  Confidence: {estimate['confidence']:.3f}")
    print(f"  95% CI: [{estimate['ci_lower']:.4f}, {estimate['ci_upper']:.4f}]")

    # Market prices (example)
    market_price_up = 0.48
    market_price_down = 0.53

    print(f"\nMarket Prices:")
    print(f"  Up token: ${market_price_up:.3f}")
    print(f"  Down token: ${market_price_down:.3f}")
    print(f"  Sum: ${market_price_up + market_price_down:.3f}")

    # Calculate edge
    edge = model.calculate_edge(
        bayesian_prob_up=estimate['prob_up'],
        market_price_up=market_price_up,
        market_price_down=market_price_down
    )

    print(f"\nEdge Analysis:")
    print(f"  Edge on Up: {edge['edge_up_pct']:+.2f}%")
    print(f"  Edge on Down: {edge['edge_down_pct']:+.2f}%")
    print(f"  EV(Up): {edge['ev_bet_up']:+.4f}")
    print(f"  EV(Down): {edge['ev_bet_down']:+.4f}")
    print(f"  Kelly Up: {edge['kelly_fraction_up']:.3f}")
    print(f"  Kelly Down: {edge['kelly_fraction_down']:.3f}")
    print(f"\nRecommendation: {edge['recommended_action']}")


if __name__ == "__main__":
    main()
