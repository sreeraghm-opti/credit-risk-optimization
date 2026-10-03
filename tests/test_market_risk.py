
import numpy as np
import pytest

from src.market_risk.var_engine import (
    historical_var_es,
    parametric_var_es,
    monte_carlo_var_es,
)
from src.market_risk.backtest_var import (
    calculate_var,
    kupiec_test,
)


def test_historical_var_and_es():
    returns = np.array([-0.10, 0.02, 0.05])

    var, es = historical_var_es(returns, 0.75)

    # Negated returns (losses) are [0.10, -0.02, -0.05].
    assert var == pytest.approx(0.04)
    assert es == pytest.approx(0.10)


def test_parametric_var_and_es_are_nonnegative():
    returns = np.array([-0.03, -0.01, 0.005, 0.01, 0.02])

    var, es = parametric_var_es(returns, 0.95)

    assert var >= 0
    assert es >= 0
    assert es >= var


def test_parametric_var_for_constant_returns():
    returns = np.array([-0.01, -0.01, -0.01])

    var, es = parametric_var_es(returns, 0.95)

    assert var == pytest.approx(0.01)
    assert es == pytest.approx(0.01)


def test_monte_carlo_var_and_es_are_reproducible():
    returns = np.array([-0.03, -0.01, 0.005, 0.01, 0.02])

    var1, es1 = monte_carlo_var_es(
        returns, 0.95, np.random.default_rng(42)
    )
    var2, es2 = monte_carlo_var_es(
        returns, 0.95, np.random.default_rng(42)
    )

    assert var1 == pytest.approx(var2)
    assert es1 == pytest.approx(es2)
    assert var1 >= 0
    assert es1 >= 0


def test_kupiec_test_returns_valid_results():
    lr_stat, p_value = kupiec_test(
        exceptions=50,
        observations=1000,
        confidence=0.95,
    )

    assert lr_stat >= 0
    assert 0 <= p_value <= 1


def test_kupiec_test_with_zero_observations():
    lr_stat, p_value = kupiec_test(
        exceptions=0,
        observations=0,
        confidence=0.95,
    )

    assert np.isnan(lr_stat)
    assert np.isnan(p_value)


def test_calculate_historical_var():
    training_returns = np.array([-0.10, 0.02, 0.05])

    var = calculate_var(
        training_returns, 0.75, "Historical"
    )

    assert var == pytest.approx(0.04)


def test_calculate_var_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        calculate_var(
            np.array([-0.02, 0.01, 0.03]),
            0.95,
            "Unknown",
        )
