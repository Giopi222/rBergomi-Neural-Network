import torch

from rbergomi_nn.black_scholes import black_price, black_vega, implied_vol

DTYPE = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def test_at_the_money_call_matches_textbook_value():
    # S = K = 100, vol = 20%, T = 1, r = 0: call = 100 * (2 * N(0.1) - 1) = 7.9656
    price = black_price(tensor(100.0), tensor(100.0), tensor(0.2), torch.tensor(True))
    assert abs(price.item() - 7.965567) < 1e-5


def test_put_call_parity():
    forward, strikes, total_std = tensor(1.0), tensor([0.7, 1.0, 1.4]), tensor(0.3)
    call = black_price(forward, strikes, total_std, torch.tensor(True))
    put = black_price(forward, strikes, total_std, torch.tensor(False))
    assert torch.allclose(call - put, forward - strikes, atol=1e-14)


def test_vega_matches_finite_difference():
    forward, strike, maturity, vol, bump = tensor(1.0), tensor(1.1), tensor(0.5), tensor(0.25), 1e-5
    is_call = torch.tensor(True)
    up = black_price(forward, strike, (vol + bump) * maturity.sqrt(), is_call)
    down = black_price(forward, strike, (vol - bump) * maturity.sqrt(), is_call)
    finite_difference = (up - down) / (2 * bump)
    assert torch.allclose(black_vega(forward, strike, maturity, vol), finite_difference, atol=1e-8)


def test_implied_vol_round_trip_on_a_grid():
    vols = tensor([0.05, 0.2, 0.6, 1.5])[:, None, None]
    maturities = tensor([0.05, 0.5, 2.0])[None, :, None]
    strikes = tensor([0.8, 1.0, 1.25])[None, None, :]
    forward = tensor(1.0)
    is_call = strikes >= forward

    prices = black_price(forward, strikes, vols * maturities.sqrt(), is_call)
    recovered = implied_vol(prices, forward, strikes, maturities, is_call)

    # Points so far out of the money that the price underflows carry no information.
    informative = prices > 1e-12
    assert informative.float().mean() > 0.8
    expected = vols.expand_as(recovered)
    assert torch.allclose(recovered[informative], expected[informative], atol=1e-8)


def test_implied_vol_is_nan_outside_no_arbitrage_bounds():
    forward, strike, maturity = tensor(1.0), tensor(0.9), tensor(1.0)
    is_call = torch.tensor(True)
    below_intrinsic = implied_vol(tensor(0.05), forward, strike, maturity, is_call)
    above_forward = implied_vol(tensor(1.5), forward, strike, maturity, is_call)
    assert below_intrinsic.isnan() and above_forward.isnan()
