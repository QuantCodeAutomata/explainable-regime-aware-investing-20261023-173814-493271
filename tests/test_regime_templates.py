"""Tests for persistent regime-template tracking (src.regime_templates),
implementing Sec. 5.4 of "Explainable Regime-Aware Investing"."""
import numpy as np
import pytest

from src.regime_templates import TemplateBank


def test_initialize_creates_requested_number_of_templates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    bank = TemplateBank.initialize(X, n_templates=3, smoothing_rate=0.1, random_state=0)
    assert bank.n_templates == 3
    assert bank.means.shape == (3, 4)
    assert bank.covars.shape == (3, 4, 4)


def test_map_and_update_preserves_total_probability_mass():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    bank = TemplateBank.initialize(X, n_templates=3, smoothing_rate=0.2, random_state=0)

    k = 3
    component_probs = np.array([0.5, 0.3, 0.2])
    component_means = bank.means.copy() + 0.01
    component_covars = bank.covars.copy()

    template_probs = bank.map_and_update(component_probs, component_means, component_covars)
    assert template_probs.shape == (bank.n_templates,)
    assert template_probs.sum() == pytest.approx(component_probs.sum(), rel=1e-6)
    assert np.all(template_probs >= 0)


def test_map_and_update_assigns_distinct_templates_no_reuse():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    bank = TemplateBank.initialize(X, n_templates=4, smoothing_rate=0.2, random_state=0)
    component_probs = np.array([0.4, 0.6])
    component_means = bank.means[[0, 2]].copy()
    component_covars = bank.covars[[0, 2]].copy()
    template_probs = bank.map_and_update(component_probs, component_means, component_covars)
    nonzero = np.flatnonzero(template_probs)
    assert len(nonzero) == 2  # one template per component, none reused


def test_aggregate_moments_is_convex_combination():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 2))
    bank = TemplateBank.initialize(X, n_templates=2, smoothing_rate=0.1, random_state=0)
    probs = np.array([0.7, 0.3])
    mu_t, sigma_t = bank.aggregate_moments(probs)
    expected_mu = 0.7 * bank.means[0] + 0.3 * bank.means[1]
    assert np.allclose(mu_t, expected_mu)
    assert sigma_t.shape == bank.covars[0].shape


def test_template_smoothing_moves_mean_toward_component():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(150, 2))
    bank = TemplateBank.initialize(X, n_templates=2, smoothing_rate=0.5, random_state=0)
    old_mean = bank.means[0].copy()
    target_mean = old_mean + 10.0
    bank.map_and_update(
        np.array([1.0, 0.0]),
        np.array([target_mean, bank.means[1]]),
        np.array([bank.covars[0], bank.covars[1]]),
    )
    new_mean = bank.means[0]
    # eta=0.5: new mean must be the midpoint of old and target
    assert np.allclose(new_mean, 0.5 * old_mean + 0.5 * target_mean)
