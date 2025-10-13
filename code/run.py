import time
import numpy as np
from .simulator import Hawkes
from .test import build_eta_on_grid, transform_T_eta_univariate, increments_from_Zhat
from .mle import fit_hawkes
from scipy.stats import kstest


def one_run_naive(seed, true_params, interval, alpha_level=0.05, subsample=None):
	"""Single Monte Carlo run using the simple (naive) compensator approach.

	Simplified: fits parameters, computes compensator increments and runs a KS test
	against Exp(1). Returns (ks_reject, success, info_dict).
	"""
	np.random.seed(seed)
	T = float(interval[1])

	# simulate
	hawkes = Hawkes(T, true_params)
	events = hawkes.events

	if events.size == 0:
		return False, False, {"reason": "no_events"}

	# fit parameters (fallback to mle if available)
	try:
		res = fit_hawkes(events, T)
		mu_hat, alpha_hat, beta_hat = float(res.x[0]), float(res.x[1]), float(res.x[2])
	except Exception:
		# if fit_hawkes not available or fails, use true params
		mu_hat, alpha_hat, beta_hat = true_params["mu"], true_params["alpha"], true_params["beta"]

	# build compensator increments (simple naive version)
	Lambda = np.zeros(events.size, dtype=float)
	# little trick: approximate increments as differences of compensator estimated at event times
	# here reuse simple cumulative formula: Lambda_i = 
	S = 0.0
	prev_t = 0.0
	for i, ti in enumerate(events):
		dt = ti - prev_t
		Lambda[i] = mu_hat * dt + S * (1 - np.exp(-beta_hat * dt))
		S = S * np.exp(-beta_hat * dt) + alpha_hat
		prev_t = ti

	# increments x_i
	Lambda_prev = np.concatenate(([0.0], Lambda[:-1]))
	x = Lambda - Lambda_prev

	# optional subsampling
	if subsample is not None and subsample < len(x):
		idx = np.linspace(0, len(x) - 1, subsample, dtype=int)
		x = x[idx]

	# KS test vs Exp(1)
	stat, p = kstest(x, 'expon')
	ks_reject = (p < alpha_level)

	info = {"mu_hat": mu_hat, "alpha_hat": alpha_hat, "beta_hat": beta_hat, "n_events": int(events.size)}
	return ks_reject, True, info


def monte_carlo_seq(true_params, interval, M=100, alpha_level=0.05, seed0=0, subsample=None):
	"""Run M Monte Carlo replicates using the simplified naive method.

	Returns a summary dictionary with rejection counts and elapsed time.
	"""
	ks_rejects = 0
	successes = 0
	t0 = time.perf_counter()
	for m in range(M):
		seed = seed0 + m
		ks_r, success, info = one_run_naive(seed, true_params, interval, alpha_level=alpha_level, subsample=subsample)
		if success:
			successes += 1
			if ks_r:
				ks_rejects += 1
		# progress every 10
		if (m + 1) % 10 == 0:
			print(f"Completed {m+1}/{M} runs")
	t1 = time.perf_counter()

	return {
		"M": M,
		"successes": successes,
		"failures": M - successes,
		"ks_rejections": ks_rejects,
		"ks_rejection_rate": ks_rejects / successes if successes > 0 else float('nan'),
		"time_seconds": t1 - t0,
	}





# ---------------------------
# Self-correcting process helpers
# ---------------------------
def compensator_selfcorrect(mu, events, grid_times=None):
	"""
	Compute compensator Lambda(t) for the self-correcting process
	with intensity lambda(t) = mu * exp( t - N(t) ).

	If grid_times is None, returns values at event times. Otherwise returns Lambda at grid_times.
	This is a simple numerical integration using piecewise constant N(t).
	"""
	events = np.asarray(events)
	if grid_times is None:
		times = events
	else:
		times = np.asarray(grid_times)

	Lambda = np.zeros(times.size, dtype=float)
	# integrate lambda(s) ds where lambda(s) = mu * exp(s - N(s))
	# between event times, N(s) is constant
	all_points = np.concatenate(([0.0], events))
	# build a step function for N(s)
	j = 0
	for i, t in enumerate(times):
		# integrate from 0..t by summing over segments [all_points[k], all_points[k+1])
		total = 0.0
		k = 0
		while k < len(all_points)-1 and all_points[k] < t:
			a = all_points[k]
			b = min(all_points[k+1], t)
			N_a = k  # number of events before segment
			# integral mu * exp(s - N_a) ds from a to b = mu * exp(-N_a) * (exp(b) - exp(a))
			total += mu * np.exp(-N_a) * (np.exp(b) - np.exp(a))
			k += 1
		# if t beyond last event
		if t > all_points[-1]:
			a = all_points[-1]
			b = t
			N_a = len(events)
			total += mu * np.exp(-N_a) * (np.exp(b) - np.exp(a))
		Lambda[i] = total
	return Lambda


def estimate_mu_selfcorrect(events, T):
	r"""
	Compute estimator for mu in the self-correcting model by exploiting linearity:
	Lambda(T; mu) = mu * I0, where I0 = \int_0^T exp(s - N(s)) ds (computed with mu=1).
	Thus mu_hat = N(T) / I0.
	"""
	events = np.asarray(events)
	n_obs = float(len(events))

	# compute I0 = integral_0^T exp(s - N(s)) ds with N(s) the observed counting process
	all_points = np.concatenate(([0.0], events))
	I0 = 0.0
	# integrate over segments [all_points[k], all_points[k+1]) where N(s)=k
	for k in range(len(all_points)-1):
		a = all_points[k]
		b = all_points[k+1]
		I0 += np.exp(-k) * (np.exp(b) - np.exp(a))
	# last segment up to T
	if events.size == 0:
		a = 0.0
		b = T
		I0 += np.exp(0) * (np.exp(b) - np.exp(a))
	else:
		a = all_points[-1]
		b = T
		I0 += np.exp(-len(events)) * (np.exp(b) - np.exp(a))

	if I0 <= 0:
		return max(1e-6, n_obs / max(1.0, T))
	mu_hat = n_obs / I0
	return float(mu_hat)


def test_selfcorrect(events, T, alpha_level=0.05):
	"""
	Test goodness-of-fit for the self-correcting model on observed events over [0,T].
	- estimate mu
	- compute compensator at event times
	- perform KS test vs Exp(1) on compensator increments
	Returns (ks_reject, info)
	"""
	if len(events) == 0:
		return False, {"reason": "no_events"}

	# estimate mu using linear estimator
	mu_hat = estimate_mu_selfcorrect(events, T)

	# compensator values at event times (use mu_hat)
	Lambda = compensator_selfcorrect(mu_hat, events)
	Lambda_prev = np.concatenate(([0.0], Lambda[:-1]))
	x = Lambda - Lambda_prev

	stat, p = kstest(x, 'expon')
	return (p < alpha_level), {"pvalue": p, "mu_hat": mu_hat, "n_events": int(len(events))}


if __name__ == '__main__':
	# small example
	T = 100.0
	true_params = {"mu": 0.5, "alpha": 1.0, "beta": 2.0}
	res = monte_carlo_seq(true_params, (0.0, T), M=50, subsample=100)
	print(res)
