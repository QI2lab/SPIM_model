import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from model_tools.analytic_forms import gaussian_mixture


# Define model parameters
r_idxs = np.arange(-50, 50, 0.1)
test_params = [6, 10000, 0, 3.0, 3500, -10, 5.0, 5000, 7, 112]
test = gaussian_mixture(r_idxs, *test_params)

data = test
first_moment = np.sum(r_idxs * data) / np.sum(data)
second_moment = np.sum(r_idxs**2 * data) / np.sum(data)
sigma = np.sqrt(second_moment - first_moment**2)
waist_guess =  sigma
bg_guess = np.mean(data)
ishift_guess = np.max(data) - bg_guess

print(bg_guess, np.median(data))
peak_idxs, peak_heights = find_peaks(test, height=bg_guess, distance=5)
peak_heights = peak_heights['peak_heights']

n_peaks = len(peak_heights)
if n_peaks==1:
    fit_initial_params = [waist_guess, peak_heights[0], r_idxs[peak_idxs[0]],
                          1, 0, 0,
                          1, 0, 0,
                          bg_guess]
    bounds = ([0, 0, np.min(r_idxs)-1,
               0, 0, np.min(r_idxs)-1,
               0, 0, np.min(r_idxs)-1, 0],
              [np.inf, np.inf, np.max(r_idxs)+2,
               np.inf, 0.1, np.max(r_idxs)+2,
               np.inf, 0.1, np.max(r_idxs)+2,
               np.inf])

elif n_peaks==2:
    fit_initial_params = [waist_guess, peak_heights[0], r_idxs[peak_idxs[0]],
                          waist_guess, peak_heights[1], r_idxs[peak_idxs[1]],
                          1, 0, 0,
                          bg_guess]

    bounds = ([0, 0, np.min(r_idxs)-1,
               0, 0, np.min(r_idxs)-1,
               0, 0, np.min(r_idxs)-1, 0],
              [np.inf, np.inf, np.max(r_idxs)+2,
               np.inf, np.inf, np.max(r_idxs)+2,
               np.inf, 0.01, np.max(r_idxs)+2,
               np.inf])

elif n_peaks>2:
    fit_initial_params = [waist_guess, peak_heights[0], r_idxs[peak_idxs[0]],
                          waist_guess, peak_heights[1], r_idxs[peak_idxs[1]],
                          waist_guess, peak_heights[2], r_idxs[peak_idxs[2]],
                          bg_guess]


    bounds = ([0, 0, np.min(r_idxs)-1,
               0, 0, np.min(r_idxs)-1,
               0, 0, np.min(r_idxs)-1, 0],
              [np.inf, np.inf, np.max(r_idxs)+2,
               np.inf, np.inf, np.max(r_idxs)+2,
               np.inf, np.inf, np.max(r_idxs)+2,
               np.inf])

print(bounds)
print(fit_initial_params)

# Fit temp_profile with guassian intensity distribution
popt, pcov = curve_fit(gaussian_mixture,
                       r_idxs,
                       test,
                       p0=fit_initial_params,
                       bounds=bounds,
                       maxfev=5000)
fit_pass = True


fig, ax =plt.subplots(1,1)
ax.plot(r_idxs, test, label='tst')
ax.plot(r_idxs, gaussian_mixture(r_idxs, *fit_initial_params), 'r', label='init')
ax.plot(r_idxs, gaussian_mixture(r_idxs, *popt), 'k', label='fit')
for peak in peak_idxs:
    ax.axvline(r_idxs[peak])
ax.legend()

gauss_params = np.reshape(popt[:-1], (3,3))
bg = popt[-1]

ws = np.array([param[0] for param in gauss_params])
Is = np.array([param[1] for param in gauss_params])
mus = np.array([param[2] for param in gauss_params])

fig, axs = plt.subplots(1,3)
axs[0].plot(ws)
axs[1].plot(Is)
axs[2].plot(mus)


central_lobe_idx = np.argmax(Is)
central_lobe_params = [p[central_lobe_idx] for p in [ws, Is, mus]]

sorted_idxs = np.argsort(Is)

if 0.5 * Is[sorted_idxs[-1]] < Is[sorted_idxs[-2]]:
    print('too large')
    popt=central_lobe_params
    popt[0]=50

print(gauss_params)
print(np.reshape(test_params[:-1], (3,3)))
print(central_lobe_params)

plt.show()