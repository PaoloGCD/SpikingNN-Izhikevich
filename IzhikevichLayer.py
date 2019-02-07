import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

# excitatory parameter definition
number_excitatory = 800
random_e = np.random.rand(number_excitatory)
exc_a = 0.02 * np.ones(number_excitatory)
exc_b = 0.2 * np.ones(number_excitatory)
exc_c = -65 + 15 * random_e ** 2
exc_d = 8 - 6 * random_e ** 2

exc_v = -65 * np.ones(number_excitatory)
exc_u = exc_b * exc_v

# inhibitory parameter definition
number_inhibitory = 200
random_i = np.random.rand(number_inhibitory)
inh_a = 0.02 + 0.08 * random_i
inh_b = 0.25 - 0.05 * random_i
inh_c = -65 * np.ones(number_inhibitory)
inh_d = 2 * np.ones(number_inhibitory)

inh_v = -65 * np.ones(number_inhibitory)
inh_u = inh_b * inh_v

# structure definition -> one layer all to all
a = np.concatenate([exc_a, inh_a])
b = np.concatenate([exc_b, inh_b])
c = np.concatenate([exc_c, inh_c])
d = np.concatenate([exc_d, inh_d])

v = np.concatenate([exc_v, inh_v])
u = np.concatenate([exc_u, inh_u])

S = np.concatenate([0.5 * np.random.rand(number_excitatory + number_inhibitory, number_excitatory),
                    -np.random.rand(number_excitatory + number_inhibitory, number_inhibitory)], axis=1)

firings=np.array([0,0])
for i in range(0, 1000):
    is_fired = v >= 30
    v[is_fired] = c[is_fired]
    u[is_fired] = u[is_fired] + d[is_fired]

    fired = np.where(is_fired)[0]
    if fired.shape[0] > 0:
        fired = fired.reshape(fired.shape[0], 1)
        firings = np.vstack([firings, np.hstack([i + 0*fired, fired])])

    I = np.concatenate([5 * np.random.normal(0, 1, number_excitatory), 2 * np.random.normal(0, 1, number_inhibitory)])
    I += np.sum(S[:, is_fired], axis=1)

    v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)
    v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)
    u += a * (b*v - u)

plt.plot(firings[:,0], firings[:,1], 'ro', markersize=1)
plt.show()










