import numpy as np
import matplotlib.pyplot as plt

k = m = 1

time = 10
timesteps = 1000
dt = time/timesteps
Lt = np.linspace(0, time, timesteps+1)

x = x2 = 1

v = (-k/m) * x * (dt/2)
v2 = (-k/m) * x * (dt/2)

Lx, Lx2, Lv, Lv2 = [x], [x2], [v], [v2]

A, w = 1, 1

# (x_n+1 - x_n) / dt = v_n+(1/2)
# -> x_n+1 = x_n + v_n+(1/2) * dt

# (v_n+(3/2) + v_n+(1/2)) / dt = F(t) - kx / m
#  -> F(t) = A sin(wt)
# -> v_n+(3/2) = v_n+(1/2) + -k/m * (x_n+1) * dt 

for t in Lt[1:]:
    x += v * dt
    x2 += v2 * dt
    v +=  (A*np.sin(w*t) - (k*x)) / m * dt
    v2 += (-k/m) * x2 * dt

    Lx.append(x)
    Lx2.append(x2)
    Lv.append(v)
    Lv2.append(v2)

plt.ylim(-4, 4)

plt.plot(Lt, Lx)
plt.plot(Lt, Lx2)
plt.show()

plt.plot(Lt, Lv)
plt.plot(Lt, Lv2)
plt.show()

# for k in [0.2, 0.5, 1, 2, 5, 8]:
#     x = 1

#     v = (-k/m) * x * (dt/2)

#     Lx, Lv = [x], [v]
#     for _ in range(timesteps):
#         x += v * dt
#         v += (-k/m) * x * dt

#         Lx.append(x)
#         Lv.append(v)
#     plt.plot(Lt, Lx)
#     plt.plot(Lt, Lv)

# plt.show()