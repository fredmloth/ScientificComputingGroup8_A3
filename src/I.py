import numpy as np
import matplotlib.pyplot as plt

k = 2
m = 1

time = 10
timesteps = 1000
dt = time/timesteps

Lt = np.linspace(0, time, timesteps+1)


x = 1

v = (-k/m) * x * (dt/2)

Lx, Lv = [x], [v]


# (x_n+1 - x_n) / dt = v_n+(1/2)
# -> x_n+1 = x_n + v_n+(1/2) * dt

# (v_n+(3/2) + v_n+(1/2)) / dt = F(x_n+1) / m
#  -> F(x) = -kx
# -> v_n+(3/2) = v_n+(1/2) + -k/m * (x_n+1) * dt 

for _ in range(timesteps):
    x += v * dt
    v += (-k/m) * x * dt

    Lx.append(x)
    Lv.append(v)

plt.plot(Lt, Lx)
plt.show()

plt.plot(Lt, Lv)
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