import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


# result = list(de(lambda x: sum(x**2) / len(x), bounds=[(-100, 100)] * 32))
# print(result[-1])


# x, f = zip(*result)
# plt.plot(f)
# plt.show()
# fig = px.line(f)
# fig.show()
fig = go.Figure()

for d in [8, 16, 32, 64]:
    it = list(de(lambda x: sum(x**2)/d, [(-100, 100)] * d, its=300))
    x, f = zip(*it)
#     # plt.plot(f, label='d={}'.format(d))
#     # fig.add_scatter(f, mode='lines')
    fig.add_trace(go.Scatter(y=f, mode='lines'))
# # plt.legend()
# # plt.show()

fig.show()
