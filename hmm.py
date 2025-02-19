import numpy as np
from hmmlearn import hmm

startprob = np.array([0.2,0.2,0.2,0.2,0.2]) # activities: standing / lying / running / sitting / walking
transmat = np.array([[0.2,0.2,0.2,0.2,0.2],
                     [1/3,1/3,0,1/3,0],
                     [1/3,0,1/3,0,1/3],
                     [1/3,1/3,0,1/3,0],
                     [1/3,0,1/3,0,1/3]])
emissionprob = np.array([[0.2,0.2,0.2,0.2,0.2],
                        [0.2,0.2,0.2,0.2,0.2],
                        [0.2,0.2,0.2,0.2,0.2],
                        [0.2,0.2,0.2,0.2,0.2],
                        [0.2,0.2,0.2,0.2,0.2]])
n_samples = 1000
model = hmm.CategoricalHMM(n_components=5, random_state=99)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

X, Z = model.sample(n_samples)

X_train = X[:X.shape[0]//2]
X_val = X[X.shape[0]//2:]

gen_prob = model.score(X_val)

best_prob = best_model = None
n_fits = 100
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(n_components=5, random_state=idx ,init_params= 's')
    model.transmat_ = np.array([np.random.dirichlet(np.ones(5)) for _ in range(5)])
    model.emissionprob_ = np.array([np.random.dirichlet(np.ones(5)) for _ in range(5)])
    model.fit(X_train)
    prob = model.score(X_val)
    if best_prob is None or prob > best_prob:
        best_model = model
        best_prob = prob

print(f'Transmission Matrix Generated:\n{model.transmat_.round(3)}\n\n'
      f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Transmission Matrix Generated:\n{model.emissionprob_.round(3)}\n\n'
      f'Transmission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')