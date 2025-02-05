import torch
import numpy as np
from hmmlearn import hmm
import time

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

startprob = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=device)
transmat = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2],
                         [1/3, 1/3, 0, 1/3, 0],
                         [1/3, 0, 1/3, 0, 1/3],
                         [1/3, 1/3, 0, 1/3, 0],
                         [1/3, 0, 1/3, 0, 1/3]], device=device)
emissionprob = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2],
                             [0.2, 0.2, 0.2, 0.2, 0.2],
                             [0.2, 0.2, 0.2, 0.2, 0.2],
                             [0.2, 0.2, 0.2, 0.2, 0.2],
                             [0.2, 0.2, 0.2, 0.2, 0.2]], device=device)

n_samples = 1000

model = hmm.CategoricalHMM(n_components=5, random_state=99)
model.startprob_ = startprob.cpu().numpy()
model.transmat_ = transmat.cpu().numpy()
model.emissionprob_ = emissionprob.cpu().numpy()

X, Z = model.sample(n_samples)

X_train = X[:X.shape[0] // 2]
X_val = X[X.shape[0] // 2:]

gen_prob = model.score(X_val)

best_prob = best_model = None
n_fits = 10000
np.random.seed(13)

for idx in range(n_fits):
    model = hmm.CategoricalHMM(n_components=5, random_state=idx, init_params='s')
    model.transmat_ = torch.tensor([np.random.dirichlet(np.ones(5)) for _ in range(5)], device=device).cpu().numpy()
    model.emissionprob_ = torch.tensor([np.random.dirichlet(np.ones(5)) for _ in range(5)], device=device).cpu().numpy()
    model.fit(X_train)
    prob = model.score(X_val)
    if best_prob is None or prob > best_prob:
        best_model = model
        best_prob = prob

end_time = time.time()



print(f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Transmission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

time_cs = end_time - start_time

print(time_cs)