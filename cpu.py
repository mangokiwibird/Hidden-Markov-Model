import numpy as np
from hmmlearn import hmm
import concurrent.futures
import time
start_time = time.time()
startprob = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) 
transmat = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
                     [1/3, 1/3, 0, 1/3, 0],
                     [1/3, 0, 1/3, 0, 1/3],
                     [1/3, 1/3, 0, 1/3, 0],
                     [1/3, 0, 1/3, 0, 1/3]])
emissionprob = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.2, 0.2, 0.2, 0.2, 0.2]])
n_samples = 1000

model = hmm.CategoricalHMM(n_components=5, random_state=99)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob


X, Z = model.sample(n_samples)

X_train = X[:X.shape[0]//2]
X_val = X[X.shape[0]//2:]


def train_model(random_state):
    model = hmm.CategoricalHMM(n_components=5, random_state=random_state, init_params='s')
    model.transmat_ = np.array([np.random.dirichlet(np.ones(5)) for _ in range(5)])
    model.emissionprob_ = np.array([np.random.dirichlet(np.ones(5)) for _ in range(5)])
    model.fit(X_train)
    prob = model.score(X_val)
    return model, prob


def run_parallel_training():
    best_prob = best_model = None
    n_fits = 10000
    np.random.seed(13)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_model, idx) for idx in range(n_fits)]

        for future in concurrent.futures.as_completed(futures):
            model, prob = future.result()
            if best_prob is None or prob > best_prob:
                best_model = model
                best_prob = prob

    return best_model
best_model = run_parallel_training()


print(f"Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n")
print(f"Transmission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n")

end_time = time.time()

time_cs = end_time - start_time

print(time_cs)