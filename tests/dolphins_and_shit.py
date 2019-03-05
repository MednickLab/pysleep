import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(0)

n = 30
temp = np.round(np.random.normal(20, 10, size=n))
num_people = np.round(np.random.normal(15, 7, size=n))
b_temp = 2
b_people = 0.5
b_intercept = 40
playfulness = np.round(b_intercept + b_temp*temp + b_people*num_people + np.random.normal(0, 5, size=n))

a = pd.DataFrame({'Temp': temp, 'People': num_people, 'Playfulness': playfulness})

mod = sm.OLS(playfulness, np.stack([np.ones(n), temp, num_people], -1))
res = mod.fit()
print(res.summary())

mod = sm.OLS(playfulness, np.stack([np.ones(n), temp], -1))
res = mod.fit()
print(res.summary())

mod = sm.OLS(playfulness, np.stack([np.ones(n), num_people], -1))
res = mod.fit()
print(res.summary())
print(a)