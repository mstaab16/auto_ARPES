import pickle
import matplotlib.pyplot as plt

with open('measurements_since_last_outlier.pkl', 'rb') as f:
    measurements_since_last_outlier = pickle.load(f)
    plt.plot(measurements_since_last_outlier)
    plt.show()
