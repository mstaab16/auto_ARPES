from gpytorch.likelihoods import likelihood
from messages import *
import numpy as np
import xarray as xr
from queue import PriorityQueue
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pickle

from predictor import fit, choose_next_position

class Experiment:
    def __init__(self, initalization):
        print(initalization)
        self.data_formats = initalization.data_formats
        self.search_axes = initalization.search_axes
        num_points_possible = np.prod([int((axis.max - axis.min) / axis.step) for axis in self.search_axes])
        self.num_points_taken = 0
        self.max_n_clusters_so_far = 6
        self.max_n_clusters = 6
        self.data = xr.Dataset()
        for data_format in self.data_formats:
            self.data[data_format.name] = xr.DataArray(
                    np.zeros([num_points_possible] + data_format.shape, dtype=data_format.dtype),
                    dims=["point"] + [f"dim{i}" for i in range(len(data_format.shape))],
                    )

        self.move_queue = PriorityQueue(maxsize=100)
        # add every combination of min and max axis positions to the queue
        self.move_queue.put((0, [axis.name for axis in self.search_axes],[axis.min for axis in self.search_axes]))
#         self.move_queue.put((0, [axis.name for axis in self.search_axes],[axis.max for axis in self.search_axes]))
#         self.move_queue.put((0, [axis.name for axis in self.search_axes],[self.search_axes[0].min, self.search_axes[1].max]))
#         self.move_queue.put((0, [axis.name for axis in self.search_axes],[self.search_axes[0].max, self.search_axes[1].min]))

#         for _ in range(6):
#              self.move_queue.put((0, [axis.name for axis in self.search_axes], [np.random.uniform(axis.min, axis.max) for axis in self.search_axes]))

        self.positions_measured = []
        self.cluster_label_history = []
        self.current_position = []
        self.measurements_since_last_outlier = [0]


        test_d1 = np.linspace(self.search_axes[0].min, self.search_axes[0].max, 91)
        test_d2 = np.linspace(self.search_axes[1].min, self.search_axes[1].max, 91)
        test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
        self.test_x = torch.tensor(np.array([test_d1.ravel(), test_d2.ravel()]).T, dtype=torch.float)

        #self.image_sender = imagezmq.ImageSender(connect_to="tcp://localhost:5551")

    def __repr__(self):
        return f"""Experiment:
                \t{[axis.name for axis in self.search_axes]}
                \t{[data_format.name for data_format in self.data_formats]}
                \tNumber of measurements (taken/max): ({self.num_points_taken})/({len(self.data.point)})
                """

    def handle_message(self, message: Message):
        if isinstance(message, QueryMessage):
            move = self.move_queue.get()
            self.positions_measured.append(move[2])
            #self.image_sender.send_image("position", np.array(self.positions_measured))
            print(self.num_points_taken, move)
            return MoveResponse(axes_to_move=move[1], position=move[2])

        if isinstance(message, InitMessage):
            return self.handle_init(message)

        if isinstance(message, DataMessage):
            return self.handle_data(message)

        if isinstance(message, ShutdownMessage):
            #self.image_sender.send_image("shutdown", np.array(self.positions_measured))
            #self.image_sender.close()
            print("Saving positions measured")
            with open("positions_measured.pkl", "wb") as f:
                pickle.dump(self.positions_measured, f)
            print("Saving cluster labels")
            with open("cluster_label_history.pkl", "wb") as f:
                pickle.dump(self.cluster_label_history, f)
            print("Saving measurements_since_last_outlier")
            with open("measurements_since_last_outlier.pkl", "wb") as f:
                pickle.dump(self.measurements_since_last_outlier[1:], f)
            print("Saving data")
            self.data.to_netcdf("data.nc")
            return OkayResponse()

    def handle_init(self, message: InitMessage):
        return OkayResponse()

    def handle_data(self, message: DataMessage):
        data = np.array(message.data)
        format_name = message.format_name
        self.data[format_name][self.num_points_taken] = data.T

        self.num_points_taken += 1
        # self.image_sender.send_image(format_name, data)
        
        prediction_response = self.predict_new_move()

        return prediction_response

    def handle_data_rand(self, message):
        self.num_points_taken += 1
        if self.num_points_taken > 6:
            self.move_queue.put((1, [axis.name for axis in self.search_axes],[np.random.uniform(axis.min, axis.max) for axis in self.search_axes]))
        image = np.array(message.data)
        #self.image_sender.send_image("ARPES", image)
        return OkayResponse()

    def handle_update(self, message):
        return OkayResponse()

    def predict_new_move(self):
        #if self.num_points_taken < 5:
        #    return OkayResponse()   
        raw_data =  self.data["ARPES"].values[:self.num_points_taken].reshape(self.num_points_taken, -1)
        self.dim_reduced_data = self.reduce_dimensions(raw_data)
        self.data_labels = self.cluster_data(self.dim_reduced_data)
        self.predicted_move = self.predict_move(raw_data.sum(axis=1), self.data_labels, self.positions_measured)
        print('adding move to queue')
        self.move_queue.put((1, [axis.name for axis in self.search_axes], self.predicted_move))
        return OkayResponse()

    def reduce_dimensions(self, data):
        print('reducing dimensions')
        #return data
        #data = (data.T/data.sum(axis=1)).T
        pca = PCA(0.8)
        pca.fit(data)
        return pca.transform(data)
    
    def index_of_max_sil_score_k(self, list_of_label_lists):
        silhouette_scores = []
        for labels in list_of_label_lists:
            silhouette_scores.append(silhouette_score(self.dim_reduced_data, labels))
        return np.argmax(silhouette_scores)

    def cluster_data(self, data):
        print('clustering data')
        if data.shape[0] < 10 + self.max_n_clusters:
            n_clusters = min(2,data.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(data)
            self.cluster_label_history.append(kmeans.labels_)
            print(f"{n_clusters} clusters")
            return kmeans.labels_
        else:
            list_of_cluster_labels = []
            list_of_k = []
            for k in range(self.max_n_clusters_so_far, self.max_n_clusters + 1):
                kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
                list_of_cluster_labels.append(kmeans.labels_)
                list_of_k.append(k)
            best_index = self.index_of_max_sil_score_k(list_of_cluster_labels)
            n_clusters = list_of_k[best_index]
            if n_clusters > self.max_n_clusters_so_far:
                print(f"New best silhouete score. Now using {n_clusters}-{self.max_n_clusters} clusters.")
                self.measurements_since_last_outlier.append(self.measurements_since_last_outlier[-1]+1)
                self.max_n_clusters_so_far = n_clusters
            else:
                self.measurements_since_last_outlier.append(0)
            self.cluster_label_history.append(list_of_cluster_labels[best_index])
            print(f"{n_clusters} clusters")
            return list_of_cluster_labels[best_index]

#TODO: track number of iterations since new cluster was needed 
#       (by silhouete score) and fit a poisson then set 
#       cutoff for when we can stopo

    def predict_move(self, counts, data_labels, positions_measured):
        print('predicting move')
        labels = np.array([counts,data_labels],dtype=np.float32).T
        model, likelihood = fit(torch.tensor(positions_measured), torch.tensor(labels))
        next_i, _ =  choose_next_position(model, likelihood, self.test_x, positions_measured)
        return self.test_x[next_i].numpy().tolist()
        


        return [np.random.uniform(axis.min, axis.max) for axis in self.search_axes]







