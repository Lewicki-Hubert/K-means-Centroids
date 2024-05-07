import random
import math


class KMeans:
    def __init__(self, data, k, labels_map):
        self.data = data
        self.k = k
        self.labels_map = labels_map
        self.clusters = [[] for _ in range(k)]
        self.centroids = [data[random.randint(0, len(data) - 1)] for _ in range(k)]

    def assign_clusters(self):
        self.clusters = [[] for _ in range(self.k)]
        for idx, point in enumerate(self.data):
            nearest_cluster = self.find_nearest_cluster(point)
            self.clusters[nearest_cluster].append(idx)

    def find_nearest_cluster(self, point):
        min_distance = float('inf')
        cluster_index = 0
        for i, centroid in enumerate(self.centroids):
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(point, centroid)))
            if distance < min_distance:
                min_distance = distance
                cluster_index = i
        return cluster_index

    def update_centroids(self):
        for i in range(self.k):
            if self.clusters[i]:
                new_centroid = [sum(self.data[idx][dim] for idx in self.clusters[i]) / len(self.clusters[i]) for dim in
                                range(len(self.data[0]))]
                self.centroids[i] = new_centroid

    def calculate_purity(self):
        for i in range(self.k):
            label_count = {}
            for idx in self.clusters[i]:
                label = self.labels_map[idx]
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

            print(f"Cluster {i + 1}: ", end="")
            total = sum(label_count.values())
            for label, count in label_count.items():
                print(f"{count / total * 100:.2f}% {label}, ", end="")
            print()

    def run(self):
        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            old_centroids = [list(centroid) for centroid in self.centroids]
            self.assign_clusters()
            self.update_centroids()
            total_distance = sum(math.sqrt(
                sum((self.data[idx][dim] - self.centroids[cluster_idx][dim]) ** 2 for dim in range(len(self.data[0]))))
                                 for cluster_idx in range(self.k) for idx in self.clusters[cluster_idx])
            print(f"Iteration: {iteration}")
            print(f"Total distance: {total_distance}")
            self.calculate_purity()
            converged = all(
                all(old == new for old, new in zip(old_centroid, new_centroid)) for old_centroid, new_centroid in
                zip(old_centroids, self.centroids))


def load_data(file_path):
    data = []
    labels_map = {}
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file.readlines()):
            # Zakładam, że dane są rozdzielone tabulatorem; dostosuj to w razie potrzeby.
            elements = line.strip().split('\t')
            # Przygotuj punkt danych, konwertując każdy element na float, pomijając ostatni element, który jest etykietą.
            point = [float(e.replace(',', '.')) for e in
                     elements[:-1]]  # Dla formatu z przecinkami jako separatorami dziesiętnymi
            data.append(point)
            labels_map[idx] = elements[-1]
    return data, labels_map


if __name__ == "__main__":
    file_path = input("Enter the path to the data file: ")
    k = int(input("Enter the number of clusters: "))
    data, labels_map = load_data(file_path)
    kmeans = KMeans(data, k, labels_map)
    kmeans.run()
