import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        labels = []
        for line in file:
            parts = line.strip().split('\t')
            # Konwersja przecinków na kropki i konwersja do float
            features = [float(x.replace(',', '.')) for x in parts[:-1]]
            data.append(features)
            labels.append(parts[-1].strip())
        return np.array(data), labels

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, clusters, k):
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def calculate_sse(data, centroids, clusters):
    distances = np.sqrt(((data - centroids[clusters])**2).sum(axis=1))
    return np.sum(distances**2)

def calculate_entropy(labels, clusters, k):
    entropy = []
    for i in range(k):
        cluster_labels = labels[clusters == i]
        _, counts = np.unique(cluster_labels, return_counts=True)
        probabilities = counts / counts.sum()
        cluster_entropy = -np.sum(probabilities * np.log(probabilities))
        entropy.append(cluster_entropy)
    return entropy

# Wczytanie danych
data, labels = load_data('iris_training.txt')

# Pobranie wartości k od użytkownika
k = int(input("Podaj liczbę klastrów k: "))

# Inicjalizacja centroidów
centroids = initialize_centroids(data, k)

# Przypisywanie punktów do klastrów i aktualizacja centroidów
for iteration in range(100):  # Maksymalnie 100 iteracji
    clusters = assign_clusters(data, centroids)
    new_centroids = update_centroids(data, clusters, k)
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids
    print(f"Iteracja {iteration+1}, SSE: {calculate_sse(data, centroids, clusters)}")

# Obliczanie entropii dla klastrów
cluster_entropy = calculate_entropy(labels, clusters, k)
for i, entropy in enumerate(cluster_entropy):
    print(f"Entropia klastra {i+1}: {entropy:.4f}")

