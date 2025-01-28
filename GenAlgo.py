import numpy as np
import folium
import openrouteservice

class GeneticAlgo:
    def __init__(self, distances, population_size, generations, mutation_rate=0.1):
        self.distances = distances
        self.n_cities = distances.shape[0]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()

    def _initialize_population(self):
        return np.array([np.random.permutation(self.n_cities) for _ in range(self.population_size)])

    def _calculate_fitness(self, individual):
        indices = np.append(individual, individual[0])
        #return 1 / np.sum(self.distances[indices[:-1], indices[1:]])
        total_distance = 0
        for i in range(len(indices) - 1):
            total_distance += self.distances[indices[i], indices[i + 1]]
        return 1 / total_distance

    def _select_parents(self):
        fitness_scores = np.array([self._calculate_fitness(ind) for ind in self.population])
        probabilities = fitness_scores / np.sum(fitness_scores)
        parent_indices = np.random.choice(self.population_size, size=2, p=probabilities)
        return self.population[parent_indices[0]], self.population[parent_indices[1]]

    def _crossover(self, parents):
        parent1, parent2 = parents
    
    # Выбираем случайные индексы для начала и конца отрезка
        start = np.random.randint(0, self.n_cities)
        end = np.random.randint(0, self.n_cities)
    
    # Убедимся, что start меньше end
        if start > end:
            start, end = end, start
    
    # Создаем ребенка с заполнением -1
        child = np.full(self.n_cities, -1)
    
    # Копируем часть от первого родителя
        child[start:end] = parent1[start:end]
    
    # Находим оставшиеся города из второго родителя
        parent2_remaining = [city for city in parent2 if city not in child]
    
    # Заполняем пустые места в ребенке оставшимися городами
        child[child == -1] = parent2_remaining
    
        return child


    def _mutate(self, individual):
        if np.random.random() < self.mutation_rate:
            i, j = np.random.choice(self.n_cities, size=2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]

    def _next_generation(self):
        new_population = []
        for _ in range(self.population_size):
            parents = self._select_parents()
            child = self._crossover(parents)
            self._mutate(child)
            new_population.append(child)
        self.population = np.array(new_population)

    def solve(self):
        best_individual = None
        best_distance = float('inf')

        for generation in range(self.generations):
            self._next_generation()

            for individual in self.population:
                distance = 1 / self._calculate_fitness(individual)
                if distance < best_distance:
                    best_distance = distance
                    best_individual = individual

            print(f"Поколение {generation + 1}: лучший путь = {best_distance}")

        return best_individual, best_distance


def get_real_distances(cities, api_key):
    client = openrouteservice.Client(key=api_key)
    n_cities = len(cities)
    distances = np.zeros((n_cities, n_cities))

    coordinates = []
    for city in cities:
        geocode = client.pelias_search(city)
        coords = geocode['features'][0]['geometry']['coordinates']
        coordinates.append(coords)

    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            try:
                route = client.directions(
                    coordinates=[coordinates[i], coordinates[j]],
                    profile='driving-car',
                    format='geojson'
                )
                distance = route['features'][0]['properties']['segments'][0]['distance']
                distances[i, j] = distance
                distances[j, i] = distance
            except openrouteservice.exceptions.ApiError as e:
                print(f"Ошибка получения расстояния между {cities[i]} и {cities[j]}: {e}")

    return distances, coordinates

def visualize_route(cities, best_path, coordinates, api_key):
    client = openrouteservice.Client(key=api_key)
    route_map = folium.Map(location=coordinates[0][::-1], zoom_start=5)

    for i in range(len(best_path)):
        start = best_path[i]
        end = best_path[(i + 1) % len(best_path)]
        try:
            route = client.directions(
                coordinates=[coordinates[start], coordinates[end]],
                profile='driving-car',
                format='geojson'
            )
            geometry = route['features'][0]['geometry']['coordinates']
            folium.PolyLine(
                [(point[1], point[0]) for point in geometry],
                color="green", weight=5, opacity=0.7
            ).add_to(route_map)
        except openrouteservice.exceptions.ApiError as e:
            print(f"Ошибка получения расстояния между {cities[start]} и {cities[end]}: {e}")

    for city, coord in zip(cities, coordinates):
        folium.Marker(location=coord[::-1], popup=city).add_to(route_map)

    route_map.save("route_map.html")
    print("Карта сохранена в файл route_map.html")

# def run_on_map():
#     API_KEY = "***"
#     cities = ["Moscow", "Ankara", "Tallinn", "Minsk", "Warszawa", "Kiev", "Bucuresti"]

#     distances, coordinates = get_real_distances(cities, API_KEY)

#     ga_tsp = GeneticAlgo(distances, population_size=20, generations=100, mutation_rate=0.1)
#     best_path, best_distance = ga_tsp.solve()

#     print("Лучший маршрут:", ' -> '.join([cities[i] for i in best_path]) + ' -> ' + cities[best_path[0]])
#     print("Длина лучшего маршрута:", best_distance)

#     visualize_route(cities, best_path, coordinates, API_KEY)
    
def run_on_random_matrix():
    n_cities = 10
    distances = np.random.randint(1, 10, size=(n_cities, n_cities))
    np.fill_diagonal(distances, 0)

    ga_tsp = GeneticAlgo(distances, population_size=20, generations=100, mutation_rate=0.1)
    best_path, best_distance = ga_tsp.solve()
    print("Матрица расстояний:")
    print(distances)
    print("Лучший путь:", *best_path)
    print("Длина лучшего пути:", best_distance)

# run_on_random_matrix()
run_on_random_matrix()