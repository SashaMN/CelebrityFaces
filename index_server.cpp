#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>

using Centroid = std::vector<double>;
using Centroids = std::vector<Centroid>;
using Descriptor = std::vector<uint64_t>;
using Descriptors = std::vector<Descriptor>;

struct IndexData {
  Centroids centroids;
  std::vector<std::vector<int>> centroid2idx;
  std::vector<std::string> filenames;
  std::vector<std::vector<double>> directions;
  std::vector<double> biases;
  Descriptors descriptors;
};

template <typename T>
std::vector<T> ReadVector(const std::string &filename) {
  std::vector<T> result;
  std::ifstream ifs(filename);
  T value = 0.0;
  while (ifs >> value) {
    result.push_back(value);
  }
  return result;
}

template <typename T>
std::vector<std::vector<T>> ReadVectors(const std::string &filename) {
  std::vector<std::vector<T>> result;
  std::ifstream ifs(filename);
  int cur_size = 0;
  for (std::string line; std::getline(ifs, line);) {
    result.emplace_back(std::vector<T>());
    ++cur_size;
    std::istringstream iss(line);
    T value = 0.0;
    while (iss >> value) {
      result[cur_size - 1].push_back(value);
    }
  } 
  return result;
}

std::vector<std::string> ReadStrings(const std::string &filename) {
  std::vector<std::string> result;
  std::ifstream ifs(filename);
  std::string cur_str;
  while (ifs >> cur_str) {
    result.push_back(cur_str);
  }
  return result;
}

std::vector<double> ReadQuery(size_t query_len=128) {
  std::vector<double> result(query_len);
  double value = 0.0;
  for (int i = 0; i < query_len; ++i) {
    std::cin >> value;
    result[i] = value;
  }
  return result;
}

IndexData ReadIndexData() {
  const std::string centroids_file = "data/index/centroids";
  const std::string centroid2idx_file = "data/index/centroid2idx";
  const std::string filenames_file = "data/index/filenames";
  const std::string directions_file = "data/index/directions";
  const std::string biases_file = "data/index/biases";
  const std::string descriptors_file = "data/index/descriptors";

  IndexData index_data;
  index_data.centroids = ReadVectors<double>(centroids_file);
  index_data.centroid2idx = ReadVectors<int>(centroid2idx_file);
  index_data.filenames = ReadStrings(filenames_file);
  index_data.directions = ReadVectors<double>(directions_file);
  index_data.biases = ReadVector<double>(biases_file);
  index_data.descriptors = ReadVectors<uint64_t>(descriptors_file);
  return index_data;
}

double l2norm(
    const std::vector<double> &first,
    const std::vector<double> &second) {
  double result = 0.0;
  for (int i = 0; i < first.size(); ++i) {
    result += (first[i] - second[i]) * (first[i] - second[i]);
  }
  return result;
}

template <typename T>
std::vector<int> ArgSort(
    const std::vector<T> &values) {
  std::vector<int> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&values](int first, int second) {
    return values[first] < values[second];
  });
  return idx;
}

std::vector<int> KNearestCentroids(
    const std::vector<double> &query,
    const Centroids &centroids,
    int count = 10) {
  auto centroids_count = centroids.size();
  std::vector<double> distances(centroids_count);
  for (int i = 0; i < centroids_count; ++i) {
    distances[i] = l2norm(query, centroids[i]);
  }
  const auto sorted_centroids = ArgSort<double>(distances);
  std::sort(distances.begin(), distances.end());
  std::vector<int> result(count);
  std::copy(sorted_centroids.begin(), sorted_centroids.begin() + count,
      result.begin());
  return result;
}

Descriptor ToBinary(const std::vector<int> &mask) {
  Descriptor result;
  for (int i = 0; i < mask.size(); i += 64) {
    uint64_t value = 0;
    for (int j = 0; j < 64; ++j) {
      if (mask[i + j]) {
        value += (static_cast<int64_t>(mask[i + j]) << j);
      }
    }
    result.push_back(value);
  }
  return result;
}

Descriptor Query2Descriptor(
    const std::vector<double> &query,
    const IndexData &index_data) {
  const auto &directions = index_data.directions;
  const auto &biases = index_data.biases;
  std::vector<int> result_mask;
  for (int i = 0; i < directions.size(); ++i) {
    double cur_value = biases[i];
    for (int j = 0; j < directions[i].size(); ++j) {
      cur_value += directions[i][j] * query[j];
    }
    result_mask.push_back(cur_value > 0);
  }
  return ToBinary(result_mask);
}

int HammingDistance(uint64_t first, uint64_t second) {
  /*
  int result = 0;
  for (int i = 0; i < first.size(); ++i) {
    result += __builtin_popcount(first[i] ^ second[i]);
  }
  return result;
  */
  return __builtin_popcount(first ^ second);
}

std::vector<int> KNearestNeighbors(
    const Descriptor &query_descriptor,
    const Descriptors &descriptors,
    const std::vector<int> &images_id,
    int count) {
  const int NUM_TABLES = 8;
  std::set<int> result;
  for (int i = 0; i < NUM_TABLES; ++i) {
    std::vector<int> distances;
    for (int image_id : images_id) {
      distances.push_back(HammingDistance(
          query_descriptor[i], descriptors[image_id][i]));
    }
    const auto sorted_args = ArgSort<int>(distances);
    for (int i = 0; i < count; ++i) {
      result.insert(images_id[sorted_args[i]]);
    }
  }
  return std::vector<int>(result.begin(), result.end());
}

std::vector<double> ShiftQuery(
    const std::vector<double> &query,
    const std::vector<double> &centroid) {
  std::vector<double> result(query.size());
  for (int i = 0; i < query.size(); ++i) {
    result[i] = query[i] - centroid[i];
  }
  return result;
}

std::vector<int> KNNInCentroids(
    const std::vector<double> &query,
    const std::vector<int> &centroids_id,
    const IndexData &index_data,
    int count = 10) {
  const auto &centroids = index_data.centroids;
  const auto &descriptors = index_data.descriptors;
  const auto &centroid2idx = index_data.centroid2idx;

  std::vector<int> result;
  for (int centroid_id : centroids_id) {
    const auto &shifted_query = ShiftQuery(query, centroids[centroid_id]);
    const auto query_descriptor = Query2Descriptor(shifted_query, index_data);
    const auto neighbors = KNearestNeighbors(
        query_descriptor, descriptors, centroid2idx[centroid_id], count);
    result.insert(result.end(), neighbors.begin(), neighbors.end());
  }
  return result;
}

int main() {
  const auto index_data = ReadIndexData();
  const auto query = ReadQuery();
  const auto neighbor_centroids = KNearestCentroids(
      query, index_data.centroids);
  const auto nearest_neighbors = KNNInCentroids(
      query, neighbor_centroids, index_data);

  for (auto neighbor : nearest_neighbors) {
    std::cout << neighbor << " ";
  }
  std::cout << std::endl;

  return 0;
}
