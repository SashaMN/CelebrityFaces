#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using Centroid = std::vector<double>;
using Centroids = std::vector<Centroid>;
using Descriptor = std::vector<uint64_t>;

template <typename T>
std::vector<std::vector<T>> ReadVectors(
    const std::string &filename) {
  std::vector<std::vector<T>> result;
  std::ifstream ifs(filename);
  int num_centroids = 0;
  for (std::string line; std::getline(ifs, line);) {
    result.emplace_back(std::vector<T>());
    ++num_centroids;
    std::istringstream iss(line);
    T value = 0.0;
    while (iss >> value) {
      result[num_centroids - 1].push_back(value);
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
    int count = 20) {
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
    const std::vector<std::vector<double>> &directions,
    const std::vector<std::vector<double>> &biases) {
  std::vector<int> result_mask;
  for (int i = 0; i < directions.size(); ++i) {
    double cur_value = biases[0][i];
    for (int j = 0; j < directions[i].size(); ++j) {
      cur_value += directions[i][j] * query[j];
    }
    result_mask.push_back(cur_value > 0);
  }
  return ToBinary(result_mask);
}

int HammingDistance(
    const Descriptor &first,
    const Descriptor &second) {
  int result = 0;
  for (int i = 0; i < first.size(); ++i) {
    result += __builtin_popcount(first[i] ^ second[i]);
  }
  return result;
}

std::vector<int> KNearestNeighbors(
    const Descriptor &query_descriptor,
    const std::vector<int> &centroids_idx,
    const std::vector<std::vector<int>> &centroid2idx,
    const std::vector<Descriptor> &descriptors,
    int count = 100) {
  std::vector<int> images_idx;
  std::vector<int> distances;
  for (int idx : centroids_idx) {
    for (int image_idx : centroid2idx[idx]) {
      images_idx.push_back(image_idx);
      distances.push_back(HammingDistance(
            query_descriptor, descriptors[image_idx]));
    }
  }
  const auto sorted_args = ArgSort<int>(distances);
  std::vector<int> result(count);
  for (int i = 0; i < count; ++i) {
    result[i] = images_idx[sorted_args[i]];
  }
  return result;
}

int main() {
  const std::string centroids_file = "data/index/centroids";
  const std::string centroid2idx_file = "data/index/centroid2idx";
  const std::string filenames_file = "data/index/filenames";
  const std::string directions_file = "data/index/directions";
  const std::string biases_file = "data/index/biases";
  const std::string descriptors_file = "data/index/descriptors";

  const auto centroids = ReadVectors<double>(centroids_file);
  const auto centroid2idx = ReadVectors<int>(centroid2idx_file);
  const auto filenames = ReadStrings(filenames_file);
  const auto directions = ReadVectors<double>(directions_file);
  const auto biases = ReadVectors<double>(biases_file);
  const auto descriptors = ReadVectors<uint64_t>(descriptors_file);
  const auto query = ReadQuery();
  const auto neighbor_centroids = KNearestCentroids(query, centroids);

  const auto query_descriptor = Query2Descriptor(query, directions, biases);

  const auto nearest_neighbors = KNearestNeighbors(
      query_descriptor, neighbor_centroids, centroid2idx, descriptors);

  for (auto neighbor : nearest_neighbors) {
    std::cout << neighbor << " ";
  }
  std::cout << std::endl;

  return 0;
}
