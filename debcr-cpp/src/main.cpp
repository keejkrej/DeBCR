#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::basic_string<ORTCHAR_T> to_ort_path(const std::string& path) {
#ifdef _WIN32
  return std::filesystem::path(path).wstring();
#else
  return path;
#endif
}

int64_t parse_dim(const char* text, const char* name) {
  try {
    const auto value = std::stoll(text);
    if (value <= 0) {
      throw std::invalid_argument("non-positive dimension");
    }
    return value;
  } catch (const std::exception& exc) {
    throw std::runtime_error(std::string("Invalid ") + name + ": " + exc.what());
  }
}

std::vector<float> read_f32_file(const std::string& path, std::size_t count) {
  std::vector<float> values(count);
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
    throw std::runtime_error("Input file does not contain the requested tensor size");
  }
  return values;
}

void write_f32_file(const std::string& path, const float* data, std::size_t count) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Failed to open output file: " + path);
  }
  output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
  if (!output) {
    throw std::runtime_error("Failed to write output file: " + path);
  }
}

std::size_t element_count(const std::vector<int64_t>& shape) {
  return static_cast<std::size_t>(
      std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: debcr-cpp <model.onnx> <input.f32> <output.f32> <N> <C> <H> <W>\n";
    return 2;
  }

  try {
    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];
    std::vector<int64_t> input_shape{
        parse_dim(argv[4], "N"),
        parse_dim(argv[5], "C"),
        parse_dim(argv[6], "H"),
        parse_dim(argv[7], "W"),
    };

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "debcr");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const auto ort_model_path = to_ort_path(model_path);
    Ort::Session session(env, ort_model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto input_values = read_f32_file(input_path, element_count(input_shape));
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_values.data(),
        input_values.size(),
        input_shape.data(),
        input_shape.size());

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    const float* output_data = outputs.front().GetTensorData<float>();
    const auto output_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const auto output_count = element_count(output_shape);
    write_f32_file(output_path, output_data, output_count);

    std::cout << "wrote " << output_count << " float32 values to " << output_path << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
