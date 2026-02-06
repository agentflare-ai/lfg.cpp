#include "lfg_model.h"
#include "lfg_inference.h"
#include <spdlog/spdlog.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cassert>
#include <cstdio>

// Simple helper to copy a file
bool copy_file(const std::string& src, const std::string& dst) {
    std::ifstream s(src, std::ios::binary);
    std::ofstream d(dst, std::ios::binary);
    if (!s || !d) return false;
    d << s.rdbuf();
    return true;
}

bool snapshot_compare_or_write(const std::string& name, const std::string& content) {
    const std::filesystem::path dir = std::filesystem::path("test_snapshots");
    std::filesystem::create_directories(dir);
    const auto snapshot_path = dir / (name + ".txt");
    const auto new_path = dir / (name + ".new.txt");

    if (!std::filesystem::exists(snapshot_path)) {
        std::ofstream out(snapshot_path);
        out << content;
        spdlog::info("Snapshot created: {}", snapshot_path.string());
        return true;
    }

    std::ifstream in(snapshot_path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    if (buffer.str() == content) {
        return true;
    }

    std::ofstream out(new_path);
    out << content;
    spdlog::error("Snapshot mismatch: {} (see {})", snapshot_path.string(), new_path.string());
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        spdlog::error("Usage: {} <path_to_valid_gguf>", argv[0]);
        return 77;
    }

    std::string src_path = argv[1];
    std::string bundle_path = "test_model.bundle";

    spdlog::info("Copying {} to {}...", src_path, bundle_path);
    if (!copy_file(src_path, bundle_path)) {
        spdlog::error("Failed to copy file to .bundle");
        return 1;
    }

    spdlog::info("Attempting to load {}...", bundle_path);
    
    lfg_model_params mparams = lfg_model_default_params();
    lfg_model* model = lfg_model_load_from_file(bundle_path.c_str(), mparams);
    
    if (!model) {
        spdlog::error("Failed to load Leap Bundle!");
        return 1;
    }

    spdlog::info("Successfully loaded Leap Bundle!");

    std::ostringstream snapshot;
    snapshot << "bundle_source=" << src_path << "\n";
    snapshot << "bundle_path=" << bundle_path << "\n";
    snapshot << "load=ok\n";
    const bool snapshot_ok = snapshot_compare_or_write("test_leap_bundle", snapshot.str());
    
    // Cleanup
    lfg_model_free(model);
    std::remove(bundle_path.c_str());

    return snapshot_ok ? 0 : 1;
}
