/*
 * wgslreplay: replay the compute pipelines a page creates from one WGSL
 * module through WebKit's own WGSL compiler, the way Safari does, and report
 * the first one Safari would reject.
 *
 * It follows WebGPU::createLibrary in Source/WebGPU/WebGPU/Pipeline.mm and the
 * checks after it in Device::createComputePipeline (ComputePipeline.mm) step
 * by step: prepare for the entry point, the override constants with their
 * lookup and range rules, code generation, then the workgroup checks against
 * the device limits. Only the last step Safari takes is missing, compiling the
 * generated Metal source, which needs macOS; --dump-msl writes that source so
 * a macOS job can compile it with xcrun metal.
 *
 * Usage: wgslreplay [options] <module.wgsl> <pipelines.txt>
 *   One pipeline a line, in creation order: "<entryPoint> [KEY=VALUE ...]".
 *   --features=a,b         WGSL features the device has (default: shader-f16)
 *   --workgroup-storage=N  maxComputeWorkgroupStorageSize (default: 32768)
 *   --invocations=N        maxComputeInvocationsPerWorkgroup (default: 1024)
 *   --apple-gpu-family=N   (default: 7)
 *   --dump-msl=PREFIX      write PREFIX.<line>.metal for each pipeline
 *
 * Output: one line a pipeline. "ok", or "fail <kind>: <detail>", where kind is
 * what Safari would show: "message" when it reports the compiler's own text,
 * "silent" when it reports only "Compute library failed creation", and
 * "invalid" when it rejects the pipeline with no text at all.
 */

#include "config.h"

#include "WGSL.h"
#include "WGSLShaderModule.h"
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <wtf/DataLog.h>
#include <wtf/FileSystem.h>
#include <wtf/text/StringBuilder.h>

namespace {

struct Options {
    const char* module { nullptr };
    const char* pipelines { nullptr };
    std::vector<std::string> features { "shader-f16" };
    size_t workgroupStorage { 32768 };
    uint64_t invocations { 1024 };
    uint64_t sizeX { 1024 }, sizeY { 1024 }, sizeZ { 64 };
    unsigned appleGPUFamily { 7 };
    std::string dumpMSL;
};

bool startsWith(const char* argument, const char* prefix, const char*& rest)
{
    size_t length = strlen(prefix);
    if (strncmp(argument, prefix, length))
        return false;
    rest = argument + length;
    return true;
}

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const char* rest;
        if (startsWith(argv[i], "--features=", rest)) {
            options.features.clear();
            std::stringstream list(rest);
            for (std::string feature; std::getline(list, feature, ',');)
                if (!feature.empty())
                    options.features.push_back(feature);
        } else if (startsWith(argv[i], "--workgroup-storage=", rest))
            options.workgroupStorage = strtoull(rest, nullptr, 10);
        else if (startsWith(argv[i], "--invocations=", rest))
            options.invocations = strtoull(rest, nullptr, 10);
        else if (startsWith(argv[i], "--apple-gpu-family=", rest))
            options.appleGPUFamily = strtoul(rest, nullptr, 10);
        else if (startsWith(argv[i], "--dump-msl=", rest))
            options.dumpMSL = rest;
        else if (!options.module)
            options.module = argv[i];
        else if (!options.pipelines)
            options.pipelines = argv[i];
        else {
            fprintf(stderr, "unexpected argument %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }
    if (!options.module || !options.pipelines) {
        fprintf(stderr, "Usage: wgslreplay [options] <module.wgsl> <pipelines.txt>\n");
        exit(EXIT_FAILURE);
    }
    return options;
}

std::optional<uint64_t> evaluatedDimension(const WGSL::ShaderModule& module, const WGSL::AST::Expression* expression, const HashMap<String, WGSL::ConstantValue>& values)
{
    if (!expression)
        return 1;
#if WGSLREPLAY_WEBKIT_7624
    UNUSED_PARAM(module);
    auto value = WGSL::evaluate(*expression, values);
#else
    auto value = WGSL::evaluate(module, *expression, values);
#endif
    if (!value)
        return std::nullopt;
    return value->integerValue();
}

// One pipeline, as createLibrary and createComputePipeline would build it.
// Returns the empty string on success, otherwise "<kind>: <detail>".
std::string replay(WGSL::ShaderModule& module, const Options& options, const std::string& entry, const std::vector<std::pair<std::string, double>>& constants, const std::string& mslPath)
{
    String entryPoint = String::fromUTF8(entry.c_str());
    auto prepareResult = WGSL::prepare(module, entryPoint, nullptr);
    if (auto* error = std::get_if<WGSL::Error>(&prepareResult))
        return "message: prepare: " + std::string(error->message().utf8().data());
    auto& result = std::get<WGSL::PrepareResult>(prepareResult);
    auto iterator = result.entryPoints.find(entryPoint);
    if (iterator == result.entryPoints.end())
        return "silent: no entry point " + entry;
    const auto& information = iterator->value;

    HashMap<String, WGSL::ConstantValue> values;
    for (const auto& [name, value] : constants) {
        String key = String::fromUTF8(name.c_str());
        auto found = information.specializationConstants.find(key);
#if WGSLREPLAY_WEBKIT_7624
        // Safari 26 (WebKit 7624): a constant for any override this entry point
        // does not use fails the whole pipeline, and with no message. WebKit
        // main later ignores it when the module declares that override.
        if (found == information.specializationConstants.end())
            return "silent: constant " + name + " is not used by entry point " + entry;
        key = found->value.mangledName;
#else
        if (found == information.specializationConstants.end()) {
            if (module.containsOverride(key))
                continue;
            return "silent: constant " + name + " names no override in the module";
        }
#endif
        switch (found->value.type) {
        case WGSL::Reflection::SpecializationConstantType::Boolean:
            values.set(key, static_cast<bool>(value));
            break;
        case WGSL::Reflection::SpecializationConstantType::Float:
            if (value < std::numeric_limits<float>::lowest() || value > std::numeric_limits<float>::max())
                return "silent: constant " + name + " out of f32 range";
            values.set(key, static_cast<float>(value));
            break;
        case WGSL::Reflection::SpecializationConstantType::Int:
            if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
                return "silent: constant " + name + " out of i32 range";
            values.set(key, static_cast<int32_t>(value));
            break;
        case WGSL::Reflection::SpecializationConstantType::Unsigned:
            if (value < 0 || value > std::numeric_limits<uint32_t>::max())
                return "silent: constant " + name + " out of u32 range";
            values.set(key, static_cast<uint32_t>(value));
            break;
        case WGSL::Reflection::SpecializationConstantType::Half: {
            constexpr double halfMax = 0x1.ffcp15;
            if (value < -halfMax || value > halfMax)
                return "silent: constant " + name + " out of f16 range";
            WGSL::half half = value;
            values.set(key, half);
            break;
        }
        }
    }

#if WGSLREPLAY_WEBKIT_7624
    // Safari 26 fills in the defaults here rather than in generate.
    for (auto& constant : information.specializationConstants) {
        auto& information = constant.value;
        if (!information.defaultValue || values.contains(information.mangledName)) {
            if (!information.defaultValue && !values.contains(information.mangledName))
                return "message: Override " + std::string(constant.key.utf8().data()) + " is used in shader but not provided";
            continue;
        }
        auto value = WGSL::evaluate(*information.defaultValue, values);
        if (!value)
            return "message: Failed to evaluate override value";
        values.add(information.mangledName, *value);
    }
#endif

    auto generation = WGSL::generate(module, result, values, WGSL::DeviceState {
        .appleGPUFamily = options.appleGPUFamily,
        .shaderValidationEnabled = false,
    });
    if (auto* error = std::get_if<WGSL::Error>(&generation))
        return "message: generate: " + std::string(error->message().utf8().data());
    if (!mslPath.empty()) {
        std::ofstream file(mslPath);
        file << std::get<String>(generation).utf8().data();
    }

    // What createComputePipeline checks once the library exists.
    if (information.specializationConstants.size() != values.size())
        return "invalid: " + std::to_string(information.specializationConstants.size()) + " overrides, "
            + std::to_string(values.size()) + " values";
    auto* compute = std::get_if<WGSL::Reflection::Compute>(&information.typedEntryPoint);
    if (!compute)
        return "invalid: not a compute entry point";
    auto width = evaluatedDimension(module, compute->workgroupSize.width, values);
    auto height = evaluatedDimension(module, compute->workgroupSize.height, values);
    auto depth = evaluatedDimension(module, compute->workgroupSize.depth, values);
    if (!width || !height || !depth)
        return "message: Failed to evaluate overrides";
    if (information.sizeForWorkgroupVariables > options.workgroupStorage)
        return "invalid: " + std::to_string(information.sizeForWorkgroupVariables) + " workgroup bytes, limit "
            + std::to_string(options.workgroupStorage);
    if (!*width || *width > options.sizeX || !*height || *height > options.sizeY || !*depth || *depth > options.sizeZ
        || *width * *height * *depth > options.invocations)
        return "invalid: workgroup size " + std::to_string(*width) + "x" + std::to_string(*height) + "x" + std::to_string(*depth);
    return { };
}

} // namespace

int main(int argc, char** argv)
{
    WTF::initializeMainThread();
    Options options = parse(argc, argv);

    auto source = FileSystem::readEntireFile(String::fromUTF8(options.module));
    if (!source) {
        fprintf(stderr, "cannot read %s\n", options.module);
        return EXIT_FAILURE;
    }
    HashSet<String> features;
    for (auto& feature : options.features)
        features.add(String::fromUTF8(feature.c_str()));
    WGSL::Configuration configuration { .supportedFeatures = WTF::move(features) };
    auto check = WGSL::staticCheck(String::fromUTF8WithLatin1Fallback(source->span()), std::nullopt, configuration);
    if (auto* failed = std::get_if<WGSL::FailedCheck>(&check)) {
        // Safari reports these at createShaderModule, with their text.
        for (const auto& error : failed->errors)
            printf("fail message: module: %s\n", error.message().utf8().data());
        return EXIT_FAILURE;
    }
    auto& module = std::get<WGSL::SuccessfulCheck>(check).ast;

    std::ifstream pipelines(options.pipelines);
    int failures = 0;
    unsigned lineNumber = 0;
    for (std::string line; std::getline(pipelines, line);) {
        if (line.empty())
            continue;
        ++lineNumber;
        std::istringstream words(line);
        std::string entry;
        words >> entry;
        std::vector<std::pair<std::string, double>> constants;
        for (std::string word; words >> word;) {
            auto equals = word.find('=');
            constants.emplace_back(word.substr(0, equals), strtod(word.c_str() + equals + 1, nullptr));
        }
        std::string msl = options.dumpMSL.empty() ? std::string() : options.dumpMSL + "." + std::to_string(lineNumber) + ".metal";
        auto outcome = replay(module.get(), options, entry, constants, msl);
        if (outcome.empty())
            printf("ok %s\n", line.c_str());
        else {
            printf("fail %s | %s\n", outcome.c_str(), line.c_str());
            ++failures;
        }
    }
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
