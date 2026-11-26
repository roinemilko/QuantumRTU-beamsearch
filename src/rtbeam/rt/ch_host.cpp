// Generic OptiX host library for triangle scenes that change at run time: the closest-hit
// min-sum backend (ch_programs.cu) and the frontier-scene experiment (fr_programs.cu).
// C ABI for ctypes. Runs in the caller's current CUDA context (CuPy's) on the legacy stream.
// The caller owns the vertex buffers; this library owns one GAS per "slot", built with
// ALLOW_UPDATE so it can be refitted. Launch parameters are an opaque block whose first
// 8 bytes receive the slot's traversable handle.

#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstring>

#include "ch_params.h"
#include "fr_params.h"

namespace {

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

constexpr int kSlots = 64;     // ring of launch-parameter slots
constexpr int kGasSlots = 32;  // one GAS per batch capacity (0-15 messages, 16-31 branch scenes)

struct Gas {
    OptixTraversableHandle handle = 0;
    CUdeviceptr output = 0, temp = 0;
    size_t output_size = 0, temp_size = 0;
    unsigned triangles = 0;
};

struct Ch {
    char* host_params = nullptr;
    size_t params_size = 0;
    CUevent slot_done[kSlots] = {};
    int next_slot = 0;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup groups[3] = {};
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    CUdeviceptr records = 0, params = 0;
    Gas gas[kGasSlots];
};

char g_log[4096];

bool fail(char* err, size_t size, const char* what, const char* detail)
{
    std::snprintf(err, size, "%s: %s\n%s", what, detail, g_log);
    return false;
}

#define OPTIX_OK(call) do { OptixResult r_ = (call); if (r_ != OPTIX_SUCCESS) return fail(err, err_size, #call, optixGetErrorName(r_)); } while (0)
#define CU_OK(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorName(r_, &s_); return fail(err, err_size, #call, s_ ? s_ : "?"); } } while (0)

struct Entries {
    const char *raygen, *closesthit, *anyhit, *miss;
    int payloads;
};

bool build_pipeline(Ch& ch, const char* ptx, size_t ptx_size, const Entries& names, char* err, size_t err_size)
{
    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = names.payloads;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    size_t log_size = sizeof(g_log);
    OPTIX_OK(optixModuleCreate(ch.context, &module_options, &pipeline_options, ptx, ptx_size, g_log, &log_size, &ch.module));

    OptixProgramGroupDesc desc[3] = {};
    desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc[0].raygen.module = ch.module;
    desc[0].raygen.entryFunctionName = names.raygen;
    desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc[1].miss.module = ch.module;
    desc[1].miss.entryFunctionName = names.miss;
    desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (names.closesthit && *names.closesthit) {
        desc[2].hitgroup.moduleCH = ch.module;
        desc[2].hitgroup.entryFunctionNameCH = names.closesthit;
    }
    if (names.anyhit && *names.anyhit) {
        desc[2].hitgroup.moduleAH = ch.module;
        desc[2].hitgroup.entryFunctionNameAH = names.anyhit;
    }
    OptixProgramGroupOptions group_options = {};
    log_size = sizeof(g_log);
    OPTIX_OK(optixProgramGroupCreate(ch.context, desc, 3, &group_options, g_log, &log_size, ch.groups));

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(g_log);
    OPTIX_OK(optixPipelineCreate(ch.context, &pipeline_options, &link_options, ch.groups, 3, g_log, &log_size, &ch.pipeline));

    Record records[3];
    for (int i = 0; i < 3; ++i) OPTIX_OK(optixSbtRecordPackHeader(ch.groups[i], &records[i]));
    CU_OK(cuMemAlloc(&ch.records, sizeof(records)));
    CU_OK(cuMemcpyHtoD(ch.records, records, sizeof(records)));
    ch.sbt.raygenRecord = ch.records;
    ch.sbt.missRecordBase = ch.records + sizeof(Record);
    ch.sbt.missRecordStrideInBytes = sizeof(Record);
    ch.sbt.missRecordCount = 1;
    ch.sbt.hitgroupRecordBase = ch.records + 2 * sizeof(Record);
    ch.sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    ch.sbt.hitgroupRecordCount = 1;
    CU_OK(cuMemAlloc(&ch.params, kSlots * ch.params_size));
    CU_OK(cuMemAllocHost(reinterpret_cast<void**>(&ch.host_params), kSlots * ch.params_size));
    for (int i = 0; i < kSlots; ++i) CU_OK(cuEventCreate(&ch.slot_done[i], CU_EVENT_DISABLE_TIMING));
    return true;
}

}  // namespace

extern "C" {

// Entry-point names may be empty ("") for an absent closest-hit or any-hit program.
void* chbp_create(const char* ptx, size_t ptx_size, size_t params_size, const char* raygen, const char* closesthit,
                  const char* anyhit, const char* miss, int payloads, char* err, size_t err_size)
{
    g_log[0] = '\0';
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current == nullptr) {
        std::snprintf(err, err_size, "no current CUDA context (initialise CuPy first)");
        return nullptr;
    }
    Ch* ch = new Ch();
    ch->params_size = params_size;
    const OptixResult init = optixInit();
    if (init != OPTIX_SUCCESS) { fail(err, err_size, "optixInit", optixGetErrorName(init)); delete ch; return nullptr; }
    OptixDeviceContextOptions options = {};
    const OptixResult created = optixDeviceContextCreate(current, &options, &ch->context);
    if (created != OPTIX_SUCCESS) { fail(err, err_size, "optixDeviceContextCreate", optixGetErrorName(created)); delete ch; return nullptr; }
    if (!build_pipeline(*ch, ptx, ptx_size, Entries{raygen, closesthit, anyhit, miss, payloads}, err, err_size)) {
        delete ch;
        return nullptr;
    }
    return ch;
}

// Build (update = 0) or refit (update = 1) GAS `slot` over `triangles` triangles at `vertices`
// (9 floats each). Asynchronous on the legacy stream. A refit needs a prior build of the same size.
int chbp_accel(void* handle, int slot, unsigned long long vertices, unsigned triangles, int update, char* err, size_t err_size)
{
    Ch& ch = *static_cast<Ch*>(handle);
    if (slot < 0 || slot >= kGasSlots) return fail(err, err_size, "chbp_accel", "bad slot");
    Gas& g = ch.gas[slot];
    if (update && g.triangles != triangles) return fail(err, err_size, "chbp_accel", "refit before build / size changed");
    const CUdeviceptr verts = vertices;
    const unsigned flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    input.triangleArray.numVertices = 3 * triangles;
    input.triangleArray.vertexBuffers = &verts;
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
    if (!update) {
        OptixAccelBufferSizes sizes = {};
        OPTIX_OK(optixAccelComputeMemoryUsage(ch.context, &options, &input, 1, &sizes));
        const size_t temp = sizes.tempSizeInBytes > sizes.tempUpdateSizeInBytes ? sizes.tempSizeInBytes : sizes.tempUpdateSizeInBytes;
        if (sizes.outputSizeInBytes > g.output_size || temp > g.temp_size) {
            CU_OK(cuCtxSynchronize());
            if (g.output) CU_OK(cuMemFree(g.output));
            if (g.temp) CU_OK(cuMemFree(g.temp));
            CU_OK(cuMemAlloc(&g.output, sizes.outputSizeInBytes));
            CU_OK(cuMemAlloc(&g.temp, temp));
            g.output_size = sizes.outputSizeInBytes;
            g.temp_size = temp;
        }
        g.triangles = triangles;
    }
    OPTIX_OK(optixAccelBuild(ch.context, 0, &options, &input, 1, g.temp, g.temp_size, g.output, g.output_size, &g.handle, nullptr, 0));
    return 1;
}

int chbp_launch(void* handle, int slot, const void* params, unsigned width, unsigned height, char* err, size_t err_size)
{
    Ch& ch = *static_cast<Ch*>(handle);
    if (slot < 0 || slot >= kGasSlots || !ch.gas[slot].handle) return fail(err, err_size, "chbp_launch", "no GAS in slot");
    const int s = ch.next_slot;
    ch.next_slot = (s + 1) % kSlots;
    CU_OK(cuEventSynchronize(ch.slot_done[s]));
    char* host = ch.host_params + s * ch.params_size;
    std::memcpy(host, params, ch.params_size);
    const unsigned long long gas = ch.gas[slot].handle;
    std::memcpy(host, &gas, sizeof(gas));
    const CUdeviceptr device = ch.params + s * ch.params_size;
    CU_OK(cuMemcpyHtoDAsync(device, host, ch.params_size, 0));
    OPTIX_OK(optixLaunch(ch.pipeline, 0, device, ch.params_size, &ch.sbt, width, height, 1));
    CU_OK(cuEventRecord(ch.slot_done[s], 0));
    return 1;
}

size_t chbp_gas_bytes(void* handle, int slot) { return static_cast<Ch*>(handle)->gas[slot].output_size; }

size_t chbp_ch_params_size() { return sizeof(ChParams); }

size_t chbp_fr_params_size() { return sizeof(FrParams); }

}  // extern "C"
