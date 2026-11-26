// Minimal OptiX host library: builds the edge-triangle GAS + pipeline once, then launches.
// C ABI for ctypes; runs in the caller's current CUDA context (CuPy's) on the legacy stream,
// so launches order with CuPy kernels. Device buffers belong to the caller (raw pointers).

#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstring>

#include "rt_params.h"

namespace {

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

constexpr int kSlots = 64;  // ring of launch-parameter slots (pinned host -> device, truly async)

struct Rt {
    RtParams* host_params = nullptr;  // pinned, kSlots entries
    CUevent slot_done[kSlots] = {};
    int next_slot = 0;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup groups[3] = {};
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixTraversableHandle gas = 0;
    CUdeviceptr gas_buffer = 0, records = 0, params = 0;
};

char g_log[4096];

bool fail(char* err, size_t size, const char* what, const char* detail)
{
    std::snprintf(err, size, "%s: %s\n%s", what, detail, g_log);
    return false;
}

#define OPTIX_OK(call) do { OptixResult r_ = (call); if (r_ != OPTIX_SUCCESS) return fail(err, err_size, #call, optixGetErrorName(r_)); } while (0)
#define CU_OK(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorName(r_, &s_); return fail(err, err_size, #call, s_ ? s_ : "?"); } } while (0)

bool build_gas(Rt& rt, const float* vertices, unsigned num_triangles, char* err, size_t err_size)
{
    const size_t bytes = sizeof(float) * 9 * num_triangles;
    CUdeviceptr verts = 0, temp = 0;
    CU_OK(cuMemAlloc(&verts, bytes));
    CU_OK(cuMemcpyHtoD(verts, vertices, bytes));
    const unsigned flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    input.triangleArray.numVertices = 3 * num_triangles;
    input.triangleArray.vertexBuffers = &verts;
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes = {};
    OPTIX_OK(optixAccelComputeMemoryUsage(rt.context, &options, &input, 1, &sizes));
    CU_OK(cuMemAlloc(&temp, sizes.tempSizeInBytes));
    CU_OK(cuMemAlloc(&rt.gas_buffer, sizes.outputSizeInBytes));
    OPTIX_OK(optixAccelBuild(rt.context, 0, &options, &input, 1, temp, sizes.tempSizeInBytes, rt.gas_buffer,
                             sizes.outputSizeInBytes, &rt.gas, nullptr, 0));
    CU_OK(cuCtxSynchronize());
    CU_OK(cuMemFree(temp));
    CU_OK(cuMemFree(verts));
    return true;
}

bool build_pipeline(Rt& rt, const char* ptx, size_t ptx_size, char* err, size_t err_size)
{
    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 2;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    size_t log_size = sizeof(g_log);
    OPTIX_OK(optixModuleCreate(rt.context, &module_options, &pipeline_options, ptx, ptx_size, g_log, &log_size, &rt.module));

    OptixProgramGroupDesc desc[3] = {};
    desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc[0].raygen.module = rt.module;
    desc[0].raygen.entryFunctionName = "__raygen__bp";
    desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc[1].miss.module = rt.module;
    desc[1].miss.entryFunctionName = "__miss__none";
    desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc[2].hitgroup.moduleAH = rt.module;
    desc[2].hitgroup.entryFunctionNameAH = "__anyhit__edge";
    OptixProgramGroupOptions group_options = {};
    log_size = sizeof(g_log);
    OPTIX_OK(optixProgramGroupCreate(rt.context, desc, 3, &group_options, g_log, &log_size, rt.groups));

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(g_log);
    OPTIX_OK(optixPipelineCreate(rt.context, &pipeline_options, &link_options, rt.groups, 3, g_log, &log_size, &rt.pipeline));

    Record records[3];
    for (int i = 0; i < 3; ++i) OPTIX_OK(optixSbtRecordPackHeader(rt.groups[i], &records[i]));
    CU_OK(cuMemAlloc(&rt.records, sizeof(records)));
    CU_OK(cuMemcpyHtoD(rt.records, records, sizeof(records)));
    rt.sbt.raygenRecord = rt.records;
    rt.sbt.missRecordBase = rt.records + sizeof(Record);
    rt.sbt.missRecordStrideInBytes = sizeof(Record);
    rt.sbt.missRecordCount = 1;
    rt.sbt.hitgroupRecordBase = rt.records + 2 * sizeof(Record);
    rt.sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    rt.sbt.hitgroupRecordCount = 1;
    CU_OK(cuMemAlloc(&rt.params, kSlots * sizeof(RtParams)));
    CU_OK(cuMemAllocHost(reinterpret_cast<void**>(&rt.host_params), kSlots * sizeof(RtParams)));
    for (int i = 0; i < kSlots; ++i) CU_OK(cuEventCreate(&rt.slot_done[i], CU_EVENT_DISABLE_TIMING));
    return true;
}

}  // namespace

extern "C" {

// vertices: 9 floats per triangle (triangle i = edge i). Returns nullptr and fills err on failure.
void* rtbp_create(const float* vertices, unsigned num_triangles, const char* ptx, size_t ptx_size, char* err, size_t err_size)
{
    g_log[0] = '\0';
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current == nullptr) {
        std::snprintf(err, err_size, "no current CUDA context (initialise CuPy first)");
        return nullptr;
    }
    Rt* rt = new Rt();
    const OptixResult init = optixInit();
    OptixDeviceContextOptions options = {};
    if (init != OPTIX_SUCCESS) { fail(err, err_size, "optixInit", optixGetErrorName(init)); delete rt; return nullptr; }
    const OptixResult created = optixDeviceContextCreate(current, &options, &rt->context);
    if (created != OPTIX_SUCCESS) { fail(err, err_size, "optixDeviceContextCreate", optixGetErrorName(created)); delete rt; return nullptr; }
    if (!build_gas(*rt, vertices, num_triangles, err, err_size) || !build_pipeline(*rt, ptx, ptx_size, err, err_size)) {
        delete rt;
        return nullptr;
    }
    return rt;
}

// Asynchronous on the legacy default stream; params->gas is overwritten. Parameters go through
// a pinned ring buffer so the copy does not synchronise the stream (pageable copies would).
int rtbp_launch(void* handle, RtParams* params, unsigned width, unsigned height, char* err, size_t err_size)
{
    Rt& rt = *static_cast<Rt*>(handle);
    const int slot = rt.next_slot;
    rt.next_slot = (slot + 1) % kSlots;
    CU_OK(cuEventSynchronize(rt.slot_done[slot]));  // the launch that last used this slot has finished
    params->gas = rt.gas;
    rt.host_params[slot] = *params;
    const CUdeviceptr device = rt.params + slot * sizeof(RtParams);
    CU_OK(cuMemcpyHtoDAsync(device, &rt.host_params[slot], sizeof(RtParams), 0));
    OPTIX_OK(optixLaunch(rt.pipeline, 0, device, sizeof(RtParams), &rt.sbt, width, height, 1));
    CU_OK(cuEventRecord(rt.slot_done[slot], 0));
    return 1;
}

size_t rtbp_params_size() { return sizeof(RtParams); }

}  // extern "C"
