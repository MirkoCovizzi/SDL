/*
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely.
*/

/* Headless test for GPU timestamp queries (SDL_GPUQueryPool).
 *
 * It records a few passes of very different cost into offscreen textures and
 * buffers, reads the pass timestamps back after the command buffer's fence
 * has signaled and prints the time spent in each pass. No window is created.
 *
 * Exit status: 0 on success (or when the selected backend does not implement
 * timestamp queries and is not Vulkan), 1 on usage errors, 2 on setup errors
 * and 3 when the timestamps are wrong.
 */

#include <stdlib.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_main.h>

/* Regenerate the shaders with testgpu/build-shaders.sh */
#include "testgpu/cube.frag.dxil.h"
#include "testgpu/cube.frag.msl.h"
#include "testgpu/cube.frag.spv.h"
#include "testgpu/cube.vert.dxil.h"
#include "testgpu/cube.vert.msl.h"
#include "testgpu/cube.vert.spv.h"

#define TESTGPU_SUPPORTED_FORMATS (SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_MSL)

#define TARGET_WIDTH   1920
#define TARGET_HEIGHT  1080
#define COPY_SIZE      (64 * 1024 * 1024)
#define NUM_ITERATIONS 3

/* Query layout, two per pass (beginning, end) */
enum
{
    PASS_COPY,
    PASS_RENDER_LIGHT,
    PASS_RENDER_HEAVY,
    PASS_RENDER_END_ONLY,
    NUM_PASSES
};
#define NUM_QUERIES (NUM_PASSES * 2)

#define LIGHT_DRAWS 1
#define HEAVY_DRAWS 1500

typedef struct VertexData
{
    float x, y, z;
    float red, green, blue;
} VertexData;

/* A quad that covers most of the render target */
static const VertexData quad_vertices[] = {
    { -0.9f, -0.9f, 0.2f, 1.0f, 0.0f, 0.0f },
    { 0.9f, -0.9f, 0.2f, 0.0f, 1.0f, 0.0f },
    { 0.9f, 0.9f, 0.2f, 0.0f, 0.0f, 1.0f },
    { -0.9f, -0.9f, 0.2f, 1.0f, 0.0f, 0.0f },
    { 0.9f, 0.9f, 0.2f, 0.0f, 0.0f, 1.0f },
    { -0.9f, 0.9f, 0.2f, 1.0f, 1.0f, 0.0f },
};

static const float identity_matrix[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

static SDL_GPUDevice *gpu_device;

static SDL_GPUShader *load_shader(bool is_vertex)
{
    SDL_GPUShaderCreateInfo createinfo;
    SDL_GPUShaderFormat format = SDL_GetGPUShaderFormats(gpu_device);

    SDL_zero(createinfo);
    createinfo.num_uniform_buffers = is_vertex ? 1 : 0;
    createinfo.stage = is_vertex ? SDL_GPU_SHADERSTAGE_VERTEX : SDL_GPU_SHADERSTAGE_FRAGMENT;

    if (format & SDL_GPU_SHADERFORMAT_DXIL) {
        createinfo.format = SDL_GPU_SHADERFORMAT_DXIL;
        createinfo.code = is_vertex ? cube_vert_dxil : cube_frag_dxil;
        createinfo.code_size = is_vertex ? cube_vert_dxil_len : cube_frag_dxil_len;
    } else if (format & SDL_GPU_SHADERFORMAT_MSL) {
        createinfo.format = SDL_GPU_SHADERFORMAT_MSL;
        createinfo.code = is_vertex ? cube_vert_msl : cube_frag_msl;
        createinfo.code_size = is_vertex ? cube_vert_msl_len : cube_frag_msl_len;
    } else {
        createinfo.format = SDL_GPU_SHADERFORMAT_SPIRV;
        createinfo.code = is_vertex ? cube_vert_spv : cube_frag_spv;
        createinfo.code_size = is_vertex ? cube_vert_spv_len : cube_frag_spv_len;
    }

    return SDL_CreateGPUShader(gpu_device, &createinfo);
}

static int quit(int rc)
{
    if (gpu_device) {
        SDL_DestroyGPUDevice(gpu_device);
        gpu_device = NULL;
    }
    SDL_Quit();
    return rc;
}

static void draw_many(SDL_GPUCommandBuffer *cmd, SDL_GPUGraphicsPipeline *pipeline,
                      SDL_GPUBuffer *vertex_buffer, SDL_GPUColorTargetInfo *color_target,
                      SDL_GPUDepthStencilTargetInfo *depth_target, int num_draws)
{
    SDL_GPUBufferBinding vertex_binding;
    SDL_GPURenderPass *pass;
    int i;

    vertex_binding.buffer = vertex_buffer;
    vertex_binding.offset = 0;

    SDL_PushGPUVertexUniformData(cmd, 0, identity_matrix, sizeof(identity_matrix));

    pass = SDL_BeginGPURenderPass(cmd, color_target, 1, depth_target);
    SDL_BindGPUGraphicsPipeline(pass, pipeline);
    SDL_BindGPUVertexBuffers(pass, 0, &vertex_binding, 1);
    for (i = 0; i < num_draws; ++i) {
        SDL_DrawGPUPrimitives(pass, SDL_arraysize(quad_vertices), 1, 0, 0);
    }
    SDL_EndGPURenderPass(pass);
}

int main(int argc, char *argv[])
{
    const char *driver = NULL;
    const char *pass_names[NUM_PASSES] = {
        "copy 64 MiB buffer",
        "render, 1 full-screen quad",
        "render, 1500 full-screen quads",
        "render, 1500 quads (end timestamp only)"
    };
    SDL_GPUQueryPoolCreateInfo pool_info;
    SDL_GPUQueryPool *pool = NULL;
    SDL_GPUBuffer *vertex_buffer = NULL;
    SDL_GPUBuffer *copy_src = NULL;
    SDL_GPUBuffer *copy_dst = NULL;
    SDL_GPUTransferBuffer *transfer = NULL;
    SDL_GPUTexture *color_texture = NULL;
    SDL_GPUTexture *depth_texture = NULL;
    SDL_GPUShader *vertex_shader, *fragment_shader;
    SDL_GPUGraphicsPipeline *pipeline = NULL;
    int i, iteration;
    int rc = 0;

    for (i = 1; i < argc; ++i) {
        if (SDL_strcmp(argv[i], "--driver") == 0 && i + 1 < argc) {
            driver = argv[++i];
        } else {
            SDL_Log("Usage: %s [--driver NAME]", argv[0]);
            return 1;
        }
    }

    /* The offscreen video driver gives us Vulkan without a display server */
    if (!SDL_GetHint(SDL_HINT_VIDEO_DRIVER) && !SDL_getenv("SDL_VIDEO_DRIVER")) {
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "offscreen");
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 2;
    }

    gpu_device = SDL_CreateGPUDevice(TESTGPU_SUPPORTED_FORMATS, true, driver);
    if (!gpu_device) {
        SDL_Log("Failed to create GPU device: %s", SDL_GetError());
        return quit(2);
    }
    SDL_Log("GPU driver: %s", SDL_GetGPUDeviceDriver(gpu_device));

    if (!SDL_GPUSupportsQueryType(gpu_device, SDL_GPU_QUERYTYPE_TIMESTAMP)) {
        if (SDL_strcmp(SDL_GetGPUDeviceDriver(gpu_device), "vulkan") == 0) {
            SDL_Log("FAIL: timestamp queries are not supported on Vulkan");
            return quit(3);
        }
        SDL_Log("Timestamp queries are not implemented by the %s backend, skipping", SDL_GetGPUDeviceDriver(gpu_device));
        return quit(0);
    }

    /* Resources */

    vertex_shader = load_shader(true);
    fragment_shader = load_shader(false);
    if (!vertex_shader || !fragment_shader) {
        SDL_Log("Failed to create shaders: %s", SDL_GetError());
        return quit(2);
    }

    {
        SDL_GPUGraphicsPipelineCreateInfo pipelinedesc;
        SDL_GPUColorTargetDescription color_target_desc;
        SDL_GPUVertexBufferDescription vertex_buffer_desc;
        SDL_GPUVertexAttribute vertex_attributes[2];

        SDL_zero(pipelinedesc);
        SDL_zero(color_target_desc);
        SDL_zero(vertex_buffer_desc);
        SDL_zeroa(vertex_attributes);

        color_target_desc.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        pipelinedesc.target_info.num_color_targets = 1;
        pipelinedesc.target_info.color_target_descriptions = &color_target_desc;
        pipelinedesc.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
        pipelinedesc.target_info.has_depth_stencil_target = true;

        /* Equal depths pass, so every quad is fully shaded and the cost scales with the draw count */
        pipelinedesc.depth_stencil_state.enable_depth_test = true;
        pipelinedesc.depth_stencil_state.enable_depth_write = true;
        pipelinedesc.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;

        pipelinedesc.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        pipelinedesc.vertex_shader = vertex_shader;
        pipelinedesc.fragment_shader = fragment_shader;

        vertex_buffer_desc.slot = 0;
        vertex_buffer_desc.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
        vertex_buffer_desc.pitch = sizeof(VertexData);

        vertex_attributes[0].location = 0;
        vertex_attributes[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
        vertex_attributes[0].offset = 0;
        vertex_attributes[1].location = 1;
        vertex_attributes[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
        vertex_attributes[1].offset = sizeof(float) * 3;

        pipelinedesc.vertex_input_state.num_vertex_buffers = 1;
        pipelinedesc.vertex_input_state.vertex_buffer_descriptions = &vertex_buffer_desc;
        pipelinedesc.vertex_input_state.num_vertex_attributes = 2;
        pipelinedesc.vertex_input_state.vertex_attributes = vertex_attributes;

        pipeline = SDL_CreateGPUGraphicsPipeline(gpu_device, &pipelinedesc);
    }
    SDL_ReleaseGPUShader(gpu_device, vertex_shader);
    SDL_ReleaseGPUShader(gpu_device, fragment_shader);
    if (!pipeline) {
        SDL_Log("Failed to create pipeline: %s", SDL_GetError());
        return quit(2);
    }

    {
        SDL_GPUBufferCreateInfo buffer_info;
        SDL_GPUTransferBufferCreateInfo transfer_info;
        SDL_GPUTextureCreateInfo texture_info;
        SDL_GPUCommandBuffer *cmd;
        SDL_GPUCopyPass *copy_pass;
        SDL_GPUTransferBufferLocation location;
        SDL_GPUBufferRegion region;
        void *map;

        SDL_zero(buffer_info);
        buffer_info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
        buffer_info.size = sizeof(quad_vertices);
        vertex_buffer = SDL_CreateGPUBuffer(gpu_device, &buffer_info);

        /* Plain buffers have no dedicated copy usage; any usage works as a copy endpoint */
        buffer_info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
        buffer_info.size = COPY_SIZE;
        copy_src = SDL_CreateGPUBuffer(gpu_device, &buffer_info);
        copy_dst = SDL_CreateGPUBuffer(gpu_device, &buffer_info);

        SDL_zero(texture_info);
        texture_info.type = SDL_GPU_TEXTURETYPE_2D;
        texture_info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        texture_info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
        texture_info.width = TARGET_WIDTH;
        texture_info.height = TARGET_HEIGHT;
        texture_info.layer_count_or_depth = 1;
        texture_info.num_levels = 1;
        color_texture = SDL_CreateGPUTexture(gpu_device, &texture_info);

        texture_info.format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
        texture_info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
        depth_texture = SDL_CreateGPUTexture(gpu_device, &texture_info);

        SDL_zero(transfer_info);
        transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
        transfer_info.size = sizeof(quad_vertices);
        transfer = SDL_CreateGPUTransferBuffer(gpu_device, &transfer_info);

        if (!vertex_buffer || !copy_src || !copy_dst || !color_texture || !depth_texture || !transfer) {
            SDL_Log("Failed to create resources: %s", SDL_GetError());
            rc = 2;
            goto cleanup;
        }

        map = SDL_MapGPUTransferBuffer(gpu_device, transfer, false);
        SDL_memcpy(map, quad_vertices, sizeof(quad_vertices));
        SDL_UnmapGPUTransferBuffer(gpu_device, transfer);

        cmd = SDL_AcquireGPUCommandBuffer(gpu_device);
        copy_pass = SDL_BeginGPUCopyPass(cmd);
        location.transfer_buffer = transfer;
        location.offset = 0;
        region.buffer = vertex_buffer;
        region.offset = 0;
        region.size = sizeof(quad_vertices);
        SDL_UploadToGPUBuffer(copy_pass, &location, &region, false);
        SDL_EndGPUCopyPass(copy_pass);
        SDL_SubmitGPUCommandBuffer(cmd);
    }

    SDL_zero(pool_info);
    pool_info.type = SDL_GPU_QUERYTYPE_TIMESTAMP;
    pool_info.num_queries = NUM_QUERIES;
    pool_info.props = SDL_CreateProperties();
    SDL_SetStringProperty(pool_info.props, SDL_PROP_GPU_QUERYPOOL_CREATE_NAME_STRING, "Pass timestamps");
    pool = SDL_CreateGPUQueryPool(gpu_device, &pool_info);
    SDL_DestroyProperties(pool_info.props);
    if (!pool) {
        SDL_Log("Failed to create query pool: %s", SDL_GetError());
        rc = 2;
        goto cleanup;
    }

    /* Record, submit, wait and read back, several times over */

    for (iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
        SDL_GPUCommandBuffer *cmd;
        SDL_GPUFence *fence;
        SDL_GPUColorTargetInfo color_target;
        SDL_GPUDepthStencilTargetInfo depth_target;
        SDL_GPUCopyPass *copy_pass;
        SDL_GPUBufferLocation src, dst;
        Uint64 ticks[NUM_QUERIES];
        Uint64 durations[NUM_PASSES];
        Uint64 unused[1];

        SDL_zero(color_target);
        color_target.texture = color_texture;
        color_target.load_op = SDL_GPU_LOADOP_CLEAR;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_zero(depth_target);
        depth_target.texture = depth_texture;
        depth_target.clear_depth = 1.0f;
        depth_target.load_op = SDL_GPU_LOADOP_CLEAR;
        depth_target.store_op = SDL_GPU_STOREOP_DONT_CARE;
        depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
        depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

        cmd = SDL_AcquireGPUCommandBuffer(gpu_device);
        if (!cmd) {
            SDL_Log("SDL_AcquireGPUCommandBuffer failed: %s", SDL_GetError());
            rc = 2;
            goto cleanup;
        }

        /* Copy pass */
        SDL_SetGPUPassTimestamps(cmd, pool, PASS_COPY * 2, PASS_COPY * 2 + 1);
        copy_pass = SDL_BeginGPUCopyPass(cmd);
        src.buffer = copy_src;
        src.offset = 0;
        dst.buffer = copy_dst;
        dst.offset = 0;
        SDL_CopyGPUBufferToBuffer(copy_pass, &src, &dst, COPY_SIZE, false);
        SDL_EndGPUCopyPass(copy_pass);

        /* A cheap render pass, then an expensive one */
        SDL_SetGPUPassTimestamps(cmd, pool, PASS_RENDER_LIGHT * 2, PASS_RENDER_LIGHT * 2 + 1);
        draw_many(cmd, pipeline, vertex_buffer, &color_target, &depth_target, LIGHT_DRAWS);

        SDL_SetGPUPassTimestamps(cmd, pool, PASS_RENDER_HEAVY * 2, PASS_RENDER_HEAVY * 2 + 1);
        draw_many(cmd, pipeline, vertex_buffer, &color_target, &depth_target, HEAVY_DRAWS);

        /* Only the end of this pass is timed */
        SDL_SetGPUPassTimestamps(cmd, pool, SDL_GPU_QUERY_INDEX_NONE, PASS_RENDER_END_ONLY * 2 + 1);
        draw_many(cmd, pipeline, vertex_buffer, &color_target, &depth_target, HEAVY_DRAWS);

        fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
        if (!fence) {
            SDL_Log("Submit failed: %s", SDL_GetError());
            rc = 2;
            goto cleanup;
        }
        SDL_WaitForGPUFences(gpu_device, true, &fence, 1);
        SDL_ReleaseGPUFence(gpu_device, fence);

        /* Query 6 was never written (the last pass has no beginning timestamp),
         * so asking for it must fail rather than return garbage */
        if (SDL_GetGPUQueryPoolResults(gpu_device, pool, PASS_RENDER_END_ONLY * 2, 1, unused)) {
            SDL_Log("FAIL: reading a query that was never written succeeded");
            rc = 3;
        } else {
            SDL_Log("Reading an unwritten query failed as expected: %s", SDL_GetError());
        }

        for (i = 0; i < NUM_PASSES; ++i) {
            /* Read the begin/end pair of each pass; the last one only has an end */
            Uint32 first = (i == PASS_RENDER_END_ONLY) ? (Uint32)(i * 2 + 1) : (Uint32)(i * 2);
            Uint32 count = (i == PASS_RENDER_END_ONLY) ? 1 : 2;
            if (!SDL_GetGPUQueryPoolResults(gpu_device, pool, first, count, &ticks[first])) {
                SDL_Log("FAIL: SDL_GetGPUQueryPoolResults failed for pass %d: %s", i, SDL_GetError());
                rc = 3;
                goto cleanup;
            }
        }

        SDL_Log("Iteration %d:", iteration);
        for (i = 0; i < PASS_RENDER_END_ONLY; ++i) {
            const Uint64 begin = ticks[i * 2];
            const Uint64 end = ticks[i * 2 + 1];
            if (end < begin) {
                SDL_Log("FAIL: pass %d (%s) has a non-monotonic timestamp pair: begin=%" SDL_PRIu64 " end=%" SDL_PRIu64,
                        i, pass_names[i], begin, end);
                rc = 3;
                durations[i] = 0;
                continue;
            }
            durations[i] = end - begin;
            SDL_Log("  pass %d, %-40s %12" SDL_PRIu64 " ns", i, pass_names[i], durations[i]);
        }
        SDL_Log("  pass %d, %-40s end timestamp at %" SDL_PRIu64 " ns", PASS_RENDER_END_ONLY, pass_names[PASS_RENDER_END_ONLY], ticks[PASS_RENDER_END_ONLY * 2 + 1]);

        /* The passes ran in submission order, so the timeline must be ordered too */
        if (ticks[PASS_RENDER_LIGHT * 2] < ticks[PASS_COPY * 2 + 1] ||
            ticks[PASS_RENDER_HEAVY * 2] < ticks[PASS_RENDER_LIGHT * 2 + 1] ||
            ticks[PASS_RENDER_END_ONLY * 2 + 1] < ticks[PASS_RENDER_HEAVY * 2 + 1]) {
            SDL_Log("FAIL: timestamps of consecutive passes are not ordered");
            rc = 3;
        }

        if (durations[PASS_COPY] == 0 || durations[PASS_RENDER_HEAVY] == 0) {
            SDL_Log("FAIL: non-trivial passes reported zero time");
            rc = 3;
        }
        if (durations[PASS_RENDER_HEAVY] <= durations[PASS_RENDER_LIGHT]) {
            SDL_Log("FAIL: the heavy pass (%" SDL_PRIu64 " ns) was not slower than the light pass (%" SDL_PRIu64 " ns)",
                    durations[PASS_RENDER_HEAVY], durations[PASS_RENDER_LIGHT]);
            rc = 3;
        }
    }

    /* Releasing a pool right after use must be safe */
    SDL_ReleaseGPUQueryPool(gpu_device, pool);
    pool = NULL;

cleanup:
    SDL_ReleaseGPUQueryPool(gpu_device, pool);
    SDL_ReleaseGPUTransferBuffer(gpu_device, transfer);
    SDL_ReleaseGPUTexture(gpu_device, color_texture);
    SDL_ReleaseGPUTexture(gpu_device, depth_texture);
    SDL_ReleaseGPUBuffer(gpu_device, vertex_buffer);
    SDL_ReleaseGPUBuffer(gpu_device, copy_src);
    SDL_ReleaseGPUBuffer(gpu_device, copy_dst);
    SDL_ReleaseGPUGraphicsPipeline(gpu_device, pipeline);

    if (rc == 0) {
        SDL_Log("PASS");
    }
    return quit(rc);
}
