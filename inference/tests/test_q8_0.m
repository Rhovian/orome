// Standalone test: Q8_0 matvec kernel correctness
// Generates known Q8_0 data, runs GPU kernel, compares against CPU reference.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device\n"); return 1; }

        // Load metallib
        NSError *error = nil;
        NSString *libPath = @"src/shaders.metallib";
        id<MTLLibrary> lib = [device newLibraryWithFile:libPath error:&error];
        if (!lib) { fprintf(stderr, "Cannot load metallib: %s\n", [[error localizedDescription] UTF8String]); return 1; }

        id<MTLFunction> fn = [lib newFunctionWithName:@"dequant_matvec_q8_0"];
        if (!fn) { fprintf(stderr, "Kernel not found\n"); return 1; }
        id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipe) { fprintf(stderr, "Pipeline error: %s\n", [[error localizedDescription] UTF8String]); return 1; }

        // Test params
        int out_dim = 64;  // small for testing
        int in_dim = 256;
        int blocks_per_row = in_dim / 32;  // 8 blocks
        int bytes_per_row = blocks_per_row * 34;  // 272 bytes
        int total_bytes = out_dim * bytes_per_row;

        // Generate Q8_0 data: scale=0.1, qs = row index (clamped to int8)
        uint8_t *q8_data = calloc(total_bytes, 1);
        float *expected = calloc(out_dim, sizeof(float));
        float *input = calloc(in_dim, sizeof(float));

        // Input: simple pattern
        for (int j = 0; j < in_dim; j++) input[j] = 0.01f * (j % 10);

        for (int row = 0; row < out_dim; row++) {
            float cpu_dot = 0;
            for (int blk = 0; blk < blocks_per_row; blk++) {
                uint8_t *block = q8_data + row * bytes_per_row + blk * 34;
                // Scale: fp16 encoding of 0.01 * (row + 1)
                float scale = 0.01f * (row + 1);
                // Convert to fp16 bytes
                uint16_t fp16;
                float tmp = scale;
                // Simple fp16 conversion
                uint32_t f32bits;
                memcpy(&f32bits, &tmp, 4);
                uint32_t sign = (f32bits >> 31) & 1;
                int32_t exp = ((f32bits >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (f32bits >> 13) & 0x3FF;
                if (exp <= 0) fp16 = 0;
                else if (exp >= 31) fp16 = (sign << 15) | (31 << 10);
                else fp16 = (sign << 15) | (exp << 10) | mant;
                block[0] = fp16 & 0xFF;
                block[1] = (fp16 >> 8) & 0xFF;

                // Quantized values: simple pattern
                for (int j = 0; j < 32; j++) {
                    int8_t q = (int8_t)((blk * 32 + j) % 127 - 63);
                    block[2 + j] = (uint8_t)q;
                    cpu_dot += scale * (float)q * input[blk * 32 + j];
                }
            }
            expected[row] = cpu_dot;
        }

        // Create Metal buffers
        id<MTLBuffer> buf_data = [device newBufferWithBytes:q8_data length:total_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_input = [device newBufferWithBytes:input length:in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_output = [device newBufferWithLength:out_dim * sizeof(float) options:MTLResourceStorageModeShared];

        // Dispatch
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_data offset:0 atIndex:0];
        [enc setBuffer:buf_input offset:0 atIndex:1];
        [enc setBuffer:buf_output offset:0 atIndex:2];
        uint od = (uint)out_dim, id_ = (uint)in_dim;
        [enc setBytes:&od length:sizeof(uint) atIndex:3];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:4];

        NSUInteger rows_per_tg = 16;
        NSUInteger tg_size = rows_per_tg * 32;
        NSUInteger num_tgs = (out_dim + rows_per_tg - 1) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Compare
        float *gpu_out = (float *)[buf_output contents];
        int pass = 1;
        float max_err = 0;
        for (int i = 0; i < out_dim; i++) {
            float err = fabsf(gpu_out[i] - expected[i]);
            if (err > max_err) max_err = err;
            if (err > 0.01f || isnan(gpu_out[i])) {
                fprintf(stderr, "MISMATCH row %d: gpu=%.6f cpu=%.6f err=%.6f\n",
                        i, gpu_out[i], expected[i], err);
                pass = 0;
                if (i > 5) break;
            }
        }
        if (pass) {
            printf("PASS: Q8_0 matvec (%d x %d), max_err=%.8f\n", out_dim, in_dim, max_err);
        } else {
            printf("FAIL: Q8_0 matvec\n");
        }

        free(q8_data); free(expected); free(input);
        return pass ? 0 : 1;
    }
}
