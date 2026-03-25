// Standalone test: Q4_K matvec kernel correctness
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@"src/shaders.metallib"] error:&error];
        if (!lib) { fprintf(stderr, "Cannot load metallib\n"); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"dequant_matvec_q4k"];
        if (!fn) { fprintf(stderr, "Kernel not found\n"); return 1; }
        id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];

        // Test: 4 rows × 256 columns (1 super-block per row)
        int out_dim = 4, in_dim = 256;
        int num_sb = in_dim / 256;  // 1 super-block per row
        int bytes_per_row = num_sb * 144;
        int total_bytes = out_dim * bytes_per_row;

        uint8_t *q4k_data = calloc(total_bytes, 1);
        float *input = calloc(in_dim, sizeof(float));
        float *expected = calloc(out_dim, sizeof(float));

        // Simple input
        for (int j = 0; j < in_dim; j++) input[j] = 0.01f;

        // Build Q4_K data manually
        for (int row = 0; row < out_dim; row++) {
            uint8_t *sb = q4k_data + row * 144;

            // d = 0.1, dmin = 0.01 (as fp16)
            float d_f = 0.1f * (row + 1);
            float dmin_f = 0.01f;
            // Convert to fp16
            uint32_t d_bits; memcpy(&d_bits, &d_f, 4);
            uint16_t d_fp16 = ((d_bits >> 16) & 0x8000) | (((((d_bits >> 23) & 0xFF) - 127 + 15) & 0x1F) << 10) | ((d_bits >> 13) & 0x3FF);
            uint32_t dm_bits; memcpy(&dm_bits, &dmin_f, 4);
            uint16_t dm_fp16 = ((dm_bits >> 16) & 0x8000) | (((((dm_bits >> 23) & 0xFF) - 127 + 15) & 0x1F) << 10) | ((dm_bits >> 13) & 0x3FF);

            sb[0] = d_fp16 & 0xFF; sb[1] = d_fp16 >> 8;
            sb[2] = dm_fp16 & 0xFF; sb[3] = dm_fp16 >> 8;

            // Scales and mins: all sub-blocks get scale=1, min=0 (6-bit packed)
            for (int i = 0; i < 4; i++) {
                sb[4 + i] = 1;  // scale[i] = 1 (low 6 bits)
                sb[8 + i] = 0;  // min[i] = 0
            }
            // Sub-blocks 4-7: packed in bytes 8-11 (low 4 bits)
            // sc[4+i] = (bytes[8+i/2] >> 4*(i%2)) & 0xF | (bytes[i] >> 6) << 4
            // With bytes[i]=1, bytes[i]>>6 = 0. bytes[8+i/2] needs low 4 bits = 1
            sb[12] = 0x11; // sub-blocks 4,5 scales
            sb[13] = 0x11; // sub-blocks 6,7 scales
            sb[14] = 0; sb[15] = 0; // mins for 4-7

            // Weights: all 5 (for each nibble)
            uint8_t *qs = sb + 16;
            for (int j = 0; j < 128; j++) qs[j] = 0x55; // each nibble = 5

            // CPU reference: value = d * sc * q - dmin * mn
            // = d_f * 1 * 5 - 0.01 * 0 = 5 * d_f for all weights
            // dot product = sum(5 * d_f * 0.01) for 256 weights = 256 * 5 * d_f * 0.01
            expected[row] = 256.0f * 5.0f * d_f * 0.01f;
        }

        id<MTLBuffer> buf_data = [device newBufferWithBytes:q4k_data length:total_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_input = [device newBufferWithBytes:input length:in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_output = [device newBufferWithLength:out_dim * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_data offset:0 atIndex:0];
        [enc setBuffer:buf_input offset:0 atIndex:1];
        [enc setBuffer:buf_output offset:0 atIndex:2];
        uint od = out_dim, id_ = in_dim;
        [enc setBytes:&od length:sizeof(uint) atIndex:3];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((out_dim + 15) / 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        [enc endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];

        float *gpu = (float *)[buf_output contents];
        int pass = 1;
        for (int i = 0; i < out_dim; i++) {
            float err = fabsf(gpu[i] - expected[i]);
            float rel = (expected[i] != 0) ? err / fabsf(expected[i]) : err;
            if (rel > 0.1f || isnan(gpu[i])) {
                fprintf(stderr, "MISMATCH row %d: gpu=%.6f cpu=%.6f err=%.6f (%.1f%%)\n",
                        i, gpu[i], expected[i], err, rel * 100);
                pass = 0;
            }
        }
        printf("%s: Q4_K matvec (%d x %d)\n", pass ? "PASS" : "FAIL", out_dim, in_dim);
        for (int i = 0; i < out_dim; i++) {
            printf("  row %d: gpu=%.6f expected=%.6f\n", i, gpu[i], expected[i]);
        }

        free(q4k_data); free(input); free(expected);
        return pass ? 0 : 1;
    }
}
