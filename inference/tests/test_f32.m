// Standalone test: F32 matvec kernel correctness
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@"src/shaders.metallib"] error:&error];
        if (!lib) { fprintf(stderr, "Cannot load metallib\n"); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"matvec_f32"];
        if (!fn) { fprintf(stderr, "Kernel not found\n"); return 1; }
        id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];

        int out_dim = 32, in_dim = 64;
        float *W = calloc(out_dim * in_dim, sizeof(float));
        float *x = calloc(in_dim, sizeof(float));
        float *expected = calloc(out_dim, sizeof(float));

        for (int i = 0; i < out_dim; i++)
            for (int j = 0; j < in_dim; j++)
                W[i * in_dim + j] = 0.01f * (i - j);
        for (int j = 0; j < in_dim; j++) x[j] = 0.1f * (j % 5);
        for (int i = 0; i < out_dim; i++) {
            float dot = 0;
            for (int j = 0; j < in_dim; j++) dot += W[i * in_dim + j] * x[j];
            expected[i] = dot;
        }

        id<MTLBuffer> buf_W = [device newBufferWithBytes:W length:out_dim * in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_x = [device newBufferWithBytes:x length:in_dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [device newBufferWithLength:out_dim * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_W offset:0 atIndex:0];
        [enc setBuffer:buf_x offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        uint od = out_dim, id_ = in_dim;
        [enc setBytes:&od length:sizeof(uint) atIndex:3];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((out_dim + 15) / 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        [enc endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];

        float *gpu = (float *)[buf_out contents];
        float max_err = 0;
        for (int i = 0; i < out_dim; i++) {
            float err = fabsf(gpu[i] - expected[i]);
            if (err > max_err) max_err = err;
            if (err > 0.01f) fprintf(stderr, "MISMATCH row %d: gpu=%.6f cpu=%.6f\n", i, gpu[i], expected[i]);
        }
        printf("%s: F32 matvec (%d x %d), max_err=%.8f\n", max_err < 0.01 ? "PASS" : "FAIL", out_dim, in_dim, max_err);

        free(W); free(x); free(expected);
        return max_err < 0.01 ? 0 : 1;
    }
}
