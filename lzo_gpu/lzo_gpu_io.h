/*
 * lzo_gpu_io.h - Optimized I/O utilities for LZO GPU
 *
 * Provides efficient file reading and writing operations, including:
 * - Multi-threaded file reading (MT-IO)
 * - Optimized buffered writing with coalescing support
 * - Zero-copy and standard copy modes
 *
 * Decoupled from compression/decompression logic for better reusability.
 */

#ifndef LZO_GPU_IO_H
#define LZO_GPU_IO_H

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Multi-threaded file read into memory buffer.
 *
 * Parameters:
 *   path       - File path to read from
 *   dest       - Destination buffer (must be pre-allocated)
 *   size       - Number of bytes to read
 *   num_threads- Number of threads to use (will be clamped to 1-32)
 *   read_us_out- Optional: returns read time in microseconds (can be NULL)
 *
 * Returns: 0 on success, -1 on failure
 */
int lzo_mt_read_file(const char* path, void* dest, size_t size, int num_threads, unsigned long* read_us_out);

/* Single-threaded file read (fallback when MT not available or desired).
 *
 * Parameters:
 *   path       - File path to read from
 *   dest       - Destination buffer (must be pre-allocated)
 *   size       - Number of bytes to read
 *   read_us_out- Optional: returns read time in microseconds (can be NULL)
 *
 * Returns: 0 on success, -1 on failure
 */
int lzo_st_read_file(const char* path, void* dest, size_t size, unsigned long* read_us_out);

/* Multi-threaded read from open file descriptor at specific offset.
 * Useful for reading compressed data starting at an offset after header.
 *
 * Parameters:
 *   fd         - Open file descriptor
 *   dest       - Destination buffer (must be pre-allocated)
 *   size       - Number of bytes to read
 *   offset     - File offset to start reading from
 *   num_threads- Number of threads to use (will be clamped to 1-32)
 *   read_us_out- Optional: returns read time in microseconds (can be NULL)
 *
 * Returns: 0 on success, -1 on failure
 */
int lzo_mt_read_file_fd(int fd, void* dest, size_t size, off_t offset, int num_threads, unsigned long* read_us_out);

/* Convenience wrapper that chooses MT or ST based on enable_mt flag.
 *
 * Parameters:
 *   path       - File path to read from
 *   dest       - Destination buffer (must be pre-allocated)
 *   size       - Number of bytes to read
 *   enable_mt  - If non-zero, use multi-threaded read
 *   num_threads- Number of threads (only used if enable_mt is true)
 *   read_us_out- Optional: returns read time in microseconds (can be NULL)
 *
 * Returns: 0 on success, -1 on failure
 */
int lzo_read_file_with_mode(const char* path, void* dest, size_t size,
                            int enable_mt, int num_threads, unsigned long* read_us_out);

/* Write compressed LZO file with optimized coalescing.
 *
 * This function writes a complete LZO compressed file including header and data blocks.
 * It supports various optimization strategies for writing the compressed data:
 * - Coalesced write: combines multiple blocks before writing
 * - Chunked coalesced write: writes in large chunks when full coalesce fails
 * - Per-block write: fallback when memory is limited
 *
 * Parameters:
 *   path              - Output file path
 *   orig_size         - Original uncompressed size
 *   blk_size          - Block size used for compression
 *   nblk              - Number of blocks
 *   lens              - Array of compressed lengths for each block
 *   sparse_data       - Source data (blocks are at offsets i*worst_blk)
 *   worst_blk         - Stride between blocks in sparse_data
 *   enable_coalesce   - Enable coalesced writes (0=disabled, 1=enabled)
 *   coalesce_chunk_mb - Chunk size in MB for chunked coalesce
 *   coalesce_max_mb   - Maximum MB for full coalesce attempt
 *   stdio_buf_mb      - stdio buffer size in MB
 *   alg_id            - Algorithm ID (0=lzo1x, 1=lzo1y, etc.)
 *   debug             - Enable debug output and profiling
 *
 * Returns: 0 on success, -1 on failure
 */
int lzo_write_compressed_file(const char* path,
                              size_t orig_size, size_t blk_size,
                              size_t nblk, const unsigned int* lens,
                              const void* sparse_data, size_t worst_blk,
                              int enable_coalesce, size_t coalesce_chunk_mb,
                              size_t coalesce_max_mb, size_t stdio_buf_mb,
                              int alg_id, int debug);

#ifdef __cplusplus
}
#endif

#endif /* LZO_GPU_IO_H */
