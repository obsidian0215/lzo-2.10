#ifndef LZO_GPU_PROTOCOL_H
#define LZO_GPU_PROTOCOL_H

#include <stdint.h>
#include <stdlib.h>
#include "timing.h"

#define LZO_MODE_COMPRESS 0
#define LZO_MODE_DECOMPRESS 1
#define LZO_DAEMON_REQUEST_MAGIC 0x4c5a4f47u
#define LZO_DAEMON_REQUEST_VERSION 2u
#define LZO_DAEMON_FLAG_RAW_BUFFER 0x1u

/* 客户端请求结构 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    char operation;          /* 'C'=COMPR, 'D'=DECOMPR */
    int alg;                 /* 0=1x, 1=1y */
    int level;               /* 10-18 */
    char input_path[1024];
    char output_path[1024];
    int block_size;
    int standard_copy;
    size_t input_size;
    uint32_t local_size;
    uint32_t flags;
} request_t;

/* 服务端响应结构 */
typedef struct {
    int status;              /* 0=成功, -1=失败 */
    char message[256];
    unsigned long time_us;   /* 处理耗时(微秒) */
    size_t out_size;         /* 输出文件大小 */
    timing_t timing;         /* 详细性能统计 */
} response_t;

#endif /* LZO_GPU_PROTOCOL_H */
