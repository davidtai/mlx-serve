/* PROFILE builds only (-Ddsv41-decode-timers=true): every command buffer MLX's GPU queue commits while a tag is set,
 * with its host commit time, Metal's GPUStartTime / GPUEndTime and its completion handler's host time
 * (dsv41_cb_timeline.mm; MLX's current buffer from dsv41_tl_mlx.cpp). Times are mach_absolute_time in ns
 * (clock_gettime_nsec_np(CLOCK_UPTIME_RAW)), the time base Metal's GPU timestamps use. Built against MLX 0.32.2's
 * CommandEncoder (one queue per stream): re-verify on every MLX bump. */
#ifndef MLX_SERVE_DSV41_CB_TIMELINE_H
#define MLX_SERVE_DSV41_CB_TIMELINE_H

#include <stdint.h>

#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t host_commit;
  uint64_t gpu_start;
  uint64_t gpu_end;
  uint64_t host_done;
  uint64_t tag;
  uint64_t status; /* MTLCommandBufferStatus at completion (4 = completed, 5 = error) */
} Dsv41tlRow;

/* MLX's current command buffer on stream s (an id<MTLCommandBuffer>), NULL when the stream has no Metal encoder. */
void *dsv41tl_mlx_buffer(mlx_stream s);
/* Hooks -commit on cmdbuf's class (once per process) and records buffers of cmdbuf's queue into rows[0..cap) from
 * index 0. 0, or -1 (no buffer / no rows), -2 (no -commit on the class). */
int dsv41tl_install(void *cmdbuf, Dsv41tlRow *rows, uint32_t cap);
/* Rows record while tag != 0 (the tag is stored in each row). */
void dsv41tl_tag(uint64_t tag);
/* committed (incl. dropped), completed (recorded rows whose handler ran), dropped (past cap). */
void dsv41tl_counts(uint32_t out[3]);
uint64_t dsv41tl_now(void);

#ifdef __cplusplus
}
#endif

#endif
