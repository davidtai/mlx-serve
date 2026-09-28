/* Native-issue read pool (q3_nativeissue.c, copied unchanged): pthread workers read each record's gate/up
 * and down spans (page-aligned F_NOCACHE preadv into a staging buffer, scatter into slot rows) and publish
 * every range's status words, then its ticket in an ordered log, then the sequence (RELEASE). The pool
 * never calls MLX. State is static: one pool per process. macOS only (mach time). */
#ifndef MLX_SERVE_Q3_NATIVEISSUE_H
#define MLX_SERVE_Q3_NATIVEISSUE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* res[t * RES_W + k]: 0 status, 1 preadv calls, 2 bytes returned, 3 payload landed, 4 errno,
 * 5 t_start (monotonic ns), 6 t_end, 7 worker index. */
#define Q3NI_RES_W 8
#define Q3NI_MAX_WORKERS 8
#define Q3NI_MAX_COMP 6
#define Q3NI_MAX_ITEMS 8
#define Q3NI_ST_PENDING (-1)
#define Q3NI_ST_OK 0
#define Q3NI_ST_SHORT 1
#define Q3NI_ST_OSERR 2
#define Q3NI_ST_DEADLINE 3
#define Q3NI_ST_SKIPPED 4

int q3ni_init(void);
int64_t q3ni_monotonic_ns(void);
/* nw workers, one page-aligned staging buffer of sbytes each (sbytes % psize == 0); res (n_tickets x RES_W),
 * log (n_log) and gauge (6) are caller-owned and start zeroed. 0, -1 (bad args / running), -ENOMEM, -2. */
int q3ni_start(int32_t nw, const uint64_t *staging_ptrs, int64_t sbytes, int64_t psize,
               int64_t *res_arr, int64_t n_tickets, int64_t *log_arr, int64_t n_log, int64_t *gauge_arr);
/* One job of n <= Q3NI_MAX_ITEMS records on tickets first .. first + 2n - 1: GU(0..n-1) then DOWN(0..n-1).
 * offsets: 2n; rows: n pointers to (ngu + ndown) destination addresses; lens: ngu + ndown. deadline -1 = none.
 * 0, -1 bad args / not running, -2 a ticket still pending, -3 queue full. */
int q3ni_submit(int32_t fd, int64_t file_size, int64_t deadline, int32_t n, int32_t ngu, int32_t ndown,
                const int64_t *offsets, const uint64_t *const *rows, const int64_t *lens, int64_t first);
int64_t q3ni_seq(void);                              /* ACQUIRE load of the completion sequence */
int64_t q3ni_wait(int64_t seen, int64_t timeout_ns); /* block until seq != seen or timeout; returns seq */
void q3ni_gauge(int64_t *out);                       /* 6 words */
int q3ni_quiesce(int64_t timeout_ns);                /* 0 quiescent, 1 timeout */
int q3ni_stop(void);                                 /* drain, join; 0, -1 not running */
int32_t q3ni_abi(void);                              /* 20260924 */

#ifdef Q3NI_INJECT
/* Test builds: scripted preadv rules keyed by the syscall's aligned offset, each used once (code 1 EINTR,
 * 2 EIO, 3 zero return, 4 truncate to arg bytes, 5 sleep arg ns), and a random per-range delay. */
void q3ni_test_rules(int32_t n, const int64_t *off, const int64_t *code, const int64_t *arg);
void q3ni_test_delay(uint64_t seed, int64_t max_ns);
#endif

#ifdef __cplusplus
}
#endif

#endif
