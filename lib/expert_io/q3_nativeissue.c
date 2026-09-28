/* DSV41_NATIVE_ISSUE (2026-09-24, lever A3 of q3-readwait-20260924.md): a native pthread pool that issues
 * each miss record's reads (page-aligned F_NOCACHE preadv into a staging buffer + the scatter copy into the
 * slot rows) with no Python and no GIL, and publishes each range's completion into a shared status array
 * plus an ordered completion log.  Python never runs on these threads.
 *
 * Ownership (q3_nativeissue_candidate.py):
 *   - the staging buffers are the stock AlignedReadBufferPool's 4 x 9 MiB mmaps, leased from that pool
 *     for the lane's lifetime (one per worker: the stock cap of 4 reads in flight is kept);
 *   - res[] (status words), log[] (completion order) and gauge[] are Python-owned ctypes arrays;
 *   - the generation thread submits one JOB per stock fill unit (one _fill record or one _fill_batch
 *     adjacency run): its ranges are GU(item 0..n-1) then DOWN(item 0..n-1), the stock per-thread
 *     order of bind_tcq_gu_reader.run.  One worker runs a job start to finish (as one stock reader
 *     thread runs one fill); a failed range marks the job's remaining ranges SKIPPED (stock: the fill
 *     thread raised and issued nothing more).
 *
 * Per range (a transcription of PositionalExpertReader._readv_range_into_impl + AlignedReadBufferPool.
 * _read_some, as nativeread/q3_nativeread.c): deadline checks at the stock points; aligned request =
 * [floor(off, page), ceil(off + pending, page)) capped at the staging size and EOF; EINTR retried inside
 * (CPython's os.preadv, PEP 475); a positive read holding only alignment padding is retried; error ->
 * errno; payload 0 -> short read; payload scattered in destination order.
 *
 * Publication: res[t] fields are written, then (under the pool mutex) log[seq % logn] = t and seq is
 * advanced with a RELEASE store, then waiters are broadcast.  A reader that loads seq with ACQUIRE
 * (q3ni_seq / q3ni_wait) sees res[t] and every slot byte the range copied.
 *
 * res[t * RES_W + k]: 0 status (-1 pending, 0 ok, 1 short, 2 os error, 3 deadline, 4 skipped),
 *   1 preadv calls, 2 bytes returned, 3 payload landed, 4 errno, 5 t_start (first syscall, monotonic ns),
 *   6 t_end, 7 worker index.
 * gauge (stock ExpertIOMetrics.enter_read/exit_read semantics, per range): 0 inflight, 1 inflight max,
 *   2 depth sum, 3 samples, 4 union wall ns, 5 busy-since ns.
 */
#include <errno.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define RES_W 8
#define MAX_WORKERS 8
#define MAX_COMP 6
#define MAX_ITEMS 8
#define ST_PENDING (-1)
#define ST_OK 0
#define ST_SHORT 1
#define ST_OSERR 2
#define ST_DEADLINE 3
#define ST_SKIPPED 4

static uint32_t tb_numer = 0, tb_denom = 0;

int q3ni_init(void) {
    mach_timebase_info_data_t tb;
    if (mach_timebase_info(&tb) != KERN_SUCCESS || tb.denom == 0) return -1;
    tb_numer = tb.numer;
    tb_denom = tb.denom;
    return 0;
}

int64_t q3ni_monotonic_ns(void) {
    uint64_t ticks = mach_absolute_time();
    int64_t t = (int64_t)ticks;
    int64_t intpart = t / tb_denom;
    int64_t rem = t % tb_denom;
    return intpart * (int64_t)tb_numer + (rem * (int64_t)tb_numer) / (int64_t)tb_denom;
}

typedef struct {
    int64_t offset;
    int32_t ndst;
    uint64_t dst[MAX_COMP];
    int64_t len[MAX_COMP];
} range_t;

typedef struct {
    int fd;
    int64_t file_size;
    int64_t deadline;
    int64_t first;      /* first ticket */
    int32_t count;      /* ranges in the job */
} job_t;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t work_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;
static pthread_t threads[MAX_WORKERS];
static char *staging[MAX_WORKERS];
static int nworkers = 0, running = 0, stopping = 0, busy = 0;
static int64_t staging_bytes = 0, page_size = 0;
static int64_t *res = 0, nt = 0, *logbuf = 0, logn = 0, *gauge = 0;
static range_t *ranges = 0;
static job_t *queue = 0;
static int64_t qcap = 0, qhead = 0, qlen = 0;
static uint64_t seq = 0;

#ifdef Q3NI_INJECT
/* Test build only: scripted preadv rules keyed by the aligned file offset of the syscall, each used once,
 * and a per-range random delay.  code 1 EINTR once (retried inside the call, as os.preadv), 2 EIO,
 * 3 zero return, 4 truncate to arg bytes, 5 sleep arg ns then a real call. */
#define MAX_RULES 16
static int64_t rule_off[MAX_RULES], rule_code[MAX_RULES], rule_arg[MAX_RULES], rule_used[MAX_RULES];
static int nrules = 0;
static uint64_t rng = 0;
static int64_t delay_max = 0;

void q3ni_test_rules(int32_t n, const int64_t *off, const int64_t *code, const int64_t *arg) {
    pthread_mutex_lock(&mu);
    nrules = n > MAX_RULES ? MAX_RULES : n;
    for (int i = 0; i < nrules; i++) { rule_off[i] = off[i]; rule_code[i] = code[i]; rule_arg[i] = arg[i]; rule_used[i] = 0; }
    pthread_mutex_unlock(&mu);
}

void q3ni_test_delay(uint64_t seed, int64_t max_ns) {
    pthread_mutex_lock(&mu);
    rng = seed ? seed : 88172645463325252ULL;
    delay_max = max_ns;
    pthread_mutex_unlock(&mu);
}

static void sleep_ns(int64_t ns) {
    struct timespec ts = { (time_t)(ns / 1000000000), (long)(ns % 1000000000) };
    nanosleep(&ts, 0);
}

static void range_delay(void) {
    int64_t d = 0;
    pthread_mutex_lock(&mu);
    if (delay_max > 0) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        d = (int64_t)(rng % (uint64_t)delay_max);
    }
    pthread_mutex_unlock(&mu);
    if (d > 0) sleep_ns(d);
}

static ssize_t do_preadv(int fd, const struct iovec *iov, int n, off_t off) {
    int64_t code = 0, arg = 0;
    pthread_mutex_lock(&mu);
    for (int i = 0; i < nrules; i++) {
        if (!rule_used[i] && rule_off[i] == (int64_t)off) { rule_used[i] = 1; code = rule_code[i]; arg = rule_arg[i]; break; }
    }
    pthread_mutex_unlock(&mu);
    if (code == 2) { errno = EIO; return -1; }
    if (code == 3) return 0;
    if (code == 4) {
        struct iovec one = iov[0];
        if ((int64_t)one.iov_len > arg) one.iov_len = (size_t)arg;
        return preadv(fd, &one, 1, off);
    }
    if (code == 5) sleep_ns(arg);
    ssize_t r;
    do { r = preadv(fd, iov, n, off); } while (r < 0 && errno == EINTR);   /* code 1: EINTR retried inside */
    return r;
}
#define RANGE_DELAY() range_delay()
#else
static ssize_t do_preadv(int fd, const struct iovec *iov, int n, off_t off) {
    ssize_t r;
    do { r = preadv(fd, iov, n, off); } while (r < 0 && errno == EINTR);
    return r;
}
#define RANGE_DELAY() ((void)0)
#endif

#define LATE(deadline) ((deadline) >= 0 && q3ni_monotonic_ns() >= (deadline))

static void gauge_enter(void) {
    pthread_mutex_lock(&mu);
    if (gauge[0] == 0) gauge[5] = q3ni_monotonic_ns();
    gauge[0] += 1;
    if (gauge[0] > gauge[1]) gauge[1] = gauge[0];
    gauge[2] += gauge[0];
    gauge[3] += 1;
    pthread_mutex_unlock(&mu);
}

static void gauge_exit(void) {
    pthread_mutex_lock(&mu);
    if (gauge[0] > 0) {
        gauge[0] -= 1;
        if (gauge[0] == 0 && gauge[5]) { gauge[4] += q3ni_monotonic_ns() - gauge[5]; gauge[5] = 0; }
    }
    pthread_mutex_unlock(&mu);
}

/* one range: stock _readv_range_into_impl loop + _read_some steps; returns the status */
static int run_range(const job_t *job, const range_t *rg, char *stage, int64_t *out) {
    int64_t calls = 0, returned = 0, read_total = 0, t_start = 0;
    int status = ST_OK;
    int32_t d = 0;
    int64_t d_off = 0;
    out[4] = 0;
    while (d < rg->ndst && rg->len[d] == 0) d++;
    while (d < rg->ndst) {
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }         /* impl loop top */
        if (LATE(job->deadline)) { status = ST_DEADLINE; break; }         /* _read_some entry */
        int64_t off = rg->offset + read_total;
        if (off >= job->file_size) { status = ST_SHORT; break; }
        int64_t requested = rg->len[d] - d_off;
        for (int32_t i = d + 1; i < rg->ndst; i++) requested += rg->len[i];
        int64_t aligned_offset = off / page_size * page_size;
        int64_t skip = off - aligned_offset;
        int64_t aligned_end = (off + requested + page_size - 1) / page_size * page_size;
        int64_t physical = staging_bytes;
        if (aligned_end - aligned_offset < physical) physical = aligned_end - aligned_offset;
        if (job->file_size - aligned_offset < physical) physical = job->file_size - aligned_offset;
        if (physical <= 0) { status = ST_SHORT; break; }
        struct iovec iov = { stage, (size_t)physical };
        ssize_t r;
        for (;;) {
            if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
            calls++;
            if (!t_start) t_start = q3ni_monotonic_ns();
            r = do_preadv(job->fd, &iov, 1, (off_t)aligned_offset);
            if (r < 0) { out[4] = errno; status = ST_OSERR; break; }
            returned += r;
            if (LATE(job->deadline)) { status = ST_DEADLINE; break; }
            if (r <= 0 || r > skip) break;
        }
        if (status) break;
        int64_t payload = r - skip;
        if (payload < 0) payload = 0;
        if (payload > requested) payload = requested;
        if (payload <= 0) { status = ST_SHORT; break; }
        const char *src = stage + skip;
        int64_t left = payload;
        while (left > 0) {
            int64_t room = rg->len[d] - d_off;
            int64_t n = left < room ? left : room;
            memcpy((char *)(uintptr_t)rg->dst[d] + d_off, src, (size_t)n);
            src += n; left -= n; d_off += n;
            if (d_off == rg->len[d]) { d++; d_off = 0; while (d < rg->ndst && rg->len[d] == 0) d++; }
        }
        read_total += payload;
    }
    out[1] = calls;
    out[2] = returned;
    out[3] = read_total;
    out[5] = t_start ? t_start : q3ni_monotonic_ns();
    out[6] = q3ni_monotonic_ns();
    return status;
}

static void publish(int64_t t) {
    /* caller holds mu; res[t] already written */
    logbuf[seq % (uint64_t)logn] = t;
    __atomic_store_n(&seq, seq + 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&done_cv);
}

static void *worker(void *arg) {
    int w = (int)(intptr_t)arg;
    char *stage = staging[w];
    for (;;) {
        pthread_mutex_lock(&mu);
        while (!stopping && qlen == 0) pthread_cond_wait(&work_cv, &mu);
        if (qlen == 0 && stopping) { pthread_mutex_unlock(&mu); return 0; }
        job_t job = queue[qhead];
        qhead = (qhead + 1) % qcap;
        qlen -= 1;
        busy += 1;
        pthread_mutex_unlock(&mu);
        int failed = 0;
        for (int32_t i = 0; i < job.count; i++) {
            int64_t t = job.first + i;
            int64_t *out = res + t * RES_W;
            if (failed) {
                out[1] = out[2] = out[3] = out[4] = 0;
                out[5] = out[6] = q3ni_monotonic_ns();
                out[7] = w;
                pthread_mutex_lock(&mu);
                out[0] = ST_SKIPPED;
                publish(t);
                pthread_mutex_unlock(&mu);
                continue;
            }
            RANGE_DELAY();
            gauge_enter();
            int status = run_range(&job, &ranges[t], stage, out);
            gauge_exit();
            out[7] = w;
            pthread_mutex_lock(&mu);
            out[0] = status;
            publish(t);
            pthread_mutex_unlock(&mu);
            if (status) failed = 1;
        }
        pthread_mutex_lock(&mu);
        busy -= 1;
        pthread_cond_broadcast(&done_cv);
        pthread_mutex_unlock(&mu);
    }
}

/* Start the pool.  staging_ptrs: nw page-aligned buffers of sbytes.  Returns 0 or -errno / -1. */
int q3ni_start(int32_t nw, const uint64_t *staging_ptrs, int64_t sbytes, int64_t psize,
               int64_t *res_arr, int64_t n_tickets, int64_t *log_arr, int64_t n_log, int64_t *gauge_arr) {
    if (tb_denom == 0 && q3ni_init() != 0) return -1;
    pthread_mutex_lock(&mu);
    if (running || nw < 1 || nw > MAX_WORKERS || psize <= 0 || sbytes % psize) { pthread_mutex_unlock(&mu); return -1; }
    ranges = (range_t *)calloc((size_t)n_tickets, sizeof(range_t));
    queue = (job_t *)calloc((size_t)n_tickets, sizeof(job_t));
    if (!ranges || !queue) { free(ranges); free(queue); ranges = 0; queue = 0; pthread_mutex_unlock(&mu); return -ENOMEM; }
    qcap = n_tickets; qhead = 0; qlen = 0;
    res = res_arr; nt = n_tickets; logbuf = log_arr; logn = n_log; gauge = gauge_arr;
    staging_bytes = sbytes; page_size = psize;
    stopping = 0; busy = 0; seq = 0;
    for (int i = 0; i < nw; i++) staging[i] = (char *)(uintptr_t)staging_ptrs[i];
    nworkers = 0;
    for (int i = 0; i < nw; i++) {
        if (pthread_create(&threads[i], 0, worker, (void *)(intptr_t)i) != 0) break;
        nworkers++;
    }
    running = 1;
    pthread_mutex_unlock(&mu);
    if (nworkers != nw) return -2;
    return 0;
}

/* One job (a stock fill unit).  offsets: 2*n (GU offsets, then DOWN offsets); rows: n pointers, each to
 * (ngu + ndown) destination addresses; lens: (ngu + ndown).  Called with the GIL held (PyDLL): never blocks
 * beyond the pool mutex.  Returns 0; -1 bad args / not running; -2 a ticket is still pending; -3 queue full. */
int q3ni_submit(int32_t fd, int64_t file_size, int64_t deadline, int32_t n, int32_t ngu, int32_t ndown,
                const int64_t *offsets, const uint64_t *const *rows, const int64_t *lens, int64_t first) {
    if (n < 1 || n > MAX_ITEMS || ngu < 1 || ndown < 1 || ngu > MAX_COMP || ndown > MAX_COMP) return -1;
    int32_t count = 2 * n;
    if (first < 0 || first + count > nt) return -1;
    pthread_mutex_lock(&mu);
    if (!running || stopping) { pthread_mutex_unlock(&mu); return -1; }
    for (int32_t i = 0; i < count; i++) {
        if (res[(first + i) * RES_W] == ST_PENDING) { pthread_mutex_unlock(&mu); return -2; }
    }
    if (qlen >= qcap) { pthread_mutex_unlock(&mu); return -3; }
    for (int32_t i = 0; i < count; i++) {
        int down = i >= n;
        int32_t item = down ? i - n : i;
        range_t *rg = &ranges[first + i];
        rg->offset = offsets[i];
        rg->ndst = down ? ndown : ngu;
        for (int32_t c = 0; c < rg->ndst; c++) {
            int32_t k = down ? ngu + c : c;
            rg->dst[c] = rows[item][k];
            rg->len[c] = lens[k];
        }
        int64_t *out = res + (first + i) * RES_W;
        for (int k = 1; k < RES_W; k++) out[k] = 0;
        out[0] = ST_PENDING;
    }
    job_t *job = &queue[(qhead + qlen) % qcap];
    job->fd = fd; job->file_size = file_size; job->deadline = deadline; job->first = first; job->count = count;
    qlen += 1;
    pthread_cond_signal(&work_cv);
    pthread_mutex_unlock(&mu);
    return 0;
}

/* ACQUIRE load of the completion sequence (PyDLL: GIL held, no blocking). */
int64_t q3ni_seq(void) {
    return (int64_t)__atomic_load_n(&seq, __ATOMIC_ACQUIRE);
}

/* Block (CDLL: GIL released) until seq != seen or timeout_ns elapses; returns seq (ACQUIRE). */
int64_t q3ni_wait(int64_t seen, int64_t timeout_ns) {
    pthread_mutex_lock(&mu);
    if ((int64_t)seq == seen && timeout_ns > 0) {
        struct timespec now, dl;
        clock_gettime(CLOCK_REALTIME, &now);
        int64_t ns = (int64_t)now.tv_nsec + timeout_ns;
        dl.tv_sec = now.tv_sec + (time_t)(ns / 1000000000);
        dl.tv_nsec = (long)(ns % 1000000000);
        while ((int64_t)seq == seen) {
            if (pthread_cond_timedwait(&done_cv, &mu, &dl) == ETIMEDOUT) break;
        }
    }
    pthread_mutex_unlock(&mu);
    return (int64_t)__atomic_load_n(&seq, __ATOMIC_ACQUIRE);
}

/* Copy the gauge under the mutex (PyDLL). */
void q3ni_gauge(int64_t *out) {
    pthread_mutex_lock(&mu);
    for (int i = 0; i < 6; i++) out[i] = gauge[i];
    pthread_mutex_unlock(&mu);
}

/* Wait (CDLL) until the queue is empty and no worker is busy.  0 = quiescent, 1 = timeout. */
int q3ni_quiesce(int64_t timeout_ns) {
    struct timespec now, dl;
    clock_gettime(CLOCK_REALTIME, &now);
    int64_t ns = (int64_t)now.tv_nsec + timeout_ns;
    dl.tv_sec = now.tv_sec + (time_t)(ns / 1000000000);
    dl.tv_nsec = (long)(ns % 1000000000);
    pthread_mutex_lock(&mu);
    int rc = 0;
    while (qlen || busy) {
        if (pthread_cond_timedwait(&done_cv, &mu, &dl) == ETIMEDOUT) { rc = (qlen || busy) ? 1 : 0; break; }
    }
    pthread_mutex_unlock(&mu);
    return rc;
}

/* Stop and join the workers (CDLL), after draining the queue.  0 ok, -1 not running. */
int q3ni_stop(void) {
    pthread_mutex_lock(&mu);
    if (!running) { pthread_mutex_unlock(&mu); return -1; }
    stopping = 1;
    pthread_cond_broadcast(&work_cv);
    pthread_mutex_unlock(&mu);
    for (int i = 0; i < nworkers; i++) pthread_join(threads[i], 0);
    pthread_mutex_lock(&mu);
    running = 0; nworkers = 0; stopping = 0;
    free(ranges); free(queue); ranges = 0; queue = 0;
    res = 0; logbuf = 0; gauge = 0;
    pthread_mutex_unlock(&mu);
    return 0;
}

int32_t q3ni_abi(void) { return 20260924; }
