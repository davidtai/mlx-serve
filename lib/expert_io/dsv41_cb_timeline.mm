// PROFILE builds only (see dsv41_cb_timeline.h). MLX exposes no hook on its command buffers' commit, so -commit is
// overridden on the concrete MTLCommandBuffer class MLX's queue hands out (class_replaceMethod on that class: a
// method it inherits is added as an override, the inherited one kept as the original). The override records a
// buffer of MLX's queue while a tag is set, adds a completed handler that reads GPUStartTime / GPUEndTime, then
// calls the original -commit. Non-ARC Objective-C++.
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <atomic>
#include <time.h>

#include "dsv41_cb_timeline.h"

typedef void (*CommitImp)(id, SEL);

static CommitImp g_orig = NULL;
static Class g_cls = Nil;
static id g_queue = nil;
static Dsv41tlRow *g_rows = NULL;
static uint32_t g_cap = 0;
static std::atomic<uint32_t> g_committed{0};
static std::atomic<uint32_t> g_completed{0};
static std::atomic<uint32_t> g_dropped{0};
static std::atomic<uint64_t> g_tag{0};

uint64_t dsv41tl_now(void) {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static void tl_commit(id self, SEL sel) {
  uint64_t tag = g_tag.load(std::memory_order_relaxed);
  if (tag != 0 && [(id<MTLCommandBuffer>)self commandQueue] == g_queue) {
    uint32_t i = g_committed.fetch_add(1);
    if (i < g_cap) {
      Dsv41tlRow *r = &g_rows[i];
      r->tag = tag;
      r->host_commit = dsv41tl_now();
      [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        r->gpu_start = (uint64_t)(cb.GPUStartTime * 1e9);
        r->gpu_end = (uint64_t)(cb.GPUEndTime * 1e9);
        r->status = (uint64_t)cb.status;
        r->host_done = dsv41tl_now();
        g_completed.fetch_add(1, std::memory_order_release);
      }];
    } else {
      g_dropped.fetch_add(1);
    }
  }
  g_orig(self, sel);
}

int dsv41tl_install(void *cmdbuf, Dsv41tlRow *rows, uint32_t cap) {
  if (cmdbuf == NULL || rows == NULL || cap == 0) return -1;
  id buf = (id)cmdbuf;
  Class cls = object_getClass(buf);
  if (g_cls == Nil) {
    Method m = class_getInstanceMethod(cls, @selector(commit));
    if (m == NULL) return -2;
    g_orig = (CommitImp)method_getImplementation(m);
    class_replaceMethod(cls, @selector(commit), (IMP)tl_commit, method_getTypeEncoding(m));
    g_cls = cls;
  } else if (cls != g_cls) {
    return -2;
  }
  g_tag.store(0);
  g_queue = [(id<MTLCommandBuffer>)buf commandQueue];
  g_rows = rows;
  g_cap = cap;
  g_committed.store(0);
  g_completed.store(0);
  g_dropped.store(0);
  return 0;
}

void dsv41tl_tag(uint64_t tag) {
  g_tag.store(tag, std::memory_order_relaxed);
}

void dsv41tl_counts(uint32_t out[3]) {
  out[0] = g_committed.load();
  out[1] = g_completed.load(std::memory_order_acquire);
  out[2] = g_dropped.load();
}
