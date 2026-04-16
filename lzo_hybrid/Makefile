ifeq ($(OS),Windows_NT)
MSYS2_USR_BIN ?= C:/msys64/usr/bin
ifneq ($(wildcard $(MSYS2_USR_BIN)/sh.exe),)
SHELL := $(MSYS2_USR_BIN)/sh.exe
export PATH := $(MSYS2_USR_BIN);$(PATH)
endif
endif

CC ?= gcc

ifeq ($(OS),Windows_NT)
EXEEXT ?=.exe
PTHREAD_LIB ?= -lpthread
OPENCL_INCLUDE_DIR ?=
OPENCL_LIB_DIR ?=
OPENCL_LIB_NAME ?= OpenCL
OPENCL_CFLAGS := $(if $(OPENCL_INCLUDE_DIR),-I$(OPENCL_INCLUDE_DIR),)
OPENCL_LDFLAGS := $(if $(OPENCL_LIB_DIR),-L$(OPENCL_LIB_DIR),) -l$(OPENCL_LIB_NAME)
CFLAGS ?= -std=gnu11 -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -DNDEBUG -DWIN32_LEAN_AND_MEAN $(OPENCL_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB)
else
EXEEXT ?=
PTHREAD_CFLAGS ?= -pthread
PTHREAD_LIB ?= -pthread
OPENCL_CFLAGS ?=
OPENCL_LDFLAGS ?=-lOpenCL
CFLAGS ?= -std=gnu11 -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -DNDEBUG $(OPENCL_CFLAGS) $(PTHREAD_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB)
endif

CPU_DIR = ../lzo_cpu
GPU_DIR = ../lzo_gpu

CPPFLAGS = -I$(CPU_DIR) -I$(CPU_DIR)/src -I$(CPU_DIR)/include -I$(CPU_DIR)/include/lzo \
           -I$(GPU_DIR)

CPU_SRCS = $(CPU_DIR)/lzo1x_1.c $(CPU_DIR)/lzo1x_1k.c $(CPU_DIR)/lzo1x_1l.c \
           $(CPU_DIR)/lzo1x_1o.c $(CPU_DIR)/lzo1x_d1.c $(CPU_DIR)/lzo1x_d2.c \
           $(CPU_DIR)/lzo1y_1.c $(CPU_DIR)/lzo1y_d1.c $(CPU_DIR)/lzo1y_d2.c \
           $(CPU_DIR)/src/lzo_init.c $(CPU_DIR)/src/lzo_ptr.c \
           $(CPU_DIR)/src/lzo_str.c $(CPU_DIR)/src/lzo_util.c $(CPU_DIR)/src/lzo_crc.c

GPU_UTILS_SRC = $(GPU_DIR)/lzo_gpu_utils.c

HYBRID_SRCS = lzo_hybrid.c lzo_hybrid_core.c

all: symlinks lzo_hybrid

lzo_hybrid: $(HYBRID_SRCS) $(GPU_UTILS_SRC) $(CPU_SRCS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@$(EXEEXT) $(HYBRID_SRCS) $(GPU_UTILS_SRC) $(CPU_SRCS) $(LDFLAGS)

symlinks:
	@for f in $(GPU_DIR)/*.cl $(GPU_DIR)/*.clbin; do \
		[ -e "$$f" ] || continue; \
		bn=$$(basename "$$f"); \
		if [ ! -e "$$bn" ]; then ln -sf "$$f" "$$bn"; fi; \
	done

clean:
	rm -f lzo_hybrid lzo_hybrid.exe *.o

distclean: clean
	rm -f *.cl *.clbin

.PHONY: all clean distclean symlinks
