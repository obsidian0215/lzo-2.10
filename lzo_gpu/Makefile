CC ?= gcc
CFLAGS ?= -O2 -Wall
LDFLAGS ?= -lOpenCL

## Automatically discover .cl sources present in this directory so we don't
## attempt to precompile kernels that aren't actually in the tree.
CL_SOURCES := $(wildcard *.cl)

all: build_kernel lzo_gpu precompile-combos

build_kernel: build_kernel.c
	$(CC) $(CFLAGS) -o build_kernel build_kernel.c $(LDFLAGS)


# build the GPU host CLI
lzo_gpu: lzo_host.c
	$(CC) $(CFLAGS) -o lzo_gpu lzo_host.c $(LDFLAGS)

precompile: build_kernel
	# If KERNEL is specified, build only that kernel (basename without .cl)
	# Usage: make precompile KERNEL=lzo1x_1
	if [ -n "$(KERNEL)" ]; then \
		src="$(KERNEL).cl"; \
		if [ ! -f "$$src" ]; then echo "Kernel source $$src not found"; exit 1; fi; \
		./build_kernel $$src $${src%%.cl}.bin; \
	else \
		$(MAKE) precompile-combos; \
	fi


precompile-all:
	# precompile each kernel source into a same-named .bin (e.g. lzo1x_1.cl -> lzo1x_1.bin)
	@if [ -z "$(CL_SOURCES)" ]; then \
		echo "No .cl kernel sources found, skipping precompile-all"; \
	else \
		for f in $(CL_SOURCES); do \
			bn=$${f%%.cl}.bin; \
			echo "Building $$f -> $$bn"; \
			./build_kernel $$f $$bn || exit 1; \
		done; \
	fi

# Build combinations of core kernels and frontend strategies into precompiled binaries.
# For each core (like lzo1x_1, lzo1x_1k, ...) and each frontend wrapper (atomic/delayed/usehost/...)
# create a temporary directory, copy the frontend there and copy the chosen core as
# `lzo1x_1.cl` so that the frontend's `#include "lzo1x_1.cl"` picks up the selected core.
# Output binaries are named `<core>_<frontend>.bin` (both basenames without .cl).
precompile-combos: build_kernel
	@./tools/precompile_combos.sh

.PHONY: help check-opencl testdata

help:
	@echo "Makefile targets for lzo_gpu:";
	@echo "  make           -> build helper, host and precompile all kernels";
	@echo "  make lzo_gpu    -> build the GPU host binary (lzo_gpu)";
	@echo "  make build_kernel -> build the kernel precompiler helper";
	@echo "  make precompile KERNEL=<name> -> precompile one kernel (basename, no .cl)";
	@echo "  make precompile-all -> precompile all kernel sources into .bin files";
	@echo "  make check-opencl -> quick compile/link check for OpenCL headers and library";
	@echo "  make testdata [OUTDIR=../samples] -> generate test data suite (calls generate-test-data.py --suite)";


check-opencl:
	@echo "Checking OpenCL installation..."
	@set -e; \
	if pkg-config --exists OpenCL 2>/dev/null; then \
		echo "OpenCL found via pkg-config:"; \
		pkg-config --modversion OpenCL || true; \
		echo "CFLAGS: $$(pkg-config --cflags OpenCL 2>/dev/null)"; \
		echo "LIBS : $$(pkg-config --libs OpenCL 2>/dev/null)"; \
	else \
		echo "OpenCL not found via pkg-config"; \
	fi; \
	if [ -f /usr/include/CL/cl.h ]; then \
		echo "OpenCL headers found at /usr/include/CL/"; \
	elif [ -f /opt/AMDAPP/include/CL/cl.h ]; then \
		echo "OpenCL headers found at /opt/AMDAPP/include/"; \
	else \
		echo "OpenCL headers not found in common locations"; \
		echo "Please install OpenCL development headers:"; \
		echo "  Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev"; \
		echo "  CentOS/RHEL: sudo yum install opencl-headers ocl-icd-devel"; \
	fi; \
	if ldconfig -p 2>/dev/null | grep -q libOpenCL; then \
		echo "OpenCL library found in ldconfig"; \
	else \
		echo "OpenCL library not found in ldconfig"; \
		echo "If you have an OpenCL ICD installed, run 'ldconfig' or ensure library path is configured."; \
	fi

# generate test data suite. Default OUTDIR=../samples so generated data lives at repo root samples/
testdata:
	@OUTDIR=$${OUTDIR:-../samples}; \
	python3 generate-test-data.py --suite --out-dir "$$OUTDIR" || (echo "failed to generate test data"; exit 1)

clean:
	rm -f build_kernel lzo_gpu *.bin

.PHONY: all build_kernel precompile precompile-all clean
