CC ?= gcc
CFLAGS ?= -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200
LDFLAGS ?= -lOpenCL -lm

## Automatically discover .cl sources present in this directory so we don't
## attempt to precompile kernels that aren't actually in the tree.
CL_SOURCES := $(wildcard *.cl)

all: build_kernel lzo_gpu lzo_gpu_daemon lzo_gpu_client precompile-combos

build_kernel: build_kernel.c lzo_gpu_utils.c
	$(CC) $(CFLAGS) -o build_kernel build_kernel.c lzo_gpu_utils.c $(LDFLAGS)


# build the GPU host CLI (standalone renamed source)
lzo_gpu: lzo_gpu_standalone.c lzo_gpu_utils.c
	$(CC) $(CFLAGS) -o lzo_gpu lzo_gpu_standalone.c lzo_gpu_utils.c $(LDFLAGS)

# build the GPU daemon
lzo_gpu_daemon: lzo_gpu_daemon.c daemon_compress.c daemon_decompress.c lzo_gpu_utils.c
	$(CC) $(CFLAGS) -o lzo_gpu_daemon lzo_gpu_daemon.c daemon_compress.c daemon_decompress.c lzo_gpu_utils.c $(LDFLAGS) -lpthread

# build the GPU client
lzo_gpu_client: lzo_gpu_client.c
	$(CC) $(CFLAGS) -o lzo_gpu_client lzo_gpu_client.c

precompile: build_kernel
# If KERNEL is specified, build only that kernel (basename without .cl)
	# Usage: make precompile KERNEL=lzo1x_12
	if [ -n "$(KERNEL)" ]; then \
		src="$(KERNEL).cl"; \
		if [ ! -f "$$src" ]; then echo "Kernel source $$src not found"; exit 1; fi; \
		./build_kernel $$src $${src%%.cl}.bin; \
	else \
		$(MAKE) precompile-combos; \
	fi


precompile-combos: build_kernel
# Generate binaries for all algorithms and dictionary sizes
	@for alg in lzo1x lzo1y; do \
		for bits in 10 11 12 14; do \
			echo "Building $${alg}_$${bits}.clbin from $${alg}.cl (D_BITS=$${bits})..."; \
			./build_kernel $${alg}.cl $${alg}_$${bits}.clbin "-I. -D D_BITS=$${bits}" || exit 1; \
		done; \
	done
# Also build decompression kernels
	@for alg in lzo1x lzo1y; do \
		echo "Building $${alg}_decomp.clbin..."; \
		./build_kernel $${alg}_decomp.cl $${alg}_decomp.clbin "-I." || exit 1; \
	done

precompile-all: precompile-combos

.PHONY: help check-opencl testdata

help:
	@echo "Makefile targets for lzo_gpu:";
	@echo "  make           -> build helper, host and precompile all kernels";
	@echo "  make lzo_gpu    -> build the GPU host binary (lzo_gpu)";
	@echo "  make build_kernel -> build the kernel precompiler helper";
	@echo "  make precompile KERNEL=<name> -> precompile one kernel (basename, no .cl)";
	@echo "  make precompile-all -> precompile all kernel sources into .bin files";
	@echo "  make check-opencl -> quick compile/link check for OpenCL headers and library";
	@echo "  make testdata [OUTDIR=../samples] -> generate test data suite (calls ../tools/generate-test-data.py --suite)";


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
	python3 ../tools/generate-test-data.py --suite --out-dir "$$OUTDIR" || (echo "failed to generate test data"; exit 1)

clean:
	rm -f build_kernel lzo_gpu lzo_gpu_daemon lzo_gpu_client *.bin *.clbin

.PHONY: all build_kernel lzo_gpu lzo_gpu_daemon lzo_gpu_client precompile precompile-all clean
