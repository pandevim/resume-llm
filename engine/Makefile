# Build a static x86_64 binary with the GuppyLM weights embedded as a
# data section. Output is engine/guppy — a self-contained ELF that the
# polyglot assembler then merges with a PDF wrapper.
#
# Requires: musl-gcc (preferred) OR gcc with glibc static. We favor musl
# because static glibc has dlopen() / NSS quirks; musl is genuinely
# self-contained.

# Prefer musl for a truly self-contained static binary; fall back to gcc
# with -static if musl-gcc isn't installed. (Note: make has a builtin
# CC=cc, so a plain `?=` doesn't pick up musl-gcc — we test explicitly.)
ifeq ($(origin CC),default)
    ifneq (,$(shell command -v musl-gcc 2>/dev/null))
        CC := musl-gcc
    else
        CC := gcc
    endif
endif

CFLAGS      ?= -O2 -std=c11 -Wall -Wextra -ffast-math -fno-stack-protector
LDFLAGS     ?= -static
LIBS        ?= -lm

WEIGHTS_BIN ?= ../model/weights.bin
OUT         ?= guppy

.PHONY: all clean check

all: $(OUT)

# Wrap the weights into an ELF object so they appear as the symbols
# _binary_weights_bin_{start,end} that inference.c references.
weights.o: $(WEIGHTS_BIN)
	cp $(WEIGHTS_BIN) weights.bin
	objcopy --input-target=binary \
	        --output-target=elf64-x86-64 \
	        --binary-architecture=i386:x86-64 \
	        --rename-section .data=.rodata,alloc,load,readonly,data,contents \
	        weights.bin weights.o
	rm -f weights.bin

inference.o: inference.c bytelevel.h
	$(CC) $(CFLAGS) -c inference.c -o inference.o

$(OUT): inference.o weights.o
	$(CC) $(LDFLAGS) -o $(OUT) inference.o weights.o $(LIBS)
	strip --strip-unneeded $(OUT) || true

check:
	@command -v musl-gcc >/dev/null 2>&1 || { \
	  echo "musl-gcc not found. Install with:"; \
	  echo "  Debian/Ubuntu:  sudo apt install musl-tools"; \
	  echo "  Arch:           sudo pacman -S musl"; \
	  exit 1; }
	@command -v objcopy >/dev/null 2>&1 || { echo "objcopy (binutils) not found"; exit 1; }
	@echo "toolchain ok"

clean:
	rm -f inference.o weights.o weights.bin $(OUT)
