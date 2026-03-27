.PHONY: all clean dist check install help

VERSION ?= 0.4.2
TYPEOUT = typeout
TYPEOUT_CPU = typeout-cpu.py
TYPEOUT_GPU = typeout-gpu.py

all: $(TYPEOUT)

$(TYPEOUT): $(TYPEOUT_CPU) $(TYPEOUT_GPU)
	VERSION=$(VERSION) ./build.sh

clean:
	rm -f $(TYPEOUT)
	rm -rf __pycache__/

dist: clean all

check: $(TYPEOUT)
	@echo "Running syntax check..."
	@shellcheck $(TYPEOUT) 2>/dev/null || echo "shellcheck not installed, skipping"
	@echo "Checking Python scripts..."
	@python3 -m py_compile $(TYPEOUT_CPU) $(TYPEOUT_GPU)
	@echo "OK"

install: $(TYPEOUT)
	@echo "Installing to ~/.local/bin/"
	@mkdir -p ~/.local/bin
	@cp $(TYPEOUT) ~/.local/bin/
	@echo "Installed. Run 'typeout --check' to verify."

help:
	@echo "make          - Build $(TYPEOUT) from Python scripts"
	@echo "make clean    - Remove built $(TYPEOUT)"
	@echo "make dist     - Clean build"
	@echo "make check    - Run syntax checks"
	@echo "make install  - Install to ~/.local/bin/"
