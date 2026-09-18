.PHONY: all clean dist check install help upgrade-deps outdated

VERSION ?= 0.5.3
TYPEOUT = typeout
TYPEOUT_CPU = typeout-cpu.py
TYPEOUT_GPU = typeout-gpu.py

all: $(TYPEOUT)

$(TYPEOUT): build.sh $(TYPEOUT_CPU) $(TYPEOUT_GPU)
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

# Re-resolve both scripts against the latest releases and raise the >= floors
# in their inline metadata (like `uv lock --upgrade`, but for lockless scripts).
upgrade-deps:
	uv run upgrade-deps.py $(TYPEOUT_CPU) $(TYPEOUT_GPU)
	@$(MAKE) --no-print-directory all

outdated:
	uv run upgrade-deps.py --dry-run $(TYPEOUT_CPU) $(TYPEOUT_GPU)

help:
	@echo "make          - Build $(TYPEOUT) from Python scripts"
	@echo "make clean    - Remove built $(TYPEOUT)"
	@echo "make dist     - Clean build"
	@echo "make check    - Run syntax checks"
	@echo "make install  - Install to ~/.local/bin/"
	@echo "make outdated - Show which dependency floors would be raised"
	@echo "make upgrade-deps - Raise dependency floors to latest and rebuild"
