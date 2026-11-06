# RecoverOps AI Makefile
# Compatible with GNU Make on Linux, macOS, and Windows (via MSYS2/WSL/PowerShell Make)

.PHONY: run test docker-build docker-run clean help

help:
	@echo "Available commands:"
	@echo "  make run          - Execute the Phase 1 main engine demonstration pipelines"
	@echo "  make test         - Run full unittest suite for Phase 1"
	@echo "  make docker-build - Build the recoverops-ai:phase1 Docker image"
	@echo "  make docker-run   - Run the Phase 1 container"
	@echo "  make clean        - Remove bytecode caches and stray database artifacts"

run:
	python -m app.main

test:
	python -m unittest discover -s tests -v

docker-build:
	docker build -t recoverops-ai:phase1 .

docker-run:
	docker run --rm recoverops-ai:phase1

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').glob('*.db*') if p.is_file()]"

