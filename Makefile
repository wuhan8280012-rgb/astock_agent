PLIST_NAME = com.wuhan.stock-agent-scheduler
PLIST_SRC = $(PLIST_NAME).plist
PLIST_DST = $(HOME)/Library/LaunchAgents/$(PLIST_NAME).plist
LEGACY_LABEL = com.wuhan.stock-agent.scheduler

.PHONY: start stop logs status dashboard test security-scan release-pack

start:
	@mkdir -p logs
	@pkill -f "from scheduler.main_scheduler import start_scheduler; start_scheduler()" 2>/dev/null || true
	@launchctl bootout gui/$$(id -u)/$(PLIST_NAME) 2>/dev/null || true
	@launchctl bootout gui/$$(id -u)/$(LEGACY_LABEL) 2>/dev/null || true
	@cp $(PLIST_SRC) $(PLIST_DST)
	@PY_BIN=$$(which python3); \
		sed -i '' "s|__PYTHON_PATH__|$$PY_BIN|g" $(PLIST_DST)
	@launchctl bootstrap gui/$$(id -u) $(PLIST_DST)
	@launchctl enable gui/$$(id -u)/$(PLIST_NAME)
	@echo "Scheduler started via LaunchAgent ($(PLIST_NAME))"

stop:
	@launchctl bootout gui/$$(id -u)/$(PLIST_NAME) 2>/dev/null || true
	@launchctl bootout gui/$$(id -u)/$(LEGACY_LABEL) 2>/dev/null || true
	@pkill -f "from scheduler.main_scheduler import start_scheduler; start_scheduler()" 2>/dev/null || true
	@echo "Scheduler stopped"

logs:
	@mkdir -p logs
	@touch logs/scheduler_stdout.log logs/scheduler_stderr.log
	@tail -f logs/scheduler_stdout.log logs/scheduler_stderr.log

status:
	@launchctl print gui/$$(id -u)/$(PLIST_NAME) 2>/dev/null | head -60 || echo "LaunchAgent not loaded"

dashboard:
	@streamlit run dashboard/app.py --server.port 8501 --server.address localhost --server.fileWatcherType none

test:
	@pytest -q

security-scan:
	@./scripts/secret_scan.sh

release-pack:
	@./scripts/package_release.sh
