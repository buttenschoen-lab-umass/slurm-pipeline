"""
SLURM job tracking with automatic cancellation on script termination.
Improved version with better signal handling for Ctrl-C.
"""

import os
import signal
import atexit
import subprocess
import threading
import time
from typing import Dict, Set, Any, Optional
from datetime import datetime


class JobTracker:
    """
    Singleton class to track all submitted SLURM jobs.
    Ensures jobs are cancelled on script exit.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.active_jobs: Set[str] = set()
            self.completed_jobs: Set[str] = set()
            self.cancelled_jobs: Set[str] = set()
            self.failed_cancellations: Set[str] = set()
            self.cancel_on_exit = True
            self.verbose = True
            self._original_handlers = {}
            self._in_signal_handler = False
            self._cleanup_lock = threading.Lock()
            self._cleanup_done = False
            self._setup_handlers()
            self.__class__._initialized = True

    def _setup_handlers(self):
        """Set up signal handlers and exit handlers."""
        # Register cleanup function
        atexit.register(self._cleanup)

        # Store original handlers and set up new ones
        # Use signal.signal() which is more reliable than signal.getsignal()
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                # Store original handler
                original = signal.signal(sig, self._signal_handler)
                self._original_handlers[sig] = original
            except (ValueError, OSError) as e:
                # Some signals might not be available in certain environments
                if self.verbose:
                    print(f"[JobTracker] Warning: Could not set handler for signal {sig}: {e}")

        # Handle SIGHUP if available (Unix-like systems)
        if hasattr(signal, 'SIGHUP'):
            try:
                original = signal.signal(signal.SIGHUP, self._signal_handler)
                self._original_handlers[signal.SIGHUP] = original
            except (ValueError, OSError):
                pass

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        # Prevent recursive signal handling
        if self._in_signal_handler:
            return

        self._in_signal_handler = True

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Received signal {signum} ({self._get_signal_name(signum)})")
            print(f"Initiating graceful shutdown...")
            print(f"{'='*60}")

        # Perform cleanup
        self._cleanup()

        # Reset flag
        self._in_signal_handler = False

        # Exit
        if self.verbose:
            print('Exiting...')

        os._exit(1)

    def _get_signal_name(self, signum):
        """Get human-readable signal name."""
        for name in dir(signal):
            if name.startswith('SIG') and not name.startswith('SIG_'):
                if getattr(signal, name) == signum:
                    return name
        return f"Signal {signum}"

    def add_job(self, job_id: str, job_name: Optional[str] = None):
        """Register a new job."""
        with self._cleanup_lock:
            self.active_jobs.add(job_id)
            if self.verbose:
                name_str = f" ({job_name})" if job_name else ""
                print(f"[JobTracker] Tracking job {job_id}{name_str} - Active jobs: {len(self.active_jobs)}")

    def complete_job(self, job_id: str):
        """Mark a job as completed."""
        with self._cleanup_lock:
            if job_id in self.active_jobs:
                self.active_jobs.remove(job_id)
                self.completed_jobs.add(job_id)
                if self.verbose:
                    print(f"[JobTracker] Job {job_id} completed - Active jobs: {len(self.active_jobs)}")

    def cancel_job(self, job_id: str, reason: str = "Script terminated") -> bool:
        """Cancel a specific job - public API."""
        with self._cleanup_lock:
            if job_id not in self.active_jobs:
                return False

            if self.verbose:
                print(f"[JobTracker] Cancelling job {job_id}: {reason}")

            success = self._cancel_single_job_unlocked(job_id)
            return success

    def cancel_all_active(self) -> Dict[str, int]:
        """Cancel all active jobs - public API."""
        with self._cleanup_lock:
            if not self.active_jobs:
                return {'cancelled': 0, 'failed': 0, 'already_done': 0}

            stats = {'cancelled': 0, 'failed': 0, 'already_done': 0}

            # Copy to avoid modification during iteration
            jobs_to_cancel = list(self.active_jobs)

            if self.verbose:
                print(f"\n[JobTracker] Cancelling {len(jobs_to_cancel)} active jobs...")

            for job_id in jobs_to_cancel:
                if self._cancel_single_job_unlocked(job_id):
                    stats['cancelled'] += 1
                else:
                    stats['failed'] += 1

            return stats

    def _cancel_single_job_unlocked(self, job_id: str) -> bool:
        """
        Cancel a single job. Assumes _cleanup_lock is already held.
        This is the core cancellation logic used by all other methods.
        """
        try:
            # Run scancel - returns 0 even for non-existent jobs
            result = subprocess.run(
                ['scancel', job_id],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Always remove from tracking regardless of result
            self.active_jobs.discard(job_id)
            self.cancelled_jobs.add(job_id)

            if self.verbose:
                print(f"[JobTracker]   Sent cancel signal to job {job_id}")

            return True

        except subprocess.TimeoutExpired:
            # Timeout - still remove from active jobs
            self.active_jobs.discard(job_id)
            self.cancelled_jobs.add(job_id)
            if self.verbose:
                print(f"[JobTracker]   Timeout cancelling job {job_id} (removed from tracking)")
            return False

        except Exception as e:
            self.active_jobs.discard(job_id)
            self.failed_cancellations.add(job_id)
            if self.verbose:
                print(f"[JobTracker]   Error cancelling job {job_id}: {e}")
            return False

    def _cleanup(self):
        """Cancel all active jobs on exit - main cleanup entry point."""
        with self._cleanup_lock:
            # Prevent multiple cleanup calls
            if self._cleanup_done:
                return

            self._cleanup_done = True

            if not self.cancel_on_exit or not self.active_jobs:
                return

            print(f"\n{'='*60}")
            print(f"[JobTracker] Cleanup: Cancelling {len(self.active_jobs)} active SLURM jobs...")
            print(f"{'='*60}")

            cancelled = 0
            failed = 0

            # Copy to avoid modification during iteration
            jobs_to_cancel = list(self.active_jobs)

            # Cancel all jobs while holding the lock
            for job_id in jobs_to_cancel:
                if self._cancel_single_job_unlocked(job_id):
                    cancelled += 1
                else:
                    failed += 1

            print(f"[JobTracker] Cleanup complete:")
            print(f"  - Cancelled: {cancelled}")
            print(f"  - Failed: {failed}")
            print(f"{'='*60}\n")

    def __del__(self):
        """Destructor to ensure cleanup happens."""
        try:
            self._cleanup()
        except:
            pass

    def disable_auto_cancel(self):
        """Disable automatic cancellation on exit."""
        self.cancel_on_exit = False
        if self.verbose:
            print("[JobTracker] Auto-cancellation disabled")

    def enable_auto_cancel(self):
        """Enable automatic cancellation on exit."""
        self.cancel_on_exit = True
        if self.verbose:
            print("[JobTracker] Auto-cancellation enabled")

    def set_verbose(self, verbose: bool):
        """Set verbosity level."""
        self.verbose = verbose

    def get_status(self) -> Dict[str, Any]:
        """Get current tracking status."""
        with self._cleanup_lock:
            return {
                'active_jobs': list(self.active_jobs),
                'completed_jobs': list(self.completed_jobs),
                'cancelled_jobs': list(self.cancelled_jobs),
                'failed_cancellations': list(self.failed_cancellations),
                'cancel_on_exit': self.cancel_on_exit,
                'stats': {
                    'active': len(self.active_jobs),
                    'completed': len(self.completed_jobs),
                    'cancelled': len(self.cancelled_jobs),
                    'failed_cancellations': len(self.failed_cancellations)
                }
            }

    def reset(self):
        """Reset tracker state (useful for testing)."""
        with self._cleanup_lock:
            self.active_jobs.clear()
            self.completed_jobs.clear()
            self.cancelled_jobs.clear()
            self.failed_cancellations.clear()
            self._cleanup_done = False
            if self.verbose:
                print("[JobTracker] State reset")


