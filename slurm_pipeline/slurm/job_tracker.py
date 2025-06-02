"""
SLURM job tracking with automatic cancellation on script termination.
"""

import os
import signal
import atexit
import subprocess
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
            self._setup_handlers()
            self.__class__._initialized = True

    def _setup_handlers(self):
        """Set up signal handlers and exit handlers."""
        # Register cleanup function
        atexit.register(self._cleanup)

        # Store original handlers and set up new ones
        for sig in [signal.SIGINT, signal.SIGTERM]:
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)

        # Handle SIGHUP if available (Unix-like systems)
        if hasattr(signal, 'SIGHUP'):
            self._original_handlers[signal.SIGHUP] = signal.signal(
                signal.SIGHUP, self._signal_handler
            )

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Received signal {signum}. Initiating graceful shutdown...")
            print(f"{'='*60}")

        self._cleanup()

        # Restore original handler and re-raise
        if signum in self._original_handlers:
            signal.signal(signum, self._original_handlers[signum])
        else:
            signal.signal(signum, signal.SIG_DFL)

        os.kill(os.getpid(), signum)

    def add_job(self, job_id: str, job_name: Optional[str] = None):
        """Register a new job."""
        self.active_jobs.add(job_id)
        if self.verbose:
            name_str = f" ({job_name})" if job_name else ""
            print(f"[JobTracker] Tracking job {job_id}{name_str} - Active jobs: {len(self.active_jobs)}")

    def complete_job(self, job_id: str):
        """Mark a job as completed."""
        if job_id in self.active_jobs:
            self.active_jobs.remove(job_id)
            self.completed_jobs.add(job_id)
            if self.verbose:
                print(f"[JobTracker] Job {job_id} completed - Active jobs: {len(self.active_jobs)}")

    def cancel_job(self, job_id: str, reason: str = "Script terminated") -> bool:
        """Cancel a specific job."""
        if job_id not in self.active_jobs:
            return False

        if self.verbose:
            print(f"[JobTracker] Cancelling job {job_id}: {reason}")

        try:
            # First check if job is still active
            check_result = subprocess.run(
                ['squeue', '-j', job_id, '--noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if not check_result.stdout.strip():
                # Job not in queue, might have completed
                self.active_jobs.remove(job_id)
                self.completed_jobs.add(job_id)
                if self.verbose:
                    print(f"[JobTracker]   Job {job_id} already completed")
                return True

            # Job is active, cancel it
            cancel_result = subprocess.run(
                ['scancel', job_id],
                capture_output=True,
                text=True,
                timeout=10
            )

            if cancel_result.returncode == 0:
                self.active_jobs.remove(job_id)
                self.cancelled_jobs.add(job_id)
                if self.verbose:
                    print(f"[JobTracker]   Successfully cancelled job {job_id}")
                return True
            else:
                self.failed_cancellations.add(job_id)
                if self.verbose:
                    print(f"[JobTracker]   Failed to cancel job {job_id}: {cancel_result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.failed_cancellations.add(job_id)
            if self.verbose:
                print(f"[JobTracker]   Timeout while cancelling job {job_id}")
            return False
        except Exception as e:
            self.failed_cancellations.add(job_id)
            if self.verbose:
                print(f"[JobTracker]   Error cancelling job {job_id}: {e}")
            return False

    def cancel_all_active(self) -> Dict[str, int]:
        """Cancel all active jobs and return statistics."""
        if not self.active_jobs:
            return {'cancelled': 0, 'failed': 0, 'already_done': 0}

        stats = {'cancelled': 0, 'failed': 0, 'already_done': 0}

        # Copy to avoid modification during iteration
        jobs_to_cancel = list(self.active_jobs)

        if self.verbose:
            print(f"\n[JobTracker] Cancelling {len(jobs_to_cancel)} active jobs...")

        for job_id in jobs_to_cancel:
            result = self.cancel_job(job_id, "Bulk cancellation")
            if job_id in self.cancelled_jobs:
                stats['cancelled'] += 1
            elif job_id in self.completed_jobs:
                stats['already_done'] += 1
            else:
                stats['failed'] += 1

        return stats

    def _cleanup(self):
        """Cancel all active jobs on exit."""
        if not self.cancel_on_exit or not self.active_jobs:
            return

        print(f"\n{'='*60}")
        print(f"[JobTracker] Cleanup: Cancelling {len(self.active_jobs)} active SLURM jobs...")
        print(f"{'='*60}")

        stats = self.cancel_all_active()

        print(f"[JobTracker] Cleanup complete:")
        print(f"  - Cancelled: {stats['cancelled']}")
        print(f"  - Already done: {stats['already_done']}")
        print(f"  - Failed: {stats['failed']}")
        print(f"{'='*60}\n")

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
        self.active_jobs.clear()
        self.completed_jobs.clear()
        self.cancelled_jobs.clear()
        self.failed_cancellations.clear()
        if self.verbose:
            print("[JobTracker] State reset")
