"""
SLURM job monitoring with real-time progress tracking.
Simplified version: One job = one ensemble, no chunk tracking needed.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from tqdm.auto import tqdm

from .job import JobState, JobInfo


class SlurmMonitor:
    """Monitor SLURM jobs with progress tracking."""

    def __init__(self, check_interval: int = 10, job_tracker=None):
        """
        Initialize SLURM monitor.

        Args:
            check_interval: Seconds between status checks
            job_tracker: Optional JobTracker instance for automatic tracking
        """
        self.check_interval = check_interval
        self.job_tracker = job_tracker

    def monitor_job(self,
                   job_id: str,
                   job_name: Optional[str] = None,
                   output_dir: Optional[str] = None,
                   show_progress: bool = True,
                   callback: Optional[callable] = None) -> JobInfo:
        """
        Monitor a single job until completion.

        Args:
            job_id: SLURM job ID
            job_name: Job name for display
            output_dir: Directory to check for completion marker
            show_progress: Show tqdm progress bar
            callback: Function to call on each update with JobInfo

        Returns:
            Final JobInfo
        """
        if show_progress:
            pbar = tqdm(total=1,
                       desc=f"{job_name or job_id}",
                       bar_format='{desc}: {elapsed} [{postfix}]')
        else:
            pbar = None

        try:
            while True:
                # Get job info
                job_info = self.get_job_info(job_id)
                if job_name:
                    job_info.job_name = job_name

                # Update progress bar
                if pbar:
                    if job_info.state == JobState.PENDING:
                        pbar.set_postfix_str("Pending in queue")
                    elif job_info.state == JobState.RUNNING:
                        elapsed = job_info.elapsed_time or timedelta(0)
                        status = f"Running - {self._format_time(elapsed)}"
                        if job_info.nodes > 0:
                            status += f" on {job_info.nodes} nodes"
                        pbar.set_postfix_str(status)
                    elif job_info.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
                        pbar.n = 1
                        pbar.refresh()
                        status_str = f"{job_info.state.value}"
                        if job_info.exit_code and job_info.exit_code != "0:0":
                            status_str += f" (exit: {job_info.exit_code})"
                        pbar.set_postfix_str(status_str)

                # Call callback if provided
                if callback:
                    callback(job_info)

                # Check if job is done
                if job_info.state in [JobState.COMPLETED, JobState.FAILED,
                                     JobState.CANCELLED, JobState.TIMEOUT]:
                    # Notify job tracker if available
                    if self.job_tracker:
                        self.job_tracker.complete_job(job_id)
                    break

                time.sleep(self.check_interval)

        finally:
            if pbar:
                pbar.close()

        return job_info

    def monitor_ensemble(self, submission_result: Dict[str, Any],
                        show_progress: bool = True) -> JobInfo:
        """
        Monitor an ensemble submission.

        Simplified: Just monitor the single job, no chunks.

        Args:
            submission_result: Result from SlurmPipeline.submit_ensemble()
            show_progress: Show progress bar

        Returns:
            Final JobInfo
        """
        return self.monitor_job(
            job_id=submission_result['job_id'],
            job_name=submission_result['job_name'],
            output_dir=submission_result.get('output_dir'),
            show_progress=show_progress
        )

    def monitor_scan(self, submission_result: Dict[str, Any],
                    show_progress: bool = True) -> JobInfo:
        """
        Monitor a parameter scan submission.

        Simplified: Monitor array job completion by counting completed parameter points.

        Args:
            submission_result: Result from SlurmPipeline.submit_parameter_scan()
            show_progress: Show progress bar

        Returns:
            Final JobInfo
        """
        job_id = submission_result['job_id']
        job_name = submission_result['job_name']
        output_dir = submission_result.get('output_dir')
        n_points = submission_result.get('n_points', 0)

        if show_progress and n_points > 0:
            pbar = tqdm(total=n_points,
                       desc=f"{job_name or job_id}",
                       unit="params")
        else:
            pbar = None

        last_completed = 0

        try:
            while True:
                # Get job info
                job_info = self.get_job_info(job_id)
                if job_name:
                    job_info.job_name = job_name

                # Count completed parameter points
                if output_dir and n_points > 0:
                    completed = self._count_completed_points(output_dir)
                    if completed > last_completed and pbar:
                        pbar.update(completed - last_completed)
                        last_completed = completed

                # Update progress bar status
                if pbar:
                    if job_info.state == JobState.PENDING:
                        pbar.set_postfix_str("Pending in queue")
                    elif job_info.state == JobState.RUNNING:
                        pbar.set_postfix_str(f"Running on {job_info.nodes} nodes")
                    elif job_info.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
                        if pbar.n < pbar.total:
                            pbar.n = pbar.total
                            pbar.refresh()
                        status_str = f"{job_info.state.value}"
                        if job_info.exit_code and job_info.exit_code != "0:0":
                            status_str += f" (exit: {job_info.exit_code})"
                        pbar.set_postfix_str(status_str)

                # Check if job is done
                if job_info.state in [JobState.COMPLETED, JobState.FAILED,
                                     JobState.CANCELLED, JobState.TIMEOUT]:
                    # Notify job tracker if available
                    if self.job_tracker:
                        self.job_tracker.complete_job(job_id)
                    break

                time.sleep(self.check_interval)

        finally:
            if pbar:
                pbar.close()

        return job_info

    def get_job_info(self, job_id: str) -> JobInfo:
        """Get current information about a job."""
        # Try squeue first for active jobs
        squeue_cmd = [
            'squeue', '-j', job_id,
            '--format=%T,%r,%S,%L,%D,%F',
            '--noheader'
        ]
        result = subprocess.run(squeue_cmd, capture_output=True, text=True)

        if result.stdout.strip():
            parts = result.stdout.strip().split(',')
            state = self._parse_state(parts[0])

            # Parse times
            start_time = None
            elapsed_time = None
            time_limit = None

            if len(parts) > 2 and parts[2] and parts[2] != 'N/A':
                try:
                    start_time = datetime.strptime(parts[2], "%Y-%m-%dT%H:%M:%S")
                    elapsed_time = datetime.now() - start_time
                except:
                    pass

            if len(parts) > 3 and parts[3] and parts[3] != 'N/A':
                time_limit = self._parse_time_limit(parts[3])

            # Check for array job
            array_size = None
            if len(parts) > 5 and '_' in parts[5]:
                # Array job - count tasks
                array_size = self._count_array_tasks(job_id)

            return JobInfo(
                job_id=job_id,
                job_name="",
                state=state,
                reason=parts[1] if len(parts) > 1 else None,
                start_time=start_time,
                elapsed_time=elapsed_time,
                time_limit=time_limit,
                nodes=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0,
                array_size=array_size
            )

        # If not in squeue, check sacct for completed jobs
        sacct_cmd = [
            'sacct', '-j', job_id,
            '--format=State,ExitCode,Start,Elapsed,JobName',
            '--noheader', '-X'
        ]
        result = subprocess.run(sacct_cmd, capture_output=True, text=True)

        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            parts = lines[0].strip().split()

            if len(parts) >= 2:
                state = self._parse_state(parts[0])
                exit_code = parts[1] if len(parts) > 1 else None

                # Parse times from sacct
                start_time = None
                elapsed_time = None
                if len(parts) > 2 and parts[2] != 'Unknown':
                    try:
                        start_time = datetime.strptime(parts[2], "%Y-%m-%dT%H:%M:%S")
                    except:
                        pass

                if len(parts) > 3:
                    elapsed_time = self._parse_elapsed(parts[3])

                job_name = parts[4] if len(parts) > 4 else ""

                return JobInfo(
                    job_id=job_id,
                    job_name=job_name,
                    state=state,
                    exit_code=exit_code,
                    start_time=start_time,
                    elapsed_time=elapsed_time
                )

        return JobInfo(job_id=job_id, job_name="", state=JobState.UNKNOWN)

    def _parse_state(self, state_str: str) -> JobState:
        """Parse SLURM state string to JobState enum."""
        state_map = {
            'PENDING': JobState.PENDING,
            'RUNNING': JobState.RUNNING,
            'COMPLETED': JobState.COMPLETED,
            'FAILED': JobState.FAILED,
            'CANCELLED': JobState.CANCELLED,
            'TIMEOUT': JobState.TIMEOUT,
        }
        return state_map.get(state_str.upper(), JobState.UNKNOWN)

    def _parse_time_limit(self, time_str: str) -> timedelta:
        """Parse SLURM time limit string (D-HH:MM:SS or HH:MM:SS)."""
        if '-' in time_str:
            days, time_part = time_str.split('-')
            days = int(days)
        else:
            days = 0
            time_part = time_str

        parts = time_part.split(':')
        hours = int(parts[0]) if len(parts) > 0 else 0
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0

        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    def _parse_elapsed(self, elapsed_str: str) -> timedelta:
        """Parse SLURM elapsed time string."""
        return self._parse_time_limit(elapsed_str)

    def _format_time(self, td: timedelta) -> str:
        """Format timedelta for display."""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _count_completed_points(self, output_dir: str) -> int:
        """Count completed parameter points by looking for completion markers."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return 0

        # Count subdirectories with completion markers
        count = 0
        for subdir in output_path.iterdir():
            if subdir.is_dir():
                marker = subdir / "completed.marker"
                if marker.exists():
                    count += 1

        return count

    def _count_array_tasks(self, job_id: str) -> Optional[int]:
        """Count array tasks for an array job."""
        # For our simplified model, we can get this from squeue
        cmd = ['squeue', '-j', job_id, '--format=%F', '--noheader']
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout.strip():
            # Count unique array task IDs
            task_ids = set()
            for line in result.stdout.strip().split('\n'):
                if '_' in line:
                    task_ids.add(line.split('_')[1])
            return len(task_ids)

        return None

    def print_job_summary(self, job_info: JobInfo):
        """Print a nice summary of job information."""
        print(f"\n{'='*50}")
        print(f"Job ID: {job_info.job_id}")
        if job_info.job_name:
            print(f"Job Name: {job_info.job_name}")
        print(f"State: {job_info.state.value}")

        if job_info.array_size:
            print(f"Array Size: {job_info.array_size}")

        if job_info.start_time:
            print(f"Start Time: {job_info.start_time}")
        if job_info.elapsed_time:
            print(f"Elapsed: {self._format_time(job_info.elapsed_time)}")
        if job_info.time_limit:
            print(f"Time Limit: {self._format_time(job_info.time_limit)}")

        if job_info.nodes > 0:
            print(f"Nodes: {job_info.nodes}")

        if job_info.exit_code and job_info.exit_code != "0:0":
            print(f"Exit Code: {job_info.exit_code}")
        if job_info.reason:
            print(f"Reason: {job_info.reason}")

        print(f"{'='*50}\n")
