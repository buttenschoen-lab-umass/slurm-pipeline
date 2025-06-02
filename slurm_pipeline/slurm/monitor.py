"""
SLURM job monitoring with real-time progress tracking.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

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
        self.active_monitors = {}
        self.job_tracker = job_tracker

    def monitor_job(self,
                   job_id: str,
                   job_name: Optional[str] = None,
                   output_dir: Optional[str] = None,
                   expected_markers: Optional[int] = None,
                   show_progress: bool = True,
                   callback: Optional[callable] = None) -> JobInfo:
        """
        Monitor a single job until completion.

        Args:
            job_id: SLURM job ID
            job_name: Job name for display
            output_dir: Directory to monitor for completion markers
            expected_markers: Expected number of completion markers (for array jobs)
            show_progress: Show tqdm progress bar
            callback: Function to call on each update with JobInfo

        Returns:
            Final JobInfo
        """
        if show_progress:
            if expected_markers:
                pbar = tqdm(total=expected_markers,
                          desc=f"{job_name or job_id}",
                          unit="tasks")
            else:
                pbar = tqdm(total=1,
                          desc=f"{job_name or job_id}",
                          bar_format='{desc}: {elapsed} [{postfix}]')
        else:
            pbar = None

        last_marker_count = 0

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
                        if output_dir and expected_markers:
                            # Count completion markers
                            marker_count = self._count_markers(output_dir, expected_markers)
                            if marker_count > last_marker_count:
                                pbar.update(marker_count - last_marker_count)
                                last_marker_count = marker_count
                            pbar.set_postfix_str(f"Running on {job_info.nodes} nodes")
                        else:
                            elapsed = job_info.elapsed_time or timedelta(0)
                            pbar.set_postfix_str(f"Running - {self._format_time(elapsed)}")
                    elif job_info.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
                        if pbar.n < pbar.total:
                            pbar.n = pbar.total
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
            expected_markers=submission_result.get('n_chunks'),
            show_progress=show_progress
        )

    def monitor_scan(self, submission_result: Dict[str, Any],
                    show_progress: bool = True) -> JobInfo:
        """
        Monitor a parameter scan submission.

        Args:
            submission_result: Result from SlurmPipeline.submit_parameter_scan()
            show_progress: Show progress bar

        Returns:
            Final JobInfo
        """
        return self.monitor_job(
            job_id=submission_result['job_id'],
            job_name=submission_result['job_name'],
            output_dir=submission_result.get('output_dir'),
            expected_markers=submission_result.get('n_points'),
            show_progress=show_progress
        )

    def monitor_multiple(self, submissions: List[Dict[str, Any]],
                        show_progress: bool = True) -> List[JobInfo]:
        """
        Monitor multiple job submissions simultaneously.

        Args:
            submissions: List of submission results
            show_progress: Show progress bars

        Returns:
            List of final JobInfo objects
        """
        if not show_progress:
            # Simple sequential monitoring without progress
            return [self.monitor_job(sub['job_id'], sub.get('job_name'),
                                   show_progress=False)
                    for sub in submissions]

        # Create multi-progress bar
        job_bars = {}
        overall_bar = tqdm(total=len(submissions), desc="Overall", position=0)

        # Initialize progress bars for each job
        for i, sub in enumerate(submissions):
            if sub.get('array_jobs') and sub.get('n_chunks'):
                total = sub['n_chunks']
            elif sub.get('n_points'):
                total = sub['n_points']
            else:
                total = 1

            job_bars[sub['job_id']] = {
                'pbar': tqdm(total=total,
                           desc=sub.get('job_name', sub['job_id']),
                           position=i+1,
                           leave=False),
                'submission': sub,
                'last_count': 0,
                'completed': False
            }

        results = []

        try:
            while len(results) < len(submissions):
                for job_id, info in job_bars.items():
                    if info['completed']:
                        continue

                    sub = info['submission']
                    job_info = self.get_job_info(job_id)

                    # Update individual progress
                    pbar = info['pbar']
                    if job_info.state == JobState.PENDING:
                        pbar.set_postfix_str("Pending")
                    elif job_info.state == JobState.RUNNING:
                        if sub.get('output_dir'):
                            # Count markers
                            if sub.get('n_chunks'):
                                marker_count = self._count_markers(
                                    sub['output_dir'], sub['n_chunks'],
                                    prefix="completed_chunk_"
                                )
                            elif sub.get('n_points'):
                                marker_count = self._count_scan_markers(
                                    sub['output_dir']
                                )
                            else:
                                marker_count = 0

                            if marker_count > info['last_count']:
                                pbar.update(marker_count - info['last_count'])
                                info['last_count'] = marker_count

                        pbar.set_postfix_str(f"Running")
                    elif job_info.state in [JobState.COMPLETED, JobState.FAILED,
                                           JobState.CANCELLED]:
                        if pbar.n < pbar.total:
                            pbar.n = pbar.total
                            pbar.refresh()
                        pbar.set_postfix_str(job_info.state.value)
                        info['completed'] = True
                        results.append(job_info)
                        overall_bar.update(1)

                        # Notify job tracker if available
                        if self.job_tracker:
                            self.job_tracker.complete_job(job_id)

                time.sleep(self.check_interval)

        finally:
            overall_bar.close()
            for info in job_bars.values():
                info['pbar'].close()

        return results

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
                # Array job format: jobid_arraytaskid
                array_size = self._get_array_size(job_id)

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
        # Same format as time limit
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

    def _count_markers(self, output_dir: str, expected: int,
                      prefix: str = "completed_chunk_") -> int:
        """Count completion marker files."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return 0

        count = 0
        for i in range(expected):
            marker = output_path / f"{prefix}{i}.marker"
            if marker.exists():
                count += 1

        return count

    def _count_scan_markers(self, output_dir: str) -> int:
        """Count parameter scan completion markers."""
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

    def _get_array_size(self, job_id: str) -> Optional[int]:
        """Get array size for an array job."""
        # Query array job details
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
            if job_info.array_completed:
                print(f"Completed: {job_info.array_completed}/{job_info.array_size}")

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
