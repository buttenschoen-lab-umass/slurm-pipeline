"""
Local tmpfs management for SLURM jobs to avoid NFS performance issues.
"""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from contextlib import contextmanager
from datetime import datetime


class TmpfsManager:
    """
    Manages local tmpfs storage for SLURM job execution.

    This class handles:
    1. Creating a local tmpfs workspace
    2. Copying input files from NFS to tmpfs
    3. Running computations in tmpfs
    4. Copying results back to NFS
    """

    def __init__(self,
                 nfs_base_dir: str,
                 job_name: str,
                 use_dev_shm: bool = True,
                 use_tmp: bool = False,
                 custom_tmpfs: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize tmpfs manager.

        Args:
            nfs_base_dir: Base NFS directory (original output directory)
            job_name: Name of the job (for creating subdirectories)
            use_dev_shm: Use /dev/shm if available (usually tmpfs)
            use_tmp: Fall back to /tmp if /dev/shm not available
            custom_tmpfs: Custom tmpfs path (e.g., /scratch/local)
            logger: Logger instance
        """
        self.nfs_base_dir = Path(nfs_base_dir)
        self.job_name = job_name
        self.use_dev_shm = use_dev_shm
        self.use_tmp = use_tmp
        self.custom_tmpfs = custom_tmpfs
        self.logger = logger or logging.getLogger(__name__)

        self.local_work_dir = None
        self.is_active = False

    def _find_tmpfs_location(self) -> Path:
        """Find the best available tmpfs location."""
        candidates = []

        # Add custom location first if specified
        if self.custom_tmpfs:
            candidates.append(Path(self.custom_tmpfs))

        # Check /dev/shm (usually tmpfs on Linux)
        if self.use_dev_shm:
            candidates.append(Path("/dev/shm"))

        # Check for other common tmpfs locations
        common_tmpfs = [
            "/scratch/local",  # Common on HPC clusters
            "/local",
            "/ltmp",
            "/scratch/$USER",  # Might need expansion
        ]

        for path_str in common_tmpfs:
            # Expand environment variables
            expanded = os.path.expandvars(path_str)
            candidates.append(Path(expanded))

        # Fall back to /tmp if allowed
        if self.use_tmp:
            candidates.append(Path("/tmp"))

        # Find first available candidate with write access
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.W_OK):
                # Test write access
                try:
                    test_file = candidate / f".tmpfs_test_{os.getpid()}"
                    test_file.touch()
                    test_file.unlink()
                    return candidate
                except Exception:
                    continue

        # If nothing found, raise error
        raise RuntimeError(
            f"No suitable tmpfs location found. Tried: {[str(c) for c in candidates]}"
        )

    def setup(self) -> Path:
        """
        Set up local tmpfs workspace.

        Returns:
            Path to local work directory
        """
        if self.is_active:
            return self.local_work_dir

        # Find tmpfs location
        tmpfs_root = self._find_tmpfs_location()
        self.logger.info(f"Using tmpfs at: {tmpfs_root}")

        # Create unique work directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")

        if array_id:
            dirname = f"{self.job_name}_{job_id}_{array_id}_{timestamp}"
        else:
            dirname = f"{self.job_name}_{job_id}_{timestamp}"

        self.local_work_dir = tmpfs_root / dirname
        self.local_work_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created local work directory: {self.local_work_dir}")

        # Check available space
        stat = os.statvfs(self.local_work_dir)
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        self.logger.info(f"Available space in tmpfs: {available_gb:.2f} GB")

        self.is_active = True
        return self.local_work_dir

    def copy_inputs(self, files: Optional[List[str]] = None) -> None:
        """
        Copy input files from NFS to local tmpfs.

        Args:
            files: Specific files to copy. If None, copies nothing.
        """
        if not self.is_active:
            raise RuntimeError("TmpfsManager not active. Call setup() first.")

        if not files:
            return

        self.logger.info(f"Copying {len(files)} input files to tmpfs...")

        for file_path in files:
            src = Path(file_path)
            if not src.exists():
                self.logger.warning(f"Input file not found: {src}")
                continue

            # Maintain relative structure if file is under nfs_base_dir
            try:
                rel_path = src.relative_to(self.nfs_base_dir)
                dst = self.local_work_dir / rel_path
            except ValueError:
                # File is outside nfs_base_dir, just copy to root
                dst = self.local_work_dir / src.name

            # Create parent directory
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            try:
                shutil.copy2(src, dst)
                self.logger.debug(f"Copied: {src} -> {dst}")
            except Exception as e:
                self.logger.error(f"Failed to copy {src}: {e}")
                raise

    def map_path(self, nfs_path: str) -> str:
        """
        Map an NFS path to the corresponding local tmpfs path.

        Args:
            nfs_path: Original NFS path

        Returns:
            Mapped local path as string
        """
        if not self.is_active:
            return nfs_path

        nfs_path = Path(nfs_path)

        # If it's already under local work dir, return as-is
        if str(nfs_path).startswith(str(self.local_work_dir)):
            return str(nfs_path)

        # Try to make it relative to nfs_base_dir
        try:
            rel_path = nfs_path.relative_to(self.nfs_base_dir)
            return str(self.local_work_dir / rel_path)
        except ValueError:
            # Path is not under nfs_base_dir
            # Put it directly under local work dir
            return str(self.local_work_dir / nfs_path.name)

    def sync_back(self,
                  exclude_patterns: Optional[List[str]] = None,
                  delete_local: bool = True) -> None:
        """
        Sync results back from tmpfs to NFS.

        Args:
            exclude_patterns: Patterns to exclude from sync (e.g., ['*.tmp', '*.log'])
            delete_local: Delete local files after sync
        """
        if not self.is_active:
            return

        self.logger.info("Syncing results back to NFS...")

        # Use rsync for efficient sync
        rsync_cmd = [
            "rsync", "-av", "--relative",
            f"{self.local_work_dir}/.",  # Source with relative paths
            f"{self.nfs_base_dir}/"      # Destination
        ]

        # Add exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                rsync_cmd.extend(["--exclude", pattern])

        try:
            # First attempt with rsync
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Rsync completed successfully")
            else:
                self.logger.warning(f"Rsync failed: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, rsync_cmd)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to Python copy
            self.logger.info("Falling back to Python-based copy...")
            self._python_sync_back(exclude_patterns)

        # Clean up local files if requested
        if delete_local and self.local_work_dir:
            try:
                shutil.rmtree(self.local_work_dir)
                self.logger.info(f"Cleaned up local directory: {self.local_work_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up local directory: {e}")

        self.is_active = False

    def _python_sync_back(self, exclude_patterns: Optional[List[str]] = None):
        """Python-based fallback for syncing files back to NFS."""
        import fnmatch

        for root, dirs, files in os.walk(self.local_work_dir):
            root_path = Path(root)

            # Skip if matches exclude pattern
            if exclude_patterns:
                skip = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(str(root_path), pattern):
                        skip = True
                        break
                if skip:
                    continue

            # Process files
            for file in files:
                # Skip if matches exclude pattern
                if exclude_patterns:
                    skip = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(file, pattern):
                            skip = True
                            break
                    if skip:
                        continue

                src = root_path / file

                # Calculate destination path
                try:
                    rel_path = src.relative_to(self.local_work_dir)
                    dst = self.nfs_base_dir / rel_path
                except ValueError:
                    # Should not happen
                    continue

                # Create parent directory
                dst.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    self.logger.error(f"Failed to copy {src} to {dst}: {e}")

    def cleanup(self):
        """Emergency cleanup method."""
        if self.local_work_dir and self.local_work_dir.exists():
            try:
                shutil.rmtree(self.local_work_dir)
                self.logger.info(f"Emergency cleanup of {self.local_work_dir}")
            except Exception as e:
                self.logger.error(f"Emergency cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures sync back even on error."""
        if exc_type is not None:
            self.logger.error(f"Exception in tmpfs context: {exc_val}")

        # Always try to sync back
        try:
            self.sync_back()
        except Exception as e:
            self.logger.error(f"Failed to sync back: {e}")
            # Don't suppress the original exception

        return False  # Don't suppress exceptions


@contextmanager
def tmpfs_context(nfs_dir: str, job_name: str, **kwargs):
    """
    Convenience context manager for tmpfs operations.

    Example:
        with tmpfs_context(output_dir, "my_job") as tmpfs:
            local_dir = tmpfs.map_path(output_dir)
            # Do work in local_dir
            # Auto-syncs back on exit
    """
    manager = TmpfsManager(nfs_dir, job_name, **kwargs)
    try:
        yield manager
    finally:
        manager.sync_back()


def get_tmpfs_info() -> Dict[str, Any]:
    """Get information about available tmpfs locations."""
    info = {
        'locations': {},
        'recommended': None
    }

    locations = [
        '/dev/shm',
        '/tmp',
        '/scratch/local',
        '/local',
        os.path.expandvars('/scratch/$USER')
    ]

    for loc in locations:
        path = Path(loc)
        if path.exists():
            try:
                stat = os.statvfs(path)
                info['locations'][loc] = {
                    'exists': True,
                    'writable': os.access(path, os.W_OK),
                    'total_gb': (stat.f_blocks * stat.f_frsize) / (1024**3),
                    'available_gb': (stat.f_bavail * stat.f_frsize) / (1024**3),
                    'mount_type': _get_mount_type(path)
                }
            except Exception as e:
                info['locations'][loc] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            info['locations'][loc] = {'exists': False}

    # Find recommended location (tmpfs with most space)
    best_size = 0
    for loc, loc_info in info['locations'].items():
        if (loc_info.get('exists') and
            loc_info.get('writable') and
            loc_info.get('mount_type') == 'tmpfs' and
            loc_info.get('available_gb', 0) > best_size):
            best_size = loc_info['available_gb']
            info['recommended'] = loc

    return info


def _get_mount_type(path: Path) -> str:
    """Get the filesystem type of a mount point."""
    try:
        result = subprocess.run(
            ['df', '-T', str(path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # Format: Filesystem Type Size Used Avail Use% Mounted
                parts = lines[1].split()
                if len(parts) > 1:
                    return parts[1]
    except Exception:
        pass
    return 'unknown'
