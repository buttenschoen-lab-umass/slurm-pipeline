"""
Debug utilities for SLURM pipeline to help diagnose issues.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class SlurmDebugger:
    """Debug utilities for SLURM pipeline issues."""

    def __init__(self, slurm_pipeline):
        """Initialize with a SlurmPipeline instance."""
        self.pipeline = slurm_pipeline

    def check_environment(self) -> Dict[str, Any]:
        """Check SLURM environment and configuration."""
        checks = {
            'slurm_available': False,
            'sbatch_path': None,
            'squeue_path': None,
            'python_path': None,
            'nfs_accessible': False,
            'nfs_writable': False,
            'issues': []
        }

        # Check SLURM commands
        for cmd in ['sbatch', 'squeue', 'sinfo']:
            result = subprocess.run(['which', cmd], capture_output=True, text=True)
            if result.returncode == 0:
                checks[f'{cmd}_path'] = result.stdout.strip()
                if cmd == 'sbatch':
                    checks['slurm_available'] = True
            else:
                checks['issues'].append(f"{cmd} not found in PATH")

        # Check Python
        result = subprocess.run(['which', 'python'], capture_output=True, text=True)
        if result.returncode == 0:
            checks['python_path'] = result.stdout.strip()
        else:
            checks['issues'].append("python not found in PATH")

        # Check NFS accessibility
        nfs_dir = self.pipeline.nfs_work_dir
        if nfs_dir.exists():
            checks['nfs_accessible'] = True

            # Try to write a test file
            test_file = nfs_dir / ".test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()
                checks['nfs_writable'] = True
            except Exception as e:
                checks['nfs_writable'] = False
                checks['issues'].append(f"Cannot write to NFS: {e}")
        else:
            checks['issues'].append(f"NFS directory not accessible: {nfs_dir}")

        # Check directory structure
        for subdir in ['inputs', 'scripts', 'outputs', 'logs']:
            dir_path = self.pipeline.nfs_work_dir / subdir
            if not dir_path.exists():
                checks['issues'].append(f"Missing directory: {dir_path}")

        return checks

    def test_submission(self, dry_run: bool = True) -> Dict[str, Any]:
        """Test job submission with a simple script."""
        print("Testing SLURM submission...")

        # Create a test script
        test_script = self.pipeline.nfs_scripts_dir / "test_submission.sbatch"

        script_content = """#!/bin/bash
#SBATCH --job-name=test_slurm_pipeline
#SBATCH --time=00:01:00
#SBATCH --mem=100M
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

echo "Test job started on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Date: $(date)"

# Test Python
python -c "print('Python is working'); import sys; print(f'Python version: {sys.version}')"

# Test file access
echo "Testing NFS access..."
ls -la

echo "Test job completed"
"""

        with open(test_script, 'w') as f:
            f.write(script_content)
        test_script.chmod(0o755)

        print(f"Created test script: {test_script}")

        if dry_run:
            print("\nDry run - script content:")
            print("-" * 60)
            print(script_content)
            print("-" * 60)
            print(f"\nTo submit manually: sbatch {test_script}")
            return {'status': 'dry_run', 'script': str(test_script)}

        # Actually submit
        result = subprocess.run(
            ['sbatch', str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(self.pipeline.nfs_scripts_dir)
        )

        response = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'script': str(test_script)
        }

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            response['job_id'] = job_id
            response['status'] = 'submitted'
            print(f"Successfully submitted test job: {job_id}")
        else:
            response['status'] = 'failed'
            print(f"Submission failed!")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")

        return response

    def check_job_submission_error(self, script_path: Path) -> Dict[str, Any]:
        """Diagnose why a job submission might have failed."""
        diagnostics = {
            'script_exists': script_path.exists(),
            'script_readable': False,
            'script_executable': False,
            'script_content': None,
            'syntax_check': None,
            'issues': []
        }

        if not diagnostics['script_exists']:
            diagnostics['issues'].append(f"Script not found: {script_path}")
            return diagnostics

        # Check permissions
        try:
            with open(script_path, 'r') as f:
                diagnostics['script_content'] = f.read()
                diagnostics['script_readable'] = True
        except Exception as e:
            diagnostics['issues'].append(f"Cannot read script: {e}")

        if os.access(script_path, os.X_OK):
            diagnostics['script_executable'] = True
        else:
            diagnostics['issues'].append("Script is not executable")

        # Check script syntax
        if diagnostics['script_readable']:
            # Check for basic SLURM directives
            content = diagnostics['script_content']
            if '#!/bin/bash' not in content:
                diagnostics['issues'].append("Missing shebang (#!/bin/bash)")
            if '#SBATCH' not in content:
                diagnostics['issues'].append("No SBATCH directives found")

            # Try bash syntax check
            result = subprocess.run(
                ['bash', '-n', str(script_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                diagnostics['syntax_check'] = 'failed'
                diagnostics['issues'].append(f"Bash syntax error: {result.stderr}")
            else:
                diagnostics['syntax_check'] = 'passed'

        return diagnostics

    def validate_runner_script(self) -> Dict[str, Any]:
        """Validate the runner script."""
        runner_path = self.pipeline.nfs_scripts_dir / "runner.py"

        validation = {
            'exists': runner_path.exists(),
            'executable': False,
            'python_syntax': None,
            'imports_ok': None,
            'issues': []
        }

        if not validation['exists']:
            validation['issues'].append(f"Runner script not found: {runner_path}")
            return validation

        if os.access(runner_path, os.X_OK):
            validation['executable'] = True
        else:
            validation['issues'].append("Runner script is not executable")

        # Check Python syntax
        result = subprocess.run(
            ['python', '-m', 'py_compile', str(runner_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            validation['python_syntax'] = 'valid'
        else:
            validation['python_syntax'] = 'invalid'
            validation['issues'].append(f"Python syntax error: {result.stderr}")

        # Test imports
        test_import_script = f"""
import sys
sys.path.insert(0, '{runner_path.parent}')
try:
    import runner
    print("Import successful")
except Exception as e:
    print(f"Import failed: {{e}}")
"""

        result = subprocess.run(
            ['python', '-c', test_import_script],
            capture_output=True,
            text=True,
            cwd=str(runner_path.parent)
        )

        if "Import successful" in result.stdout:
            validation['imports_ok'] = True
        else:
            validation['imports_ok'] = False
            validation['issues'].append(f"Import test failed: {result.stdout} {result.stderr}")

        return validation

    def run_diagnostics(self, job_id: Optional[str] = None) -> None:
        """Run full diagnostics and print report."""
        print("=" * 70)
        print("SLURM PIPELINE DIAGNOSTICS")
        print("=" * 70)

        # Environment check
        print("\n1. ENVIRONMENT CHECK")
        print("-" * 30)
        env_check = self.check_environment()

        print(f"SLURM available: {env_check['slurm_available']}")
        print(f"NFS accessible: {env_check['nfs_accessible']}")
        print(f"NFS writable: {env_check['nfs_writable']}")

        if env_check['sbatch_path']:
            print(f"sbatch path: {env_check['sbatch_path']}")
        if env_check['python_path']:
            print(f"Python path: {env_check['python_path']}")

        if env_check['issues']:
            print("\nIssues found:")
            for issue in env_check['issues']:
                print(f"  - {issue}")

        # Runner validation
        print("\n2. RUNNER SCRIPT VALIDATION")
        print("-" * 30)
        runner_check = self.validate_runner_script()

        print(f"Runner exists: {runner_check['exists']}")
        print(f"Runner executable: {runner_check['executable']}")
        print(f"Python syntax: {runner_check['python_syntax']}")
        print(f"Imports OK: {runner_check['imports_ok']}")

        if runner_check['issues']:
            print("\nIssues found:")
            for issue in runner_check['issues']:
                print(f"  - {issue}")

        # Job-specific diagnostics
        if job_id:
            print(f"\n3. JOB {job_id} DIAGNOSTICS")
            print("-" * 30)

            # Check job status
            result = subprocess.run(
                ['squeue', '-j', job_id, '--format=%T,%r'],
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                print(f"Job status: {result.stdout.strip()}")
            else:
                # Check completed jobs
                result = subprocess.run(
                    ['sacct', '-j', job_id, '--format=State,ExitCode,Elapsed', '--noheader'],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    print(f"Job history: {result.stdout.strip()}")
                else:
                    print("Job not found in queue or history")

        print("\n" + "=" * 70)

    def test_pickle_serialization(self, obj: Any) -> bool:
        """Test if an object can be pickled and unpickled."""
        import pickle
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                pickle.dump(obj, tmp)
                tmp_path = tmp.name

            with open(tmp_path, 'rb') as f:
                loaded = pickle.load(f)

            os.unlink(tmp_path)
            print("Object serialization: OK")
            return True

        except Exception as e:
            print(f"Object serialization: FAILED - {e}")
            return False


def add_debug_to_pipeline(pipeline_class):
    """Add debug method to SlurmPipeline class."""
    def debug(self):
        """Get debug interface for this pipeline."""
        return SlurmDebugger(self)

    pipeline_class.debug = debug
    return pipeline_class
