# SLURM Pipeline

A Python module for running simulation pipelines on SLURM clusters with automatic distribution, monitoring, and result collection.

## Features

- **Automatic work distribution** across SLURM array jobs
- **Real-time progress monitoring** with tqdm progress bars
- **NFS-aware** file handling for cluster environments
- **Hierarchical execution**: Single simulations → Ensembles → Parameter scans
- **Automatic post-processing** and result collection
- **Built-in diagnostics** for debugging cluster issues

## Installation

### Option 1: Install from source
```bash
git clone https://github.com/yourusername/slurm-pipeline.git
cd slurm-pipeline
pip install -e .
```

### Option 2: Install on cluster nodes
```bash
# On login node
cd slurm-pipeline
python setup.py sdist

# Copy to shared location
cp dist/slurm-pipeline-0.1.0.tar.gz /shared/software/

# On compute nodes (or in your SLURM script)
pip install --user /shared/software/slurm-pipeline-0.1.0.tar.gz
```

## Quick Start

```python
from slurm_pipeline import (
    SimulationPipeline, Ensemble, ParameterScan,
    SlurmConfig, SlurmPipeline
)

# 1. Define your model
class MyModel:
    def __init__(self, param1=1.0):
        self.param1 = param1

    def random_ic(self):
        return np.random.randn(100)

    def simulate(self, T, initial_state, dt=0.01):
        # Your simulation code
        pass

# 2. Create an ensemble
ensemble = Ensemble(
    MyModel,
    params={'param1': 2.0},
    integrator_params={'dt': 0.01}
)

# 3. Submit to SLURM
slurm = SlurmPipeline(nfs_work_dir="/scratch/$USER/simulations")
result = slurm.submit_ensemble(
    ensemble,
    n_simulations=1000,
    T=100.0,
    array_jobs=True,
    sims_per_job=50,  # 20 array jobs
    wait_for_completion=True
)
```

## Directory Structure

The module creates this structure on your NFS mount:
```
/your/nfs/path/slurm_pipeline/
├── inputs/          # Serialized objects and configs
├── scripts/         # SLURM batch scripts and runner.py
├── outputs/         # Simulation results
├── logs/           # SLURM stdout/stderr
└── submissions/    # Job submission records
```

## Module Structure

```
slurm_pipeline/
├── core/           # Core simulation components
│   ├── pipeline.py      # Single simulation pipeline
│   ├── ensemble.py      # Ensemble of simulations
│   └── parameter_scan.py # Parameter space scanning
├── slurm/          # SLURM-specific components
│   ├── config.py        # SLURM job configuration
│   ├── pipeline.py      # Main SLURM pipeline
│   └── monitor.py       # Job monitoring
├── diagnostics/    # Debugging tools
└── tests/          # Test suite
```

## Usage Examples

### Single Simulation
```python
pipeline = SimulationPipeline(MyModel, params={'a': 1.0})
result = slurm.submit_pipeline(pipeline, T=50.0)
```

### Ensemble with Analysis
```python
def analyze_rotation(time, solution, model):
    # Your analysis code
    return {'rotation': 'CW'}

ensemble = Ensemble(
    MyModel,
    params={'temperature': 0.5},
    analysis_functions={'rotation': analyze_rotation}
)

result = slurm.submit_ensemble(
    ensemble,
    n_simulations=500,
    wait_for_completion=True
)
print(f"Rotation statistics: {result['analysis']['rotation_statistics']}")
```

### Parameter Scan
```python
scan = ParameterScan(
    MyModel,
    base_params={'a': 1.0},
    scan_params={
        'temperature': [0.1, 0.5, 1.0, 2.0],
        'field_strength': [0, 1, 2]
    }
)

result = slurm.submit_parameter_scan(
    scan,
    n_simulations_per_point=100,
    array_jobs=True
)
```

### Monitoring
```python
# Submit without waiting
submission = slurm.submit_ensemble(ensemble, n_simulations=1000)

# Check status later
status = slurm.check_job_status(submission['job_id'])

# Monitor and collect when ready
result = slurm.monitor_and_collect(submission)
```

## Diagnostics

```bash
# Run diagnostics
slurm-pipeline-diagnose --nfs-dir /scratch/$USER/slurm_pipeline

# Test submission
slurm-pipeline-diagnose --test-submit

# Run test suite
slurm-pipeline-test --quick
```

## Configuration

### SLURM Configuration
```python
config = SlurmConfig(
    partition="gpu",
    time="12:00:00",
    mem="64G",
    cpus_per_task=8,
    gres="gpu:1",
    modules=["cuda/11.8", "python/3.10"],
    conda_env="simulations"
)

slurm.submit_ensemble(ensemble, slurm_config=config, ...)
```

### Custom NFS Path
```python
slurm = SlurmPipeline(nfs_work_dir="/custom/nfs/path")
```

## Requirements

- Python ≥ 3.7
- NumPy ≥ 1.19
- Matplotlib ≥ 3.3
- tqdm ≥ 4.60
- SLURM cluster with shared NFS storage

## License

MIT License - see LICENSE file for details.
