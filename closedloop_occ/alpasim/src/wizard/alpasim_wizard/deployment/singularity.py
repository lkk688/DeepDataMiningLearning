# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Singularity/Apptainer deployment strategy.

This backend targets SLURM clusters that provide Singularity/Apptainer instead
of Docker (the local backend) or enroot + pyxis (the SLURM backend).

It deliberately mirrors :class:`SlurmDeployment`: every microservice is launched
as an ``srun --overlap`` job step pinned to the wizard's own node, so the
services are co-located and reach each other over ``localhost`` (ports
``baseport``, ``baseport+1``, ...). The *only* difference is how a container is
entered -- instead of pyxis flags (``--container-image`` / ``--container-mounts``
/ ``--container-writable`` ...) this backend runs ``singularity exec`` as the
job-step program. Because of that, all of the launch/wait/cleanup orchestration
(:meth:`deploy`, :meth:`wait_for_containers`, :meth:`get_missing_containers`,
``scancel`` cleanup) is inherited unchanged from :class:`SlurmDeployment` and
only :meth:`_to_slurm_run` is overridden.

Notes on parity with the pyxis backend:

* ``--container-image=<sqsh>``      -> ``singularity exec ... <sif>``
* ``--container-mounts=A:B,C:D``    -> ``--bind A:B,C:D``
* ``--container-writable``          -> ``--writable-tmpfs``
* ``--container-workdir=X``         -> ``--pwd X``
* ``--no-container-remap-root``     -> default (Apptainer runs as the user)
* (pyxis remap-to-root default)     -> ``--fakeroot`` when ``remap_root`` is set
* GPU selection                     -> ``--nv`` plus the same in-command
  ``export CUDA_VISIBLE_DEVICES=...`` used by the pyxis backend.

The trailing ``bash -c "<payload>"`` (command + escaping) is byte-for-byte the
same as :class:`SlurmDeployment`, so quoting/escaping semantics are preserved;
only the container-entry tokens that precede it change.
"""

from __future__ import annotations

import logging
import os
import socket

from ..schema import RunMode
from ..services import ContainerDefinition
from ..utils import ensure_sif_path
from .slurm import SlurmDeployment

logger = logging.getLogger(__name__)


class SingularityDeployment(SlurmDeployment):
    """Deployment strategy using Singularity/Apptainer under SLURM.

    Inherits the full launch/wait/cleanup workflow from :class:`SlurmDeployment`
    (including the inherited constructor, which builds the container set with
    ``use_address_string="0.0.0.0"`` -- the single-node/localhost model) and
    overrides only :meth:`_to_slurm_run`.
    """

    def _to_slurm_run(
        self,
        container: ContainerDefinition,
        mode: RunMode,
    ) -> str:
        """Generate the ``srun`` command that launches a container via singularity.

        Mirrors :meth:`SlurmDeployment._to_slurm_run` exactly except for the
        container-entry mechanism: the pyxis ``--container-*`` srun flags are
        replaced by a ``singularity exec ...`` invocation placed immediately
        before the (identical) ``bash -c "<payload>"``.

        Args:
            container: ContainerDefinition instance.
            mode: RunMode (ONESHOT or SERVER).

        Returns:
            SLURM srun command string.
        """
        # LOCAL (no-SLURM) support: on a bare interactive node (our single H100)
        # there is no SLURM allocation, so slurm_job_id is None. In that case we
        # drop the `srun --overlap` wrapper entirely and run `apptainer exec`
        # directly as a background subprocess (dispatch_command uses Popen); all
        # services still co-locate on this node and talk over localhost.
        # No SLURM allocation -> slurm_job_id is None or 0 (wizard.create sets it
        # to int(SLURM_JOB_ID or "0") when unset). Either means "run locally".
        no_slurm = not container.context.cfg.wizard.slurm_job_id
        slurm_job_id = container.context.cfg.wizard.slurm_job_id or "local"

        s_log = (
            f"{container.context.cfg.wizard.log_dir}/txt-logs/"
            f"out-{slurm_job_id}-{container.uuid}-log.txt"
        )

        sif = ensure_sif_path(
            container.service_config.image,
            list(container.context.cfg.wizard.sifcaches),
        )

        # Pin the GPU from inside the container, set last so it wins over any
        # CUDA_VISIBLE_DEVICES inherited from the SLURM job environment. This
        # matches SlurmDeployment, which also sets it in-command (SLURM otherwise
        # overrides an exported CUDA_VISIBLE_DEVICES). ``--nv`` exposes the host
        # GPUs/driver; this export narrows them to the assigned device.
        s_gpu = (
            f"export CUDA_VISIBLE_DEVICES={container.gpu};"
            if container.gpu is not None
            else ""
        )

        # Environment variables, split exactly like SlurmDeployment:
        #  - 'VAR=value' -> exported inside the container (value visible in logs).
        #  - 'VAR'       -> pass-through from the host. Apptainer forwards the
        #                   host environment into the container by default, so
        #                   these reach the service without exposing their values
        #                   on the command line (secure for secrets). We do NOT
        #                   pass --cleanenv, which would strip that forwarding.
        env_export_set = [e for e in (container.environments or []) if "=" in e]
        s_env_exports = (
            " ".join(f"export {e};" for e in env_export_set) + " "
            if env_export_set
            else ""
        )

        s_mnt = ",".join([v.to_str() for v in container.volumes])

        # Pin child srun steps to the wizard's node so services are co-located
        # and reachable via localhost (identical rationale and source to
        # SlurmDeployment).
        current_node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()

        singularity_bin = os.environ.get("ALPASIM_SINGULARITY_BIN", "singularity")

        if no_slurm:
            # Direct on this node: no srun, no node-pinning. dispatch_command
            # runs this via Popen(shell=True); non-blocking steps background
            # naturally, the runtime step blocks. Output is redirected to the
            # per-service log (srun's --output no longer does that for us).
            cmd = ""
        else:
            cmd = r"srun --verbose --overlap "
            cmd += f"--job-name={self._get_slurm_step_name(container)} "
            cmd += f"--nodes=1 --ntasks=1 --nodelist={current_node} "
            if container.context.cfg.wizard.slurm_cpu_bind_none:
                cmd += "--cpu-bind=none "
            cmd += f"--output={s_log} --error={s_log} "

        # Container entry: ``singularity exec`` replaces the pyxis --container-*
        # flags. Everything after the image is the program run inside it.
        cmd += f"{singularity_bin} exec --nv --writable-tmpfs "
        if s_mnt:
            cmd += f"--bind {s_mnt} "
        if container.workdir is not None:
            cmd += f"--pwd {container.workdir} "
        if container.service_config.remap_root:
            # pyxis remaps to root by default; the Singularity equivalent is
            # --fakeroot. (No OSS service sets remap_root=True today.)
            cmd += "--fakeroot "
        cmd += f"{sif} "

        escaped_command = container.command.replace("$$", r"\$")

        if mode in (RunMode.ONESHOT, RunMode.SERVER):
            cmd += f'bash -c "{s_gpu}{s_env_exports}{escaped_command}"'
            if no_slurm:
                # srun used to capture stdout/err to s_log; do it ourselves.
                cmd += f" > {s_log} 2>&1"
        else:
            raise ValueError(f"Unknown run mode: {mode}")
        return cmd

    # ---- LOCAL (no-SLURM) overrides -------------------------------------
    # When there is no SLURM allocation, step names and cleanup cannot use
    # slurm_job_id / scancel. We key everything off the (unique per run) host
    # log_dir, which appears in every service's `apptainer exec ... --bind
    # <log_dir>:...` command line, so `pkill -f` can find and TERM them.

    def _local(self) -> bool:
        return not self.context.cfg.wizard.slurm_job_id

    def _get_slurm_step_name(self, container: ContainerDefinition) -> str:
        if self._local():
            return f"alpasim-local-{container.uuid}"
        return super()._get_slurm_step_name(container)

    def _get_slurm_cleanup_command(self, container: ContainerDefinition) -> str:
        if self._local():
            # Kill this run's co-located service processes by their unique
            # log_dir marker (present in the apptainer --bind args). Matching
            # `apptainer exec` avoids killing the wizard itself.
            log_dir = self.context.cfg.wizard.log_dir
            return f"pkill -TERM -f 'apptainer exec .*{log_dir}' || true"
        return super()._get_slurm_cleanup_command(container)
