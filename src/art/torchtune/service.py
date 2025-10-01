import asyncio
import glob
import logging
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import AsyncIterator

import torch
import torchtune
from safetensors.torch import load_file
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..preprocessing.pack import DiskPackedTensors
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers
from .batch import Batch

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a console handler if one doesn't exist
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


@dataclass
class TorchtuneService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False
    _llm_task: asyncio.Task[AsyncLLM] | None = field(default=None, init=False)

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        await openai_server_task(
            engine=await self.llm,
            config=dev.get_openai_server_config(
                model_name=self.model_name,
                base_model=self.get_last_checkpoint_dir() or self.base_model,
                log_file=f"{self.output_dir}/logs/vllm.log",
                config=config,
            ),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def train(
            self,
            disk_packed_tensors: DiskPackedTensors,
            config: types.TrainConfig,
            _config: dev.TrainConfig,
            verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        logger.info(f"Starting training with output_dir: {self.output_dir}")
        logger.info(f"Torchtune args: {self.torchtune_args}")

        # Track if we've started yielding results
        yielded_any = False

        try:
            # Yield an initial status to establish the async generator connection
            yield {"status": "initializing", "num_gradient_steps": 0}
            yielded_any = True

            # Get LLM instance
            logger.info("Getting LLM instance...")
            try:
                llm = await self.llm
                logger.info("Got LLM instance successfully")
            except Exception as e:
                logger.error(f"Failed to get LLM instance: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            pids_path = f"{self.output_dir}/pids.txt"
            # reset the pids file
            with open(pids_path, "w") as f:
                f.write("")
            weights_path = "/dev/shm/weights.safetensors"
            # remove the weights file if it exists
            Path(weights_path).unlink(missing_ok=True)
            async_weight_syncing = self.torchtune_args.get("async_weight_syncing", False)
            logger.info(f"Async weight syncing: {async_weight_syncing}")

            # start putting the workers to sleep
            logger.info("Creating sleep task for workers...")
            try:
                sleep_task = asyncio.create_task(
                    run_on_workers(
                        llm,
                        sleep,
                        # level=1 if llm.output_processor.has_unfinished_requests() else 2,
                        level=1,
                        pids_path=pids_path,
                        weights_path=None if async_weight_syncing else weights_path,
                        profile=verbose,
                    )
                )
                logger.info("Sleep task created successfully")
            except Exception as e:
                logger.error(f"Failed to create sleep task: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

            # wait for the workers to write their pids twice, indicating that they are asleep
            logger.info("Waiting for workers to sleep...")
            timeout_counter = 0
            max_timeout = 300  # Increase timeout to 120 seconds
            expected_worker_count = torch.cuda.device_count() if hasattr(torch, 'cuda') else 2
            logger.info(f"Expecting {expected_worker_count} workers to report sleep status")

            while True:
                try:
                    # Check if the sleep task has failed
                    if sleep_task.done():
                        exc = sleep_task.exception()
                        if exc:
                            logger.error(f"Sleep task failed with exception: {exc}")
                            raise exc

                    if not os.path.exists(pids_path):
                        logger.warning(f"PIDs file doesn't exist yet: {pids_path}")
                        await asyncio.sleep(0.25)
                        timeout_counter += 0.25
                        if timeout_counter > max_timeout:
                            raise TimeoutError(f"Timeout waiting for PIDs file after {max_timeout}s")
                        continue

                    with open(pids_path, 'r') as f:
                        pid_lines = f.read().splitlines()
                    pids = Counter(pid_lines)

                    # Check if all expected workers have written their PIDs twice
                    if all(count == 2 for count in pids.values()):
                        logger.info(f"All {expected_worker_count} workers reported as asleep")
                        break

                    await asyncio.sleep(0.25)
                    timeout_counter += 0.25
                    if timeout_counter > max_timeout:
                        logger.error(f"Current PID status after timeout: {dict(pids)}")
                        raise TimeoutError(f"Timeout waiting for workers to sleep after {max_timeout}s")

                except asyncio.CancelledError:
                    logger.error("PID waiting was cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error while waiting for workers to sleep: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
            self._is_sleeping = True
            logger.info("Workers are asleep")

            # acquire the train process and queue
            logger.info("Getting train process...")
            train_process = await self.train_process
            logger.info("Getting train queue...")
            train_queue = await self.train_queue
            # write the batch to communicate with the train process
            batch_file = f"{self.output_dir}/batches.jsonl"
            logger.info(f"Writing batch to {batch_file}")
            with open(batch_file, "a") as f:
                f.write(
                    Batch(
                        disk_packed_tensors=disk_packed_tensors,
                        config=config,
                        dev_config=_config,
                    ).model_dump_json()
                    + "\n"
                )

            # consume the batch gradient step results
            logger.info("Starting to consume training results...")
            num_gradient_steps = -1
            while num_gradient_steps != 0:
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(train_queue.get()),
                        asyncio.create_task(train_process.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    result = task.result()
                    if isinstance(result, dict):
                        result["num_gradient_steps"] = int(result["num_gradient_steps"])
                        if num_gradient_steps == -1:
                            num_gradient_steps = result["num_gradient_steps"]
                            logger.info(f"Total gradient steps to process: {num_gradient_steps}")
                        logger.debug(f"Yielding result: {result}")
                        yield result
                        yielded_any = True
                    else:
                        logger.error(f"Train process exited with code: {result}")
                        # Try to get more info about the failure
                        if train_process.returncode is not None:
                            logger.error(f"Process return code: {train_process.returncode}")
                        raise RuntimeError(
                            f"Train process exited early. See {self.output_dir}/logs/train.log for details."
                        )
                num_gradient_steps -= 1

            # wait for the workers to wake up
            logger.info("Waiting for workers to wake up...")
            await sleep_task
            self._is_sleeping = False

            # update the weights after wake up if async_weight_syncing is enabled
            if async_weight_syncing:
                asyncio.create_task(self.update_worker_weights(llm, weights_path, verbose))
            else:
                # remove the weights file
                Path(weights_path).unlink(missing_ok=True)

        except asyncio.CancelledError:
            logger.error("Training was cancelled")
            logger.error(f"Yielded any results: {yielded_any}")
            # If we were cancelled but haven't yielded anything yet, this might be due to
            # the ASGI response being terminated by an external error
            if not yielded_any:
                logger.error("Training cancelled before yielding any results - possible ASGI termination")
            raise
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Yielded any results: {yielded_any}")
            raise

    async def update_worker_weights(
            self, llm: AsyncLLM, weights_path: str, profile: bool
    ) -> None:
        while True:
            if os.path.exists(weights_path):
                break
            else:
                time.sleep(1)
                continue
        await run_on_workers(
            llm,
            update_weights,
            weights_path=weights_path,
            profile=profile,
        )
        # remove the weights file
        Path(weights_path).unlink(missing_ok=True)

    @property
    def torchtune_args(self) -> dev.TorchtuneArgs:
        torchtune_args = self.config.get("torchtune_args")
        assert torchtune_args is not None, (
            'TorchtuneService created without config["torchtune_args"]'
        )
        return torchtune_args

    @property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        logger.debug("Getting llm property")
        if self._llm_task is None:
            logger.info("Creating LLM task...")
            try:
                engine_args = self.config.get("engine_args", {})
                logger.info(f"Engine args: {engine_args}")

                # Create the get_llm task
                self._llm_task = asyncio.create_task(
                    get_llm(AsyncEngineArgs(**engine_args))  # type: ignore
                )
                logger.info("LLM task created successfully")
            except Exception as e:
                logger.error(f"Failed to create LLM task: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        return self._llm_task

    @cached_property
    def train_queue(self) -> asyncio.Task[asyncio.Queue[dict[str, float]]]:
        return asyncio.create_task(self.get_train_queue())

    @cached_property
    def train_process(self) -> asyncio.Task[asyncio.subprocess.Process]:
        return asyncio.create_task(self.get_train_process())

    async def get_train_process(self) -> asyncio.subprocess.Process:
        logger.info("=== Starting get_train_process ===")

        try:
            # Migrate existing checkpoints to new structure if needed
            from ..local.checkpoints import migrate_checkpoints_to_new_structure
            migrate_checkpoints_to_new_structure(self.output_dir)

            batch_file = f"{self.output_dir}/batches.jsonl"
            Path(batch_file).unlink(missing_ok=True)
            logger.info(f"Cleared batch file: {batch_file}")

            # Get checkpoint directory
            logger.info("Getting checkpoint directory...")
            checkpoint_dir = await self.get_checkpoint_dir()
            logger.info(f"Checkpoint directory: {checkpoint_dir}")

            # Check if checkpoint directory exists
            if not os.path.exists(checkpoint_dir):
                logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

            torchtune_args = self.torchtune_args
            logger.info(f"Torchtune args: {torchtune_args}")

            # Get the list of safetensor files
            safetensor_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
            logger.info(f"Found {len(safetensor_files)} safetensor files: {safetensor_files}")

            if not safetensor_files:
                logger.error(f"No .safetensors files found in {checkpoint_dir}")
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

            checkpoint_files = [os.path.basename(f) for f in safetensor_files]
            checkpoint_files_str = "[" + ", ".join(f'"{f}"' for f in checkpoint_files) + "]"
            logger.info(f"Checkpoint files string: {checkpoint_files_str}")

            def model_dir(model: str) -> str:
                for prefix in [
                    "llama3_1",
                    "llama3_2_vision",
                    "llama3_2",
                    "llama3_3",
                    "qwen2_5",
                ]:
                    if model.startswith(prefix):
                        return prefix
                return model.split("_")[0]

            # Check torchtune installation
            torchtune_path = os.path.dirname(torchtune.__file__)
            tune_cli_path = f"{torchtune_path}/_cli/tune.py"
            logger.info(f"Torchtune path: {torchtune_path}")
            logger.info(f"Tune CLI path: {tune_cli_path}")

            if not os.path.exists(tune_cli_path):
                logger.error(f"Torchtune CLI not found at: {tune_cli_path}")
                raise FileNotFoundError(f"Torchtune CLI not found: {tune_cli_path}")

            # Check config file
            config_path = f"{os.path.dirname(__file__)}/config.yaml"
            if not os.path.exists(config_path):
                logger.error(f"Config file not found at: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")

            # Ensure logs directory exists
            logs_dir = f"{self.output_dir}/logs"
            os.makedirs(logs_dir, exist_ok=True)
            logger.info(f"Created logs directory: {logs_dir}")

            # Build the command
            model_component = f"torchtune.models.{model_dir(torchtune_args['model'])}.{torchtune_args['model']}"
            logger.info(f"Model component: {model_component}")

            program_and_args = [
                sys.executable,  # Use current Python interpreter
                tune_cli_path,
                "run",
                "--nproc-per-node",
                str(torch.cuda.device_count()),
                "art.torchtune.recipe.FullFinetuneRecipeDistributed",
                "--config",
                config_path,
                f"model._component_={model_component}",
                f"checkpointer.checkpoint_dir={checkpoint_dir}",
                f"checkpointer.checkpoint_files={checkpoint_files_str}",
                f"checkpointer.model_type={torchtune_args['model_type']}",
                f"tensor_parallel_dim={torchtune_args.get('tensor_parallel_dim', 1)}",
                f"context_parallel_dim={torchtune_args.get('context_parallel_dim', 1)}",
                f"output_dir={self.output_dir}",
                "clip_grad_norm=0.1",
                "metric_logger._component_=torchtune.training.metric_logging.StdoutLogger",
                "metric_logger.log_dir=null",
                f"enable_activation_offloading={torchtune_args.get('enable_activation_offloading', False)}",
            ]

            logger.info("Full command to execute:")
            logger.info(" ".join(program_and_args))

            # Create the subprocess
            logger.info("Creating subprocess...")
            process = await asyncio.subprocess.create_subprocess_exec(
                *program_and_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Ensure unbuffered output
            )

            logger.info(f"Subprocess created with PID: {process.pid}")
            return process

        except Exception as e:
            logger.error(f"Failed to create train process: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_train_queue(self) -> asyncio.Queue[dict[str, float]]:
        logger.info("Getting train queue...")

        try:
            process = await self.train_process
            queue = asyncio.Queue()

            # Create logs directory if it doesn't exist
            logs_dir = f"{self.output_dir}/logs"
            os.makedirs(logs_dir, exist_ok=True)
            train_log_path = f"{logs_dir}/train.log"
            logger.info(f"Train log will be written to: {train_log_path}")

            async def read(reader: asyncio.StreamReader, stream_name: str) -> None:
                try:
                    async for line in reader:
                        line_str = line.decode("utf-8")
                        with open(train_log_path, "a") as f:
                            f.write(f"[{stream_name}] {line_str}")

                        # Also log the first few lines to console for debugging
                        if stream_name == "stderr":
                            logger.error(f"Train process stderr: {line_str.strip()}")

                        line_str = line_str.strip()
                        if line_str.startswith("Step ") and " | " in line_str:
                            parts = line_str.split(" | ", 1)
                            metrics: dict[str, float] = {}
                            if len(parts) > 1:
                                for metric in parts[1].split():
                                    if ":" in metric:
                                        name, value = metric.split(":", 1)
                                        try:
                                            metrics[name] = float(value)
                                        except ValueError:
                                            # Skip non-numeric values to match the return type
                                            pass
                            await queue.put(metrics)
                except Exception as e:
                    logger.error(f"Error reading from {stream_name}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

            assert process.stdout and process.stderr
            asyncio.create_task(read(process.stdout, "stdout"))
            asyncio.create_task(read(process.stderr, "stderr"))

            # Check if process is still running after a short delay
            await asyncio.sleep(0.5)
            if process.returncode is not None:
                logger.error(f"Train process exited immediately with code: {process.returncode}")
                # Try to read any output that was produced
                if process.stderr:
                    stderr_output = await process.stderr.read()
                    if stderr_output:
                        logger.error(f"Stderr output: {stderr_output.decode('utf-8')}")
                if process.stdout:
                    stdout_output = await process.stdout.read()
                    if stdout_output:
                        logger.error(f"Stdout output: {stdout_output.decode('utf-8')}")

            return queue

        except Exception as e:
            logger.error(f"Failed to get train queue: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def get_checkpoint_dir(self) -> str:
        logger.info("Getting checkpoint directory...")

        # Use the last of any existing checkpoints to resume training
        if last_checkpoint_dir := self.get_last_checkpoint_dir():
            logger.info(f"Using existing checkpoint directory: {last_checkpoint_dir}")
            return last_checkpoint_dir

        # Check if self.base_model is a directory
        if os.path.isdir(self.base_model):
            logger.info(f"Base model is a directory: {self.base_model}")
            return self.base_model

        # Otherwise, assume it's a HuggingFace model id and download it
        logger.info(f"Downloading HuggingFace model: {self.base_model}")

        try:
            process = await asyncio.subprocess.create_subprocess_exec(
                "huggingface-cli",
                "download",
                self.base_model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to download model. Return code: {process.returncode}")
                logger.error(f"Stderr: {stderr.decode('utf-8')}")
                raise RuntimeError(f"Failed to download model {self.base_model}")

            output_lines = stdout.decode("utf-8").splitlines()
            if not output_lines:
                logger.error("No output from huggingface-cli download")
                raise RuntimeError("huggingface-cli download produced no output")

            checkpoint_dir = output_lines[-1].strip()
            logger.info(f"Downloaded model to: {checkpoint_dir}")
            return checkpoint_dir

        except Exception as e:
            logger.error(f"Failed to get checkpoint directory: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_last_checkpoint_dir(self) -> str | None:
        from ..local.checkpoints import get_last_checkpoint_dir

        return get_last_checkpoint_dir(self.output_dir)


def sleep(
        *, level: int, pids_path: str, weights_path: str | None, profile: bool
) -> None:
    """
    Put the worker to sleep until the new model weights are loaded.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        weights_path: The path to the weights file.
        profile: Whether to profile
    """
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.v1.worker.gpu_worker import logger

    with open(pids_path, "a") as f:
        f.write(f"{os.getpid()}\n")
    worker = get_worker()
    allocator = CuMemAllocator.get_instance()
    try:
        if not (profile and worker.rank == 0):
            logger.setLevel(logging.CRITICAL)
        setattr(allocator, "_override_tags", {"weights", "kv_cache"})
        with worker.time("sleep"):
            worker.sleep(level)
        with open(pids_path, "a") as f:
            f.write(f"{os.getpid()}\n")
        weights = None
        while True:
            if weights_path:
                # wait for the weights file to be created
                try:
                    with worker.time("load_file"):
                        weights = load_file(weights_path)
                    break
                except FileNotFoundError:
                    time.sleep(1)
                    continue
            elif os.path.exists(pids_path):
                time.sleep(1)
                continue
            else:
                # no pids file indicates we can wake up
                break
        with worker.time("wake_up"):
            worker.wake_up()
        if weights is None:
            return
        with worker.time("load_weights"):
            worker.model_runner.model.load_weights(weights.items())  # type: ignore
    finally:
        logger.setLevel(logging.INFO)
        delattr(allocator, "_override_tags")


def update_weights(weights_path: str, profile: bool) -> None:
    from vllm.v1.worker.gpu_worker import logger

    worker = get_worker()
    try:
        if not (profile and worker.rank == 0):
            logger.setLevel(logging.CRITICAL)
        with worker.time("load_file"):
            weights = load_file(weights_path)
        with worker.time("load_weights"):
            worker.model_runner.model.load_weights(weights.items())  # type: ignore
    finally:
        logger.setLevel(logging.INFO)
