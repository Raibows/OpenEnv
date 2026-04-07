"""
Scale benchmark: find the max concurrent sessions on this machine.

Runs the RL stress test at increasing scales, recording metrics at each level.
Stops when failure rate exceeds 5% or system memory exceeds 90%.

Usage:
    # Terminal 1: Start server
    PYTHONPATH=src:envs uv run uvicorn \
        envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899

    # Terminal 2: Run benchmark
    PYTHONPATH=src:envs uv run python \
        envs/agent_world_model_env/example_scale_benchmark.py
"""

import asyncio
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx
import psutil
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scale_bench")

BASE_URL = "http://localhost:8899"

SCENARIOS = [
    "e_commerce_33",
    "inventory_management_7",
    "document_management_5",
    "billing_payments_3",
    "hris_employee_management_1",
]

# Fixed parameters for all scales
TURNS_PER_EPISODE = 3  # keep short to focus on concurrency, not interaction time
LLM_THINK_MIN = 0.05
LLM_THINK_MAX = 0.3
CLIENT_TIMEOUT = 600.0
CONNECT_TIMEOUT = 120.0


@dataclass
class SessionResult:
    success: bool = False
    connect_s: float = 0.0
    reset_s: float = 0.0
    list_tools_s: float = 0.0
    tool_call_latencies: list[float] = field(default_factory=list)
    done_s: float = 0.0
    total_s: float = 0.0
    error: str | None = None


class ResourceTracker:
    def __init__(self, server_pid: int | None, interval: float = 2.0):
        self._server_pid = server_pid
        self._interval = interval
        self._samples: list[dict] = []
        self._task: asyncio.Task | None = None
        self._client_proc = psutil.Process(os.getpid())

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while True:
            s = {
                "cpu": psutil.cpu_percent(interval=0),
                "mem_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "mem_pct": psutil.virtual_memory().percent,
                "client_mb": round(self._client_proc.memory_info().rss / (1024**2), 1),
            }
            if self._server_pid:
                try:
                    sp = psutil.Process(self._server_pid)
                    children = sp.children(recursive=True)
                    mem = sp.memory_info().rss
                    for c in children:
                        try:
                            mem += c.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    s["server_mb"] = round(mem / (1024**2), 1)
                    s["subprocs"] = len(children)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._samples.append(s)
            await asyncio.sleep(self._interval)

    def summary(self) -> dict:
        if not self._samples:
            return {}
        def stats(vals):
            return {"mean": round(statistics.mean(vals), 1), "max": round(max(vals), 1)}
        cpu = [s["cpu"] for s in self._samples]
        mem = [s["mem_used_gb"] for s in self._samples]
        mem_pct = [s["mem_pct"] for s in self._samples]
        client = [s["client_mb"] for s in self._samples]
        r = {
            "cpu_pct": stats(cpu),
            "mem_gb": {"mean": round(statistics.mean(mem), 1), "peak": round(max(mem), 1)},
            "mem_pct_peak": round(max(mem_pct), 1),
            "client_mb": stats(client),
        }
        srv = [s["server_mb"] for s in self._samples if "server_mb" in s]
        if srv:
            r["server_mb"] = {"mean": round(statistics.mean(srv), 1), "peak": round(max(srv), 1)}
        subs = [s["subprocs"] for s in self._samples if "subprocs" in s]
        if subs:
            r["subproc_peak"] = max(subs)
        return r


def lat_stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "p50": round(s[len(s)//2], 3),
        "p90": round(s[int(len(s)*0.9)], 3),
        "p99": round(s[int(len(s)*0.99)], 3),
        "max": round(max(s), 3),
        "mean": round(statistics.mean(s), 3),
    }


async def run_episode(
    scenario: str, task_idx: int,
    reset_sem: asyncio.Semaphore,
    counters: dict,
) -> SessionResult:
    r = SessionResult()
    t_start = time.monotonic()
    env = AWMEnv(base_url=BASE_URL, message_timeout_s=CLIENT_TIMEOUT, connect_timeout_s=CONNECT_TIMEOUT)
    phase = "init"
    try:
        async with reset_sem:
            phase = "connect"
            t0 = time.monotonic()
            await env.connect()
            r.connect_s = time.monotonic() - t0

            phase = "reset"
            t0 = time.monotonic()
            result = await env.reset(scenario=scenario, task_idx=task_idx)
            r.reset_s = time.monotonic() - t0
            if result.observation.reward_type not in ("reset_ok", "reset_warning"):
                r.error = f"reset: {result.observation.error}"
                counters["fail"] += 1
                counters["done"] += 1
                return r
            counters["resets"] += 1

        phase = "list_tools"
        t0 = time.monotonic()
        result = await env.step(ListToolsAction())
        r.list_tools_s = time.monotonic() - t0
        tools = []
        obs = result.observation
        if hasattr(obs, "tools") and obs.tools:
            tools = [t.get("name", "") for t in obs.tools if isinstance(t, dict)]
        if not tools:
            tools = ["unknown"]

        for turn in range(TURNS_PER_EPISODE):
            phase = f"turn_{turn}"
            await asyncio.sleep(random.uniform(LLM_THINK_MIN, LLM_THINK_MAX))
            t0 = time.monotonic()
            try:
                await env.step(CallToolAction(tool_name=random.choice(tools), arguments={}))
            except Exception:
                pass
            r.tool_call_latencies.append(time.monotonic() - t0)

        phase = "done"
        t0 = time.monotonic()
        await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
        r.done_s = time.monotonic() - t0
        r.success = True
        counters["ok"] += 1
    except Exception as e:
        r.error = f"[{phase}] {type(e).__name__}: {str(e)[:120]}"
        counters["fail"] += 1
    finally:
        r.total_s = time.monotonic() - t_start
        counters["done"] += 1
        try:
            await env.close()
        except Exception:
            pass
    return r


async def progress(counters, total, tracker, interval=15.0):
    start = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        e = time.monotonic() - start
        last = tracker._samples[-1] if tracker._samples else {}
        log.info(
            f"  [{e:.0f}s] done={counters['done']}/{total} "
            f"ok={counters['ok']} fail={counters['fail']} "
            f"resets={counters['resets']} | "
            f"cpu={last.get('cpu','?')}% mem={last.get('mem_used_gb','?')}GB "
            f"server={last.get('server_mb','?')}MB subs={last.get('subprocs','?')}"
        )


async def find_server_pid() -> int | None:
    from urllib.parse import urlparse
    port = str(urlparse(BASE_URL).port or "8899")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "uvicorn" in cmdline and port in cmdline:
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


async def run_scale(scale: int, concurrency: int) -> dict:
    """Run one scale test, return metrics dict."""
    log.info(f"{'='*70}")
    log.info(f"SCALE={scale}  concurrency={concurrency}  turns={TURNS_PER_EPISODE}")
    log.info(f"{'='*70}")

    # Verify server
    async with httpx.AsyncClient() as c:
        resp = await c.get(f"{BASE_URL}/docs", timeout=10)
        resp.raise_for_status()

    server_pid = await find_server_pid()
    tracker = ResourceTracker(server_pid)
    tracker.start()

    counters = {"done": 0, "ok": 0, "fail": 0, "resets": 0}
    reset_sem = asyncio.Semaphore(concurrency)

    prog = asyncio.create_task(progress(counters, scale, tracker))
    wall_start = time.monotonic()

    tasks = []
    for i in range(scale):
        sc = SCENARIOS[i % len(SCENARIOS)]
        tasks.append(run_episode(sc, i % 10, reset_sem, counters))

    results = await asyncio.gather(*tasks)
    wall_s = time.monotonic() - wall_start

    prog.cancel()
    try:
        await prog
    except asyncio.CancelledError:
        pass
    await tracker.stop()

    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    res = tracker.summary()

    metrics = {
        "scale": scale,
        "concurrency": concurrency,
        "ok": len(ok),
        "fail": len(failed),
        "fail_pct": round(len(failed) / scale * 100, 1),
        "wall_s": round(wall_s, 1),
        "connect": lat_stats([r.connect_s for r in ok]),
        "reset": lat_stats([r.reset_s for r in ok]),
        "list_tools": lat_stats([r.list_tools_s for r in ok]),
        "tool_call": lat_stats([lat for r in ok for lat in r.tool_call_latencies]),
        "done": lat_stats([r.done_s for r in ok]),
        "episode_total": lat_stats([r.total_s for r in ok]),
        "resources": res,
    }

    log.info(f"  OK={len(ok)}/{scale} fail={len(failed)} wall={wall_s:.1f}s")
    log.info(f"  tool_call: {json.dumps(metrics['tool_call'])}")
    log.info(f"  resources: {json.dumps(res)}")
    if failed:
        # Show first few failure types
        err_types: dict[str, int] = {}
        for r in failed:
            key = (r.error or "unknown")[:60]
            err_types[key] = err_types.get(key, 0) + 1
        for err, cnt in sorted(err_types.items(), key=lambda x: -x[1])[:3]:
            log.info(f"  failure: {err} (x{cnt})")

    return metrics


async def main():
    scales = [2000, 3000, 4000, 5000, 6000, 8000]
    concurrency = 128  # reset concurrency, fixed

    log.info(f"Scale benchmark: {scales}")
    log.info(f"Machine: {psutil.cpu_count()} cores, {psutil.virtual_memory().total // (1024**3)}GB RAM")

    all_metrics = []

    for scale in scales:
        # Check memory before starting
        mem_pct = psutil.virtual_memory().percent
        if mem_pct > 80:
            log.warning(f"Memory at {mem_pct}%, waiting 30s for cleanup...")
            await asyncio.sleep(30)
            mem_pct = psutil.virtual_memory().percent
            if mem_pct > 85:
                log.error(f"Memory still at {mem_pct}%, stopping benchmark.")
                break

        metrics = await run_scale(scale, concurrency)
        all_metrics.append(metrics)

        # Stop if failure rate too high
        if metrics["fail_pct"] > 5:
            log.warning(f"Failure rate {metrics['fail_pct']}% > 5%, stopping.")
            break

        # Stop if memory peaked too high
        mem_peak = metrics["resources"].get("mem_pct_peak", 0)
        if mem_peak > 92:
            log.warning(f"Memory peaked at {mem_peak}%, stopping.")
            break

        # Cool down between scales
        if scale != scales[-1]:
            log.info("Cooling down 15s...")
            await asyncio.sleep(15)

    # Final summary table
    log.info("")
    log.info("=" * 100)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 100)
    header = (
        f"{'scale':>6s} {'ok':>5s} {'fail':>5s} {'fail%':>6s} {'wall':>7s} "
        f"{'tc_p50':>7s} {'tc_p90':>7s} {'tc_p99':>7s} "
        f"{'mem_avg':>8s} {'mem_peak':>9s} {'srv_peak':>9s} {'subs':>5s} "
        f"{'cpu_avg':>7s}"
    )
    log.info(header)
    log.info("-" * len(header))
    for m in all_metrics:
        tc = m["tool_call"]
        res = m["resources"]
        log.info(
            f"{m['scale']:>6d} {m['ok']:>5d} {m['fail']:>5d} {m['fail_pct']:>5.1f}% "
            f"{m['wall_s']:>6.0f}s "
            f"{tc.get('p50','-'):>7s} {tc.get('p90','-'):>7s} {tc.get('p99','-'):>7s} "
            f"{res.get('mem_gb',{}).get('mean','-'):>7s}G "
            f"{res.get('mem_gb',{}).get('peak','-'):>8s}G "
            f"{res.get('server_mb',{}).get('peak','-'):>8s}M "
            f"{res.get('subproc_peak','-'):>5s} "
            f"{res.get('cpu_pct',{}).get('mean','-'):>6s}%"
        )
    log.info("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
