"""
Test max session capacity + idle timeout freeing session slots.

The IdleWebSocketMiddleware closes WebSocket connections that are idle
for longer than max_idle_time, which frees the session slot in http_server.

Scenario:
    max_concurrent_envs=10, max_idle_time=10s
    1. Create 8 sessions (8/10 slots)
    2. Interact with 6, leave 2 idle
    3. Wait for idle timeout to close the 2 idle WebSockets (~10s)
    4. Create 4 new sessions (6 + 4 = 10 slots) — should succeed

Usage:
    # Terminal 1:
    MAX_CONCURRENT_ENVS=10 MAX_IDLE_TIME=10 pueue add -- /path/to/run_server.sh

    # Terminal 2:
    PYTHONPATH=OpenEnv/src:OpenEnv/envs uv run python \
        OpenEnv/envs/agent_world_model_env/example_usage_capacity.py
"""

import asyncio
import time

import httpx
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv

BASE_URL = "http://localhost:8899"


async def get_stats() -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/stats")
        return resp.json()


async def create_session(label: str, task_idx: int) -> AWMEnv | None:
    env = AWMEnv(base_url=BASE_URL, message_timeout_s=300)
    try:
        await env.connect()
        result = await env.reset(scenario="e_commerce_33", task_idx=task_idx)
        if result.observation.task is None:
            print(f"  [{label}] Reset failed: {result.observation.error}")
            await env.close()
            return None
        print(f"  [{label}] Created OK (tools={result.observation.num_tools})")
        return env
    except Exception as e:
        print(f"  [{label}] Failed: {type(e).__name__}: {str(e)[:120]}")
        try:
            await env.close()
        except Exception:
            pass
        return None


async def keep_alive(envs: dict[str, AWMEnv], labels: list[str]):
    """Send a list_tools to keep sessions active."""
    for label in labels:
        if label in envs:
            try:
                await envs[label].step(ListToolsAction())
            except Exception:
                pass


async def main():
    stats = await get_stats()
    print(f"Server: max_idle_time={stats['default_max_idle_time']}s")
    idle_time = stats["default_max_idle_time"]
    assert idle_time <= 15, (
        f"max_idle_time should be <=15s for this test, got {idle_time}. "
        f"Start server with: MAX_IDLE_TIME=10"
    )
    print()

    # =================================================================
    # Phase 1: Create 8 sessions sequentially
    # =================================================================
    print("=" * 70)
    print("PHASE 1: Create 8 sessions (8/10 slots)")
    print("=" * 70)

    envs: dict[str, AWMEnv] = {}
    for i in range(8):
        label = f"s{i}"
        env = await create_session(label, task_idx=i)
        if env:
            envs[label] = env
        # Keep all existing sessions alive while creating new ones
        # (sequential creation can take longer than idle_timeout)
        await keep_alive(envs, list(envs.keys()))

    stats = await get_stats()
    print(
        f"\n  sessions={stats['total_sessions']}, subprocesses={stats['active_subprocesses']}"
    )
    assert stats["total_sessions"] == 8, (
        f"Expected 8 sessions, got {stats['total_sessions']}"
    )
    print()

    active_labels = [f"s{i}" for i in range(6)]
    idle_labels = ["s6", "s7"]

    # =================================================================
    # Phase 2: Interact with s0-s5, leave s6+s7 idle
    # Keep s0-s5 alive while waiting for s6+s7 to timeout
    # =================================================================
    print("=" * 70)
    print("PHASE 2: Keep s0-s5 active, wait for s6+s7 idle timeout")
    print("=" * 70)

    # Interact with s0-s5 immediately to mark them active
    await keep_alive(envs, active_labels)
    print(f"  Interacted with {active_labels}")
    print(f"  Idle (no interaction): {idle_labels}")
    print(f"  Waiting for idle timeout + middleware disconnect...")

    t0 = time.time()
    slots_freed = False
    for tick in range(10):  # up to 50s
        await asyncio.sleep(5)
        elapsed = time.time() - t0

        # Keep s0-s5 alive
        await keep_alive(envs, active_labels)

        stats = await get_stats()
        print(
            f"  t+{elapsed:.0f}s: sessions={stats['total_sessions']}, "
            f"subprocesses={stats['active_subprocesses']}"
        )

        if stats["total_sessions"] <= 6:
            print(f"  -> Idle sessions disconnected! (slots freed)")
            slots_freed = True
            break

    if not slots_freed:
        print("  FAIL: Idle sessions were not disconnected in time")
        # Cleanup and exit
        for env in envs.values():
            try:
                await env.close()
            except Exception:
                pass
        return

    print()

    # =================================================================
    # Phase 3: Create 4 new sessions (should succeed: 6 + 4 = 10)
    # =================================================================
    print("=" * 70)
    print("PHASE 3: Create 4 new sessions (6 active + 4 new = 10 max)")
    print("=" * 70)

    # Remove closed idle envs from our tracking
    for label in idle_labels:
        envs.pop(label, None)

    new_envs: dict[str, AWMEnv] = {}
    for i in range(4):
        label = f"new{i}"
        env = await create_session(label, task_idx=i)
        if env:
            new_envs[label] = env
        # Keep ALL sessions alive during creation
        await keep_alive(envs, active_labels)
        await keep_alive(new_envs, list(new_envs.keys()))

    stats = await get_stats()
    print(
        f"\n  Created {len(new_envs)}/4 new sessions, "
        f"total_sessions={stats['total_sessions']}"
    )
    assert len(new_envs) == 4, f"Expected 4 new sessions, got {len(new_envs)}"
    assert stats["total_sessions"] == 10, (
        f"Expected 10 total, got {stats['total_sessions']}"
    )
    print("  SUCCESS: All 4 new sessions created within capacity limit")
    print()

    # =================================================================
    # Cleanup
    # =================================================================
    print("=" * 70)
    print("CLEANUP")
    print("=" * 70)

    for label, env in {**envs, **new_envs}.items():
        try:
            await env.step(
                CallToolAction(tool_name="done", arguments={"keep_session": False})
            )
        except Exception:
            pass
        try:
            await env.close()
        except Exception:
            pass

    await asyncio.sleep(1)
    stats = await get_stats()
    print(f"  Final: {stats}")

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
