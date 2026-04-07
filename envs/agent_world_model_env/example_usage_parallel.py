"""
Test parallel session behavior and idle cleanup.

Usage:
    # Terminal 1: Start server with short idle timeout for testing
    PYTHONPATH=src:envs uv run python -m awm_grpo.server \
        --max_concurrent_envs 100 --max_idle_time 30 --port 8899

    # Terminal 2: Run this test
    cd OpenEnv/envs/agent_world_model_env
    PYTHONPATH=../../src:.. uv run python example_usage_parallel.py
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


async def run_single_session(
    session_id: int, scenario: str, task_idx: int, max_idle_time: float | None = None
):
    """Run a single session: reset, list_tools, call a tool, verify, done."""
    env = AWMEnv(base_url=BASE_URL, message_timeout_s=120)
    try:
        await env.connect()

        reset_kwargs = {"scenario": scenario, "task_idx": task_idx}
        if max_idle_time is not None:
            reset_kwargs["max_idle_time"] = max_idle_time

        result = await env.reset(**reset_kwargs)
        if result.observation.task is None:
            print(f"  [session {session_id}] Reset failed: {result.observation.error}")
            return False

        print(
            f"  [session {session_id}] Reset OK: {scenario}/{task_idx}, tools={result.observation.num_tools}"
        )

        # list tools
        result = await env.step(ListToolsAction())
        num_tools = (
            len(result.observation.tools) if hasattr(result.observation, "tools") else 0
        )
        print(f"  [session {session_id}] Listed {num_tools} tools")

        # verify
        result = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": "test"},
            )
        )
        print(
            f"  [session {session_id}] Verify: reward_type={result.observation.reward_type}, reward={result.reward}"
        )

        # done
        result = await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": False})
        )
        print(f"  [session {session_id}] Done: {result.done}")
        return True
    except Exception as e:
        print(f"  [session {session_id}] Error ({type(e).__name__}): {e}")
        return False
    finally:
        try:
            await env.close()
        except Exception:
            pass


async def test_parallel_sessions():
    """Test 1: Run two batches of 4 sessions in parallel (8 total)."""
    print("=" * 70)
    print("TEST 1: Parallel sessions (2 batches × 4)")
    print("=" * 70)

    stats = await get_stats()
    print(f"Before: {stats}")

    total_success = 0
    for batch in range(2):
        start_idx = batch * 4
        start = time.time()
        tasks = [
            run_single_session(start_idx + i, "e_commerce_33", start_idx + i)
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start

        successes = sum(1 for r in results if r is True)
        total_success += successes
        print(f"  Batch {batch + 1}: {successes}/4 in {elapsed:.1f}s")

    print(f"\nTotal: {total_success}/8")

    stats = await get_stats()
    print(f"After: {stats}")
    print()


async def test_idle_cleanup():
    """Test 2: Create sessions, let them idle, check cleanup."""
    print("=" * 70)
    print("TEST 2: Idle cleanup (sessions with short max_idle_time)")
    print("=" * 70)

    # Create 4 sessions with 15s idle timeout, do reset but DON'T call done
    envs = []
    for i in range(4):
        env = AWMEnv(base_url=BASE_URL, message_timeout_s=120)
        await env.connect()
        result = await env.reset(scenario="e_commerce_33", task_idx=i, max_idle_time=15)
        if result.observation.task is not None:
            print(f"  [idle-{i}] Reset OK, task={result.observation.task[:60]}...")
            envs.append(env)
        else:
            print(f"  [idle-{i}] Reset failed: {result.observation.error}")
            await env.close()

    stats = await get_stats()
    print(f"\nAfter creating {len(envs)} idle sessions: {stats}")

    # Wait for idle cleanup (15s idle + 30s scan interval = ~45s worst case)
    print("\nWaiting for idle cleanup (checking every 10s)...")
    for wait_round in range(6):
        await asyncio.sleep(10)
        stats = await get_stats()
        print(f"  t+{(wait_round + 1) * 10}s: {stats}")
        if stats["active_subprocesses"] == 0:
            print("  All idle subprocesses cleaned up!")
            break

    # Now try to step on an idle-cleaned session — should get error
    if envs:
        print("\nStepping on idle-cleaned session (expect error)...")
        try:
            result = await envs[0].step(ListToolsAction())
            if hasattr(result.observation, "error") and result.observation.error:
                print(f"  Got expected error: {result.observation.error}")
            elif (
                hasattr(result.observation, "tools")
                and len(result.observation.tools) == 0
            ):
                print(f"  Got empty tools (subprocess was killed)")
            else:
                print(f"  Unexpected: got result with tools")
        except Exception as e:
            print(f"  Got expected exception: {e}")

    # Cleanup
    for env in envs:
        try:
            await env.close()
        except Exception:
            pass

    stats = await get_stats()
    print(f"\nAfter cleanup: {stats}")
    print()


async def test_many_sessions_sequential():
    """Test 3: Create many sessions sequentially to verify no leaks."""
    print("=" * 70)
    print("TEST 3: Sequential sessions (leak check)")
    print("=" * 70)

    stats_before = await get_stats()
    print(f"Before: {stats_before}")

    for i in range(10):
        env = AWMEnv(base_url=BASE_URL, message_timeout_s=120)
        await env.connect()
        result = await env.reset(scenario="e_commerce_33", task_idx=i % 10)
        if result.observation.task is not None:
            await env.step(ListToolsAction())
            await env.step(
                CallToolAction(tool_name="done", arguments={"keep_session": False})
            )
        await env.close()

    stats_after = await get_stats()
    print(f"After 10 sequential sessions: {stats_after}")

    leaked = stats_after["total_sessions"] - stats_before["total_sessions"]
    if leaked == 0:
        print("No session leaks detected!")
    else:
        print(f"WARNING: {leaked} sessions leaked")
    print()


async def main():
    print(f"Testing against {BASE_URL}")
    stats = await get_stats()
    print(f"Server stats: {stats}")
    print()

    await test_parallel_sessions()
    await test_idle_cleanup()
    await test_many_sessions_sequential()

    print("=" * 70)
    print("All tests completed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
