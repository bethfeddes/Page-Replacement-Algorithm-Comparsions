import random
import matplotlib.pyplot as plt
from collections import deque


# =========================================================
# GLOBAL EXPERIMENT PARAMETERS
# =========================================================
PHASE_LEN = 300
NUM_PHASES = 60
N = PHASE_LEN * NUM_PHASES  # 18,000 references

WS_SIZE = 12
UNIVERSE = 400
NOISE = 0.02

MEMORY_LIMIT = 50  # Graphs and printed averages only use avg memory <= 50


# =========================================================
# FIFO PAGE REPLACEMENT
# =========================================================
def fifo_sim(refs, frame_limit):
    memory = set()
    queue = deque()
    faults = 0
    mem_sum = 0

    for page in refs:
        if page not in memory:
            faults += 1

            if len(memory) >= frame_limit:
                victim = queue.popleft()
                memory.remove(victim)

            memory.add(page)
            queue.append(page)

        mem_sum += len(memory)

    return {
        "faults": faults,
        "fault_rate": faults / len(refs),
        "avg_memory": mem_sum / len(refs)
    }


# =========================================================
# LRU PAGE REPLACEMENT
# =========================================================
def lru_sim(refs, frame_limit):
    memory = set()
    last_used = {}
    faults = 0
    time = 0
    mem_sum = 0

    for page in refs:
        time += 1

        if page not in memory:
            faults += 1

            if len(memory) >= frame_limit:
                victim = min(memory, key=lambda p: last_used[p])
                memory.remove(victim)

            memory.add(page)

        last_used[page] = time
        mem_sum += len(memory)

    return {
        "faults": faults,
        "fault_rate": faults / len(refs),
        "avg_memory": mem_sum / len(refs)
    }


# =========================================================
# TRUE WORKING SET MODEL
# =========================================================
def working_set_sim(refs, delta):
    """
    Denning-style Working Set:
    W(t, Δ) = pages referenced in the last Δ references.

    delta is measured in references, not frames.
    Working Set does not use a fixed frame limit.
    """
    memory = set()
    last_used = {}
    faults = 0
    time = 0
    mem_sum = 0

    for page in refs:
        time += 1

        # Remove pages outside W(t, Δ)
        for page_in_memory in list(memory):
            if time - last_used[page_in_memory] > delta:
                memory.remove(page_in_memory)

        if page not in memory:
            faults += 1

        memory.add(page)
        last_used[page] = time

        mem_sum += len(memory)

    return {
        "faults": faults,
        "fault_rate": faults / len(refs),
        "avg_memory": mem_sum / len(refs)
    }


# =========================================================
# WORKLOADS / TEST CASES
# =========================================================
def stable_locality_pattern(
    n=N,
    working_set_size=WS_SIZE,
    universe=UNIVERSE,
    noise=NOISE
):
    """
    Test Case 1: Stable locality.

    Most references come from one hot working set.
    Occasional references go outside the hot set.

    Hot pages are offset to avoid overlap with other workload page ranges.
    The offset does not affect algorithm behavior.
    """
    hot_pages = list(range(500, 500 + working_set_size))
    cold_pages = list(range(0, universe))

    refs = []

    for _ in range(n):
        if random.random() < noise:
            refs.append(random.choice(cold_pages))
        else:
            refs.append(random.choice(hot_pages))

    return refs


def recurring_locality_shift_pattern(
    n=N,
    phase_len=PHASE_LEN,
    ws_size=WS_SIZE,
    universe=UNIVERSE,
    noise=NOISE
):
    """
    Test Case 2: Recurring locality shift.

    Pattern:
        A -> B -> A -> C -> A -> B -> A -> D

    Each phase lasts phase_len references.
    Each phase uses ws_size active pages.
    """
    phase_bases = [0, 100, 0, 200, 0, 100, 0, 300]

    refs = []
    phase_index = 0

    while len(refs) < n:
        base = phase_bases[phase_index % len(phase_bases)]
        current_ws = list(range(base, base + ws_size))

        for _ in range(phase_len):
            if random.random() < noise:
                refs.append(random.randint(0, universe - 1))
            else:
                refs.append(random.choice(current_ws))

            if len(refs) >= n:
                break

        phase_index += 1

    return refs[:n]


def random_pattern(n=N, universe=UNIVERSE):
    """
    Test Case 3: Random access.

    No intentional locality.
    This acts as a baseline where replacement policy should matter less.
    """
    return [random.randint(0, universe - 1) for _ in range(n)]


# =========================================================
# WORKING SET DELTA SELECTION
# =========================================================
def workload_based_deltas(workload_type, phase_len=PHASE_LEN):
    """
    Chooses Working Set Δ values based on workload structure.

    Δ is measured in references, not frames.
    These ranges include small, medium, and larger windows so the
    Working Set curve does not collapse into one cluster.
    """

    small_deltas = list(range(1, 31))

    if workload_type == "stable_locality":
        # Stable locality has a 12-page hot set with 2% noise.
        # Larger Δ values show how occasional noise pages increase memory usage.
        return sorted(set(
            small_deltas +
            list(range(50, 601, 50)) +
            list(range(700, 1501, 100))
        ))

    elif workload_type == "locality_shift":
        # Phase length is 300 references.
        # Test sub-phase, one-phase, two-phase, three-phase, and four-phase windows.
        step = max(1, phase_len // 12)
        return sorted(set(
            small_deltas +
            list(range(step, phase_len * 4 + 1, step))
        ))

    elif workload_type == "random":
        # Random has no locality, so Working Set memory rises quickly with Δ.
        # Use a broad range so the curve is visible.
        return sorted(set(
            small_deltas +
            list(range(50, 1201, 50))
        ))

    else:
        step = max(1, phase_len // 12)
        return sorted(set(
            small_deltas +
            list(range(step, phase_len * 4 + 1, step))
        ))


# =========================================================
# EXPERIMENT 1:
# FIFO/LRU FRAME SIZE VS FAULT RATE
# =========================================================
def plot_fifo_lru_frame_fault(refs, title, max_frames=20):
    frames = list(range(2, max_frames + 1))
    fifo_rates = []
    lru_rates = []

    for frame_size in frames:
        fifo = fifo_sim(refs, frame_size)
        lru = lru_sim(refs, frame_size)

        fifo_rates.append(fifo["fault_rate"])
        lru_rates.append(lru["fault_rate"])

    plt.figure(figsize=(10, 6))
    plt.plot(frames, fifo_rates, marker="o", markersize=3, label="FIFO")
    plt.plot(frames, lru_rates, marker="o", markersize=3, label="LRU")

    plt.xlabel("Frame Size")
    plt.ylabel("Page Fault Rate")
    plt.title(title)
    plt.xticks(range(2, max_frames + 1, 2))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# EXPERIMENT 2:
# FIFO/LRU/WORKING SET FAULT RATE VS AVERAGE MEMORY
# =========================================================
def fifo_lru_memory_curves(refs, max_frames=80, memory_limit=MEMORY_LIMIT):
    """
    FIFO and LRU use fixed frame sizes.

    Only points with average memory usage <= memory_limit are returned.
    This keeps the plotted graph consistent with the printed averages.
    """
    fifo_points = []
    lru_points = []

    for frame_size in range(2, max_frames + 1):
        fifo = fifo_sim(refs, frame_size)
        lru = lru_sim(refs, frame_size)

        if fifo["avg_memory"] <= memory_limit:
            fifo_points.append((fifo["avg_memory"], fifo["fault_rate"]))

        if lru["avg_memory"] <= memory_limit:
            lru_points.append((lru["avg_memory"], lru["fault_rate"]))

    return fifo_points, lru_points


def working_set_memory_curve(refs, workload_type, memory_limit=MEMORY_LIMIT):
    """
    Working Set uses Δ values, not frames.

    Only points with average memory usage <= memory_limit are returned.
    This keeps the plotted graph consistent with the printed averages.
    """
    ws_points = []

    for delta in workload_based_deltas(workload_type):
        ws = working_set_sim(refs, delta)

        if ws["avg_memory"] <= memory_limit:
            ws_points.append((ws["avg_memory"], ws["fault_rate"]))

    return ws_points


def plot_fault_vs_memory(
    refs,
    title,
    workload_type,
    max_frames_for_fifo_lru=80,
    memory_limit=MEMORY_LIMIT
):
    fifo_points, lru_points = fifo_lru_memory_curves(
        refs,
        max_frames=max_frames_for_fifo_lru,
        memory_limit=memory_limit
    )

    ws_points = working_set_memory_curve(
        refs,
        workload_type=workload_type,
        memory_limit=memory_limit
    )

    fifo_points.sort()
    lru_points.sort()
    ws_points.sort()

    if not fifo_points or not lru_points or not ws_points:
        print(f"Could not plot {title}: missing data within memory limit.")
        return

    fifo_memory, fifo_fault = zip(*fifo_points)
    lru_memory, lru_fault = zip(*lru_points)
    ws_memory, ws_fault = zip(*ws_points)

    plt.figure(figsize=(10, 6))
    plt.plot(fifo_memory, fifo_fault, marker="o", markersize=3, label="FIFO")
    plt.plot(lru_memory, lru_fault, marker="o", markersize=3, label="LRU")
    plt.plot(ws_memory, ws_fault, marker="o", markersize=3, label="Working Set")

    plt.xlabel("Average Memory Usage (pages)")
    plt.ylabel("Page Fault Rate")
    plt.title(title)
    plt.xlim(0, memory_limit)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# PERFORMANCE SUMMARY
# =========================================================
def print_summary(refs, workload_name, workload_type, memory_limit=MEMORY_LIMIT):
    """
    Prints one average summary table for each workload.

    Only results with average memory usage <= memory_limit are included.
    This matches graphs cropped to average memory usage of 50 pages.

    FIFO and LRU are tested across frame sizes k = 2 to 80.
    Working Set is tested across workload-specific delta values.
    """
    print(f"\n=== {workload_name} Average Summary ===")
    print(f"Total references: {len(refs)}")
    print(f"Only including results where average memory usage <= {memory_limit}")

    fifo_fault_rates = []
    fifo_avg_memories = []
    fifo_faults = []

    lru_fault_rates = []
    lru_avg_memories = []
    lru_faults = []

    for frame_size in range(2, 81):
        fifo = fifo_sim(refs, frame_size)
        lru = lru_sim(refs, frame_size)

        if fifo["avg_memory"] <= memory_limit:
            fifo_fault_rates.append(fifo["fault_rate"])
            fifo_avg_memories.append(fifo["avg_memory"])
            fifo_faults.append(fifo["faults"])

        if lru["avg_memory"] <= memory_limit:
            lru_fault_rates.append(lru["fault_rate"])
            lru_avg_memories.append(lru["avg_memory"])
            lru_faults.append(lru["faults"])

    ws_fault_rates = []
    ws_avg_memories = []
    ws_faults = []

    for delta in workload_based_deltas(workload_type):
        ws = working_set_sim(refs, delta)

        if ws["avg_memory"] <= memory_limit:
            ws_fault_rates.append(ws["fault_rate"])
            ws_avg_memories.append(ws["avg_memory"])
            ws_faults.append(ws["faults"])

    print()
    print(
        f"{'Algorithm':<15} | "
        f"{'Avg Fault Rate':>15} | "
        f"{'Avg Memory':>12} | "
        f"{'Avg Faults':>10} | "
        f"{'Points':>6}"
    )
    print("-" * 75)

    if fifo_fault_rates:
        print(
            f"{'FIFO':<15} | "
            f"{sum(fifo_fault_rates) / len(fifo_fault_rates):15.4f} | "
            f"{sum(fifo_avg_memories) / len(fifo_avg_memories):12.2f} | "
            f"{sum(fifo_faults) / len(fifo_faults):10.2f} | "
            f"{len(fifo_fault_rates):6d}"
        )
    else:
        print(f"{'FIFO':<15} | {'No points within memory limit':>56}")

    if lru_fault_rates:
        print(
            f"{'LRU':<15} | "
            f"{sum(lru_fault_rates) / len(lru_fault_rates):15.4f} | "
            f"{sum(lru_avg_memories) / len(lru_avg_memories):12.2f} | "
            f"{sum(lru_faults) / len(lru_faults):10.2f} | "
            f"{len(lru_fault_rates):6d}"
        )
    else:
        print(f"{'LRU':<15} | {'No points within memory limit':>56}")

    if ws_fault_rates:
        print(
            f"{'Working Set':<15} | "
            f"{sum(ws_fault_rates) / len(ws_fault_rates):15.4f} | "
            f"{sum(ws_avg_memories) / len(ws_avg_memories):12.2f} | "
            f"{sum(ws_faults) / len(ws_faults):10.2f} | "
            f"{len(ws_fault_rates):6d}"
        )
    else:
        print(f"{'Working Set':<15} | {'No points within memory limit':>56}")


# =========================================================
# RUN EXPERIMENTS FOR ONE WORKLOAD
# =========================================================
def run_workload_experiments(refs, workload_name, workload_type):
    print_summary(
        refs=refs,
        workload_name=workload_name,
        workload_type=workload_type,
        memory_limit=MEMORY_LIMIT
    )

    plot_fifo_lru_frame_fault(
        refs,
        title=f"{workload_name}: FIFO vs LRU — Frame Size vs Fault Rate",
        max_frames=20
    )

    plot_fault_vs_memory(
        refs,
        title=f"{workload_name}: Fault Rate vs Average Memory Usage",
        workload_type=workload_type,
        max_frames_for_fifo_lru=80,
        memory_limit=MEMORY_LIMIT
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    random.seed(42)

    test_cases = [
        {
            "name": "Stable Locality Pattern",
            "type": "stable_locality",
            "refs": stable_locality_pattern(
                n=N,
                working_set_size=WS_SIZE,
                universe=UNIVERSE,
                noise=NOISE
            )
        },
        {
            "name": "Recurring Locality Shift Pattern",
            "type": "locality_shift",
            "refs": recurring_locality_shift_pattern(
                n=N,
                phase_len=PHASE_LEN,
                ws_size=WS_SIZE,
                universe=UNIVERSE,
                noise=NOISE
            )
        },
        {
            "name": "Random Pattern",
            "type": "random",
            "refs": random_pattern(
                n=N,
                universe=UNIVERSE
            )
        }
    ]

    for test in test_cases:
        run_workload_experiments(
            refs=test["refs"],
            workload_name=test["name"],
            workload_type=test["type"]
        )
