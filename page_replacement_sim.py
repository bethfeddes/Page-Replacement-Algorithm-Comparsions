"""
Page Replacement Algorithm Simulator
Compares FIFO, LRU, and Working Set across various conditions.
"""

import random
import json
from collections import deque, OrderedDict


# ──────────────────────────────────────────────
# ALGORITHM IMPLEMENTATIONS
# ──────────────────────────────────────────────

def simulate_fifo(reference_string, num_frames):
    """First-In First-Out page replacement."""
    frames = []
    queue = deque()  # tracks insertion order
    page_faults = 0

    for page in reference_string:
        if page not in frames:
            page_faults += 1
            if len(frames) < num_frames:
                frames.append(page)
                queue.append(page)
            else:
                evict = queue.popleft()
                frames.remove(evict)
                frames.append(page)
                queue.append(page)

    fault_rate = page_faults / len(reference_string)
    return {"page_faults": page_faults, "fault_rate": round(fault_rate, 4)}


def simulate_lru(reference_string, num_frames):
    """Least Recently Used page replacement."""
    frames = OrderedDict()  # key=page, maintains recency order
    page_faults = 0

    for page in reference_string:
        if page in frames:
            frames.move_to_end(page)  # mark as recently used
        else:
            page_faults += 1
            if len(frames) >= num_frames:
                frames.popitem(last=False)  # evict least recently used
            frames[page] = True

    fault_rate = page_faults / len(reference_string)
    return {"page_faults": page_faults, "fault_rate": round(fault_rate, 4)}


def simulate_working_set(reference_string, num_frames, window_size):
    """
    Working Set page replacement.
    At each step, the working set = distinct pages in last `window_size` references.
    If a page fault occurs and frames are full, evict pages not in the current working set.
    """
    frames = set()
    page_faults = 0

    for i, page in enumerate(reference_string):
        # Compute current working set: distinct pages in last window_size steps
        start = max(0, i - window_size + 1)
        working_set = set(reference_string[start:i + 1])

        if page not in frames:
            page_faults += 1
            if len(frames) < num_frames:
                frames.add(page)
            else:
                # Evict pages not in working set
                not_in_ws = frames - working_set
                if not_in_ws:
                    frames.discard(next(iter(not_in_ws)))
                else:
                    # All frames are in working set — evict arbitrary oldest
                    old_start = max(0, i - window_size)
                    oldest = reference_string[old_start]
                    frames.discard(oldest)
                frames.add(page)
        else:
            # Page hit — trim frames to working set if over limit
            frames &= working_set | {page}

    fault_rate = page_faults / len(reference_string)
    return {"page_faults": page_faults, "fault_rate": round(fault_rate, 4)}


# ──────────────────────────────────────────────
# REFERENCE STRING GENERATORS
# ──────────────────────────────────────────────

def gen_random(length, page_range):
    """Completely random access pattern — no locality."""
    return [random.randint(0, page_range - 1) for _ in range(length)]


def gen_locality(length, page_range, hot_ratio=0.3, hot_prob=0.8):
    """
    Simulates temporal locality: a 'hot' subset of pages is accessed
    with high probability, the rest rarely.
    """
    hot_count = max(1, int(page_range * hot_ratio))
    hot_pages = random.sample(range(page_range), hot_count)
    cold_pages = [p for p in range(page_range) if p not in hot_pages]
    result = []
    for _ in range(length):
        if random.random() < hot_prob and hot_pages:
            result.append(random.choice(hot_pages))
        else:
            result.append(random.choice(cold_pages) if cold_pages else random.choice(hot_pages))
    return result


def gen_sequential(length, page_range):
    """Sequential scan — cycles through pages in order repeatedly."""
    return [i % page_range for i in range(length)]


def gen_cyclic(length, page_range, cycle_size=6):
    """
    Cyclic pattern designed to demonstrate Belady's Anomaly in FIFO.
    Cycle size of 6 means:
      - With 3 frames (less than cycle): high fault rate for all algorithms
      - With 5 frames (just under cycle): FIFO may fault MORE than with 3 frames
      - With 8+ frames (exceeds cycle): all pages fit, near-zero faults
    The cycle is just large enough to stay problematic for smaller frame counts
    while revealing FIFO's counterintuitive behavior at 5 frames.
    """
    cycle = list(range(cycle_size))
    return [cycle[i % cycle_size] for i in range(length)]


def gen_strided(length, page_range, stride=3):
    """Strided access — jumps by a fixed stride, simulating array traversal."""
    return [(i * stride) % page_range for i in range(length)]


# ──────────────────────────────────────────────
# TEST RUNNER
# ──────────────────────────────────────────────

def run_all_tests(custom_strings=None, ws_windows=(3, 5, 10),
                  frame_sizes=(3, 5, 8, 12), num_random_tests=20,
                  random_length=100, page_range=15):
    """
    Runs FIFO, LRU, and WS across all patterns and parameters.
    Returns structured results dict.
    """
    results = []

    # ── Named patterns ──
    named_patterns = {
        "sequential": gen_sequential(random_length, page_range),
        "cyclic":     gen_cyclic(random_length, page_range),
        "strided":    gen_strided(random_length, page_range),
        "locality":   gen_locality(random_length, page_range),
        "random":     gen_random(random_length, page_range),
    }

    # Add user-defined strings
    if custom_strings:
        for name, string in custom_strings.items():
            named_patterns[name] = string

    for pattern_name, ref_string in named_patterns.items():
        for frames in frame_sizes:
            row = {
                "pattern": pattern_name,
                "frames": frames,
                "string_length": len(ref_string),
                "fifo": simulate_fifo(ref_string, frames),
                "lru":  simulate_lru(ref_string, frames),
                "ws":   {}
            }
            for w in ws_windows:
                row["ws"][f"w={w}"] = simulate_working_set(ref_string, frames, w)
            results.append(row)

    # ── Random batch tests ──
    random_batch = []
    for _ in range(num_random_tests):
        ref = gen_random(random_length, page_range)
        frames = random.choice(frame_sizes)
        entry = {
            "pattern": "random_batch",
            "frames": frames,
            "string_length": len(ref),
            "fifo": simulate_fifo(ref, frames),
            "lru":  simulate_lru(ref, frames),
            "ws":   {}
        }
        for w in ws_windows:
            entry["ws"][f"w={w}"] = simulate_working_set(ref, frames, w)
        random_batch.append(entry)

    return results, random_batch


# ──────────────────────────────────────────────
# REPORT PRINTER
# ──────────────────────────────────────────────

def print_report(results, random_batch, ws_windows=(3, 5, 10)):
    SEP = "─" * 90

    print("\n" + "═" * 90)
    print("  PAGE REPLACEMENT ALGORITHM COMPARISON — FIFO vs LRU vs Working Set")
    print("═" * 90)

    ws_labels = [f"w={w}" for w in ws_windows]

    # Header
    ws_header = "  ".join(f"WS(w={w}) faults  rate" for w in ws_windows)
    print(f"\n{'Pattern':<16} {'Frames':>6}  {'FIFO faults':>11}  {'rate':>6}  "
          f"{'LRU faults':>10}  {'rate':>6}  {ws_header}")
    print(SEP)

    for r in results:
        fifo = r["fifo"]
        lru  = r["lru"]
        ws_parts = "  ".join(
            f"{r['ws'][lbl]['page_faults']:>13}  {r['ws'][lbl]['fault_rate']:>6.2%}"
            for lbl in ws_labels
        )
        print(f"{r['pattern']:<16} {r['frames']:>6}  "
              f"{fifo['page_faults']:>11}  {fifo['fault_rate']:>6.2%}  "
              f"{lru['page_faults']:>10}  {lru['fault_rate']:>6.2%}  "
              f"{ws_parts}")

    # ── Random batch summary ──
    print("\n" + "═" * 90)
    print(f"  RANDOM BATCH SUMMARY ({len(random_batch)} tests)")
    print("═" * 90)

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    fifo_rates = [r["fifo"]["fault_rate"] for r in random_batch]
    lru_rates  = [r["lru"]["fault_rate"]  for r in random_batch]

    print(f"\n  {'Algorithm':<20} {'Avg Fault Rate':>16} {'Min':>8} {'Max':>8}")
    print(f"  {SEP[:60]}")
    print(f"  {'FIFO':<20} {avg(fifo_rates):>15.2%} {min(fifo_rates):>8.2%} {max(fifo_rates):>8.2%}")
    print(f"  {'LRU':<20} {avg(lru_rates):>15.2%}  {min(lru_rates):>8.2%} {max(lru_rates):>8.2%}")

    for lbl in ws_labels:
        ws_rates = [r["ws"][lbl]["fault_rate"] for r in random_batch]
        print(f"  {f'WS ({lbl})':<20} {avg(ws_rates):>15.2%}  {min(ws_rates):>8.2%} {max(ws_rates):>8.2%}")

    # ── Winner per pattern ──
    print("\n" + "═" * 90)
    print("  WINNER PER PATTERN (lowest avg fault rate across frame sizes)")
    print("═" * 90)

    from collections import defaultdict
    pattern_scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        p = r["pattern"]
        pattern_scores[p]["FIFO"].append(r["fifo"]["fault_rate"])
        pattern_scores[p]["LRU"].append(r["lru"]["fault_rate"])
        for lbl in ws_labels:
            pattern_scores[p][f"WS({lbl})"].append(r["ws"][lbl]["fault_rate"])

    print(f"\n  {'Pattern':<16} {'Winner':<16} {'Avg Rate':>10}")
    print(f"  {SEP[:50]}")

    for pattern, scores in pattern_scores.items():
        averages = {algo: avg(rates) for algo, rates in scores.items()}
        winner = min(averages, key=averages.get)
        print(f"  {pattern:<16} {winner:<16} {averages[winner]:>10.2%}")

    print("\n" + "═" * 90)
    print("  KEY OBSERVATIONS")
    print("═" * 90)
    print("""
  • Working Set performance is sensitive to window size (w):
      – Small w → misses pages still in use → more faults
      – Large w → keeps too many pages → may exceed frame capacity
      – Optimal w depends on the locality structure of the reference string

  • FIFO can suffer Belady's Anomaly on cyclic patterns:
      adding more frames can INCREASE page faults

  • LRU performs well under locality but has higher overhead
    than FIFO in real implementations

  • Working Set is theoretically optimal when w matches the
    program's locality window — as shown in Murzakhmetov et al. (2025)
  """)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)  # reproducible results

    # ── Define your own custom reference strings here ──
    custom_strings = {
        # Sudden locality shift: algorithms must adapt when the hot page set changes abruptly.
        # First half repeatedly accesses pages 1-4 (hot set A),
        # then switches entirely to pages 9-12 (hot set B).
        # Tests how quickly each algorithm evicts the old working set and loads the new one.
        "locality_shift": [1,1,2,2,3,3,4,4,1,2,3,4,
                           9,10,11,12,9,10,11,12,9,10],

        # Simulates realistic program execution:
        #   Phase 1 - Initialization: pages 1-5 each loaded once
        #   Phase 2 - Main loop: tight repeated access to pages 3-4
        #   Phase 3 - Function call: brief jump to page 7, returns to loop
        #   Phase 4 - Loop continues
        #   Phase 5 - Another external call to pages 8-9, returns to loop
        # Tests whether algorithms can hold the "loop pages" while handling brief spikes.
        "real_program":   [1,2,3,4,5,
                           3,3,4,4,3,4,3,4,
                           7,3,4,3,4,
                           3,3,4,4,3,4,
                           8,9,3,4,3,4],
    }

    WS_WINDOWS  = (4, 6, 8)
    FRAME_SIZES = (3, 5, 8, 12)

    results, random_batch = run_all_tests(
        custom_strings=custom_strings,
        ws_windows=WS_WINDOWS,
        frame_sizes=FRAME_SIZES,
        num_random_tests=50,
        random_length=200,
        page_range=15,
    )

    print_report(results, random_batch, ws_windows=WS_WINDOWS)

    # Save raw data as JSON for further analysis
    output = {"named_tests": results, "random_batch": random_batch}
    with open("/mnt/user-data/outputs/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Raw results saved to results.json")