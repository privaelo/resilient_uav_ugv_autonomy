# air-ground-ops

A ROS 2 / Gazebo testbed for multi-robot informative path planning on a heterogeneous UAV–UGV team.

Allocation and information gathering are treated as one decision. The team maintains a posterior over target locations and scores assignments on how much expected undetected target mass they remove.

<!-- <img width="800" height="450" alt="MRTA_Hungarian1-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/4bfb70d4-baef-46f8-9a5a-67278a692008" /> -->
<img width="800" height="450" alt="MRTA_Demo2-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/656d0441-0bed-4bb2-9930-257ad6e9f9cc" />

---

## Question

Classical multi-robot task allocation assumes the task set is known. Something upstream has already decided where the work is. When targets must be found before they can be acted on, that assumption does unearned work: it separates the decision of where to look from the decision of who goes where.

This testbed removes it and measures what changes.

---

## Formulation

**Environment.** Targets are drawn from an inhomogeneous Poisson process over a 2D domain `D ⊂ R²` with latent intensity field `λ₀(l)`. The field is unknown to the team.

**Sensing.** Robot `i` at pose `x_i` detects a target at `l` with instantaneous detection rate

```text
γ_i(l ; x_i) = ρ_i · g_i(l ; x_i)
```

`ρ_i` is the platform's maximum detection capability. `g_i` is the geometric footprint term. UAV and UGV are parameterized asymmetrically: the UAV has a wide footprint and low `ρ`, the UGV a narrow footprint and high `ρ`.

**Belief.** Conditioned on no detection at `l` through time `t`, the posterior over *undetected* targets remains Poisson — thinning a Poisson process yields a Poisson process — with intensity

```text
λ_t(l) = λ₀(l) · exp( −∫₀ᵗ Σ_i γ_i(l ; x_i(τ)) dτ )
```

No target list is ever materialized. Targets are regions of posterior mass.

**Undetected mass.** At time `t`, the expected target mass the team has not observed:

```text
U(t) = ∫_D λ_t(l) dl
```

**Objective.** Minimize integrated undetected mass over the mission:

```text
J = ∫₀ᵀ U(t) dt
```

Search and confirmation are commensurable under this objective. Sweeping an empty region moves `J` little. Confirming a high-`λ` region moves it a lot. Re-sweeping an already-cleared region moves it not at all — the exponential has already decayed there — so there is no incentive to loiter on high-`λ` cells instead of covering ground.

**Discretization.** On an `H×W` grid with cell area `A`, the entire belief update is elementwise. No particle filter, no sampling:

```python
lam *= np.exp(-dt * gamma_sum)   # gamma_sum: H×W, summed over robots
U = lam.sum() * A
J += U * dt
```

---

## Allocation arms

All arms optimize on the same environment, the same sensing model, and are scored on the same `J`.

| Arm | Score | What it isolates | Status |
|---|---|---|---|
| A — Hungarian on thresholded task list | Euclidean distance to thresholded centroids | Standard decoupled practice | Runs, on ground-truth targets |
| B — Hungarian on `ΔJ` | `c_ij = −ΔJ_i(task_j)`, optimal matching | Information coupling only — same solver as A | Planned |
| C — Greedy on `ΔJ` | Marginal `ΔJ`, sequential | Myopic sequencing vs. optimal matching | Planned |

Arm B exists because A→C changes the solver and the information at the same time, so any gap between those two is unattributable. Holding the solver fixed and varying only what it is told is what isolates the claim.

The thresholding rule that converts a posterior into a task list is a design decision, not a detail. A weak threshold produces a strawman baseline. The rule is documented, held fixed across runs, and reported with a sensitivity sweep.

---

## Allocation policy

- **Horizon.** Myopic greedy.
- **Trigger.** Event-triggered on belief divergence, not fixed-rate replanning.
- **Commitment.** Assignments are divertible mid-transit. A UGV that disconfirms a region en route can be redirected before arrival.

**Approximation guarantees, by constraint class:**

| Setting | Constraint | Greedy bound |
|---|---|---|
| Static sensor placement, budget `k` | Cardinality | `1 − 1/e` |
| One-shot assignment, each robot ≤ 1 task | Partition matroid | `1/2` |
| Mobile sensors over trajectories | Routing / travel budget | No constant factor without further structure |

The mobile case is submodular orienteering. Whether a useful guarantee survives there is an open question, not a settled one.

---

## Metrics

- `J` — integrated undetected mass. Primary.
- Time to first detection, per target.
- Fraction of posterior mass resolved at mission end.
- Reallocation count. Divertible assignments raise a thrashing question that the count exposes.

---

## Status

The formulation above is the specification. What currently runs is the decoupled baseline that it will be measured against.

| Component | State |
|---|---|
| Sim bringup, 1 UAV + 3 UGVs, namespaced topics | Implemented |
| UAV target detection → broadcast | Implemented — Gazebo `LogicalCamera`, deterministic, over three static SDF targets |
| Hungarian allocation → UGV navigation | Implemented — Euclidean cost, solved once and cached; APF turn-then-drive controller |
| RViz assignment markers, terminal demo display | Implemented |
| Comms disruption layer (`clean` / `drop` / `delay` / `blackout`) | Implemented, runs standalone, not yet wired into the allocation path |
| Belief core — `λ` grid, thinning update, sensing model, seeded Poisson sampling | Implemented as a library (`ipp_core`), unit-tested, not yet wired into the sim |
| Mobile UAV, runtime target spawning, `J` logging | Not started |
| Arms B and C | Not started |
| Decentralized auction | Not started — no auction code exists in this repo |
| Learned allocator | Not started |

There is no posterior in the loop yet. `ipp_core` computes `λ_t`, `U(t)`, and `J` offline against a seeded intensity field, but nothing in the running system consumes it. Arm A allocates directly against detected target positions on a ground-truth list, with no posterior, no `J`, and no capability belief — which is why it is described as the baseline rather than as a working instance of the formulation.

---

## Platform

- ROS 2 Jazzy
- Gazebo Harmonic
- Ubuntu 24.04
- 1 UAV (static aerial observer; flight dynamics out of scope), 3 UGVs (diff-drive-equivalent, velocity-controlled)
- Python (`rclpy`) · URDF/Xacro + SDF

## Packages

| Package | Role |
|---|---|
| `multi_robot_bringup` | Top-level launch, world SDF, RViz config |
| `uav_description` | UAV URDF/Xacro + `robot_state_publisher` launch |
| `ugv_description` | UGV URDF (RViz) + SDF with VelocityControl + OdometryPublisher plugins |
| `ipp_core` | Belief core — intensity field, Poisson target sampling, sensing model, thinning posterior and `J`. Pure numpy, no `rclpy` |
| `uav_observer` | Target detection node — broadcasts detections on `/uav_1/targets`. Direct detection today; posterior-based detection is the next milestone |
| `task_allocator` | Hungarian allocator (blind baseline arm) + `CapabilityBelief`, an information-form Gaussian over `ρ_i` that is unit-tested and deliberately unwired (see Scope) |
| `ugv_nav` | Goal follower with potential-field obstacle avoidance, RViz marker node, demo display node |
| `comm_layer` | Comms disruption layer (clean / drop / delay / blackout). Implemented and runnable standalone; re-integration is the point at which coupled allocation gets tested against a posterior fragmented across the team |

---

## Running it

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

The belief core is verifiable without the simulator. Its tests assert the properties the
formulation rests on — that `U(t)` is monotone non-increasing under arbitrary motion, that a
stationary sensor's marginal return decays, and that a seed reproduces a target realization
exactly:

```bash
colcon test --packages-select ipp_core && colcon test-result --verbose
```

### Baseline sim (Gazebo only)

```bash
ros2 launch multi_robot_bringup simulation.launch.py use_rviz:=false
```

### Demo (Hungarian blind baseline)

Two terminals:

```bash
# Terminal 1
# Optionally add the flag 'paused:=true' if you don't want to start the simulation on launch
ros2 launch multi_robot_bringup simulation.launch.py \
  use_rviz:=true use_uav_observer:=true use_allocator:=true

# Terminal 2
ros2 run ugv_nav demo_display_node
```

This runs the blind Hungarian baseline only — targets are detected and assigned directly, with no posterior, no `J`, and no capability belief in the loop.

### Comms disruption

Runs standalone on a synthetic mission topic pair; not yet part of the allocation flow.

```bash
ros2 launch multi_robot_bringup simulation.launch.py \
  use_rviz:=false use_mission_comms:=true use_network_sim:=true network_scenario:=drop
```

Scenarios: `clean`, `drop`, `delay`, `blackout`. Schema in `ros2_ws/src/comm_layer/comm_layer/mission_schema.md`.

---

## Scope

This repository is a multi-robot informative path planning testbed. Adversarial and pursuit-evasion settings are out of scope. Interaction-aware planning is out of scope.

Obstacles are added only where they create platform asymmetry — ground routes the UAV can overfly, occlusion the UGV resolves and the UAV does not. Navigation realism for its own sake is out of scope.

> The current obstacle field (9 hardcoded AABB blocks/barriers, avoided via an artificial potential field) predates this principle and does not yet reflect it — it is general urban obstacle avoidance, not asymmetry-producing. `ipp_core.sensing` already models the asymmetry (the UGV's footprint is occluded, the UAV's is not), so the mechanism exists; what is unverified is whether the layout puts occlusion anywhere the posterior mass actually is. Redesign is deferred, and must happen before any comparison numbers are generated from this world.

Capability uncertainty is out of scope for this testbed. Under `J` as defined, the undetected-mass integrand is a product over independent factors each linear in its own `ρ_i`, so `E[Π_i(1 − ρ_i g_i(l))] = Π_i(1 − μ_i g_i(l))` exactly — the variance never enters and a belief-aware allocator selects the same assignment as a mean plug-in. It becomes load-bearing only under a risk-aware objective. That line of work lives in [risk-aware-sensor-placement](https://github.com/privaelo/risk-aware-sensor-placement).

---

## Open

1. Does the greedy guarantee survive mobile sensors? The set function is over trajectories, not locations. If submodularity breaks, that is a finding.
2. Does a structure-constrained learned allocator beat an unconstrained one on `J`?
3. Under what belief-divergence threshold does divertible assignment start to thrash?
4. Does the coupling survive decentralization? A decentralized auction with bids computed on marginal reduction in `J`, and no agent holding the global belief, would test whether coupled allocation degrades gracefully when the posterior is fragmented across the team. This has to be built — there is no auction implementation in this repo today.
