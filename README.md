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

**Environment.** Targets are drawn from an inhomogeneous Poisson process over a 2D domain with latent intensity field `λ(l)`. The field is unknown to the team.

**Belief.** The team maintains a posterior `λ_t(l)` over the field, updated from observations. No target list is ever materialized. Targets are regions of posterior mass.

**Sensing.** Robot `i` at pose `x_i` detects a target at `l` with probability

```text
γ_i(l) = ρ_i · g_i(l ; x_i)
```

`ρ_i` is the platform's maximum detection capability. `g_i` is the geometric footprint term. UAV and UGV are parameterized asymmetrically: the UAV has a wide footprint and low `ρ`, the UGV a narrow footprint and high `ρ`.

**Undetected mass.** At time `t`, the expected target mass the team has not observed:

```text
U(t) = ∫ λ_t(l) · Π_i (1 − γ_i(l)) dl
```

**Objective.** Minimize integrated undetected mass over the mission:

```text
J = ∫₀ᵀ U(t) dt
```

Search and confirmation are commensurable under this objective. Sweeping an empty region moves `J` little. Confirming a high-`λ` region moves it a lot.

---

## Allocation arms

All arms optimize on the same environment, the same sensing model, and are scored on the same `J`.

| Arm | Information use | Status |
| --- | --- | --- |
| Hungarian | Blind. Requires the posterior thresholded into a task list. | Runs end-to-end as a blind baseline. Threshold rule not yet formalized — currently allocates directly against detected target positions, not a posterior. |
| Greedy on `J` | Coupled. Scores marginal reduction in integrated undetected mass. | Planned — needs the belief field, sensing model, and `J` computed first. |
| Learned allocator | Coupled. Structural priors imposed on the policy class. | Planned. |
| Decentralized auction | Coupled. Bids on marginal `ΔJ`, no agent holds global belief. | In progress. |

The thresholding rule that converts a posterior into a task list is a design decision, not a detail. A weak threshold produces a strawman baseline. The rule must be documented and held fixed across runs once formalized.

---

## Allocation policy

- **Horizon.** Myopic greedy. The objective's submodularity for static sensors gives a `(1 − 1/e)` guarantee. Whether it survives mobile sensors is open.
- **Trigger.** Event-triggered on belief divergence, not fixed-rate replanning.
- **Commitment.** Assignments are divertible mid-transit. A UGV that disconfirms a region en route can be redirected before arrival.

---

## Capability belief

Each robot's `ρ_i` is not assumed known. The allocator maintains a Gaussian belief `ρ_i ~ N(μ_i, σ_i²)` updated by conjugate Gaussian updates from execution outcomes.

These structural results are established in [risk-aware-sensor-placement](https://github.com/privaelo/risk-aware-sensor-placement) and constrain the design here:

- Void probability is convex in each capability coordinate.
- The Jensen gap `E[ν] ≥ ν(μ)` scales as `½σ²ν(μ)ΣB_i²`, additive across independent units.
- Submodularity is preserved in expected coverage. It is not preserved in the void probability.

Status: prototyped (`CapabilityBelief` in `task_allocator`, information-form Gaussian update, unit-tested). Not yet wired into any allocation arm.

---

## Metrics

- `J` — integrated undetected mass. Primary.
- Time to first detection, per target.
- Fraction of posterior mass resolved at mission end.
- Reallocation count. Divertible assignments raise a thrashing question that the count exposes.

---

## Platform

- ROS 2 Jazzy
- Gazebo Harmonic
- 1 UAV (static aerial observer), 3 UGVs (diff-drive-equivalent, velocity-controlled)
- Python

## Packages

| Package | Role |
|---|---|
| `multi_robot_bringup` | Top-level launch, world SDF, RViz config |
| `uav_description` | UAV URDF/Xacro + `robot_state_publisher` launch |
| `ugv_description` | UGV URDF (RViz) + SDF with VelocityControl + OdometryPublisher plugins |
| `uav_observer` | Target detection node — broadcasts discovered targets on `/uav_1/targets` (direct detection today; posterior-based detection is the belief-machinery milestone) |
| `task_allocator` | Hungarian allocator (blind baseline arm) + `CapabilityBelief` prototype (Gaussian belief over `ρ_i`) |
| `ugv_nav` | Goal follower with potential-field obstacle avoidance, RViz marker node, demo display node |
| `comm_layer` | Comms disruption layer (clean / drop / delay / blackout) — planned; re-integration point for testing whether coupled allocation degrades gracefully when the posterior is fragmented across the team |

---

## Scope

This repository is a multi-robot informative path planning testbed. Adversarial and pursuit-evasion settings are out of scope. Interaction-aware planning is out of scope.

Obstacles are added only where they create platform asymmetry — ground routes the UAV can overfly, occlusion the UGV resolves and the UAV does not. Navigation realism for its own sake is out of scope.

> The current obstacle field (9 hardcoded AABB blocks/barriers, avoided via an artificial potential field) predates this principle and does not yet reflect it — it's general urban obstacle avoidance, not asymmetry-producing. Left as-is for now; redesign is deferred, not urgent.

---

## Running it

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
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

This demo runs the blind Hungarian baseline only — targets are detected and assigned directly, with no posterior, no `J`, and no capability belief in the loop yet.

---

## Open

1. Does the greedy guarantee survive mobile sensors? The set function is over trajectories, not locations. If submodularity breaks, that is a finding.
2. Does a structure-constrained learned allocator beat an unconstrained one on `J`?
3. Under what belief-divergence threshold does divertible assignment start to thrash?
4. Does the coupling survive decentralization? An auction implementation is in progress in this repo. Bids computed on marginal reduction in `J`, with no agent holding the global belief, would test whether coupled allocation degrades gracefully when the posterior is fragmented across the team.
