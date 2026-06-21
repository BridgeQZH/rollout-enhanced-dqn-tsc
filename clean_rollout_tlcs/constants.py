"""Centralized constants for the rollout-enhanced TLCS project.

Every magic number used by the environment, the transition model ``f``, the stage
cost ``g`` and the rollout controllers lives here so that the rest of the codebase
contains no unexplained literals.

The intersection is the canonical single 4-way junction inherited from the
AndreaVidali / Vidali thesis SUMO network: four 750 m incoming arms, each with one
dedicated left-turn lane and three through/right lanes, controlled by a single
traffic light with four green phases (plus their yellows).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and filenames
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS_PATH = Path("settings")
DEFAULT_MODEL_PATH = Path("models")
DEFAULT_EVAL_PATH = Path("evaluation")

TRAINING_SETTINGS_FILE = Path("training_settings.yaml")
EVAL_SETTINGS_FILE = Path("eval_settings.yaml")
SEEDS_FILE = Path("seeds.yaml")

MODEL_FILE = Path("trained_model.pt")
CALIBRATION_FILE = Path("fg_calibration.json")

# Snapshot of the settings YAML and a JSON manifest written into each run folder.
CONFIG_SNAPSHOT_FILE = Path("training_settings.yaml")
RUN_MANIFEST_FILE = Path("run_manifest.json")

DEFAULT_TEST_FOLDER = "test"

# Route file referenced by the SUMO ``.sumocfg``; regenerated each episode.
ROUTES_FILE = Path("intersection/episode_routes.rou.xml")

# 2x2 grid network assets (Phase 2.3 multi-agent). The route file is regenerated
# per episode (gitignored); the net and sumocfg are committed.
GRID2X2_DIR = Path("clean_rollout_tlcs/networks/grid2x2")
GRID2X2_NET = GRID2X2_DIR / "grid2x2.net.xml"
GRID2X2_SUMOCFG = GRID2X2_DIR / "grid2x2.sumocfg"
GRID2X2_ROUTES = GRID2X2_DIR / "grid2x2_routes.rou.xml"

# Header for the generated route file: one vehicle type plus all 12 OD routes.
ROUTES_FILE_HEADER = """<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>"""  # noqa: E501

# Straight-through versus turning OD routes (used by the route generator).
STRAIGHT_ROUTES: tuple[str, ...] = ("W_E", "E_W", "N_S", "S_N")
TURN_ROUTES: tuple[str, ...] = ("W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E")

# ---------------------------------------------------------------------------
# Traffic-light phases and action mapping
# ---------------------------------------------------------------------------

# Phase indices follow the <tlLogic> order in environment.net.xml.
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7

# Discrete action -> SUMO green phase code.
ACTION_TO_TL_PHASE: dict[int, int] = {
    0: PHASE_NS_GREEN,  # North-South through / right
    1: PHASE_NSL_GREEN,  # North-South left
    2: PHASE_EW_GREEN,  # East-West through / right
    3: PHASE_EWL_GREEN,  # East-West left
}

# Green phase code -> matching yellow phase code, inserted on phase change.
TL_GREEN_TO_YELLOW: dict[int, int] = {
    PHASE_NS_GREEN: PHASE_NS_YELLOW,
    PHASE_NSL_GREEN: PHASE_NSL_YELLOW,
    PHASE_EW_GREEN: PHASE_EW_YELLOW,
    PHASE_EWL_GREEN: PHASE_EWL_YELLOW,
}

TRAFFIC_LIGHT_ID = "TL"

# ---------------------------------------------------------------------------
# RL state / action space (12-dimensional lane-count state)
# ---------------------------------------------------------------------------

STATE_SIZE = 12  # 4 arms x 3 lane groups (right+through, through, left)
NUM_ACTIONS = 4  # matches keys in ACTION_TO_TL_PHASE

# Human-readable label for each state index; used in verify/debug dumps.
LANE_GROUP_LABELS: tuple[str, ...] = (
    "N0",  # 0  North  right+through lane (N2TL_0)
    "N12",  # 1  North  through lanes     (N2TL_1, N2TL_2)
    "N3",  # 2  North  left lane          (N2TL_3)
    "S0",  # 3  South  right+through lane (S2TL_0)
    "S12",  # 4  South  through lanes     (S2TL_1, S2TL_2)
    "S3",  # 5  South  left lane          (S2TL_3)
    "E0",  # 6  East   right+through lane (E2TL_0)
    "E12",  # 7  East   through lanes     (E2TL_1, E2TL_2)
    "E3",  # 8  East   left lane          (E2TL_3)
    "W0",  # 9  West   right+through lane (W2TL_0)
    "W12",  # 10 West   through lanes     (W2TL_1, W2TL_2)
    "W3",  # 11 West   left lane          (W2TL_3)
)

# Number of physical lanes mapped onto each state index. The "through" group of
# each arm aggregates two SUMO lanes; all others map to a single lane. The
# transition model f uses this to scale per-lane saturation discharge.
LANES_PER_GROUP: tuple[int, ...] = (1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1)

# Map each incoming SUMO lane id to its state index.
LANE_ID_TO_STATE_INDEX: dict[str, int] = {
    # North
    "N2TL_0": 0,
    "N2TL_1": 1,
    "N2TL_2": 1,
    "N2TL_3": 2,
    # South
    "S2TL_0": 3,
    "S2TL_1": 4,
    "S2TL_2": 4,
    "S2TL_3": 5,
    # East
    "E2TL_0": 6,
    "E2TL_1": 7,
    "E2TL_2": 7,
    "E2TL_3": 8,
    # West
    "W2TL_0": 9,
    "W2TL_1": 10,
    "W2TL_2": 10,
    "W2TL_3": 11,
}

# Incoming edges towards the junction (used for waiting-time bookkeeping).
INCOMING_EDGES: tuple[str, ...] = ("N2TL", "S2TL", "E2TL", "W2TL")

# ---------------------------------------------------------------------------
# Service structure  (transition indicators A(i, u))
# ---------------------------------------------------------------------------

# For each action u, the set of state indices that receive green (discharge).
SERVED_LANES: dict[int, frozenset[int]] = {
    0: frozenset({0, 1, 3, 4}),  # NS through/right
    1: frozenset({2, 5}),  # NS left
    2: frozenset({6, 7, 9, 10}),  # EW through/right
    3: frozenset({8, 11}),  # EW left
}

# Dense 0/1 service-indicator matrix A[u][i] == 1 iff lane group i is served by u.
SERVICE_INDICATOR: tuple[tuple[int, ...], ...] = tuple(
    tuple(1 if i in SERVED_LANES[u] else 0 for i in range(STATE_SIZE))
    for u in range(NUM_ACTIONS)
)

# ---------------------------------------------------------------------------
# Lane geometry, detector placement and free-flow speed (for the lag τ = d / v)
# ---------------------------------------------------------------------------

# Edge length in environment.net.xml (m).
ROAD_MAX_LENGTH = 750.0

# Distance-to-junction (m) within which a vehicle is counted into the state
# vector, i.e. treated as part of the standing/approaching queue at the stop line.
STOPLINE_ZONE_M = 100.0

# Distance-to-junction (m) of the virtual upstream loop detector. A vehicle is
# registered by the detector when it first appears farther than this from the
# stop line (i.e. right after entering the 750 m arm).
DETECTOR_DISTANCE_M = 636.0

# Free-flow speed on the incoming arms (m/s) read from environment.net.xml
# (13.89 m/s == 50 km/h on the 750 m approach lanes).
V_FREE_FLOW_MS = 13.89

# Default detector->stop-line travel-time lag (s):  τ = DETECTOR_DISTANCE_M / V_FREE_FLOW_MS.
# A vehicle sensed upstream now reaches the stop line ~τ seconds later, so the
# arrival rate that matters at the stop line is the detector rate measured τ ago.
DEFAULT_DETECTOR_LAG_TAU = DETECTOR_DISTANCE_M / V_FREE_FLOW_MS  # ≈ 45.79 s

# Sliding-window length (s) over which the causal arrival-rate estimator averages.
DEFAULT_DETECTOR_WINDOW_S = 100

# ---------------------------------------------------------------------------
# Transition-model (f) physical defaults  — calibrated against SUMO in Phase 2
# ---------------------------------------------------------------------------

# Saturation discharge rate per physical lane during green (veh/s/lane).
# ~0.5 veh/s/lane corresponds to a ~1800 veh/h saturation flow.
DEFAULT_SATURATION_FLOW = 0.5

# Startup lost time at green onset (s) — reduced discharge while queued vehicles
# accelerate; applied only when the phase has just changed.
DEFAULT_STARTUP_LOST_TIME = 2.0

# Physical vehicle footprint used to bound queue storage (m). Matches the
# standard_car length + minGap declared in ROUTES_FILE_HEADER.
VEHICLE_LENGTH_M = 5.0
MIN_GAP_M = 2.5
VEHICLE_FOOTPRINT_M = VEHICLE_LENGTH_M + MIN_GAP_M  # 7.5 m

# Maximum vehicles a single lane can hold within the stop-line detection zone;
# the transition model clips predicted queues to this capacity per lane group.
MAX_QUEUE_PER_LANE = STOPLINE_ZONE_M / VEHICLE_FOOTPRINT_M  # ≈ 13.33 veh

# Per-state-index storage capacity C_i = lanes_in_group * MAX_QUEUE_PER_LANE.
LANE_QUEUE_CAPACITY: tuple[float, ...] = tuple(
    lanes * MAX_QUEUE_PER_LANE for lanes in LANES_PER_GROUP
)

# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------

# Number of distinct traffic seeds in the paired benchmark harness.
DEFAULT_N_EVAL_SEEDS = 30

# Traffic-generation parameters for the Weibull arrival profile.
WEIBULL_SHAPE = 2.0
DEFAULT_DEPART_SPEED = 10.0
