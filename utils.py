"""
SUMO utility helpers: TLS discovery, occupancy collection, optimization, and visualization.

The functions in this module encapsulate the "latest" experimentation toolkit that we use
during SUMO integration. They intentionally rely on the global constants from config.py so
that scripts and notebooks stay in sync without duplicating knobs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import traci

from config import CYCLE_TIME, MAX_PHASE_DURATION, MIN_PHASE_DURATION, PROXIMITY_THRESHOLD

# Optional mapping between TLS IDs and the lanes controlled by each phase. Fill it from
# a scenario-specific config before calling optimize_phases to unlock per-phase queues.
TLS_PHASE_LANES: Dict[str, Dict[int, List[str]]] = {}


def get_junction_info(traci_client, junction_id: str) -> Optional[Dict[str, object]]:
    """Возвращает основную информацию о перекрестке из SUMO."""
    try:
        position = traci_client.junction.getPosition(junction_id)
        junction_type = traci_client.junction.getType(junction_id)
        shape = traci_client.junction.getShape(junction_id)
        return {
            "id": junction_id,
            "position": position,
            "type": junction_type,
            "shape": shape,
        }
    except Exception as exc:  # pragma: no cover - SUMO connection errors
        print(f"Ошибка при получении информации о перекрестке {junction_id}: {exc}")
        return None


def get_traffic_light_info(traci_client, tls_id: str) -> Optional[Dict[str, object]]:
    """Получаем расширенный набор сведений по конкретному светофору."""
    try:
        controlled_lanes = traci_client.trafficlight.getControlledLanes(tls_id)
        controlled_links = traci_client.trafficlight.getControlledLinks(tls_id)
        program = traci_client.trafficlight.getProgram(tls_id)
        complete_programs = traci_client.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        phase = traci_client.trafficlight.getPhase(tls_id)
        state = traci_client.trafficlight.getRedYellowGreenState(tls_id)
        junctions = []
        if "#" in tls_id:
            parts = tls_id.split("_")
            for part in parts:
                if part.isdigit():
                    junction_info = get_junction_info(traci_client, part)
                    if junction_info:
                        junctions.append(junction_info)
        return {
            "id": tls_id,
            "controlled_lanes": controlled_lanes,
            "controlled_links": controlled_links,
            "program": program,
            "complete_programs": complete_programs,
            "phase": phase,
            "state": state,
            "is_cluster": "#" in tls_id,
            "junctions": junctions,
        }
    except Exception as exc:  # pragma: no cover - SUMO connection errors
        print(f"Ошибка при получении информации о светофоре {tls_id}: {exc}")
        return None


def extract_junctions_from_cluster(cluster_id: str) -> List[str]:
    """Извлекает ID перекрестков из идентификатора кластера."""
    if "#" not in cluster_id:
        return [cluster_id]
    parts = cluster_id.split("_")
    junctions: List[str] = []
    for part in parts[1:]:
        if "#" in part:
            break
        junctions.append(part)
    return junctions


def select_traffic_light(
    traci_client,
    tls_ids: Sequence[str],
    tls_id: Optional[str] = None,
) -> Optional[str]:
    """Интерактивный выбор светофора или кластера."""
    if not tls_ids:
        print("Нет доступных светофоров в сети")
        return None
    if tls_id and tls_id in tls_ids:
        print(f"Выбран светофор: {tls_id}")
        return tls_id

    print("Доступные светофоры:")
    for idx, tls in enumerate(tls_ids):
        is_cluster = "#" in tls
        cluster_info = " (кластер)" if is_cluster else ""
        print(f"{idx + 1}. {tls}{cluster_info}")
        if is_cluster:
            junctions = extract_junctions_from_cluster(tls)
            print(f" Перекрестки в кластере: {', '.join(junctions)}")
        try:
            state = traci_client.trafficlight.getRedYellowGreenState(tls)
            controlled_lanes = traci_client.trafficlight.getControlledLanes(tls)
            print(f" Количество контролируемых полос: {len(controlled_lanes)}")
            print(f" Текущее состояние: {state}")
        except Exception as exc:  # pragma: no cover - SUMO connection errors
            print(f" Ошибка при получении информации о светофоре: {exc}")
    try:
        choice = int(input("Выберите номер светофора (или Enter для выбора первого): ") or "1")
        if 1 <= choice <= len(tls_ids):
            selected_tls = tls_ids[choice - 1]
            print(f"Выбран светофор: {selected_tls}")
            if "#" in selected_tls:
                print("ВНИМАНИЕ: Выбран кластер светофоров.")
                print("Для корректной работы рекомендуется использовать весь кластер.")
                use_cluster = input("Использовать весь кластер? (y/n, по умолчанию y): ").lower() != "n"
                if not use_cluster:
                    junctions = extract_junctions_from_cluster(selected_tls)
                    print("Перекрестки в кластере:")
                    for idx, junction in enumerate(junctions):
                        print(f"{idx + 1}. {junction}")
                    sub_choice = input("Выберите номер перекрестка: ")
                    if sub_choice and sub_choice.isdigit():
                        sub_choice_int = int(sub_choice)
                        if 1 <= sub_choice_int <= len(junctions):
                            junction_id = junctions[sub_choice_int - 1]
                            print(f"Выбран перекресток: {junction_id}")
                            print("ВНИМАНИЕ: Управление отдельным перекрестком может не работать корректно.")
                            print("Если возникнут ошибки, попробуйте использовать весь кластер.")
                            return junction_id
            return selected_tls
        print(f"Неверный выбор. Выбран первый светофор: {tls_ids[0]}")
        return tls_ids[0]
    except ValueError:
        print(f"Неверный ввод. Выбран первый светофор: {tls_ids[0]}")
        return tls_ids[0]


def detect_near_miss() -> Tuple[int, float]:
    """Детектирует near-miss события на основе TTC < 2 сек с фильтрацией по дистанции."""
    vehicles = traci.vehicle.getIDList()
    near_miss_count = 0
    risk_metrics: List[float] = []
    for i, veh1 in enumerate(vehicles):
        pos1 = np.array(traci.vehicle.getPosition(veh1))
        speed1 = traci.vehicle.getSpeed(veh1)
        for veh2 in vehicles[i + 1 :]:
            pos2 = np.array(traci.vehicle.getPosition(veh2))
            dist = np.linalg.norm(pos1 - pos2)
            if dist > PROXIMITY_THRESHOLD:
                continue
            speed2 = traci.vehicle.getSpeed(veh2)
            rel_speed = abs(speed1 - speed2)
            if rel_speed > 0 and dist / rel_speed < 2.0:
                near_miss_count += 1
                risk_metrics.append(dist / rel_speed)
    avg_risk = float(np.mean(risk_metrics)) if risk_metrics else 0.0
    return near_miss_count, avg_risk


_program_counter = 0


def collect_phase_inflow_outflow(tls_id: str, logic) -> Tuple[np.ndarray, np.ndarray]:
    """
    Сбор реального притока и оттока машин за последние N секунд по каждому подходу.
    Возвращает inflow/outflow массивы длины len(phases).
    """
    n = len(logic.phases)
    inflow = np.zeros(n)
    outflow = np.zeros(n)

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    for link in controlled_links:
        if not link:
            continue
        lane = link[0][0]
        try:
            edge = traci.lane.getEdgeID(lane)
            entered = traci.edge.getLastStepVehicleNumberEntered(edge)
            left = traci.edge.getLastStepVehicleNumberLeft(edge)
        except Exception:
            continue

        for phase_idx, phase in enumerate(logic.phases):
            state = phase.state
            try:
                link_index = controlled_links.index(link)
            except ValueError:
                continue
            if link_index >= len(state):
                continue
            if state[link_index] in "Gg":
                inflow[phase_idx] += entered
                outflow[phase_idx] += left

    if inflow.sum() == 0:
        inflow = np.ones(n)
    if outflow.sum() == 0:
        outflow = np.ones(n)

    return inflow, outflow


def optimize_tls_durations(tls_id: str, logic, near_miss_count: int, avg_risk: float) -> List[int]:
    """
    Простая, стабильная и эффективная адаптация длительностей фаз.
    """
    MIN_GREEN = max(10, MIN_PHASE_DURATION)
    YELLOW_TIME = 4
    ALL_RED_TIME = 2
    FIXED_LOSS_PER_CYCLE = YELLOW_TIME * 2 + ALL_RED_TIME
    TARGET_CYCLE = CYCLE_TIME

    phases = logic.phases
    n = len(phases)
    if n == 0:
        return [int(p.duration) for p in phases]

    green_phase_indices: List[int] = []
    for idx, ph in enumerate(phases):
        if any(c in "Gg" for c in ph.state):
            green_phase_indices.append(idx)

    num_green = len(green_phase_indices)
    if num_green == 0:
        return [int(p.duration) for p in phases]

    occupancy_per_green_phase: List[float] = []
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for green_idx in green_phase_indices:
        phase = phases[green_idx]
        halting_sum = 0.0
        veh_sum = 0.0
        occ_max = 0.0

        for link_idx, link in enumerate(controlled_links):
            if link_idx >= len(phase.state) or not link:
                continue
            if phase.state[link_idx] in "Gg":
                for lane_info in link:
                    if not lane_info:
                        continue
                    lane_id = lane_info[0]
                    try:
                        halting_sum += traci.lane.getLastStepHaltingNumber(lane_id)
                    except Exception:
                        pass
                    try:
                        veh_sum += traci.lane.getLastStepVehicleNumber(lane_id)
                    except Exception:
                        pass
                    try:
                        occ = traci.lane.getLastStepOccupancy(lane_id)
                        occ_max = max(occ_max, occ)
                    except Exception:
                        pass

        if halting_sum > 0:
            load = halting_sum
        elif veh_sum > 0:
            load = veh_sum * 0.5
        else:
            load = occ_max * 10.0
        occupancy_per_green_phase.append(max(load, 0.1))

    occupancy_per_green_phase = np.array(occupancy_per_green_phase)

    if not hasattr(optimize_tls_durations, "_smoothed_occupancy"):
        optimize_tls_durations._smoothed_occupancy = {}  # type: ignore[attr-defined]

    key = tls_id
    alpha = 0.2
    smoothed_map = optimize_tls_durations._smoothed_occupancy  # type: ignore[attr-defined]
    if key not in smoothed_map:
        smoothed = occupancy_per_green_phase.copy()
    else:
        old = smoothed_map[key]
        if len(old) == len(occupancy_per_green_phase):
            smoothed = (1 - alpha) * old + alpha * occupancy_per_green_phase
        else:
            smoothed = occupancy_per_green_phase.copy()
    smoothed_map[key] = smoothed

    total_demand = smoothed.sum()
    if total_demand == 0:
        total_demand = 1.0

    available_green_time = TARGET_CYCLE - FIXED_LOSS_PER_CYCLE
    if available_green_time < num_green * MIN_GREEN:
        available_green_time = num_green * MIN_GREEN

    effective_green = (smoothed / total_demand) * available_green_time
    effective_green = np.maximum(effective_green, MIN_GREEN)
    effective_green = np.minimum(effective_green, MAX_PHASE_DURATION)

    current_sum = effective_green.sum()
    if current_sum > available_green_time:
        effective_green *= available_green_time / current_sum
    effective_green = np.maximum(effective_green, MIN_GREEN)
    effective_green = np.round(effective_green).astype(int)

    new_durations = [int(p.duration) for p in phases]
    green_ptr = 0
    for idx, _ in enumerate(phases):
        if idx in green_phase_indices:
            new_durations[idx] = int(effective_green[green_ptr])
            green_ptr += 1
        else:
            new_durations[idx] = max(3, min(6, new_durations[idx]))

    actual_cycle = sum(new_durations)
    diff = actual_cycle - TARGET_CYCLE
    if abs(diff) > 5 and diff != 0:
        adjustable = [
            (phase_idx, new_durations[phase_idx], smoothed[idx] if idx < len(smoothed) else 0)
            for idx, phase_idx in enumerate(green_phase_indices)
        ]
        adjustable.sort(key=lambda x: x[2], reverse=(diff > 0))

        adj_idx = 0
        while diff != 0 and adj_idx < len(adjustable):
            idx = adjustable[adj_idx][0]
            step = 1 if diff < 0 else -1
            candidate = new_durations[idx] - step
            if MIN_GREEN <= candidate <= MAX_PHASE_DURATION:
                new_durations[idx] = candidate
                diff += step
            adj_idx += 1

    old_durations = [int(p.duration) for p in phases]
    for idx in range(n):
        old = old_durations[idx]
        new = new_durations[idx]
        if old > 0:
            ratio = new / old
            if ratio < 0.7:
                new_durations[idx] = int(old * 0.7)
            elif ratio > 1.3:
                new_durations[idx] = int(old * 1.3)

    return new_durations


def apply_phase_durations(tls_id: str, logic, new_durations: Sequence[int]) -> bool:
    """
    Меняет длительность ТЕКУЩЕЙ фазы, оставаясь совместимым с SUMO actuated логикой.
    """
    try:
        current_phase_idx = traci.trafficlight.getPhase(tls_id)
        current_time_left = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()

        if current_time_left < 6:
            return False

        try:
            current_state = logic.phases[current_phase_idx].state
        except Exception:
            current_state = ""
        if not any(c in "Gg" for c in current_state):
            return False

        old_duration = int(logic.phases[current_phase_idx].duration)
        new_duration = int(new_durations[current_phase_idx])
        delta = new_duration - old_duration

        if abs(delta) > 25:
            delta = 25 if delta > 0 else -25

        if current_time_left + delta < 8:
            delta = 8 - current_time_left

        if delta != 0:
            desired_remaining = current_time_left + delta
            desired_remaining = max(8, desired_remaining)
            desired_remaining = min(MAX_PHASE_DURATION, desired_remaining)
            traci.trafficlight.setPhaseDuration(tls_id, desired_remaining)
            print(
                f"Фаза {current_phase_idx}: целевой остаток {desired_remaining:.1f}с "
                f"(было {current_time_left:.1f}с, Δ={delta:+d}с)"
            )
            return True
        return False
    except Exception as exc:  # pragma: no cover - SUMO connection errors
        print(f"apply_phase_durations error: {exc}")
        return False


def set_static_program_for_opt(tls_id: str) -> bool:
    """
    Переводит светофор в полностью статический режим, чтобы адаптация работала корректно.
    """
    from traci import trafficlight as tl

    try:
        all_logics = tl.getAllProgramLogics(tls_id)
        current_logic = all_logics[0]
        static_phases = []
        for phase in current_logic.phases:
            duration = max(int(phase.duration), MIN_PHASE_DURATION)
            static_phases.append(
                tl.Phase(
                    duration=duration,
                    state=phase.state,
                    minDur=duration,
                    maxDur=duration,
                    name=getattr(phase, "name", None),
                )
            )
        static_logic = tl.Logic(
            programID="fixed_baseline_for_opt",
            type=0,
            currentPhaseIndex=tl.getPhase(tls_id),
            phases=static_phases,
        )
        tl.setCompleteRedYellowGreenDefinition(tls_id, static_logic)
        tl.setProgram(tls_id, "fixed_baseline_for_opt")
        return True
    except Exception as exc:  # pragma: no cover - SUMO connection errors
        print(f"Не удалось перевести светофор в static режим: {exc}")
        return False


def set_semi_static_bounds(tls_id: str) -> bool:
    """
    Включает actuated-программу, но ограничивает minDur/maxDur для зелёных фаз.
    """
    from traci import trafficlight as tl

    try:
        all_logics = tl.getAllProgramLogics(tls_id)
        current_logic = all_logics[0]
        bounded_phases = []
        for phase in current_logic.phases:
            duration = max(int(phase.duration), MIN_PHASE_DURATION)
            is_green = any(c in "Gg" for c in phase.state)
            if is_green:
                min_d = max(MIN_PHASE_DURATION, 6)
                max_d = max(min(MAX_PHASE_DURATION, duration), MIN_PHASE_DURATION)
                bounded_phases.append(
                    tl.Phase(
                        duration=duration,
                        state=phase.state,
                        minDur=min_d,
                        maxDur=max_d,
                        name=getattr(phase, "name", None),
                    )
                )
            else:
                bounded_phases.append(
                    tl.Phase(
                        duration=duration,
                        state=phase.state,
                        minDur=duration,
                        maxDur=duration,
                        name=getattr(phase, "name", None),
                    )
                )

        bounded_logic = tl.Logic(
            programID="semi_actuated_bounded",
            type=1,
            currentPhaseIndex=tl.getPhase(tls_id),
            phases=bounded_phases,
        )
        tl.setCompleteRedYellowGreenDefinition(tls_id, bounded_logic)
        tl.setProgram(tls_id, "semi_actuated_bounded")
        return True
    except Exception as exc:  # pragma: no cover - SUMO connection errors
        print(f"Не удалось установить semi-actuated режим: {exc}")
        return False


def optimize_phases(cluster_tls_ids: Sequence[str], cluster_phases: Dict[str, Sequence]) -> Dict[str, List[float]]:
    """
    LP оптимизация длительностей фаз: зеленый распределяется пропорционально очередям.
    """
    optimized_durations: Dict[str, List[float]] = {}

    MIN_GREEN = 5
    MAX_GREEN = 60
    CYCLE_LENGTH = 90

    for tls_id in cluster_tls_ids:
        if tls_id not in TLS_PHASE_LANES:
            print(f"[WARN] Нет lane→phase маппинга для {tls_id}, пропускаю.")
            continue

        phase_lane_map = TLS_PHASE_LANES[tls_id]
        phases = cluster_phases[tls_id]
        n = len(phases)

        flows: List[float] = []
        for phase_index in range(n):
            lanes = phase_lane_map.get(phase_index, [])
            queue = 0.0

            for lane in lanes:
                try:
                    q = traci.lane.getLastStepHaltingNumber(lane)
                    queue += q
                except traci.TraCIException:
                    pass

            flows.append(queue)

        flows_arr = np.array(flows, dtype=float)

        if np.sum(flows_arr) == 0:
            optimized_durations[tls_id] = [CYCLE_LENGTH / n] * n
            continue

        x = cp.Variable(n)
        objective = cp.Maximize(cp.sum(cp.multiply(flows_arr, x)))
        constraints = [
            cp.sum(x) == CYCLE_LENGTH,
            x >= MIN_GREEN,
            x <= MAX_GREEN,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if x.value is None:
            print(f"[ERROR] LP не решилась для {tls_id}.")
            optimized_durations[tls_id] = [CYCLE_LENGTH / n] * n
            continue

        durations = x.value.tolist()
        optimized_durations[tls_id] = durations

        print(f"\nOPTIMIZED {tls_id}:")
        for idx, duration in enumerate(durations):
            print(f"  Фаза {idx}: {duration:.2f} сек (очередь={flows_arr[idx]})")

    return optimized_durations


def visualize_results(risk_history: Sequence[float]) -> None:
    """Простая визуализация тренда риска."""
    plt.plot(risk_history)
    plt.xlabel("Time steps")
    plt.ylabel("Avg Risk")
    plt.title("Risk Trend")
    plt.savefig("risk_trend.png")
    print("Visualization saved to risk_trend.png")


def analyze_tlslog(tls_id: str, tlslog_path: str) -> Optional[Dict[str, float]]:
    """Анализирует tlslog.xml и возвращает средние длительности по состояниям."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(tlslog_path)
    except Exception:
        return None
    root = tree.getroot()
    events: List[Tuple[float, str]] = []
    for evt in root.findall(".//tlsState"):
        if evt.get("id") == tls_id:
            try:
                t = float(evt.get("time"))
                s = evt.get("state")
            except Exception:
                continue
            events.append((t, s))
    if len(events) < 2:
        return None
    durations_by_state: Dict[str, float] = {}
    counts_by_state: Dict[str, int] = {}
    for i in range(len(events) - 1):
        t0, s0 = events[i]
        t1, _ = events[i + 1]
        dur = max(0.0, t1 - t0)
        durations_by_state[s0] = durations_by_state.get(s0, 0.0) + dur
        counts_by_state[s0] = counts_by_state.get(s0, 0) + 1
    avg_by_state = {
        state: round(durations_by_state[state] / counts_by_state[state], 2)
        for state in durations_by_state
    }
    return avg_by_state


__all__ = [
    "TLS_PHASE_LANES",
    "analyze_tlslog",
    "apply_phase_durations",
    "collect_phase_inflow_outflow",
    "detect_near_miss",
    "extract_junctions_from_cluster",
    "get_junction_info",
    "get_traffic_light_info",
    "optimize_phases",
    "optimize_tls_durations",
    "select_traffic_light",
    "set_semi_static_bounds",
    "set_static_program_for_opt",
    "visualize_results",
]
