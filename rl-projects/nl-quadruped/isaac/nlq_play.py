# NLQ integration script — based on Isaac Lab's rsl_rl/play.py (Isaac Lab 3.0-beta2).
# 차이점은 "# === NLQ ===" 로 표시된 블록뿐. env/checkpoint 로딩은 원본 play.py 그대로(검증된 경로).
#
# 핵심: 정책이 보는 velocity_commands 는 obs 벡터의 index 9:12 (obs manager layout 기준).
#       매 스텝 그 슬라이스를 우리 go-to-goal 명령(vx, vy, yaw_rate)으로 덮어써서 로봇을 운전한다.
#       command manager 의 랜덤 샘플러와 싸우지 않는 가장 안전한 주입법.
#
# 배치 위치: 반드시 ~/IsaacLab/scripts/reinforcement_learning/rsl_rl/ 안에 둘 것 (import cli_args 때문).
#
# 사용:
#   3a) 상수 전진:   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/nlq_play.py \
#                      --task=Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 1 --headless \
#                      --nlq_mode constant --video --video_length 400
#   3b) 고정 목표:   ... --nlq_mode goto --goal 3 3 --video --video_length 600

"""Script to play a checkpoint of an RL agent from RSL-RL, driven by NLQ go-to-goal commands."""

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# === NLQ ===================================================================
# 컨트롤러 로직 inline (controller.py 의 go_to_goal 과 동일). 의존성 0, import 경로 문제 회피.
VEL_CMD_SLICE = slice(9, 12)  # obs 벡터에서 velocity_commands 위치 (base_lin/ang/gravity 다음)


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_from_quat_wxyz(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _go_to_goal(x, y, yaw, gx, gy, vmax):
    """(vx_forward, vy, yaw_rate) — base frame.

    항상 0.4*vmax 이상 전진하며 조향(자동차식). 제자리 회전(vx=0)은 locomotion 정책에
    out-of-distribution 이라 불안정 → 걸으면서 도는 in-distribution 동작으로 유도.
    """
    err = _wrap_to_pi(math.atan2(gy - y, gx - x) - yaw)
    yaw_rate = max(-1.0, min(1.0, 1.5 * err))
    vx = vmax * (0.4 + 0.6 * max(0.0, math.cos(err)))  # 0.4*vmax ~ vmax, 절대 0 안 됨
    return vx, 0.0, yaw_rate
# === /NLQ ==================================================================


# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play an RSL-RL agent driven by NLQ commands.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an external callback.")
# === NLQ ===
parser.add_argument("--nlq_mode", type=str, default="goto", choices=["constant", "goto"], help="NLQ command mode")
parser.add_argument("--goal", type=float, nargs=2, default=[3.0, 3.0], help="Goal (x y) in env-local meters")
parser.add_argument("--vmax", type=float, default=0.7, help="Forward speed command magnitude (m/s)")
parser.add_argument("--goal_tol", type=float, default=0.3, help="Goal reach tolerance (m)")
parser.add_argument("--robot_name", type=str, default="robot", help="Scene articulation key for the robot")
# === /NLQ ===
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent, driven by NLQ go-to-goal commands."""
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "nlq"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording NLQ video.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # (원본의 JIT/ONNX export 블록은 데모에 불필요해서 생략)

        dt = env.unwrapped.step_dt

        # === NLQ === 통합 셋업 ===============================================
        base_env = env.unwrapped
        robot = base_env.scene[args_cli.robot_name]
        try:
            origin = base_env.scene.env_origins[0]
            ox, oy = float(origin[0]), float(origin[1])
        except Exception:
            ox, oy = 0.0, 0.0
        gx, gy = float(args_cli.goal[0]), float(args_cli.goal[1])

        # 명령 주입은 command manager(source of truth)로 — obs 인덱스에 의존하지 않음.
        cmd_term = base_env.command_manager.get_term("base_velocity")
        for _attr, _val in (("resampling_time_range", (1.0e9, 1.0e9)),
                            ("heading_command", False),
                            ("rel_standing_envs", 0.0)):
            try:
                setattr(cmd_term.cfg, _attr, _val)
            except Exception as _e:
                print(f"[NLQ] warn: cfg.{_attr} 설정 실패: {_e}", flush=True)

        def _set_cmd(vx, vy, yr):
            val = torch.tensor([vx, vy, yr], device=base_env.device, dtype=torch.float32)
            if hasattr(cmd_term, "vel_command_b"):
                cmd_term.vel_command_b[:, :3] = val
            try:
                base_env.command_manager.get_command("base_velocity")[:, :3] = val
            except Exception:
                pass

        print(f"[NLQ] mode={args_cli.nlq_mode} goal=({gx},{gy}) vmax={args_cli.vmax} "
              f"origin=({ox:.2f},{oy:.2f}) cmd_term={type(cmd_term).__name__} "
              f"has_vel_command_b={hasattr(cmd_term, 'vel_command_b')}", flush=True)

        # 목표 시각 마커(빨간 기둥) — point-instancer(VisualizationMarkers) 대신 정적 USD prim 직접 스폰.
        # 이 빌드에서 instancer 가 렌더 안 되는 이슈 회피. 실패해도 데모는 계속.
        if args_cli.nlq_mode == "goto":
            try:
                import isaaclab.sim as sim_utils
                _mk = sim_utils.CuboidCfg(
                    size=(0.4, 0.4, 1.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                )
                _mk.func("/World/nlq_goal_marker", _mk, translation=(gx + ox, gy + oy, 0.75))
                print("[NLQ] goal marker spawned (static cuboid)", flush=True)
            except Exception as _e:
                print(f"[NLQ] warn: goal marker 생성 실패(데모는 계속): {_e}", flush=True)

        reached = False
        _dbg_done = False
        # === /NLQ ============================================================

        obs = env.get_observations()
        timestep = 0
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    # === NLQ === 매 스텝 명령 계산 후 obs[:,9:12] 덮어쓰기 =====
                    if args_cli.nlq_mode == "constant":
                        vx, vy, yr = args_cli.vmax, 0.0, 0.0
                        x = y = yaw = dist = 0.0
                    else:  # goto
                        pos = robot.data.root_pos_w[0]
                        x = float(pos[0]) - ox
                        y = float(pos[1]) - oy
                        # Isaac Lab 공식 변환 (root_quat_w 와 일관). 손수 짠 수식은 성분순서가 안 맞았음.
                        _, _, _yaw_t = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
                        yaw = _wrap_to_pi(float(_yaw_t[0]))
                        dist = math.hypot(gx - x, gy - y)
                        if dist < args_cli.goal_tol:
                            vx, vy, yr = 0.0, 0.0, 0.0
                            if not reached:
                                print(f"[NLQ] goal reached at ({x:.2f},{y:.2f}) after {timestep} steps", flush=True)
                                reached = True
                        else:
                            vx, vy, yr = _go_to_goal(x, y, yaw, gx, gy, args_cli.vmax)

                    _set_cmd(vx, vy, yr)  # command manager 에 주입 (obs 는 env 가 정확한 자리에 채움)

                    if not _dbg_done:
                        _dbg_done = True
                        try:
                            readback = base_env.command_manager.get_command("base_velocity")[0].tolist()
                        except Exception as _e:
                            readback = f"<err {_e}>"
                        print(f"[NLQ-DBG] set=({vx:.2f},{vy:.2f},{yr:.2f}) readback={readback}", flush=True)

                    if timestep % 50 == 0:
                        avz = float(robot.data.root_ang_vel_w[0, 2])
                        lv = robot.data.root_lin_vel_w[0]
                        spd = float((lv[0] ** 2 + lv[1] ** 2) ** 0.5)
                        try:
                            gz = float(robot.data.projected_gravity_b[0, 2])  # 서있으면 ~-1, 넘어지면 ~0
                        except Exception:
                            gz = 0.0
                        print(f"[NLQ] t={timestep} cmd=({vx:.2f},{vy:.2f},{yr:.2f}) "
                              f"meas_yawrate={avz:.2f} meas_speed={spd:.2f} gravz={gz:.2f} "
                              f"pos=({x:.2f},{y:.2f}) yaw={yaw:.2f} dist={dist:.2f}", flush=True)
                    # === /NLQ ===

                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        # rsl-rl < 4.0.0: reset via the policy network
                        with contextlib.suppress(Exception):
                            runner.alg.policy.reset(dones)

                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break
                else:
                    timestep += 1

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
