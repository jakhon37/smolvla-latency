"""Real observations for the real-weights checks: frames of the dataset a checkpoint was trained on.

    python tools/real_inputs.py --out real_inputs.npz        # lerobot/libero, episodes 0,423,846,1269, 4 frames each

Loads the episodes through LeRobotDataset, which downloads only the data and video files holding them, and takes
frames floor(k * L / n), k = 0..n-1, of each (L the episode length, n --frames-per-episode). The frames are what
LeRobotDataset returns: both cameras decoded with its default video backend, float32 in [0, 1], channel-first,
under the dataset's camera keys (the checkpoint's preprocessor renames them). They are stored as uint8, which is
exact (the decoder's uint8 divided by 255; checked here), with the state, the task string and the demonstrated
actions that follow each frame. A JSON file next to the npz records the selection, the sha256 of every array, and
the decoder (torchcodec and its FFmpeg libraries), which decides the frames' bytes.

``load_observations`` turns the file back into un-batched observation dicts for the policy preprocessor.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_EPISODES = "0,423,846,1269"  # spread over lerobot/libero: one episode in each LIBERO suite's block


def frame_indices(length: int, n: int) -> list[int]:
    return [k * length // n for k in range(n)]


def select(dataset, episodes: list[int], frames_per_episode: int, chunk: int) -> tuple[dict, list[dict]]:
    """Arrays and per-frame records for the chosen frames of an episode-filtered LeRobotDataset."""
    hf = dataset.hf_dataset
    ep_col = np.asarray(hf["episode_index"])
    fr_col = np.asarray(hf["frame_index"])
    cams = list(dataset.meta.camera_keys)
    arrays: dict[str, list] = {f"images.{c}": [] for c in cams}
    arrays.update(state=[], demo_actions=[], demo_valid=[])
    records = []
    for ep in episodes:
        rows = np.flatnonzero(ep_col == ep)
        if rows.size == 0:
            raise SystemExit(f"episode {ep} is not in the loaded dataset")
        length = int(rows.size)
        actions = torch.stack([hf[int(r)]["action"] for r in rows]).numpy()
        for f in frame_indices(length, frames_per_episode):
            row = int(rows[np.flatnonzero(fr_col[rows] == f)[0]])
            item = dataset[row]
            for c in cams:
                x = item[c].numpy()
                u8 = np.round(x * 255.0).astype(np.uint8)
                if not np.array_equal(u8.astype(np.float32) / np.float32(255.0), x):
                    raise SystemExit(f"{c} of episode {ep} frame {f} is not a uint8 frame divided by 255")
                arrays[f"images.{c}"].append(u8)
            arrays["state"].append(item["observation.state"].numpy().astype(np.float32))
            demo = np.zeros((chunk, actions.shape[1]), np.float32)
            valid = np.zeros(chunk, bool)
            n = min(chunk, length - f)
            demo[:n], valid[:n] = actions[f:f + n], True
            arrays["demo_actions"].append(demo)
            arrays["demo_valid"].append(valid)
            records.append({"index": len(records), "episode": ep, "frame_index": f, "episode_length": length,
                            "dataset_index": int(item["index"]), "timestamp": float(item["timestamp"]),
                            "task_index": int(item["task_index"]), "task": item["task"]})
    return {k: np.stack(v) for k, v in arrays.items()}, records


def load_observations(npz_path: Path) -> tuple[list[dict], dict]:
    """Un-batched observations (dataset camera keys, float32 [0, 1] CHW, state, task) and the selection record."""
    data = np.load(npz_path)
    selection = json.loads(Path(npz_path).with_suffix(".json").read_text())
    obs = []
    for rec in selection["frames"]:
        i = rec["index"]
        o = {c: torch.from_numpy(data[f"images.{c}"][i].astype(np.float32) / np.float32(255.0))
             for c in selection["camera_keys"]}
        o["observation.state"] = torch.from_numpy(data["state"][i].copy())
        o["task"] = rec["task"]
        obs.append(o)
    return obs, selection


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lerobot/libero", help="the dataset the checkpoint was trained on")
    ap.add_argument("--revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    ap.add_argument("--root", type=Path, default=None, help="local directory for the dataset files")
    ap.add_argument("--episodes", default=DEFAULT_EPISODES)
    ap.add_argument("--frames-per-episode", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50, help="demonstrated actions kept after each frame")
    ap.add_argument("--out", type=Path, required=True, help=".npz; the selection goes to the same name .json")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from run_info import code_version, decoder_versions, sha256_array, system_info

    episodes = [int(e) for e in args.episodes.split(",")]
    ds = LeRobotDataset(args.dataset, root=args.root, episodes=episodes, revision=args.revision)
    arrays, records = select(ds, episodes, args.frames_per_episode, args.chunk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    selection = {
        "dataset": args.dataset, "revision": ds.revision, "codebase_version": ds.meta.info.get("codebase_version"),
        "fps": ds.meta.fps, "video_backend": ds._video_backend, "camera_keys": list(ds.meta.camera_keys),
        "episodes": episodes, "frames_per_episode": args.frames_per_episode,
        "frame_rule": "floor(k * episode_length / frames_per_episode), k = 0..frames_per_episode-1",
        "files": sorted(ds.reader.get_episodes_file_paths()),
        "demo_actions": f"the {args.chunk} dataset actions from each frame on, within its episode (demo_valid)",
        "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": sha256_array(v)}
                   for k, v in arrays.items()},
        "decoder": decoder_versions(),
        "code": code_version(),
        "system": system_info(),
        "frames": records,
    }
    args.out.with_suffix(".json").write_text(json.dumps(selection, indent=1))
    for r in records:
        print(f"{r['index']:2d}  episode {r['episode']:4d}  frame {r['frame_index']:3d}/{r['episode_length']}  "
              f"{r['task']}")
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
