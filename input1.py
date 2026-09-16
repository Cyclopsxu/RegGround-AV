import json
from pathlib import Path

nuscenes_mini = "v1.0-mini"
fixture_dir = Path("tests/layer1/fixtures")

# 读取所有表
with open(f"{nuscenes_mini}/ego_pose.json") as f:
    ego_poses = json.load(f)
with open(f"{nuscenes_mini}/sample.json") as f:
    samples = json.load(f)
with open(f"{nuscenes_mini}/sample_data.json") as f:
    sample_data = json.load(f)

# 建立索引
ego_pose_by_token = {ep["token"]: ep for ep in ego_poses}
sample_data_by_sample = {}
for sd in sample_data:
    sid = sd["sample_token"]
    sample_data_by_sample.setdefault(sid, []).append(sd)

# 取第一个 sample，往后截 8 帧 ego_pose
first_sample = samples[0]
sample_token = first_sample["token"]

# 找这个 sample 对应的 ego_pose token（通过 sample_data）
sds = sorted(
    [sd for sd in sample_data_by_sample.get(sample_token, [])
     if sd["is_key_frame"]],
    key=lambda x: x["timestamp"]
)
print(f"sample_token: {sample_token}")
print(f"关联 sample_data 数量: {len(sds)}")

if sds:
    ep_token = sds[0]["ego_pose_token"]
    anchor_ep = ego_pose_by_token[ep_token]
    anchor_ts = anchor_ep["timestamp"]

    # 取时间上最近的 8 条 ego_pose 作为"未来帧"
    future_eps = sorted(
        [ep for ep in ego_poses if ep["timestamp"] >= anchor_ts],
        key=lambda x: x["timestamp"]
    )[:8]

    print(f"截取 ego_pose: {len(future_eps)} 条")
    print(f"时间跨度: {(future_eps[-1]['timestamp'] - future_eps[0]['timestamp'])/1e6:.2f} 秒")

    # 存 fixture
    with open(fixture_dir / "sample_ego_poses.json", "w") as f:
        json.dump(future_eps, f, indent=2)
    print("✅ sample_ego_poses.json 已写入")

    # 同时存一个最小 sample fixture
    sample_fixture = {
        "sample_token": sample_token,
        "ego_pose_token": ep_token,
        "timestamp": anchor_ts,
        "keyframe_index": 0
    }
    with open(fixture_dir / "sample_scene_no_signal.json", "w") as f:
        json.dump(sample_fixture, f, indent=2)
    print("✅ sample_scene_no_signal.json 已写入")