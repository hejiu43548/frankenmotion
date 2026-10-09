"""Nine distinct additions; units refer to human SMPL kinematics, not G1 scale."""

import torch, math

NEW = {
    "squat": dict(
        pattern=r"\bsquat",
        unit="m pelvis drop",
        bounds=[0.10, 0.45],
        prompt="A person squats down with both feet flat on the ground and then stands upright again.",
    ),
    "bow": dict(
        pattern=r"\bbow(?:ing|s)?\b",
        unit="rad forward torso angle",
        bounds=[0.20, 0.90],
        prompt="A person bows forward politely with straight legs and then returns upright.",
    ),
    "clap": dict(
        pattern=r"\bclap",
        unit="m hand separation amplitude",
        bounds=[0.20, 0.70],
        prompt="A person stands in place and repeatedly claps both hands together in front of the chest.",
    ),
    "point": dict(
        pattern=r"\bpoint(?:ing|s)?\b",
        unit="m right wrist forward reach",
        bounds=[0.25, 0.65],
        prompt="A person stands still, extends the right arm forward to point at something, then lowers the arm.",
    ),
    "stretch": dict(
        pattern=r"\bstretch",
        unit="m bilateral wrist elevation relative pelvis",
        bounds=[0.50, 1.00],
        prompt="A person stretches both arms high overhead, holds the stretch, and lowers both arms again.",
    ),
    "twist": dict(
        pattern=r"\btwist|\brotat.*(?:torso|waist)|(?:torso|waist).*\brotat",
        unit="rad torso twist relative pelvis",
        bounds=[0.15, 0.65],
        prompt="A person stands with feet planted and twists the upper body left and right at the waist.",
    ),
    "march": dict(
        pattern=r"\bmarch",
        unit="m ankle lift from standing",
        bounds=[0.06, 0.22],
        prompt="A person marches in place, alternately raising the left and right knees, and then stands still.",
    ),
    "jog": dict(
        pattern=r"\bjog|\brun(?:ning)?\b",
        unit="m/s root path speed",
        bounds=[0.80, 2.00],
        prompt="A person jogs forward with a steady relaxed running gait.",
    ),
    "arm_circle": dict(
        pattern=r"(?:arm|shoulder).*circl|circl.*(?:arm|shoulder)",
        unit="m wrist vertical excursion",
        bounds=[0.60, 1.30],
        prompt="A person stands still and makes large continuous circles with both arms at the shoulders.",
    ),
}


def quantities(p):
    """p is canonical B,T,24,3; all measurements differentiable."""
    root = p[:, :, 0]
    torso = (p[:, :, 16] + p[:, :, 17]) / 2 - root
    lateral = p[:, :, 16, :2] - p[:, :, 17, :2]
    pelvis = p[:, :, 1, :2] - p[:, :, 2, :2]
    angle = torch.atan2(lateral[..., 1], lateral[..., 0]) - torch.atan2(
        pelvis[..., 1], pelvis[..., 0]
    )
    angle = torch.atan2(angle.sin(), angle.cos())
    hands = torch.linalg.vector_norm(p[:, :, 20] - p[:, :, 21], dim=-1)
    feet = p[:, :, [7, 8], 2]
    feet = feet - feet[:, :1]
    return {
        "squat": root[:, :5, 2].mean(1) - root[:, :, 2].amin(1),
        "bow": torch.atan2(torso[..., 0], torso[..., 2]).amax(1),
        "clap": hands.amax(1) - hands.amin(1),
        "point": (p[:, :, 21, 0] - root[:, :, 0]).amax(1),
        "stretch": ((p[:, :, 20, 2] + p[:, :, 21, 2]) / 2 - root[:, :, 2]).amax(1),
        "twist": angle.abs().amax(1),
        "march": feet.amax(1).mean(1),
        "jog": torch.linalg.vector_norm(root[:, 1:, :2] - root[:, :-1, :2], dim=-1).sum(
            1
        )
        / ((p.shape[1] - 1) / 20),
        "arm_circle": (
            p[:, :, [20, 21], 2].amax(1) - p[:, :, [20, 21], 2].amin(1)
        ).mean(1),
    }
