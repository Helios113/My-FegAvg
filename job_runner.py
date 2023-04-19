import subprocess

commands = [
    [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "results/test1/fed/experiement.yaml",
    ],
    [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "results/test1/non_fed/experiement.yaml",
    ],
]

for cmd in commands:
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
