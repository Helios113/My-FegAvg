import subprocess

commands = [
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "E6/p/fed/experiment.yaml",
    # ],
    [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "NEW_DATA/i3/fed/experiment.yaml",
    ],
    [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "NEW_DATA/i3/non/experiment.yaml",
    ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "E6/d/fed/experiment.yaml",
    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "E6/d/non/experiment.yaml",
    # ],
]

for cmd in commands:
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

# 9 and 11d not running