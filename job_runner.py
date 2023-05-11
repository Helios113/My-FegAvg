import subprocess

commands = [
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "E6/p/fed/experiment.yaml",
    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "NEW_DATA/g/fed/experiment.yaml",
    # ],
   [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "NEW_DATA/g/non/experiment.yaml",
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