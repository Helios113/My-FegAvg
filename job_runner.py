import subprocess

commands = [
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "NEW_DATA_M/a/fed/experiment.yaml",
    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "NEW_DATA_M/a/locl/experiment.yaml",

    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",top
    #     "NEW_DATA_M/a/non/experiment.yaml",

    # ]
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "NEW_DATA/joshua/fed/experiment.yaml",

    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "NEW_DATA/emmanuelF3_2/fed/experiment.yaml",

    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "NEW_DATA/emmanuelF3_2/fed1/experiment.yaml",

    # ],
    # [
    #     "python",
    #     "/home/preslav/Projects/My-FegAvg/server.py",
    #     "--paramPath",
    #     "T1/c0/fed/ex.yaml",

    # ],
    [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "T2/g1-1/fed/ex.yaml",

    ],
    [
        "python",
        "/home/preslav/Projects/My-FegAvg/server.py",
        "--paramPath",
        "T2/g1-1/non/ex.yaml",

    ]
]

for cmd in commands:
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    
    
# try devices frm different locations like, gyro1, mag2 and acc3 to fit with the eagle, human, seal analogy