import os
def create_dummy_repo():
    os.makedirs("my_repo", exist_ok=True)
    with open("my_repo/main.py", "w") as f:
        f.write("import numpy as np\nimport model\nnp.random.seed(42)\nmodel.run()")
    with open("my_repo/model.py", "w") as f:
        f.write("import numpy as np\ndef run():\n    print(np.random.rand())")
    with open("my_repo/bad.py", "w") as f:
        f.write("import numpy as np\nprint(np.random.rand())")

create_dummy_repo()